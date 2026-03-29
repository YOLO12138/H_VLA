"""
ManiFlow Multi-Modal Policy (v2 — Pointmap)
===========================================
Multi-camera RGB + dense XYZ pointmap + robot state.

Camera keys and image resolution are derived from shape_meta.obs:
  - Keys ending with "_rgb"   → RGB cameras
  - Keys ending with "_ptmap" → pointmap cameras
  - Resolution from the shape [3, H, W]

Visual token counts are computed dynamically via encoder probe forward
passes, so changing backbone or resolution auto-updates visual_cond_len.

State injection is configurable: "token" (state as extra token in visual
sequence) or "timestep_concat" (state concatenated with timestep embedding
in DiTX).

Language injection is also configurable: "token" (language tokens
concatenated with visual tokens) or "timestep_concat" (sentence embedding
concatenated with timestep embedding).

Inherits flow-matching / consistency-training / EMA / ODE from
ManiFlowTransformerPointcloudPolicy.
"""

from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import reduce
from termcolor import cprint

from maniflow.model.common.normalizer import LinearNormalizer
from maniflow.policy.base_policy import BasePolicy
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy
from maniflow.common.pytorch_util import dict_apply
from maniflow.common.model_util import print_params
from maniflow.model.diffusion.ditx import DiTX
from maniflow.model.common.sample_util import *

# ---------------------------------------------------------------------------
# Multi-modal observation encoder
# ---------------------------------------------------------------------------


class MultiModalObsEncoder(nn.Module):
    """
    Siamese 2D encoder for N-camera RGB + N-camera XYZ pointmap.

    Camera keys, image resolution, pointmap bounds, and state MLP hidden dim
    are all configurable.  Token counts per camera are determined dynamically
    via probe forward passes during __init__.

    Two encoder_type modes:
      "spatial": features_only=True → spatial tokens per camera
      "pooled":  num_classes=0      → 1 global token per camera

    Returns (vis_cond_no_state, state_emb).
    """

    def __init__(
        self,
        rgb_backbone: str = "vit_base_patch16_clip_224.openai",
        ptmap_backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        obs_feature_dim: int = 768,
        state_dim: int = 28,
        propr_mask_prob: float = 0.0,
        encoder_type: str = "spatial",
        use_ptmap: bool = True,
        rgb_keys: Optional[List[str]] = None,
        ptmap_keys: Optional[List[str]] = None,
        img_size: int = 224,
        ptmap_min: Optional[List[float]] = None,
        ptmap_max: Optional[List[float]] = None,
        state_mlp_hidden_dim: int = 256,
        rgb_mean: Optional[List[float]] = None,
        rgb_std: Optional[List[float]] = None,
        random_crop_ratio: float = 0.95,
        random_rotation_degrees: float = 5.0,
        color_jitter: Optional[Dict] = None,
    ):
        super().__init__()

        assert encoder_type in ("spatial", "pooled"), f"encoder_type must be 'spatial' or 'pooled', got '{encoder_type}'"

        self.obs_feature_dim = obs_feature_dim
        self.propr_mask_prob = propr_mask_prob
        self.encoder_type = encoder_type
        self.use_ptmap = use_ptmap
        self.img_size = img_size

        # Camera keys (derived from shape_meta by the policy, or defaults)
        self.rgb_keys = rgb_keys or ["head_rgb", "wrist_r_rgb", "wrist_l_rgb"]
        self.ptmap_keys = ptmap_keys or ["head_ptmap", "wrist_r_ptmap", "wrist_l_ptmap"]
        self.n_rgb_cams = len(self.rgb_keys)
        self.n_ptmap_cams = len(self.ptmap_keys)

        # Pointmap normalization bounds
        _ptmap_min = ptmap_min or [-1.0, -1.0, 0.0]
        _ptmap_max = ptmap_max or [1.0, 1.0, 2.0]

        # ---- RGB backbone ----
        if encoder_type == "spatial":
            self.rgb_encoder = timm.create_model(
                f"hf_hub:timm/{rgb_backbone}",
                pretrained=True,
                features_only=True,
                out_indices=[-1],
            )
            with torch.no_grad():
                probe = self.rgb_encoder(torch.zeros(1, 3, img_size, img_size))[0]  # [1, C, H, W]
                C_rgb = probe.shape[1]
                self.tokens_per_rgb_cam = probe.shape[2] * probe.shape[3]
        else:
            self.rgb_encoder = timm.create_model(
                f"hf_hub:timm/{rgb_backbone}",
                pretrained=True,
                num_classes=0,
            )
            with torch.no_grad():
                probe = self.rgb_encoder(torch.zeros(1, 3, img_size, img_size))  # [1, C]
                C_rgb = probe.shape[1]
                self.tokens_per_rgb_cam = 1
        self.rgb_proj = nn.Linear(C_rgb, obs_feature_dim) if C_rgb != obs_feature_dim else nn.Identity()

        # ---- Pointmap backbone ----
        if use_ptmap:
            if encoder_type == "spatial":
                self.ptmap_encoder = timm.create_model(
                    ptmap_backbone,
                    pretrained=True,
                    features_only=True,
                    out_indices=[-1],
                )
                with torch.no_grad():
                    probe = self.ptmap_encoder(torch.zeros(1, 3, img_size, img_size))[0]
                    C_pt = probe.shape[1]
                    self.tokens_per_ptmap_cam = probe.shape[2] * probe.shape[3]
            else:
                self.ptmap_encoder = timm.create_model(
                    ptmap_backbone,
                    pretrained=True,
                    num_classes=0,
                )
                with torch.no_grad():
                    probe = self.ptmap_encoder(torch.zeros(1, 3, img_size, img_size))
                    C_pt = probe.shape[1]
                    self.tokens_per_ptmap_cam = 1
            self.ptmap_proj = nn.Linear(C_pt, obs_feature_dim) if C_pt != obs_feature_dim else nn.Identity()

            # Pointmap normalization buffers
            self.register_buffer(
                "ptmap_min",
                torch.tensor(_ptmap_min, dtype=torch.float32).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "ptmap_max",
                torch.tensor(_ptmap_max, dtype=torch.float32).view(1, 3, 1, 1),
            )
        else:
            self.tokens_per_ptmap_cam = 0

        # ---- RGB normalization: explicit config > auto-detect from backbone > CLIP defaults ----
        _clip_mean = [0.48145466, 0.4578275, 0.40821073]
        _clip_std = [0.26862954, 0.26130258, 0.27577711]
        if rgb_mean is not None:
            _rgb_mean = rgb_mean
        elif hasattr(self.rgb_encoder, "pretrained_cfg"):
            _rgb_mean = list(self.rgb_encoder.pretrained_cfg.get("mean", _clip_mean))
        else:
            _rgb_mean = _clip_mean
        if rgb_std is not None:
            _rgb_std = rgb_std
        elif hasattr(self.rgb_encoder, "pretrained_cfg"):
            _rgb_std = list(self.rgb_encoder.pretrained_cfg.get("std", _clip_std))
        else:
            _rgb_std = _clip_std
        self.register_buffer("img_mean", torch.tensor(_rgb_mean).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(_rgb_std).view(1, 3, 1, 1))
        cprint(f"[MultiModalObsEncoder] RGB norm mean={_rgb_mean} std={_rgb_std}", "cyan")

        # ---- Image augmentation (RGB only, applied during training) ----
        _cj = color_jitter or {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}
        crop_size = int(img_size * random_crop_ratio)
        self.rgb_transform = nn.Sequential(
            T.RandomCrop(size=crop_size),
            T.Resize(size=img_size, antialias=True),
            T.RandomRotation(degrees=[-random_rotation_degrees, random_rotation_degrees], expand=False),
            T.ColorJitter(**_cj),
        )

        # ---- State MLP ----
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, state_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(state_mlp_hidden_dim, obs_feature_dim),
        )

        cprint(
            f"[MultiModalObsEncoder] encoder_type={encoder_type} use_ptmap={use_ptmap} "
            f"rgb={rgb_backbone} ({self.n_rgb_cams} cams, {self.tokens_per_rgb_cam} tok/cam) "
            f"ptmap={ptmap_backbone if use_ptmap else 'disabled'}"
            f"{f' ({self.n_ptmap_cams} cams, {self.tokens_per_ptmap_cam} tok/cam)' if use_ptmap else ''} "
            f"img_size={img_size} obs_feature_dim={obs_feature_dim}",
            "cyan",
        )

    @property
    def visual_token_count(self) -> int:
        """Total visual tokens (without state token)."""
        count = self.n_rgb_cams * self.tokens_per_rgb_cam
        if self.use_ptmap:
            count += self.n_ptmap_cams * self.tokens_per_ptmap_cam
        return count

    def _aggregate_spatial(self, feat_4d: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] → [B, H*W, C]"""
        _B, _C, _H, _W = feat_4d.shape
        return feat_4d.flatten(2).transpose(1, 2)

    def output_shape(self) -> int:
        return self.obs_feature_dim

    def forward(self, obs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (vis_cond_no_state, state_emb).
        """
        B = obs["agent_pos"].shape[0]

        # ---- 1. RGB path ----
        rgb_imgs = torch.stack([obs[k] for k in self.rgb_keys], dim=1)
        rgb_flat = rgb_imgs.reshape(B * self.n_rgb_cams, 3, self.img_size, self.img_size)
        if self.training:
            rgb_flat = self.rgb_transform(rgb_flat)
        rgb_flat = (rgb_flat - self.img_mean) / self.img_std
        rgb_feat_raw = self.rgb_encoder(rgb_flat)
        if self.encoder_type == "spatial":
            rgb_spatial = self._aggregate_spatial(rgb_feat_raw[0])
        else:
            rgb_spatial = rgb_feat_raw.unsqueeze(1)
        rgb_spatial = self.rgb_proj(rgb_spatial)
        L_rgb = rgb_spatial.shape[1]
        rgb_tokens = rgb_spatial.reshape(B, self.n_rgb_cams * L_rgb, self.obs_feature_dim)

        # ---- 2. Pointmap path ----
        if self.use_ptmap:
            ptmaps_raw = torch.stack([obs[k] for k in self.ptmap_keys], dim=1)
            ptmaps_raw = ptmaps_raw.reshape(B * self.n_ptmap_cams, 3, self.img_size, self.img_size)
            ptmaps_clipped = torch.clamp(ptmaps_raw, self.ptmap_min, self.ptmap_max)
            ptmaps_norm = 2.0 * (ptmaps_clipped - self.ptmap_min) / (self.ptmap_max - self.ptmap_min) - 1.0
            ptmap_feat_raw = self.ptmap_encoder(ptmaps_norm)
            if self.encoder_type == "spatial":
                ptmap_spatial = self._aggregate_spatial(ptmap_feat_raw[0])
            else:
                ptmap_spatial = ptmap_feat_raw.unsqueeze(1)
            ptmap_spatial = self.ptmap_proj(ptmap_spatial)
            L_pt = ptmap_spatial.shape[1]
            ptmap_tokens = ptmap_spatial.reshape(B, self.n_ptmap_cams * L_pt, self.obs_feature_dim)

        # ---- 3. State path ----
        agent_pos = obs["agent_pos"]
        if self.training and self.propr_mask_prob > 0:
            mask = (torch.rand(B, 1, device=agent_pos.device) > self.propr_mask_prob).float()
            agent_pos = agent_pos * mask
        state_emb = self.state_mlp(agent_pos)

        if self.use_ptmap:
            vis_cond_no_state = torch.cat([rgb_tokens, ptmap_tokens], dim=1)
        else:
            vis_cond_no_state = rgb_tokens
        return vis_cond_no_state, state_emb


# ---------------------------------------------------------------------------
# Multi-modal policy
# ---------------------------------------------------------------------------


class ManiFlowTransformerMultiModalPolicy(ManiFlowTransformerPointcloudPolicy):
    """
    Multi-camera RGB + pointmap policy with configurable camera keys, image
    resolution, state/language injection modes, and dynamic visual token counts.
    """

    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps=None,
        obs_as_global_cond: bool = True,
        diffusion_timestep_embed_dim: int = 256,
        diffusion_target_t_embed_dim: int = 256,
        visual_cond_len: Optional[int] = None,
        n_layer: int = 12,
        n_head: int = 8,
        n_emb: int = 768,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        block_type: str = "DiTX",
        encoder_output_dim: int = 768,
        rgb_backbone: str = "vit_base_patch16_clip_224.openai",
        ptmap_backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        propr_mask_prob: float = 0.0,
        encoder_type: str = "spatial",
        state_injection_mode: str = "token",
        lang_injection_mode: str = "token",
        pre_norm_modality: bool = False,
        language_conditioned: bool = False,
        flow_batch_ratio: float = 0.75,
        consistency_batch_ratio: float = 0.25,
        denoise_timesteps: int = 10,
        sample_t_mode_flow: str = "beta",
        sample_t_mode_consistency: str = "discrete",
        sample_dt_mode_consistency: str = "uniform",
        sample_target_t_mode: str = "relative",
        use_pc_color: bool = True,
        use_ptmap: bool = True,
        state_mlp_hidden_dim: int = 256,
        ptmap_min: Optional[List[float]] = None,
        ptmap_max: Optional[List[float]] = None,
        rgb_mean: Optional[List[float]] = None,
        rgb_std: Optional[List[float]] = None,
        random_crop_ratio: float = 0.95,
        random_rotation_degrees: float = 5.0,
        color_jitter: Optional[Dict] = None,
        **kwargs,
    ):
        super(ManiFlowTransformerPointcloudPolicy, self).__init__()
        self.use_pc_color = use_pc_color

        action_shape = shape_meta["action"]["shape"]
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        state_dim = shape_meta["obs"]["agent_pos"]["shape"][0]

        # Derive camera keys from shape_meta
        rgb_keys = sorted(k for k in shape_meta["obs"] if k.endswith("_rgb"))
        ptmap_keys = sorted(k for k in shape_meta["obs"] if k.endswith("_ptmap"))
        if not rgb_keys:
            raise ValueError("shape_meta.obs must contain at least one key ending with '_rgb'")

        # Derive image resolution from shape_meta
        img_size = shape_meta["obs"][rgb_keys[0]]["shape"][-1]

        obs_encoder = MultiModalObsEncoder(
            rgb_backbone=rgb_backbone,
            ptmap_backbone=ptmap_backbone,
            obs_feature_dim=encoder_output_dim,
            state_dim=state_dim,
            propr_mask_prob=propr_mask_prob,
            encoder_type=encoder_type,
            use_ptmap=use_ptmap,
            rgb_keys=rgb_keys,
            ptmap_keys=ptmap_keys,
            img_size=img_size,
            ptmap_min=ptmap_min,
            ptmap_max=ptmap_max,
            state_mlp_hidden_dim=state_mlp_hidden_dim,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            random_crop_ratio=random_crop_ratio,
            random_rotation_degrees=random_rotation_degrees,
            color_jitter=color_jitter,
        )
        obs_feature_dim = obs_encoder.output_shape()

        # Compute visual_cond_len from encoder if not explicitly overridden
        if visual_cond_len is None:
            base_vis_len = obs_encoder.visual_token_count
            visual_cond_len = base_vis_len if state_injection_mode == "timestep_concat" else base_vis_len + 1

        model = DiTX(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=obs_feature_dim,
            visual_cond_len=visual_cond_len,
            diffusion_timestep_embed_dim=diffusion_timestep_embed_dim,
            diffusion_target_t_embed_dim=diffusion_target_t_embed_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            block_type=block_type,
            pre_norm_modality=pre_norm_modality,
            language_conditioned=language_conditioned,
            state_injection_mode=state_injection_mode,
            lang_injection_mode=lang_injection_mode,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.language_conditioned = language_conditioned
        self.state_injection_mode = state_injection_mode
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps
        self.flow_batch_ratio = flow_batch_ratio
        self.consistency_batch_ratio = consistency_batch_ratio
        assert flow_batch_ratio + consistency_batch_ratio == 1.0, "Sum of batch ratios must equal 1.0"
        self.denoise_timesteps = denoise_timesteps
        self.sample_t_mode_flow = sample_t_mode_flow
        self.sample_t_mode_consistency = sample_t_mode_consistency
        self.sample_dt_mode_consistency = sample_dt_mode_consistency
        self.sample_target_t_mode = sample_target_t_mode
        assert self.sample_target_t_mode in (
            "absolute",
            "relative",
        ), "sample_target_t_mode must be 'absolute' or 'relative'"

        cprint("[ManiFlowTransformerMultiModalPolicy] Initialized", "yellow")
        cprint(f"  obs_feature_dim     : {obs_feature_dim}", "yellow")
        cprint(f"  visual_cond_len     : {visual_cond_len}", "yellow")
        cprint(f"  state_injection_mode: {state_injection_mode}", "yellow")
        cprint(f"  lang_injection_mode : {lang_injection_mode}", "yellow")
        cprint(f"  action_dim          : {action_dim}", "yellow")
        cprint(f"  horizon             : {horizon}", "yellow")
        cprint(f"  rgb_keys            : {rgb_keys}", "yellow")
        cprint(f"  ptmap_keys          : {ptmap_keys}", "yellow")
        cprint(f"  img_size            : {img_size}", "yellow")
        print_params(self)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Override: MultiModalObsEncoder returns (vis_cond_no_state, state_emb), not a single tensor.
        Build vis_cond and state_emb per state_injection_mode, then call parent's sampling flow.
        """
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        To = self.n_obs_steps
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        lang_cond = None
        if self.language_conditioned:
            lang_cond = nobs.get("task_name", None)
            assert lang_cond is not None, "Language goal is required"

        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]).to(device),
        )
        vis_cond_no_state, state_emb = self.obs_encoder(this_nobs)

        if self.state_injection_mode == "token":
            vis_cond = torch.cat([vis_cond_no_state, state_emb.unsqueeze(1)], dim=1)
            state_emb_arg = None
        else:
            vis_cond = vis_cond_no_state
            state_emb_arg = state_emb

        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        nsample = self.conditional_sample(
            cond_data,
            vis_cond=vis_cond,
            lang_cond=lang_cond,
            state_emb=state_emb_arg,
            **self.kwargs,
        )
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {
            "action": action,
            "action_pred": action_pred,
        }

    def compute_loss(self, batch, ema_model=None, **kwargs):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"]).to(self.device)

        batch_size = nactions.shape[0]
        lang_cond = None
        if self.language_conditioned:
            lang_cond = nobs.get("task_name", None)
            assert lang_cond is not None, "Language goal is required"

        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]).to(self.device),
        )
        vis_cond_no_state, state_emb = self.obs_encoder(this_nobs)

        if self.state_injection_mode == "token":
            vis_cond = torch.cat([vis_cond_no_state, state_emb.unsqueeze(1)], dim=1)
            state_emb_slice = None
        else:
            vis_cond = vis_cond_no_state
            state_emb_slice = state_emb

        vis_cond = vis_cond.reshape(batch_size, -1, self.obs_feature_dim)

        # ---- feature std (debug) ----
        feature_std = vis_cond.float().std().item()
        feature_dim_std = vis_cond.float().std(dim=(0, 1)).mean().item()

        flow_batchsize = int(batch_size * self.flow_batch_ratio)
        consistency_batchsize = int(batch_size * self.consistency_batch_ratio)

        # ---- flow matching loss ----
        flow_target_dict = self.get_flow_velocity(
            nactions[:flow_batchsize],
            vis_cond=vis_cond[:flow_batchsize],
            lang_cond=lang_cond[:flow_batchsize] if lang_cond is not None else None,
        )
        v_flow_pred = self.model(
            sample=flow_target_dict["x_t"],
            timestep=flow_target_dict["t"].squeeze(),
            target_t=flow_target_dict["target_t"].squeeze(),
            vis_cond=vis_cond[:flow_batchsize],
            lang_cond=flow_target_dict["lang_cond"][:flow_batchsize] if lang_cond is not None else None,
            state_emb=state_emb_slice[:flow_batchsize] if state_emb_slice is not None else None,
        )
        v_flow_pred_magnitude = torch.sqrt(torch.mean(v_flow_pred**2)).item()

        # ---- consistency training loss ----
        consistency_target_dict = self.get_consistency_velocity(
            nactions[flow_batchsize : flow_batchsize + consistency_batchsize],
            vis_cond=vis_cond[flow_batchsize : flow_batchsize + consistency_batchsize],
            lang_cond=(
                lang_cond[flow_batchsize : flow_batchsize + consistency_batchsize] if lang_cond is not None else None
            ),
            ema_model=ema_model,
            state_emb=(
                state_emb_slice[flow_batchsize : flow_batchsize + consistency_batchsize]
                if state_emb_slice is not None
                else None
            ),
        )
        v_ct_pred = self.model(
            sample=consistency_target_dict["x_t"],
            timestep=consistency_target_dict["t"].squeeze(),
            target_t=consistency_target_dict["target_t"].squeeze(),
            vis_cond=vis_cond[flow_batchsize : flow_batchsize + consistency_batchsize],
            lang_cond=(
                lang_cond[flow_batchsize : flow_batchsize + consistency_batchsize] if lang_cond is not None else None
            ),
            state_emb=(
                state_emb_slice[flow_batchsize : flow_batchsize + consistency_batchsize]
                if state_emb_slice is not None
                else None
            ),
        )
        v_ct_pred_magnitude = torch.sqrt(torch.mean(v_ct_pred**2)).item()

        # ---- combine losses ----
        loss = 0.0
        v_flow_target = flow_target_dict["v_target"]
        loss_flow = F.mse_loss(v_flow_pred, v_flow_target, reduction="none")
        loss_flow = reduce(loss_flow, "b ... -> b (...)", "mean")
        loss += loss_flow.mean()
        loss_flow = loss_flow.mean().item()

        v_ct_target = consistency_target_dict["v_target"]
        loss_ct = F.mse_loss(v_ct_pred, v_ct_target, reduction="none")
        loss_ct = reduce(loss_ct, "b ... -> b (...)", "mean")
        loss += loss_ct.mean()
        loss_ct = loss_ct.mean().item()

        loss = loss.mean()
        loss_dict = {
            "loss_flow": loss_flow,
            "loss_ct": loss_ct,
            "v_flow_pred_magnitude": v_flow_pred_magnitude,
            "v_ct_pred_magnitude": v_ct_pred_magnitude,
            "bc_loss": loss.item(),
            "feature_std": feature_std,
            "feature_dim_std": feature_dim_std,
        }
        return loss, loss_dict
