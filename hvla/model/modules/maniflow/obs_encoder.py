"""
Multi-modal observation encoder (standalone, no maniflow dependency).

Siamese 2D encoder for N-camera RGB + N-camera XYZ pointmap + robot state.
Extracted from maniflow.policy.maniflow_multimodal_policy.MultiModalObsEncoder.
"""

from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from termcolor import cprint


class MultiModalObsEncoder(nn.Module):
    """
    Siamese 2D encoder for N-camera RGB + N-camera XYZ pointmap.

    Camera keys, image resolution, pointmap bounds, and state MLP hidden dim
    are all configurable.  Token counts per camera are determined dynamically
    via probe forward passes during __init__.

    Two encoder_type modes:
      "spatial": features_only=True -> spatial tokens per camera
      "pooled":  num_classes=0      -> 1 global token per camera

    Returns dict: {"rgb_tokens", "ptmap_tokens", "state_emb"}.
    """

    def __init__(
        self,
        rgb_backbone: str = "vit_base_patch16_clip_224.openai",
        ptmap_backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        state_dim: int = 28,
        propr_mask_prob: float = 0.0,
        encoder_type: str = "spatial",
        use_rgb: bool = True,
        use_ptmap: bool = True,
        rgb_keys: Optional[List[str]] = None,
        ptmap_keys: Optional[List[str]] = None,
        img_size: int = 224,
        ptmap_min: Optional[List[float]] = None,
        ptmap_max: Optional[List[float]] = None,
        rgb_mean: Optional[List[float]] = None,
        rgb_std: Optional[List[float]] = None,
        random_crop_ratio: float = 0.95,
        random_rotation_degrees: float = 5.0,
        color_jitter: Optional[Dict] = None,
    ):
        super().__init__()

        assert encoder_type in ("spatial", "pooled"), f"encoder_type must be 'spatial' or 'pooled', got '{encoder_type}'"
        assert use_rgb or use_ptmap, "At least one of use_rgb or use_ptmap must be True"

        self.propr_mask_prob = propr_mask_prob
        self.encoder_type = encoder_type
        self.use_rgb = use_rgb
        self.use_ptmap = use_ptmap
        self.img_size = img_size

        # Camera keys (derived from shape_meta by the policy, or defaults)
        self.rgb_keys = (rgb_keys or ["head_rgb", "wrist_r_rgb", "wrist_l_rgb"]) if use_rgb else []
        self.ptmap_keys = ptmap_keys or ["head_ptmap", "wrist_r_ptmap", "wrist_l_ptmap"]
        self.n_rgb_cams = len(self.rgb_keys)
        self.n_ptmap_cams = len(self.ptmap_keys)

        # Pointmap normalization bounds
        _ptmap_min = ptmap_min or [-1.0, -1.0, 0.0]
        _ptmap_max = ptmap_max or [1.0, 1.0, 2.0]

        # ---- RGB backbone ----
        if use_rgb:
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
            self.rgb_feature_dim = C_rgb
        else:
            self.tokens_per_rgb_cam = 0
            self.rgb_feature_dim = 0

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
            self.ptmap_feature_dim = C_pt

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

        # ---- RGB normalization & augmentation ----
        if use_rgb:
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

            _cj = color_jitter or {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}
            crop_size = int(img_size * random_crop_ratio)
            self.rgb_transform = nn.Sequential(
                T.RandomCrop(size=crop_size),
                T.Resize(size=img_size, antialias=True),
                T.RandomRotation(degrees=[-random_rotation_degrees, random_rotation_degrees], expand=False),
                T.ColorJitter(**_cj),
            )

        self.state_dim = state_dim
        # NOTE: state adaptor and per-modality LayerNorm are applied inside DiTX.

        cprint(
            f"[MultiModalObsEncoder] encoder_type={encoder_type} use_rgb={use_rgb} use_ptmap={use_ptmap} "
            f"rgb={rgb_backbone if use_rgb else 'disabled'}"
            f"{f' ({self.n_rgb_cams} cams, {self.tokens_per_rgb_cam} tok/cam)' if use_rgb else ''} "
            f"ptmap={ptmap_backbone if use_ptmap else 'disabled'}"
            f"{f' ({self.n_ptmap_cams} cams, {self.tokens_per_ptmap_cam} tok/cam)' if use_ptmap else ''} "
            f"img_size={img_size} state_dim={state_dim}",
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
        """[B, C, H, W] -> [B, H*W, C]"""
        _B, _C, _H, _W = feat_4d.shape
        return feat_4d.flatten(2).transpose(1, 2)

    def forward(self, obs: dict) -> dict:
        """
        Returns dict with per-modality raw features:
          "rgb_tokens":  (B, n_rgb_cams * L_rgb, C_rgb)   or None
          "ptmap_tokens": (B, n_ptmap_cams * L_pt, C_pt)  or None
          "state_emb":   (B, state_dim)
        """
        B = obs["agent_pos"].shape[0]

        # ---- 1. RGB path (backbone only, no projection) ----
        rgb_tokens = None
        if self.use_rgb:
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
            L_rgb = rgb_spatial.shape[1]
            rgb_tokens = rgb_spatial.reshape(B, self.n_rgb_cams * L_rgb, self.rgb_feature_dim)

        # ---- 2. Pointmap path (backbone only, no projection) ----
        ptmap_tokens = None
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
            L_pt = ptmap_spatial.shape[1]
            ptmap_tokens = ptmap_spatial.reshape(B, self.n_ptmap_cams * L_pt, self.ptmap_feature_dim)

        # ---- 3. State path (masking only, adaptor is in DiTX) ----
        agent_pos = obs["agent_pos"]
        if self.training and self.propr_mask_prob > 0:
            mask = (torch.rand(B, 1, device=agent_pos.device) > self.propr_mask_prob).float()
            agent_pos = agent_pos * mask

        return {
            "rgb_tokens": rgb_tokens,
            "ptmap_tokens": ptmap_tokens,
            "state_emb": agent_pos,
        }
