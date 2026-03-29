"""
ManiFlow Pointcloud Framework
Wraps ManiFlowTransformerPointcloudPolicy for training within hvla infrastructure.

Input batch format (from yamrobot_collate_fn):
    {
        "obs": {
            "point_cloud": Tensor [B, T_obs, N, 6]   XYZRGB, pre-sampled to max_points
            "agent_pos":   Tensor [B, T_obs, state_dim]  q99-normalised state
        },
        "action": Tensor [B, T_a, action_dim]           q99-normalised actions
        "lang":   List[str]                             task instructions (unused if language_conditioned=False)
    }

Output of forward():
    {"action_loss": Tensor scalar}   compatible with VLATrainer._train_step
"""

import copy
from typing import Optional

import torch

from hvla.model.framework.base_framework import baseframework
from hvla.model.tools import FRAMEWORK_REGISTRY
from hvla.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class _PCEncoderCfg:
    """
    Thin config proxy: supports attribute access, mutation, .get(), and **-unpacking.
    DP3Encoder expects a config object that supports all four operations.
    """

    def __init__(self, d: dict):
        self.__dict__.update(d)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def keys(self):
        return (k for k in self.__dict__ if not k.startswith("_"))

    def __iter__(self):
        return self.keys()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __len__(self):
        return len(self.__dict__)


@FRAMEWORK_REGISTRY.register("ManiFlowPointcloud")
class ManiFlowPointcloudFramework(baseframework):
    """
    hvla framework wrapper around ManiFlowTransformerPointcloudPolicy.

    Consistency Flow training (75 % flow + 25 % consistency) with EMA shadow model.
    Data is assumed to be already q99-normalised by the builder / collate_fn,
    so the policy's LinearNormalizer is set to identity.

    EMA is updated at the start of every forward() call.  Because gradient
    accumulation is typically 1 for this pipeline the update cadence matches
    the optimizer step cadence.  If gradient_accumulation_steps > 1 the EMA
    will be updated slightly more often than ideal, but converges to the same
    result at training scales of 10 K+ steps.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------ #
        # Import here to avoid mandatory ManiFlow dep for other frameworks    #
        # ------------------------------------------------------------------ #
        from maniflow.policy.maniflow_pointcloud_policy import (
            ManiFlowTransformerPointcloudPolicy,
        )
        from maniflow.model.diffusion.ema_model import EMAModel
        from maniflow.model.common.normalizer import (
            LinearNormalizer,
            SingleFieldLinearNormalizer,
        )
        from omegaconf import OmegaConf

        fw = config.framework  # shorthand

        # ------------------------------------------------------------------ #
        # shape_meta: convert from OmegaConf to plain Python                 #
        # ------------------------------------------------------------------ #
        shape_meta = OmegaConf.to_container(fw.shape_meta, resolve=True)

        # ------------------------------------------------------------------ #
        # pointcloud_encoder_cfg: needs .get() + attr access + **-unpacking  #
        # ------------------------------------------------------------------ #
        pc_cfg_dict = OmegaConf.to_container(fw.pointcloud_encoder_cfg, resolve=True)
        # Fill in encoder_output_dim alias used in maniflow configs
        pc_cfg_dict.setdefault("out_channels", fw.encoder_output_dim)
        pc_cfg_dict.setdefault("num_points", fw.visual_cond_len)
        pc_cfg = _PCEncoderCfg(pc_cfg_dict)

        # ------------------------------------------------------------------ #
        # Build policy                                                        #
        # ------------------------------------------------------------------ #
        self.policy = ManiFlowTransformerPointcloudPolicy(
            shape_meta=shape_meta,
            horizon=fw.horizon,
            n_action_steps=fw.n_action_steps,
            n_obs_steps=fw.n_obs_steps,
            num_inference_steps=fw.num_inference_steps,
            obs_as_global_cond=fw.obs_as_global_cond,
            n_layer=fw.n_layer,
            n_head=fw.n_head,
            n_emb=fw.n_emb,
            visual_cond_len=fw.visual_cond_len,
            qkv_bias=fw.get("qkv_bias", False),
            qk_norm=fw.get("qk_norm", False),
            encoder_type=fw.encoder_type,
            encoder_output_dim=fw.encoder_output_dim,
            use_pc_color=fw.use_pc_color,
            pointnet_type=fw.pointnet_type,
            pointcloud_encoder_cfg=pc_cfg,
            downsample_points=fw.downsample_points,
            pre_norm_modality=fw.get("pre_norm_modality", False),
            language_conditioned=fw.get("language_conditioned", False),
            flow_batch_ratio=fw.get("flow_batch_ratio", 0.75),
            consistency_batch_ratio=fw.get("consistency_batch_ratio", 0.25),
            sample_t_mode_flow=fw.get("sample_t_mode_flow", "beta"),
            sample_t_mode_consistency=fw.get("sample_t_mode_consistency", "discrete"),
            sample_dt_mode_consistency=fw.get("sample_dt_mode_consistency", "uniform"),
            sample_target_t_mode=fw.get("sample_target_t_mode", "relative"),
            denoise_timesteps=fw.get("denoise_timesteps", 10),
            diffusion_timestep_embed_dim=fw.get("diffusion_timestep_embed_dim", 128),
            diffusion_target_t_embed_dim=fw.get("diffusion_target_t_embed_dim", 128),
        )

        # ------------------------------------------------------------------ #
        # Identity normalizer: data is pre-normalised by builder / collate   #
        # ------------------------------------------------------------------ #
        normalizer = LinearNormalizer()
        identity = SingleFieldLinearNormalizer.create_identity()
        for key in ("action", "point_cloud", "agent_pos"):
            normalizer.params_dict[key] = identity.params_dict
        self.policy.set_normalizer(normalizer)

        # ------------------------------------------------------------------ #
        # EMA shadow model                                                    #
        # ------------------------------------------------------------------ #
        ema_cfg = OmegaConf.to_container(fw.get("ema", {}), resolve=True) if fw.get("ema") else {}
        self.ema = EMAModel(
            model=copy.deepcopy(self.policy),
            update_after_step=ema_cfg.get("update_after_step", 0),
            inv_gamma=ema_cfg.get("inv_gamma", 1.0),
            power=ema_cfg.get("power", 0.75),
            min_value=ema_cfg.get("min_value", 0.0),
            max_value=ema_cfg.get("max_value", 0.9999),
        )

    # ---------------------------------------------------------------------- #
    # Forward                                                                 #
    # ---------------------------------------------------------------------- #
    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: dict with keys "obs", "action", "lang" produced by yamrobot_collate_fn.

        Returns:
            {"action_loss": scalar Tensor}
        """
        # EMA update is handled in the training loop (_train_step) after the
        # optimizer step, so it uses the freshly updated weights.
        loss, loss_dict = self.policy.compute_loss(batch, ema_model=self.ema.averaged_model)

        logger.debug(
            "loss=%.4f  flow=%.4f  ct=%.4f",
            loss.item(),
            loss_dict.get("loss_flow", 0.0),
            loss_dict.get("loss_ct", 0.0),
        )

        return {"action_loss": loss, **{f"mf_{k}": v for k, v in loss_dict.items()}}
