# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [suport topdowm processing, suport param from config].

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ColorJitter

from hvla.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from hvla.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from hvla.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from hvla.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag


def collate_fn(batch):
    return batch


# Camera order produced by LeRobotSingleDataset._pack_sample for AgiBotDataConfig:
#   primary cameras (no "wrist" in key name) come first, then wrist cameras.
#   For AgiBot: video.head → index 0, video.left_wrist → 1, video.right_wrist → 2
AGIBOT_CAMERA_NAMES = ["head", "left_wrist", "right_wrist"]


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a float32 tensor [C, H, W] in [0, 1]."""
    arr = np.array(img, dtype=np.float32) / 255.0   # [H, W, C]
    return torch.from_numpy(arr).permute(2, 0, 1)    # [C, H, W]


def maniflow_collate_fn(
    batch: list[dict],
    camera_names: list[str] = AGIBOT_CAMERA_NAMES,
) -> dict:
    """Collate a list of LeRobotSingleDataset samples into a ManiFlow-compatible batch.

    Each sample (from LeRobotSingleDataset.__getitem__) contains:
        "image":    List[PIL.Image]           – one image per camera, 224×224
        "lang":     str                       – task instruction string
        "action":   np.ndarray [T_a, 30] f16 – q99-normalised actions
        "state":    np.ndarray [T_s, 30] f16 – q99-normalised state (if include_state=True)

    Returns a dict compatible with ManiFlowTransformerImagePolicy.compute_loss:
        obs:
            <cam_name>: tensor [B, 1, 3, H, W] float32 in [0, 1]
            state:      tensor [B, T_s, 30]    float32          (if available)
        action:         tensor [B, T_a, 30]    float32
        lang:           List[str]
    """
    obs: dict[str, torch.Tensor] = {}

    # Images: each sample["image"] is a list of PIL Images, one per camera.
    # Unsqueeze T_obs dim (=1) so shape becomes [1, C, H, W], then stack to [B, 1, C, H, W].
    for cam_idx, cam_name in enumerate(camera_names):
        tensors = [_pil_to_tensor(s["image"][cam_idx]).unsqueeze(0) for s in batch]
        obs[cam_name] = torch.stack(tensors)  # [B, 1, C, H, W]

    # State (optional – only present when include_state=True in data_cfg)
    if "state" in batch[0]:
        state_arr = np.stack([np.asarray(s["state"], dtype=np.float32) for s in batch])
        obs["state"] = torch.from_numpy(state_arr)  # [B, T_s, 30]

    # Actions
    action_arr = np.stack([np.asarray(s["action"], dtype=np.float32) for s in batch])
    action = torch.from_numpy(action_arr)  # [B, T_a, 30]

    # Language
    lang = [s["lang"] for s in batch]

    return {"obs": obs, "action": action, "lang": lang}

# ---------------------------------------------------------------------------
# YAM robot (ZED stereo camera, lossless dual-row depth encoding)
# ---------------------------------------------------------------------------

# Camera names exposed by YAMDataConfig (video_keys order):
#   video.head_rgb, video.wrist_r_rgb, video.wrist_l_rgb, video.head_depth,
#   video.wrist_r_depth, video.wrist_l_depth
#
# _pack_sample() in LeRobotSingleDataset re-orders images as:
#   primary (non-"wrist") cameras first, then wrist cameras.
#   For YAM: [head_rgb, head_depth, wrist_r_rgb, wrist_l_rgb, wrist_r_depth, wrist_l_depth]
#            indices [0, 1, 2, 3, 4, 5]
#
# All cameras use --eye left (672×376 single left eye). No crop needed.
# See: build_lerobot_dataset_yam.py §1 NOTE.
YAM_CAMERA_NAMES = ["head_rgb", "wrist_r_rgb", "wrist_l_rgb"]
YAM_HEAD_RGB_IDX     = 0   # head_rgb     non-wrist → index 0
YAM_HEAD_DEPTH_IDX   = 1   # head_depth   non-wrist → index 1
YAM_WRIST_R_RGB_IDX  = 2   # wrist_r_rgb  wrist     → index 2
YAM_WRIST_L_RGB_IDX  = 3   # wrist_l_rgb  wrist     → index 3
YAM_WRIST_R_DEPTH_IDX = 4  # wrist_r_depth wrist    → index 4
YAM_WRIST_L_DEPTH_IDX = 5  # wrist_l_depth wrist    → index 5

# ManiFlow paper (Appendix, Point Cloud Augmentation):
#   "In real-world experiments, color jitter augmentation becomes essential for generalizing to
#    environment changes and preventing overfitting to specific lighting conditions. We apply the
#    same color jitter parameters as in image augmentation to the RGB in point clouds with 0.2
#    probability." (brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)
_YAM_COLOR_JITTER = ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)
_YAM_COLOR_JITTER_PROB = 0.2


@lru_cache(maxsize=8)
def _get_pixel_grid(H: int, W: int):
    """Cache pixel coordinate grids (uu, vv) for a given image resolution.

    Both _depth_to_xyzrgb and _depth_to_pointmap call meshgrid on every
    sample.  Since camera resolution is constant within a dataset, caching
    by (H, W) eliminates the repeated allocation.  The returned arrays are
    read-only (element-wise ops create new arrays, so there is no aliasing
    risk across workers — each worker process has its own lru_cache).
    """
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    return np.meshgrid(u, v)   # uu: (H, W), vv: (H, W)


@lru_cache(maxsize=8)
def _load_camera_info(path: str) -> dict:
    """Load and cache camera intrinsics from camera_info.json.

    Expected JSON fields: fx, fy, cx, cy, depth_scale (mm→m, default 1000.0).
    """
    with open(path) as f:
        info = json.load(f)
    return info


def _decode_lossless_depth_frame(pil_img: Image.Image) -> np.ndarray:
    """Decode a lossless dual-row depth frame back to uint16 depth (mm).

    Encoding (written by the YAM builder):
        frame shape: (2H, W, 3) uint8
        lo = frame[:H, :, 0]   lower 8 bits of depth
        hi = frame[H:, :, 0]   upper 8 bits of depth
        depth_mm = (hi.astype(uint16) << 8) | lo.astype(uint16)

    Args:
        pil_img: PIL Image decoded from head_depth.mp4 (2H rows, W cols, RGB).

    Returns:
        depth_mm: np.ndarray shape (H, W) dtype uint16, millimetres.
    """
    frame = np.array(pil_img, dtype=np.uint8)   # (2H, W, 3)
    H = frame.shape[0] // 2
    lo = frame[:H, :, 0].astype(np.uint16)
    hi = frame[H:, :, 0].astype(np.uint16)
    return (hi << 8) | lo                        # (H, W) uint16


def _depth_to_xyzrgb(
    depth_mm: np.ndarray,
    rgb: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 1000.0,
    max_depth_m: float = 3.0,
) -> np.ndarray:
    """Back-project a depth map to an XYZRGB point cloud.

    Args:
        depth_mm: (H, W) uint16 depth in millimetres.
        rgb:      (H, W, 3) float32 RGB in [0, 1].
        fx/fy/cx/cy: camera intrinsics.
        depth_scale: divisor converting mm → metres (default 1000.0).
        max_depth_m: discard points beyond this distance.

    Returns:
        xyzrgb: (N, 6) float32 array.  N is the number of valid pixels.
    """
    H, W = depth_mm.shape
    depth_m = depth_mm.astype(np.float32) / depth_scale

    uu, vv = _get_pixel_grid(H, W)

    z = depth_m
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z

    valid = (z > 0.0) & (z < max_depth_m)
    xyz = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    rgb_valid = rgb[valid]                          # (N, 3)
    return np.concatenate([xyz, rgb_valid], axis=-1)  # (N, 6)


def _depth_to_pointmap(
    depth_mm: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 1000.0,
    max_depth_m: float = 2.0,
    target_h: int = 224,
    target_w: int = 224,
) -> torch.Tensor:
    """Back-project depth to dense XYZ pointmap and resize to target size.

    Args:
        depth_mm: (H, W) uint16 depth in millimetres.
        fx/fy/cx/cy: camera intrinsics.
        depth_scale: divisor converting mm -> metres.
        max_depth_m: valid depth upper bound in metres.
        target_h/target_w: output pointmap resolution.

    Returns:
        pointmap: torch.Tensor [3, target_h, target_w], float32.
    """
    H, W = depth_mm.shape
    depth_m = depth_mm.astype(np.float32) / depth_scale

    uu, vv = _get_pixel_grid(H, W)

    z = depth_m
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z

    valid = (z > 0.01) & (z < max_depth_m)
    x[~valid] = 0.0
    y[~valid] = 0.0
    z[~valid] = 0.0

    ptmap = np.stack([x, y, z], axis=0).astype(np.float32)  # (3, H, W)
    ptmap_t = torch.from_numpy(ptmap).unsqueeze(0)          # (1, 3, H, W)
    ptmap_t = F.interpolate(ptmap_t, size=(target_h, target_w), mode="nearest")
    return ptmap_t.squeeze(0)                               # (3, target_h, target_w)


def yamrobot_collate_fn(
    batch: list[dict],
    camera_info: dict,
    max_points: int = 10_000,
    depth_cam_idx: int = YAM_HEAD_DEPTH_IDX,
    rgb_cam_idx: int = YAM_HEAD_RGB_IDX,
    rgb_camera_names: list[str] = YAM_CAMERA_NAMES,
    augment: bool = True,
) -> dict:
    """Collate YAM robot samples into a ManiFlowTransformerPointcloudPolicy batch.

    Each sample (from LeRobotSingleDataset.__getitem__) contains:
        "image":   List[PIL.Image]  — cameras in YAMDataConfig.video_keys order
                   [head_rgb, wrist_r_rgb, wrist_l_rgb, head_depth, wrist_r_depth, wrist_l_depth]
        "state":   np.ndarray [T_obs, 14]   q99-normalised YAM state
        "action":  np.ndarray [T_a,   14]   q99-normalised actions
        "lang":    str

    Returns a dict for ManiFlowTransformerPointcloudPolicy.compute_loss:
        obs:
            point_cloud: Tensor [B, T_obs, N, 6]   XYZRGB, pre-sampled to max_points
            agent_pos:   Tensor [B, T_obs, 14]      normalised state
        action:          Tensor [B, T_a, 14]
        lang:            List[str]

    Args:
        batch:         List of raw dataset samples.
        camera_info:   Dict with keys: fx, fy, cx, cy, depth_scale.
                       Load from {dataset_root}/meta/head_camera_info.json.
        max_points:    Random pre-sample limit before GPU FPS (default 10 000).
        depth_cam_idx: Index of the depth PIL image in sample["image"] list.
        rgb_cam_idx:   Index of the head RGB image used for colorising the cloud.
        rgb_camera_names: Camera names for the RGB images (for completeness, unused here).
        augment:       If True, apply color jitter to RGB with prob=0.2 (training).
                       Set to False for evaluation/inference.
    """
    fx = camera_info["fx"]
    fy = camera_info["fy"]
    cx = camera_info["cx"]
    cy = camera_info["cy"]
    depth_scale = camera_info.get("depth_scale", 1000.0)

    obs: dict[str, torch.Tensor] = {}

    # ---- Build point clouds ------------------------------------------------
    pc_list = []
    for s in batch:
        depth_pil = s["image"][depth_cam_idx]
        rgb_pil   = s["image"][rgb_cam_idx]

        # ManiFlow paper: apply color jitter to RGB with prob=0.2 during training.
        # Jitter is applied on the PIL image before float conversion so it acts on
        # all valid pixels uniformly (same transform per frame, not per point).
        if augment and random.random() < _YAM_COLOR_JITTER_PROB:
            rgb_pil = _YAM_COLOR_JITTER(rgb_pil)

        depth_mm = _decode_lossless_depth_frame(depth_pil)    # (H, W) uint16
        rgb_f32  = np.array(rgb_pil, dtype=np.float32) / 255.0  # (H, W, 3)

        xyzrgb = _depth_to_xyzrgb(depth_mm, rgb_f32, fx, fy, cx, cy, depth_scale)

        # Random pre-sample to max_points (CPU); GPU FPS to visual_cond_len
        # happens inside DP3Encoder.forward() via fps_torch.
        N = xyzrgb.shape[0]
        if N > max_points:
            idx = np.random.choice(N, max_points, replace=False)
            xyzrgb = xyzrgb[idx]
        elif N < max_points:
            # Pad with zeros so all samples have the same N.
            pad = np.zeros((max_points - N, 6), dtype=np.float32)
            xyzrgb = np.concatenate([xyzrgb, pad], axis=0)

        pc_list.append(xyzrgb)                                  # (max_points, 6)

    # Stack: [B, max_points, 6] → unsqueeze T_obs dim → [B, 1, max_points, 6]
    pc_arr = np.stack(pc_list, axis=0).astype(np.float32)       # (B, N, 6)
    obs["point_cloud"] = torch.from_numpy(pc_arr).unsqueeze(1)  # (B, 1, N, 6)

    # ---- State → agent_pos (rename for DP3Encoder) -------------------------
    if "state" in batch[0]:
        state_arr = np.stack(
            [np.asarray(s["state"], dtype=np.float32) for s in batch]
        )  # (B, T_obs, 14)
        obs["agent_pos"] = torch.from_numpy(state_arr)

    # ---- Actions ------------------------------------------------------------
    action_arr = np.stack(
        [np.asarray(s["action"], dtype=np.float32) for s in batch]
    )  # (B, T_a, 14)
    action = torch.from_numpy(action_arr)

    # ---- Language -----------------------------------------------------------
    lang = [s["lang"] for s in batch]

    return {"obs": obs, "action": action, "lang": lang}


def yamrobot_multimodal_collate_fn(
    batch: list[dict],
    head_camera_info: dict,
    wrist_r_camera_info: dict,
    wrist_l_camera_info: dict,
    ptmap_size: int = 224,
    head_max_depth_m: float = 2.0,
    wrist_max_depth_m: float = 2.0,
    augment: bool = True,
    include_ptmap: bool = True,
) -> dict:
    """Collate YAM samples into RGB + dense XYZ pointmap multimodal batch.

    image list from _pack_sample (6 entries, see YAM_*_IDX constants):
        [head_rgb(0), head_depth(1), wrist_r_rgb(2), wrist_l_rgb(3),
         wrist_r_depth(4), wrist_l_depth(5)]

    Returns:
        obs:
            head_rgb:      Tensor [B, 1, 3, 224, 224]
            wrist_r_rgb:   Tensor [B, 1, 3, 224, 224]
            wrist_l_rgb:   Tensor [B, 1, 3, 224, 224]
            head_ptmap:    Tensor [B, 1, 3, ptmap_size, ptmap_size]  (only if include_ptmap=True)
            wrist_r_ptmap: Tensor [B, 1, 3, ptmap_size, ptmap_size]  (only if include_ptmap=True)
            wrist_l_ptmap: Tensor [B, 1, 3, ptmap_size, ptmap_size]  (only if include_ptmap=True)
            agent_pos:   Tensor [B, 1, 14]
        action: Tensor [B, 16, 14]
        lang:   List[str]

    Args:
        head_camera_info:    dict with fx/fy/cx/cy/depth_scale for head camera.
        wrist_r_camera_info: dict with fx/fy/cx/cy/depth_scale for right wrist.
        wrist_l_camera_info: dict with fx/fy/cx/cy/depth_scale for left wrist.
        ptmap_size:          target pointmap resolution.
        head_max_depth_m:    depth clip for head camera (default 2.0 m).
        wrist_max_depth_m:   depth clip for wrist cameras (default 2.0 m).
        augment:             apply colour jitter to each RGB with prob=0.2 (training).
        include_ptmap:       include XYZ pointmap in obs (default True). Set False for RGB-only training.
    """
    def _extract_intrinsics(cam_info: dict):
        return (
            cam_info["fx"], cam_info["fy"],
            cam_info["cx"], cam_info["cy"],
            cam_info.get("depth_scale", 1000.0),
        )

    def _make_ptmap(
        depth_pil: Image.Image,
        fx: float, fy: float, cx: float, cy: float,
        depth_scale: float,
        max_depth_m: float,
    ) -> torch.Tensor:
        """Decode depth -> XYZ pointmap [3, ptmap_size, ptmap_size]."""
        depth_mm = _decode_lossless_depth_frame(depth_pil)          # (H, W) uint16
        return _depth_to_pointmap(
            depth_mm=depth_mm,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            depth_scale=depth_scale,
            max_depth_m=max_depth_m,
            target_h=ptmap_size,
            target_w=ptmap_size,
        )

    h_fx,  h_fy,  h_cx,  h_cy,  h_ds  = _extract_intrinsics(head_camera_info)
    wr_fx, wr_fy, wr_cx, wr_cy, wr_ds = _extract_intrinsics(wrist_r_camera_info)
    wl_fx, wl_fy, wl_cx, wl_cy, wl_ds = _extract_intrinsics(wrist_l_camera_info)

    head_rgb_list    = []
    wrist_r_rgb_list = []
    wrist_l_rgb_list = []
    head_ptmap_list    = []
    wrist_r_ptmap_list = []
    wrist_l_ptmap_list = []

    for s in batch:
        imgs = s["image"]

        # --- extract PIL images (already 224×224 for RGB, original res for depth)
        head_rgb_pil    = imgs[YAM_HEAD_RGB_IDX]
        wrist_r_rgb_pil = imgs[YAM_WRIST_R_RGB_IDX]
        wrist_l_rgb_pil = imgs[YAM_WRIST_L_RGB_IDX]

        # RGB augmentation (crop, rotation, color jitter) is applied inside
        # MultiModalObsEncoder.forward() during training, not in the collate.

        # --- RGB tensors [3, 224, 224] in [0, 1]  (ImageNet norm done in encoder)
        head_rgb_list.append(_pil_to_tensor(head_rgb_pil))
        wrist_r_rgb_list.append(_pil_to_tensor(wrist_r_rgb_pil))
        wrist_l_rgb_list.append(_pil_to_tensor(wrist_l_rgb_pil))

        # --- pointmaps: decode depth -> dense XYZ (skipped when include_ptmap=False)
        if include_ptmap:
            head_depth_pil    = imgs[YAM_HEAD_DEPTH_IDX]
            wrist_r_depth_pil = imgs[YAM_WRIST_R_DEPTH_IDX]
            wrist_l_depth_pil = imgs[YAM_WRIST_L_DEPTH_IDX]
            head_ptmap_list.append(
                _make_ptmap(head_depth_pil, h_fx,  h_fy,  h_cx,  h_cy,  h_ds,  head_max_depth_m)
            )
            wrist_r_ptmap_list.append(
                _make_ptmap(wrist_r_depth_pil, wr_fx, wr_fy, wr_cx, wr_cy, wr_ds, wrist_max_depth_m)
            )
            wrist_l_ptmap_list.append(
                _make_ptmap(wrist_l_depth_pil, wl_fx, wl_fy, wl_cx, wl_cy, wl_ds, wrist_max_depth_m)
            )

    def _stack_rgb(pil_list: list) -> torch.Tensor:
        """Stack [3,H,W] tensors → [B, 1, 3, H, W]."""
        return torch.stack(pil_list).unsqueeze(1)   # [B, 1, 3, 224, 224]

    def _stack_ptmap(ptmap_list: list[torch.Tensor]) -> torch.Tensor:
        """Stack [3,H,W] tensors -> [B, 1, 3, H, W]."""
        return torch.stack(ptmap_list).unsqueeze(1)

    obs: dict[str, torch.Tensor] = {
        "head_rgb":    _stack_rgb(head_rgb_list),
        "wrist_r_rgb": _stack_rgb(wrist_r_rgb_list),
        "wrist_l_rgb": _stack_rgb(wrist_l_rgb_list),
    }
    if include_ptmap:
        obs["head_ptmap"]    = _stack_ptmap(head_ptmap_list)
        obs["wrist_r_ptmap"] = _stack_ptmap(wrist_r_ptmap_list)
        obs["wrist_l_ptmap"] = _stack_ptmap(wrist_l_ptmap_list)

    if "state" in batch[0]:
        state_arr = np.stack(
            [np.asarray(s["state"], dtype=np.float32) for s in batch]
        )  # (B, T_obs, 14)
        obs["agent_pos"] = torch.from_numpy(state_arr)

    action_arr = np.stack(
        [np.asarray(s["action"], dtype=np.float32) for s in batch]
    )  # (B, T_a, 14)

    return {
        "obs":    obs,
        "action": torch.from_numpy(action_arr),
        "lang":   [s["lang"] for s in batch],
    }


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    
    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"
    # frames_root: path to pre-decoded JPEG/PNG frames; empty string / None → fallback to MP4
    frames_root = data_cfg.get("frames_root", None) if data_cfg else None
    if frames_root == "":
        frames_root = None
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend, # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
        frames_root=frames_root,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg), d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )



if __name__ == "__main__":

    # import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./hvla/config/training/hvla_cotrain_behavior.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()
    args.config_yaml = "./examples/MultiRobot/train_files/hvla_cotrain_multiRobot.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # cfg.datasets.vla_data.data_mix = "robotwin"
    vla_dataset_cfg = cfg.datasets.vla_data
    # cfg.datasets.vla_data.include_state = True
    vla_dataset_cfg.task_id = 1
    for task_id in ["all"]:
        vla_dataset_cfg.task_id = task_id
        print(f"Testing Task ID: {task_id}")
        dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        # dataset
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm
    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # print(batch)
        # print(1)
        if count > 100:
            break
        count += 1
        pass