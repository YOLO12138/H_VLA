"""
YAM Policy Server — Demo Client
================================
Sends an observation to the running policy server, receives the predicted
action chunk, denormalizes it, and prints the result.

By default uses fake random data. Pass --obs_dir to load a real frame
extracted by scripts/extract_yam_frames.py.

Usage:
    # Terminal 1 — start the server
    bash scripts/run_scripts/eval_maniflow_multimodal.sh <ckpt.pt> 0 6678

    # Terminal 2 — run with fake data
    python examples/YAM/eval_files/demo_client.py \
        --ckpt_path <ckpt.pt> --host 127.0.0.1 --port 6678

    # Terminal 2 — run with real dataset frame
    python examples/YAM/eval_files/demo_client.py \
        --ckpt_path <ckpt.pt> --host 127.0.0.1 --port 6678 \
        --obs_dir demo_dataset/00000
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODEL_IMG_SIZE = 224

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from hvla.model.tools import read_mode_config

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def depth_to_pointmap(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    min_depth_m: float = 0.01,
    max_depth_m: float = 2.0,
    target_h: int = MODEL_IMG_SIZE,
    target_w: int = MODEL_IMG_SIZE,
) -> np.ndarray:
    """Back-project a depth map to a dense XYZ pointmap and resize.

    Args:
        depth_m: (H, W) float32 depth in **metres** (live ZED sensor output).
        intrinsics: (3, 3) camera intrinsic matrix.
        min_depth_m / max_depth_m: valid depth bounds.
        target_h / target_w: output spatial resolution.

    Returns:
        (3, target_h, target_w) float32 XYZ pointmap in camera frame.
    """
    h, w = depth_m.shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    z = depth_m.astype(np.float32)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    valid = (z > min_depth_m) & (z < max_depth_m)
    x[~valid] = 0.0
    y[~valid] = 0.0
    z[~valid] = 0.0

    # (H, W, 3) → resize → (target_h, target_w, 3) → (3, H, W)
    ptmap_hwc = np.stack([x, y, z], axis=-1)
    ptmap_hwc = cv2.resize(ptmap_hwc, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return ptmap_hwc.transpose(2, 0, 1).astype(np.float32)  # (3, H, W)


def q99_normalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Map raw values to ~[-1, 1] via q99 normalization."""
    span = q99 - q01
    mask = span != 0
    result = np.zeros_like(x, dtype=np.float32)
    result[..., mask] = 2.0 * (x[..., mask] - q01[mask]) / span[mask] - 1.0
    result[..., ~mask] = x[..., ~mask]
    return np.clip(result, -1.0, 1.0)


def q99_denormalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Map normalized ~[-1, 1] values back to original scale."""
    span = q99 - q01
    mask = span != 0
    clipped = np.clip(x, -1.0, 1.0)
    result = np.broadcast_to(q01, clipped.shape).copy().astype(np.float32)
    result[..., mask] = 0.5 * (clipped[..., mask] + 1.0) * span[mask] + q01[mask]
    return result



def preprocess_rgb(img_hwc: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize uint8 HWC → float32 (1,1,3,H,W) in [0,1]."""
    img = cv2.resize(img_hwc, (size, size), interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, None]


def preprocess_depth(depth_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Depth (H,W) float32 metres → pointmap (1,1,3,224,224)."""
    ptmap = depth_to_pointmap(depth_m, intrinsics)  # (3,224,224)
    return ptmap[None, None]


def load_norm_stats(ckpt_path: str):
    """Load q01/q99 stats for action and state from checkpoint directory."""
    cfg, norm_stats = read_mode_config(ckpt_path)
    action_mode = cfg.get("datasets", {}).get("vla_data", {}).get("action_mode", "abs")
    unnorm_key = next(iter(norm_stats))
    logger.info("Using norm stats key: %s, action_mode: %s", unnorm_key, action_mode)
    stats = norm_stats[unnorm_key]

    # Action stats — handle both new per-mode format and old flat format
    if action_mode in stats:
        action_stats = stats[action_mode].get("action", stats[action_mode])
    elif "action" in stats:
        action_stats = stats["action"]
    else:
        raise ValueError(f"Cannot find action stats under key '{unnorm_key}'")

    action_q01 = np.array(action_stats["q01"], dtype=np.float32)
    action_q99 = np.array(action_stats["q99"], dtype=np.float32)

    # State stats
    state_stats = stats.get("state") or stats.get("abs", {}).get("state")
    if state_stats is None:
        raise ValueError(f"Cannot find state stats under key '{unnorm_key}'")
    state_q01 = np.array(state_stats["q01"], dtype=np.float32)
    state_q99 = np.array(state_stats["q99"], dtype=np.float32)

    return action_q01, action_q99, state_q01, state_q99, action_mode


def load_obs_from_dir(obs_dir: str) -> dict:
    """Load a real frame extracted by scripts/extract_yam_frames.py.

    Expects inside obs_dir:
        head_rgb.png / wrist_r_rgb.png / wrist_l_rgb.png  — uint8 RGB
        head_depth.png / wrist_r_depth.png / wrist_l_depth.png  — uint16 mm
        state_action.npz  — state (T_obs, 14), action (T_action, 14)
        (optional) head_camera_info.json  — fx/fy/cx/cy intrinsics
    """
    d = Path(obs_dir)

    def load_rgb(name):
        return np.array(Image.open(d / name).convert("RGB"), dtype=np.uint8)

    def load_depth_m(name):
        # Stored as uint16 millimetres; convert to float32 metres
        return np.array(Image.open(d / name), dtype=np.float32) / 1000.0

    state = np.load(d / "state_action.npz")["state"]
    state = state[0] if state.ndim == 2 else state  # take first obs step → (14,)

    # Load intrinsics — check frame dir first, then parent (output root from extract script)
    cam_info_path = next(
        (p for p in (d / "head_camera_info.json", d.parent / "head_camera_info.json") if p.exists()),
        None,
    )
    if cam_info_path is not None:
        info = json.loads(cam_info_path.read_text())
        intrinsics = np.array([[info["fx"], 0., info["cx"]],
                               [0., info["fy"], info["cy"]],
                               [0., 0., 1.]], dtype=np.float64)
    else:
        logger.warning("head_camera_info.json not found near %s — using placeholder intrinsics", obs_dir)
        intrinsics = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])

    return {
        "head_rgb":      load_rgb("head_rgb.png"),
        "wrist_r_rgb":   load_rgb("wrist_r_rgb.png"),
        "wrist_l_rgb":   load_rgb("wrist_l_rgb.png"),
        "head_depth":    load_depth_m("head_depth.png"),
        "wrist_r_depth": load_depth_m("wrist_r_depth.png"),
        "wrist_l_depth": load_depth_m("wrist_l_depth.png"),
        "state":         state.astype(np.float32),
        "intrinsics":    intrinsics,
    }


def make_fake_obs() -> dict:
    """Fake observation dict used when no --obs_dir is provided."""
    return {
        "head_rgb":      np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "wrist_r_rgb":   np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "wrist_l_rgb":   np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "head_depth":    np.random.uniform(0.3, 2.0, (480, 640)).astype(np.float32),
        "wrist_r_depth": np.random.uniform(0.3, 2.0, (480, 640)).astype(np.float32),
        "wrist_l_depth": np.random.uniform(0.3, 2.0, (480, 640)).astype(np.float32),
        "state":         np.zeros(14, dtype=np.float32),
        "intrinsics":    np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]]),
    }


def main():
    parser = argparse.ArgumentParser(description="YAM policy server demo client")
    parser.add_argument("--ckpt_path", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6678)
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    parser.add_argument("--obs_dir", default=None, help="Frame directory from extract_yam_frames.py (omit for fake data)")
    args = parser.parse_args()

    # Load norm stats
    action_q01, action_q99, state_q01, state_q99, action_mode = load_norm_stats(args.ckpt_path)
    logger.info("Loaded norm stats — action dim: %d, state dim: %d, action_mode: %s", len(action_q01), len(state_q01), action_mode)

    # Build observation
    if args.obs_dir is not None:
        logger.info("Loading real obs from %s", args.obs_dir)
        obs = load_obs_from_dir(args.obs_dir)
    else:
        logger.info("No --obs_dir given, using fake random data")
        obs = make_fake_obs()

    # Normalize state
    norm_state = q99_normalize(obs["state"], state_q01, state_q99)  # (14,)
    logger.info("Raw state   : %s", np.round(obs["state"], 3))
    logger.info("Norm state  : %s", np.round(norm_state, 3))

    # Preprocess all modalities
    server_obs = {
        "head_rgb":      preprocess_rgb(obs["head_rgb"]),
        "wrist_r_rgb":   preprocess_rgb(obs["wrist_r_rgb"]),
        "wrist_l_rgb":   preprocess_rgb(obs["wrist_l_rgb"]),
        "head_ptmap":    preprocess_depth(obs["head_depth"],    obs["intrinsics"]),
        "wrist_r_ptmap": preprocess_depth(obs["wrist_r_depth"], obs["intrinsics"]),
        "wrist_l_ptmap": preprocess_depth(obs["wrist_l_depth"], obs["intrinsics"]),
        "agent_pos":     norm_state[None, None],  # (1, 1, 14)
    }

    # Connect and send
    logger.info("Connecting to server at %s:%d ...", args.host, args.port)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    logger.info("Connected. Server metadata: %s", client.get_server_metadata())

    msg = {"examples": {"obs": server_obs}, "num_ddim_steps": args.num_ddim_steps}
    response = client.predict_action(msg)

    # Handle response
    if not response.get("ok", False):
        logger.error("Server error: %s", response.get("error", response))
        return

    norm_actions = response["data"]["normalized_actions"][0]  # (16, 14)
    actions = q99_denormalize(norm_actions, action_q01, action_q99)

    if action_mode == "rel":
        actions = actions + obs["state"][np.newaxis]
    elif action_mode == "delta":
        abs_actions = np.zeros_like(actions)
        abs_actions[0] = actions[0] + obs["state"]
        for t in range(1, len(actions)):
            abs_actions[t] = actions[t] + abs_actions[t - 1]
        actions = abs_actions

    print("\n--- Results ---")
    print(f"normalized_actions shape : {norm_actions.shape}")
    print(f"normalized_actions[0]    : {np.round(norm_actions[0], 3)}")
    print(f"actions (abs)[0]         : {np.round(actions[0], 3)}")
    print(f"actions range            : [{actions.min():.4f}, {actions.max():.4f}]")

    client.close()


if __name__ == "__main__":
    main()
