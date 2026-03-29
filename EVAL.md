# YAM ManiFlowMultiModal — Evaluation Guide

End-to-end demo evaluation for the bimanual YAM robot policy.
No real robot required — runs against a live policy server with real dataset frames.

---

## Overview

Evaluation follows a two-terminal pattern:

```
Terminal 1: Policy Server (GPU)
  └── loads checkpoint → serves predict_action over WebSocket

Terminal 2: Demo Client
  └── loads a dataset frame → normalizes → sends to server
      → receives normalized actions → denormalizes → prints
```

---

## Prerequisites


Checkpoint directory structure expected by `read_mode_config`:
```
<run_dir>/
├── config.yaml
├── dataset_statistics.json
└── final_model/
    └── model.safetensors     ← pass this path to --ckpt_path
```

The dummy checkpoint at `ckpts/` already has this layout.

---

## Eval Dataset Format

A pre-extracted `demo_dataset/` is included in the repo. Each frame directory contains:

```
demo_dataset/
├── head_camera_info.json       ← camera intrinsics
├── wrist_r_camera_info.json
├── wrist_l_camera_info.json
├── 00000/
│   ├── head_rgb.png            ← uint8 RGB
│   ├── wrist_r_rgb.png
│   ├── wrist_l_rgb.png
│   ├── head_depth.png          ← uint16 depth in millimetres
│   ├── wrist_r_depth.png
│   ├── wrist_l_depth.png
│   └── state_action.npz        ← state (T_obs, 14) + action (T_action, 14) float32
├── 00001/
│   └── ...
```

> **Note:** Depth PNGs are stored as uint16 mm. The demo client converts to float32 metres (`/1000.0`) automatically.

---

## Step 1 — Start the Policy Server (Terminal 1)

```bash
bash scripts/run_scripts/eval_maniflow_multimodal.sh \
    ckpts/final_model/model.safetensors 0 6678
```

Arguments: `<checkpoint_path> <gpu_id> <port>`

Wait for `server running ...` before proceeding.
Logs are saved to `ckpts/final_model/output_server/`.

---

## Step 2 — Run the Demo Client (Terminal 2)

**With a real dataset frame:**
```bash
python examples/YAM/eval_files/demo_client.py \
    --ckpt_path ckpts/final_model/model.safetensors \
    --port 6678 \
    --obs_dir demo_dataset/00000
```

**With fake random data (no dataset needed):**
```bash
python examples/YAM/eval_files/demo_client.py \
    --ckpt_path ckpts/final_model/model.safetensors \
    --port 6678
```

**Iterate over multiple frames:**
```bash
for i in $(seq -f "demo_dataset/%05g" 0 9); do
    python examples/YAM/eval_files/demo_client.py \
        --ckpt_path ckpts/final_model/model.safetensors \
        --port 6678 \
        --obs_dir $i
done
```

---

## Expected Output

```
--- Results ---
normalized_actions shape : (16, 14)
normalized_actions[0]    : [ 1.148 -0.231 -0.559  0.217  ...]
actions (denormalized)[0]: [ 0.362  0.887  0.497 -0.325  ...]
actions range            : [-0.3300, 0.9791]
```

- **`normalized_actions`** — raw model output in `[-1, 1]` (values slightly outside are clipped during denorm)
- **`actions (denormalized)[0]`** — first of 16 predicted steps, in joint-space (radians / binary gripper)
- **Action layout:** `[left_joint×6, left_grip, right_joint×6, right_grip]` (14D)
- **Valid range** is roughly within `[q01, q99]` per dimension from `dataset_statistics.json`

---

## Client CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--ckpt_path` | required | Path to `.pt` or `.safetensors` checkpoint |
| `--host` | `127.0.0.1` | Server hostname |
| `--port` | `6678` | Server port |
| `--num_ddim_steps` | `10` | DDIM denoising steps |
| `--obs_dir` | None (fake data) | Frame directory from `demo_dataset/` |

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/run_scripts/eval_maniflow_multimodal.sh` | Launch policy server |
| `examples/YAM/eval_files/demo_client.py` | Demo client (normalize → infer → denormalize → print) |
| `examples/YAM/eval_files/model2yam_interface.py` | Full robot client with chunking (used by demo_client internally) |
| `deployment/model_server/server_policy.py` | WebSocket policy server |
| `ckpts/dataset_statistics.json` | Normalization stats (q01/q99 per dim for action and state) |
