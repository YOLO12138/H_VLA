# HVLA

> Built on top of [starVLA](https://github.com/starVLA/starVLA). A modular codebase for developing and training Vision-Language-Action models for robot learning.

HVLA is a modular, plug-and-play framework for building and training VLA (Vision-Language-Action) models for robot learning. Each component — VLM backbone, action head, dataloader, trainer — follows high cohesion / low coupling design, enabling rapid prototyping and independent debugging.

---

## Quick Start (ManiFlow Multi-Modal)

### 1. Installation

See [INSTALL.md](INSTALL.md) for full setup instructions (conda env, flash-attn, PyTorch3D, ManiFlow package).

### 2. Prepare Data

HVLA uses the [GR00T style LeRobot V2.0 dataset format](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md):

```
your_dataset/
├── meta/
│   ├── info.json                     # Dataset metadata
│   ├── episodes.jsonl                # Episode list with lengths
│   ├── tasks.jsonl                   # Task descriptions
│   ├── modality.json                 # State/action dimension mapping
│   ├── stats.json                    # Normalization statistics
│   ├── head_camera_info.json         # Camera intrinsics (fx, fy, cx, cy, depth_scale)
│   ├── wrist_r_camera_info.json      # One JSON per depth camera
│   └── wrist_l_camera_info.json
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── episode_000001.parquet
└── videos/
    └── chunk-000/
        └── observation.images.<camera_name>/
            ├── episode_000000.mp4
            └── episode_000001.mp4
```

#### YAM robot modalities

The default YAM maniflow policy uses a **dual-arm YAM robot** with the following sensor and action layout:

| Modality | Keys | Dims |
|----------|------|------|
| RGB cameras | `head_rgb`, `wrist_r_rgb`, `wrist_l_rgb` | 3 × RGB video |
| Depth cameras | `head_depth`, `wrist_r_depth`, `wrist_l_depth` | 3 × depth video |
| State (`observation.state`) | left_joint[0:6], left_grip[6], right_joint[7:13], right_grip[13] | 14D float32 |
| Action (`action`) | left_joint[0:6], left_grip[6], right_joint[7:13], right_grip[13] | 14D float32 |

State and action are both stored as **absolute** in dataset, joint positions + gripper width, concatenated in the same order.

#### Parquet file contents

| Column | Type | Description |
|--------|------|-------------|
| `observation.state` | float32 array | Concatenated state (joint positions, gripper, etc.) |
| `action` | float32 array | Concatenated action |
| `timestamp` | float | Timestep timestamp |
| `episode_index` | int | Episode ID |
| `index` | int | Global step index |
| `annotation.human.action.task_description` | int | Task index -> maps to `meta/tasks.jsonl` |



#### Depth & Pointmap data

ManiFlow uses dense XYZ pointmaps built from depth cameras. The depth-to-pointmap conversion happens in the collate function at data loading time. Each depth camera needs a `meta/<camera>_camera_info.json` with intrinsics:

```json
{
  "fx": 265.59, "fy": 265.59,
  "cx": 336.0, "cy": 188.0,
  "depth_scale": 1000.0
}
```

### 3. Configure

The full ManiFlow config is at [`examples/YAM/train_maniflow_multimodal.yaml`](examples/YAM/train_maniflow_multimodal.yaml).

### 4. Train

```bash
# Edit paths in the script first, then:
bash scripts/run_scripts/run_maniflow_multimodal.sh

# Or override via env vars:
NUM_GPUS=1 DATA_ROOT=/my/data bash scripts/run_scripts/run_maniflow_multimodal.sh

# Or launch directly:
accelerate launch \
  --config_file hvla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  hvla/training/train_hvla.py \
  --config_yaml examples/YAM/train_maniflow_multimodal.yaml \
  --datasets.vla_data.data_root_dir /path/to/data
```

Any config parameter can be overridden from the command line (OmegaConf):

```bash
accelerate launch ... \
  --framework.n_layer 8 \
  --framework.encoder_type spatial \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.max_train_steps 50000
```

### 5. Evaluate


**Sim evaluation** uses a two-step workflow: policy server + dummy client. See [EVAL.md](EVAL.md) for the full guide including a no-robot demo using the included `demo_dataset/`.

## Codebase Structure

```
hvla/
├── model/
│   ├── framework/          # Full VLA model assemblies (single external API surface)
│   │   ├── ManiFlowMultiModal.py  # DiT with multi-camera RGB + pointmap
│   │   ├── ManiFlow.py            # DiT with pointcloud
│   │   └── base_framework.py      # Base class (PreTrainedModel)
│   └── modules/
│       ├── maniflow/              # ManiFlow policy, DiTX backbone, encoders
│       ├── action_model/          # Action prediction heads (flow-matching, MLP, FAST, DiT)
│       ├── vlm/                   # VLM backbone wrappers (Qwen, Florence)
│       └── projector/             # Vision-to-action projections (QFormer)
├── dataloader/
│   ├── lerobot_datasets.py        # Dataset loading + collate functions
│   └── gr00t_lerobot/
│       ├── data_config.py         # Robot-specific configs (action dims, cameras, normalization)
│       └── mixtures.py            # Dataset mixture definitions
├── training/
│   ├── train_hvla.py              # VLA training (main)
│   ├── train_hvla_cotrain.py      # VLA + VLM co-training
│   └── train_starvlm.py          # VLM-only training
└── config/
    └── deepseeds/deepspeed_zero2.yaml

```



---

## Citation

HVLA is forked from [StarVLA](https://github.com/starVLA/starVLA) and released under the MIT License.

```bibtex
@misc{starvla2025,
  title        = {StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing},
  author       = {starVLA Contributors},
  year         = {2025},
  url          = {https://github.com/starVLA/starVLA},
  howpublished = {GitHub repository},
}
```

## Acknowledgements

This project builds upon:
- [StarVLA](https://github.com/starVLA/starVLA)
- [LeRobot](https://github.com/huggingface/lerobot)
- [GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [Qwen-VL](https://github.com/QwenLM/Qwen3-VL)
