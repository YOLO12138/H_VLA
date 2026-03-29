#!/bin/bash
# ManiFlow Multi-Modal Eval Script
# ==================================
# Two-step eval: (1) start policy server, (2) connect a simulator.
#
# === Step 1 — Start policy server (this script) ===
#   bash scripts/run_scripts/eval_maniflow_multimodal.sh <checkpoint_path> [gpu_id] [port]
#
# === Step 2 — Connect a simulator (in a separate terminal) ===
#
# RoboTwin example:
#   conda activate robotwin
#   cd examples/Robotwin/eval_files
#   bash eval.sh <task_name> demo_clean my_test 0 0
#   # Edit deploy_policy.yml to set your checkpoint path and port.
#   # See examples/Robotwin/README.md for full task list.
#
# SimplerEnv example:
#   conda activate simpler_env
#   python examples/SimplerEnv/eval_files/start_simpler_env.py \
#       --ckpt-path <checkpoint_path> --port 5694 \
#       --robot widowx --policy-setup widowx_bridge \
#       --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0

set -euo pipefail

CKPT_PATH=${1:?"Usage: eval_maniflow_multimodal.sh <checkpoint_path> [gpu_id] [port]"}
GPU_ID=${2:-0}
PORT=${3:-5694}

export PYTHONPATH=$(pwd):${PYTHONPATH:-}

echo "============================================"
echo "ManiFlow Policy Server"
echo "  Checkpoint: ${CKPT_PATH}"
echo "  GPU:        ${GPU_ID}"
echo "  Port:       ${PORT}"
echo "============================================"

# Build output directory for logs
ckpt_dir=$(dirname "${CKPT_PATH}")
ckpt_base=$(basename "${CKPT_PATH}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${PORT}.log"

echo "Log: ${log_file}"
echo "Waiting for sim client to connect on port ${PORT} ..."

CUDA_VISIBLE_DEVICES=${GPU_ID} python deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --port ${PORT} \
    --use_bf16 \
    2>&1 | tee "${log_file}"
