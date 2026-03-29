#!/bin/bash
# Launch the HVLA policy server for YAM bimanual ManiFlowMultiModal.
# Usage: bash examples/YAM/eval_files/run_policy_server.sh

cd "$(git rev-parse --show-toplevel)" || exit 1
export PYTHONPATH=$(pwd):${PYTHONPATH}

# --- User config ---
your_ckpt=results/Checkpoints/maniflow_multimodal/checkpoints/steps_XXXX_pytorch_model.pt
gpu_id=0
port=6678
# export DEBUG=true

# --- Launch server ---
CUDA_VISIBLE_DEVICES=$gpu_id python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16
