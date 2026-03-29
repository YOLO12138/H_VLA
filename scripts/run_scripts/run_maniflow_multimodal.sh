#!/bin/bash
# ManiFlow Multi-Modal Training Script
# =====================================
# Usage:  bash scripts/run_scripts/run_maniflow_multimodal.sh
# Override: NUM_GPUS=1 DATA_ROOT=/my/data bash scripts/run_scripts/run_maniflow_multimodal.sh

# Communication NIC options for multi-node training (uncomment and modify if needed)
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

###########################################################################################
# === Modify the following variables for your environment ===
Framework_name=ManiFlowMultiModal
config_yaml=./examples/YAM/train_maniflow_multimodal.yaml
run_root_dir=${RUN_ROOT_DIR:-./results/Checkpoints}
data_root_dir=${DATA_ROOT:-/path/to/data}
data_mix=${DATA_MIX:-yam_ai2}
run_id=${RUN_ID:-maniflow_${data_mix}}
NUM_GPUS=${NUM_GPUS:-8}
grad_accum=${GRAD_ACCUM:-1}
# === End of configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=${grad_accum}

accelerate launch \
  --config_file hvla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_GPUS} \
  hvla/training/train_hvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --trainer.gradient_accumulation_steps ${grad_accum} \
  "$@"


# Resume from checkpoint:
# accelerate launch \
#   --config_file hvla/config/deepseeds/deepspeed_zero2.yaml \
#   --num_processes ${NUM_GPUS} \
#   hvla/training/train_hvla.py \
#   --config_yaml ${config_yaml} \
#   --framework.name ${Framework_name} \
#   --datasets.vla_data.data_root_dir ${data_root_dir} \
#   --datasets.vla_data.data_mix ${data_mix} \
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id} \
#   --trainer.is_resume true

# Multi-node (SLURM):
# accelerate launch \
#   --config_file hvla/config/deepseeds/deepspeed_zero2.yaml \
#   --main_process_ip $MASTER_ADDR \
#   --main_process_port $MASTER_PORT \
#   --machine_rank $SLURM_PROCID \
#   --num_machines $SLURM_NNODES \
#   --num_processes ${TOTAL_GPUS} \
#   hvla/training/train_hvla.py \
#   --config_yaml ${config_yaml} \
#   --framework.name ${Framework_name} \
#   --datasets.vla_data.data_root_dir ${data_root_dir} \
#   --datasets.vla_data.data_mix ${data_mix} \
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id}
