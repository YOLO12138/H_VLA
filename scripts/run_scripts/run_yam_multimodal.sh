

# Communication NIC options for multi-node training (uncomment and modify if needed)
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

###########################################################################################
# === Please modify the following paths according to your environment ===
## Data
# PERFORMANCE NOTE: For best I/O throughput, pre-copy dataset to node-local NVMe before launch:
#   rsync -av --progress /gscratch/scrubbed/yuzhi/dataset/lerobot_yam /tmp/yam_dataset/
#   Then set: data_root_dir=/tmp/yam_dataset/
data_root_dir=/home/yuzhi/dataset
data_mix=yam   # replace with your full dataset mix name
frames_root: /home/yuzhi/dataset/lerobot_yam_frames
# Global batch = per_device_batch_size(8) × num_gpus(4) × grad_accum(4) = 128.
# This reduces AllReduce frequency by 4×, amortising the PCIe communication cost.
grad_accum=1
## To save
run_root_dir=results/Checkpoints
run_id=yam_multimodal_v2
output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/
# === End of environment variable configuration ===
###########################################################################################

export PYTHONPATH=/mmfs1/home/jliu63/project/yam_demo/ManiFlow_Policy/ManiFlow:$PYTHONPATH

# Accelerate reads this env var to set gradient_accumulation_steps in AcceleratorState.
# This is required because train_hvla.py initialises Accelerator() at module import
# time (before CLI args are parsed), so we cannot pass grad_accum via constructor arg.
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=${grad_accum}

accelerate launch \
  --config_file hvla/hvla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  hvla/hvla/training/train_hvla.py \
  --config_yaml hvla/examples/YAM/train_yam_multimodal.yaml \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.frames_root ${frames_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --trainer.max_train_steps 17500 \
  --trainer.save_interval 5000 \
  --trainer.eval_interval 500 \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.gradient_accumulation_steps ${grad_accum}


# resume from checkpoint example:
# accelerate launch \
#   --config_file hvla/hvla/config/deepseeds/deepspeed_zero2.yaml \
#   --num_processes 8 \
#   hvla/hvla/training/train_hvla.py \
#   --config_yaml hvla/examples/YAM/train_yam_multimodal.yaml \
#   --datasets.vla_data.data_root_dir ${data_root_dir} \
#   --datasets.vla_data.data_mix ${data_mix} \
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id} \
#   --trainer.is_resume true


# multi-node launch example:
# accelerate launch \
#   --config_file hvla/hvla/config/deepseeds/deepspeed_zero2.yaml \
#   --main_process_ip $MASTER_ADDR \
#   --main_process_port $MASTER_PORT \
#   --machine_rank $SLURM_PROCID \
#   --num_machines $SLURM_NNODES \
#   --num_processes=${TOTAL_GPUS} \
#   hvla/hvla/training/train_hvla.py \
#   --config_yaml hvla/examples/YAM/train_yam_multimodal.yaml \
#   --datasets.vla_data.data_root_dir ${data_root_dir} \
#   --datasets.vla_data.data_mix ${data_mix} \
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id}
