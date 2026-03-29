

cd /mnt/petrelfs/yejinhui/Projects/hvla
export PYTHONPATH=$(pwd):${PYTHONPATH}

port=6678
gpu_id=2
# export DEBUG=true
export hvla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/hvla/bin/python

your_ckpt=./results/Checkpoints/1208_bridge_rt_1_Qwen3PI/final_model/pytorch_model.pt

#### build output directory #####
ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"


#### run server #####
CUDA_VISIBLE_DEVICES=${gpu_id} ${hvla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    2>&1 | tee "${log_file}"