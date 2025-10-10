#!/bin/bash
#SBATCH --partition=flame
#SBATCH --job-name=verl_sglang
#SBATCH --gres=gpu:8         # Request 8 H100 GPUs for Training 
#SBATCH --output=logs/prompt_gen_flame_sglang_%J.out
#SBATCH --error=logs/prompt_gen_flame_sglang_%J.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5-00:00:00
#SBATCH --mem=1000G

## SGLang fast version using proven baseline architecture with prefill/decode disaggregation
# sbatch prompt_gen_grpo_flame_sglang_f1_explore.sh /project/flame/emilyx/data/prompt-gen/verl_preprocess_v8_multipass_meta3_test20 Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct qwen_7b_v8_multipass_meta3_flame_sglang_f1_explore
# sbatch prompt_gen_grpo_flame_sglang_f1_explore.sh /project/flame/emilyx/data/prompt-gen/verl_preprocess_v8_multipass_test20 meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B-Instruct llama3_8b_v8_multipass_custom_sglang_f1_explore

set -euo pipefail
echo "ARGS: $1 $2 $3 $4"

# Cleanup function that runs on exit (success, failure, or cancellation)
cleanup() {
    echo "Running cleanup..."
    # Stop SGLang processes
    kill $REWARD_ROUTER_PID 2>/dev/null || true
    for p in "${REWARD_PIDS[@]}"; do kill $p 2>/dev/null || true; done
    
    # Clean up temporary cache
    echo "Cleaning up temporary cache..."
    rm -rf /tmp/hf_cache_${SLURM_JOB_ID} || true
    
    # Unmount GCS bucket
    fusermount -u /tmp/gcs-bucket-${SLURM_JOB_ID} || true
    rmdir /tmp/gcs-bucket-${SLURM_JOB_ID} || true
    
    echo "Cleanup completed."
}

# Set trap to run cleanup on EXIT (success, failure, kill, etc.)
trap cleanup EXIT

export HF_HUB_ENABLE_HF_TRANSFER=1
unset NCCL_P2P_DISABLE

# Optional: Setup cloud storage for checkpoints (if you have GCS)
# If you don't have cloud storage, comment this out and change checkpoint paths to local directories
mkdir -p /tmp/gcs-bucket-${SLURM_JOB_ID}
gcsfuse your-bucket-name /tmp/gcs-bucket-${SLURM_JOB_ID}  # TODO: Replace with your GCS bucket name

# Set environment variables - use /tmp to avoid disk space issues on /project/flame
export HF_DATASETS_CACHE="/tmp/hf_cache_${SLURM_JOB_ID}"
export TRANSFORMERS_CACHE="/tmp/hf_cache_${SLURM_JOB_ID}/hub"
export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID}/hub"
export HF_HUB_CACHE="/tmp/hf_cache_${SLURM_JOB_ID}/hub"
export TMPDIR="/tmp/tmp_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="/tmp/xdg_cache_${SLURM_JOB_ID}"
export HF_DATASETS_TRUST_REMOTE_CODE=True
unset ROCR_VISIBLE_DEVICES

# Configure gcloud (only needed if using GCS)
# gcloud config set core/disable_file_logging true
# gcloud config set storage/rsync_files_directory /tmp/gcloud_rsync_files
# gcloud config set storage/tracker_files_directory /tmp/gcloud_tracker_files

# Input arguments
if [[ -z "$1" || -z "$2" || -z "$3" || -z "$4" ]]; then
    echo "Usage: $0 <data_directory> <model_name> <reward_model_name> <experiment_name>"
    exit 1
fi

data_dir="$1"
model_path="$2"
reward_model_name="$3"
experiment_name="$4"
wandb_experiment_name="${experiment_name}_${SLURM_JOB_ID}"

export WANDB_RESUME=allow

train_path=$data_dir/train.parquet
test_path=$data_dir/validation.parquet

echo "Configuration:"
echo "  Train data: $train_path"
echo "  Test data: $test_path"
echo "  Actor model: $model_path"
echo "  Reward model: $reward_model_name"
echo "  Experiment: $experiment_name"
echo "  Wandb: $wandb_experiment_name"

# SGLang Configuration (matching proven baseline)
REWARD_ROUTER_PORT=8100
REWARD_BASE_PORT=8300
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.9}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-4096}
SCHEDULE_CONSERVATIVENESS=${SCHEDULE_CONSERVATIVENESS:-0.3}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-256}
DTYPE=${SGLANG_DTYPE:-bfloat16}
CTX_LEN=${CTX_LEN:-8192}

# Set environment variables for reward function
export SGLANG_SERVER_ADDR="localhost"
export SGLANG_SERVER_PORT="$REWARD_ROUTER_PORT"
export SGLANG_MODEL_NAME="$reward_model_name"

echo "SGLang Configuration:"
echo "  Router port: $REWARD_ROUTER_PORT"
echo "  Worker base port: $REWARD_BASE_PORT"
echo "  Max running requests: $MAX_RUNNING_REQUESTS"
echo "  Memory fraction: $MEM_FRACTION_STATIC"
echo "  Context length: $CTX_LEN"

mkdir -p logs

# Launch 4 SGLang workers for reward model (GPUs 4-7, same as baseline)
REWARD_PIDS=()

# Prefill workers (GPUs 4-5) - specialized for prompt processing
for idx in 0 1; do
  GPU=$(($idx + 4))
  PORT=$(($REWARD_BASE_PORT + $idx))
  echo "Starting REWARD PREFILL worker gpu=$GPU port=$PORT"
  CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
    --model-path "$reward_model_name" \
    --host 0.0.0.0 \
    --port $PORT \
    --dtype $DTYPE \
    --context-length $CTX_LEN \
    --mem-fraction-static $MEM_FRACTION_STATIC \
    --chunked-prefill-size $CHUNKED_PREFILL_SIZE \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    --schedule-conservativeness $SCHEDULE_CONSERVATIVENESS \
    --cuda-graph-max-bs $CUDA_GRAPH_MAX_BS \
    --enable-tokenizer-batch-encode \
    > logs/reward_prefill_worker_${PORT}.log 2>&1 & REWARD_PIDS+=($!)
done

# Decode workers (GPUs 6-7) - specialized for short token generation  
for idx in 2 3; do
  GPU=$(($idx + 4))
  PORT=$(($REWARD_BASE_PORT + $idx))
  echo "Starting REWARD DECODE worker gpu=$GPU port=$PORT"
  CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
    --model-path "$reward_model_name" \
    --host 0.0.0.0 \
    --port $PORT \
    --dtype $DTYPE \
    --context-length $CTX_LEN \
    --mem-fraction-static $MEM_FRACTION_STATIC \
    --chunked-prefill-size $CHUNKED_PREFILL_SIZE \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    --schedule-conservativeness $SCHEDULE_CONSERVATIVENESS \
    --cuda-graph-max-bs $CUDA_GRAPH_MAX_BS \
    --enable-tokenizer-batch-encode \
    > logs/reward_decode_worker_${PORT}.log 2>&1 & REWARD_PIDS+=($!)
done

# Launch SGLang Router for reward model
echo "Starting REWARD Router on port $REWARD_ROUTER_PORT"
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prometheus-port 29100 \
  --host 0.0.0.0 \
  --port $REWARD_ROUTER_PORT \
  --policy power_of_two \
  --prefill http://127.0.0.1:$(($REWARD_BASE_PORT+0)) \
  --prefill http://127.0.0.1:$(($REWARD_BASE_PORT+1)) \
  --decode http://127.0.0.1:$(($REWARD_BASE_PORT+2)) \
  --decode http://127.0.0.1:$(($REWARD_BASE_PORT+3)) \
  > logs/reward_router_${REWARD_ROUTER_PORT}.log 2>&1 & REWARD_ROUTER_PID=$!

# Health check for SGLang router
echo "Waiting for SGLang reward router to be ready..."
until curl -sSf http://127.0.0.1:${REWARD_ROUTER_PORT}/v1/models >/dev/null; do
  echo "  waiting for reward router..."
  sleep 5
done
echo "SGLang reward router is up and running!"

# Run GRPO training with SGLang reward function
echo "Starting GRPO training with SGLang reward function..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.033 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.4 \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='VERL' \
    trainer.experiment_name=$wandb_experiment_name \
    trainer.default_local_dir=/tmp/gcs-bucket-${SLURM_JOB_ID}/checkpoints/VERL/$experiment_name \  # TODO: Change to your storage path if not using GCS
    trainer.validation_data_dir=/tmp/gcs-bucket-${SLURM_JOB_ID}/validation_data/VERL/$experiment_name \
    trainer.rollout_data_dir=/tmp/gcs-bucket-${SLURM_JOB_ID}/rollout_data/VERL/$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=15 \
    trainer.total_epochs=30 \
    reward_model.reward_manager=batch \
    custom_reward_function.path=verl/utils/reward_score/prompt_gen_batch_sglang_v4_fast_f1.py \
    custom_reward_function.name=compute_score_batch

echo "SGLang GRPO training completed!"
# Note: cleanup() function will be called automatically via trap on EXIT