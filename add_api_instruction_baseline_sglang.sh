#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=verl_api_instr_sglang
#SBATCH --gres=gpu:L40S:2         
#SBATCH --output=logs/add_api_instruction_sglang_%J.out
#SBATCH --error=logs/add_api_instruction_sglang_%J.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=100G

# Script to add API-based instruction baseline to existing results using SGLang
# Usage: sbatch add_api_instruction_baseline_sglang.sh [test|validation|train] [api_model_name] [prediction_model_name] [existing_output_path] [baseline_name] [baseline_type]
# Example: sbatch add_api_instruction_baseline_sglang.sh validation "litellm_proxy/neulab/qwen3-8b" Qwen/Qwen2.5-7B-Instruct out/out4/baseline_eval_v3_qwen2_5_7b_validation_sglang "qwen3-8b-generated_instruction" "api_generated_instruction"

# sbatch add_api_instruction_baseline_sglang.sh validation \
#   "litellm_proxy/neulab/qwen3-235b-a22b" \
#   Qwen/Qwen2.5-7B-Instruct \
#   out/out4/baseline_eval_v3_qwen2_5_7b_validation_sglang \
#   "qwen3-235b-generated_instruction3" \
#   "api_generated_instruction3"

# Run api_generated_instruction3+icl method:
# sbatch add_api_instruction_baseline_sglang.sh validation \
#   "litellm_proxy/neulab/qwen3-235b-a22b" \
#   Qwen/Qwen2.5-7B-Instruct \
#   out/out4/baseline_eval_v3_qwen2_5_7b_validation_sglang \
#   "qwen3-235b-a22b_generated_instruction3+icl" \
#   "api_generated_instruction3+icl"

set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
unset NCCL_P2P_DISABLE

# Check if dataset split argument is provided
if [ $# -eq 0 ]; then
    DATASET_SPLIT="validation"
else
    DATASET_SPLIT="$1"
fi

# Check if API model name is provided
if [ $# -lt 2 ]; then
    echo "Error: Please provide the API model name"
    echo "Usage: sbatch add_api_instruction_baseline_sglang.sh [test|validation|train] [api_model_name] [prediction_model_name] [existing_output_path] [baseline_name]"
    exit 1
else
    API_MODEL_NAME="$2"
fi

# Check if prediction model name is provided
if [ $# -lt 3 ]; then
    echo "Error: Please provide the prediction model name"
    echo "Usage: sbatch add_api_instruction_baseline_sglang.sh [test|validation|train] [api_model_name] [prediction_model_name] [existing_output_path] [baseline_name]"
    exit 1
else
    PREDICTION_MODEL_NAME="$3"
fi

# Check if existing output path is provided
if [ $# -lt 4 ]; then
    echo "Error: Please provide the existing output path"
    echo "Usage: sbatch add_api_instruction_baseline_sglang.sh [test|validation|train] [api_model_name] [prediction_model_name] [existing_output_path] [baseline_name]"
    exit 1
else
    EXISTING_OUTPUT_PATH="$4"
fi

# Check if baseline name is provided
if [ $# -lt 5 ]; then
    echo "Error: Please provide the baseline name"
    echo "Usage: sbatch add_api_instruction_baseline_sglang.sh [test|validation|train] [api_model_name] [prediction_model_name] [existing_output_path] [baseline_name] [baseline_type]"
    exit 1
else
    BASELINE_NAME="$5"
fi

# Check if baseline type is provided
if [ $# -lt 6 ]; then
    echo "Error: Please provide the baseline type"
    echo "Usage: sbatch add_api_instruction_baseline_sglang.sh [test|validation|train] [api_model_name] [prediction_model_name] [existing_output_path] [baseline_name] [baseline_type]"
    echo "Valid baseline types: api_generated_instruction, api_generated_instruction3"
    exit 1
else
    BASELINE_TYPE="$6"
fi

# Configuration
PRED_ROUTER_PORT=8000
PRED_BASE_PORT=8200
DATASET_PATH="/data/user_data/emilyx/prompt_gen/v3_3_processed/${DATASET_SPLIT}" # TODO replace with dir that you download from https://huggingface.co/datasets/milli19/promptmii-dataset

# SGLang configuration
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.9}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-4096}
SCHEDULE_CONSERVATIVENESS=${SCHEDULE_CONSERVATIVENESS:-0.3}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-256}
DTYPE=${SGLANG_DTYPE:-bfloat16}
CTX_LEN=${CTX_LEN:-32767}

echo "Configuration:"
echo "  Dataset split: $DATASET_SPLIT"
echo "  API model name: $API_MODEL_NAME"
echo "  Prediction model name: $PREDICTION_MODEL_NAME"
echo "  Existing output path: $EXISTING_OUTPUT_PATH"
echo "  Baseline name: $BASELINE_NAME"
echo "  Baseline type: $BASELINE_TYPE"
echo "  Prediction router port: $PRED_ROUTER_PORT"
echo "  Prediction worker ports: ${PRED_BASE_PORT}..$(($PRED_BASE_PORT+1))"
echo "  Dataset path: $DATASET_PATH"
echo "  Adding API baseline: $BASELINE_NAME"
echo "  mem-fraction-static: $MEM_FRACTION_STATIC"
echo "  chunked-prefill-size: $CHUNKED_PREFILL_SIZE"
echo "  max-running-requests: $MAX_RUNNING_REQUESTS"
echo "  schedule-conservativeness: $SCHEDULE_CONSERVATIVENESS"
echo "  cuda-graph-max-bs: $CUDA_GRAPH_MAX_BS"
echo "  dtype: $DTYPE, ctx: $CTX_LEN"

mkdir -p logs
mkdir -p "$EXISTING_OUTPUT_PATH"

# Launch 2 workers for Prediction role (GPUs 0–1)
PRED_PIDS=()
# Prefill worker (GPU 0)
GPU=0
PORT=$PRED_BASE_PORT
echo "Starting PREDICTION PREFILL worker gpu=$GPU port=$PORT"
CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
  --model-path "$PREDICTION_MODEL_NAME" \
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
  > logs/pred_prefill_worker_${PORT}.log 2>&1 & PRED_PIDS+=($!)

# Decode worker (GPU 1)
GPU=1
PORT=$(($PRED_BASE_PORT + 1))
echo "Starting PREDICTION DECODE worker gpu=$GPU port=$PORT"
CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
  --model-path "$PREDICTION_MODEL_NAME" \
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
  > logs/pred_decode_worker_${PORT}.log 2>&1 & PRED_PIDS+=($!)

# Launch Router for Prediction
echo "Starting PREDICTION Router on port $PRED_ROUTER_PORT"
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prometheus-port 28000 \
  --host 0.0.0.0 \
  --port $PRED_ROUTER_PORT \
  --policy power_of_two \
  --prefill http://127.0.0.1:$(($PRED_BASE_PORT+0)) \
  --decode http://127.0.0.1:$(($PRED_BASE_PORT+1)) \
  > logs/pred_router_${PRED_ROUTER_PORT}.log 2>&1 & PRED_ROUTER_PID=$!

# Health check
echo "Waiting for prediction router to be ready..."
until curl -sSf http://127.0.0.1:${PRED_ROUTER_PORT}/v1/models >/dev/null; do
  echo "  waiting pred router..."
  sleep 5
done
echo "Prediction router is up."

# Run the modular baseline evaluation script with API model for instruction generation
# Only run the API baseline
python baseline_eval_v3_modular.py \
  --instruction_model "$API_MODEL_NAME" \
  --prediction_model "$PREDICTION_MODEL_NAME" \
  --vllm_url "http://localhost:${PRED_ROUTER_PORT}/v1/chat/completions" \
  --prediction_vllm_url "http://localhost:${PRED_ROUTER_PORT}/v1/chat/completions" \
  --output_path "$EXISTING_OUTPUT_PATH" \
  --dataset_path "$DATASET_PATH" \
  --max_datasets 100 \
  --baselines "$BASELINE_TYPE" \
  --custom_baseline_name "$BASELINE_NAME" \
  --random_seed 42 \
  --max_tokens_instruction 2048 \
  --max_tokens_prediction 10

# Cleanup
echo "Stopping prediction router and workers..."
kill $PRED_ROUTER_PID || true
for p in "${PRED_PIDS[@]}"; do kill $p || true; done

# Run F1 macro analysis
F1_ANALYSIS_OUTPUT_DIR="$EXISTING_OUTPUT_PATH/analysis_f1_macro"
python analyze_baseline_results_f1_macro.py \
  --input_path "$EXISTING_OUTPUT_PATH" \
  --output_dir "$F1_ANALYSIS_OUTPUT_DIR"

echo "Job completed successfully!"
