#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=verl_trained_instr_sglang
#SBATCH --gres=gpu:L40S:4         
#SBATCH --output=logs/add_trained_instruction_sglang_%J.out
#SBATCH --error=logs/add_trained_instruction_sglang_%J.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=100G

# Script to add trained_generated_instruction baseline to existing results using SGLang
# Usage: sbatch add_trained_instruction_baseline_sglang.sh [test|validation|train] [trained_model_path] [prediction_model_name] [existing_output_path] [trained_baseline_name] [baseline_type] [random_seed]

### Llama - validation
# sbatch add_trained_instruction_baseline_sglang.sh \
#   validation \
#   milli19/promptmii-qwen-2.5-7b-instruct \
#   meta-llama/Meta-Llama-3.1-8B-Instruct \
#   out/out4/baseline_eval_v3_llama3_1_8b_validation_sglang \
#   trained_generated_instruction

### Qwen - validation  
# sbatch add_trained_instruction_baseline_sglang.sh \
#   validation \
#   milli19/promptmii-llama-3.1-8b-instruct \
#   Qwen/Qwen2.5-7B-Instruct \
#   out/out4/baseline_eval_v3_qwen2_5_7b_validation_sglang \
#   trained_generated_instruction

set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
unset NCCL_P2P_DISABLE

# Check if baseline type is provided
if [ $# -lt 6 ]; then
    BASELINE_TYPE="trained_generated_instruction"
else
    BASELINE_TYPE="$6"
fi

# Set random seed. base is 42
if [ $# -lt 7 ]; then
    RANDOM_SEED=42
else
    RANDOM_SEED="$7"
fi

# Check if dataset split argument is provided
if [ $# -eq 0 ]; then
    DATASET_SPLIT="test"
else
    DATASET_SPLIT="$1"
fi

# Check if trained model path is provided
if [ $# -lt 2 ]; then
    echo "Error: Please provide the trained model path"
    echo "Usage: sbatch add_trained_instruction_baseline_sglang.sh [test|validation|train] [trained_model_path] [prediction_model_name] [existing_output_path] [trained_baseline_name] [baseline_type] [random_seed]"
    exit 1
else
    TRAINED_MODEL_PATH="$2"
fi

# Check if prediction model name is provided
if [ $# -lt 3 ]; then
    echo "Error: Please provide the prediction model name"
    echo "Usage: sbatch add_trained_instruction_baseline_sglang.sh [test|validation|train] [trained_model_path] [prediction_model_name] [existing_output_path] [trained_baseline_name] [baseline_type] [random_seed]"
    exit 1
else
    PREDICTION_MODEL_NAME="$3"
fi

# Check if existing output path is provided
if [ $# -lt 4 ]; then
    echo "Error: Please provide the existing output path"
    echo "Usage: sbatch add_trained_instruction_baseline_sglang.sh [test|validation|train] [trained_model_path] [prediction_model_name] [existing_output_path] [trained_baseline_name] [baseline_type] [random_seed]"
    exit 1
else
    EXISTING_OUTPUT_PATH="$4"
fi

# Check if trained baseline name is provided
if [ $# -lt 5 ]; then
    echo "Error: Please provide the trained baseline name"
    echo "Usage: sbatch add_trained_instruction_baseline_sglang.sh [test|validation|train] [trained_model_path] [prediction_model_name] [existing_output_path] [trained_baseline_name] [baseline_type] [random_seed]"
    exit 1
else
    TRAINED_BASELINE_NAME="$5"
fi

# Configuration
TRAINED_ROUTER_PORT=8000
TRAINED_BASE_PORT=8200
PRED_ROUTER_PORT=8100
PRED_BASE_PORT=8300
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
echo "  Trained model path: $TRAINED_MODEL_PATH"
echo "  Prediction model name: $PREDICTION_MODEL_NAME"
echo "  Existing output path: $EXISTING_OUTPUT_PATH"
echo "  Trained baseline name: $TRAINED_BASELINE_NAME"
echo "  Baseline type: $BASELINE_TYPE"
echo "  Random seed: $RANDOM_SEED"
echo "  Trained router port: $TRAINED_ROUTER_PORT"
echo "  Trained worker ports: ${TRAINED_BASE_PORT}..$(($TRAINED_BASE_PORT+1))"
echo "  Prediction router port: $PRED_ROUTER_PORT"
echo "  Prediction worker ports: ${PRED_BASE_PORT}..$(($PRED_BASE_PORT+1))"
echo "  Dataset path: $DATASET_PATH"
echo "  Adding baseline: $TRAINED_BASELINE_NAME"
echo "  mem-fraction-static: $MEM_FRACTION_STATIC"
echo "  chunked-prefill-size: $CHUNKED_PREFILL_SIZE"
echo "  max-running-requests: $MAX_RUNNING_REQUESTS"
echo "  schedule-conservativeness: $SCHEDULE_CONSERVATIVENESS"
echo "  cuda-graph-max-bs: $CUDA_GRAPH_MAX_BS"
echo "  dtype: $DTYPE, ctx: $CTX_LEN"

mkdir -p logs
mkdir -p "$EXISTING_OUTPUT_PATH"

# Launch 2 workers for Trained model (GPUs 0-1)
TRAINED_PIDS=()
# Prefill worker (GPU 0)
GPU=0
PORT=$TRAINED_BASE_PORT
echo "Starting TRAINED PREFILL worker gpu=$GPU port=$PORT"
CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
  --model-path "$TRAINED_MODEL_PATH" \
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
  > logs/trained_prefill_worker_${PORT}.log 2>&1 & TRAINED_PIDS+=($!)

# Decode worker (GPU 1)
GPU=1
PORT=$(($TRAINED_BASE_PORT + 1))
echo "Starting TRAINED DECODE worker gpu=$GPU port=$PORT"
CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
  --model-path "$TRAINED_MODEL_PATH" \
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
  > logs/trained_decode_worker_${PORT}.log 2>&1 & TRAINED_PIDS+=($!)

# Launch Router for Trained model
echo "Starting TRAINED Router on port $TRAINED_ROUTER_PORT"
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prometheus-port 28001 \
  --host 0.0.0.0 \
  --port $TRAINED_ROUTER_PORT \
  --policy power_of_two \
  --prefill http://127.0.0.1:$(($TRAINED_BASE_PORT+0)) \
  --decode http://127.0.0.1:$(($TRAINED_BASE_PORT+1)) \
  > logs/trained_router_${TRAINED_ROUTER_PORT}.log 2>&1 & TRAINED_ROUTER_PID=$!

# Launch 2 workers for Prediction model (GPUs 2-3)
PRED_PIDS=()
# Prefill worker (GPU 2)
GPU=2
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

# Decode worker (GPU 3)
GPU=3
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

# Launch Router for Prediction model
echo "Starting PREDICTION Router on port $PRED_ROUTER_PORT"
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prometheus-port 28002 \
  --host 0.0.0.0 \
  --port $PRED_ROUTER_PORT \
  --policy power_of_two \
  --prefill http://127.0.0.1:$(($PRED_BASE_PORT+0)) \
  --decode http://127.0.0.1:$(($PRED_BASE_PORT+1)) \
  > logs/pred_router_${PRED_ROUTER_PORT}.log 2>&1 & PRED_ROUTER_PID=$!

# Health check for trained model router
echo "Waiting for trained model router to be ready..."
until curl -sSf http://127.0.0.1:${TRAINED_ROUTER_PORT}/v1/models >/dev/null; do
  echo "  waiting trained router..."
  sleep 5
done
echo "Trained model router is up."

# Health check for prediction model router
echo "Waiting for prediction model router to be ready..."
until curl -sSf http://127.0.0.1:${PRED_ROUTER_PORT}/v1/models >/dev/null; do
  echo "  waiting prediction router..."
  sleep 5
done
echo "Prediction model router is up."

# Run the modular baseline evaluation script with trained model for instruction generation
# Only run the trained_generated_instruction baseline
python baseline_eval_v3_modular.py \
  --instruction_model "$TRAINED_MODEL_PATH" \
  --prediction_model "$PREDICTION_MODEL_NAME" \
  --trained_instruction_model "$TRAINED_MODEL_PATH" \
  --custom_baseline_name "$TRAINED_BASELINE_NAME" \
  --vllm_url "http://localhost:${TRAINED_ROUTER_PORT}/v1/chat/completions" \
  --prediction_vllm_url "http://localhost:${PRED_ROUTER_PORT}/v1/chat/completions" \
  --output_path "$EXISTING_OUTPUT_PATH" \
  --dataset_path "$DATASET_PATH" \
  --max_datasets 100 \
  --baselines "$BASELINE_TYPE" \
  --random_seed $RANDOM_SEED \
  --max_tokens_instruction 2048 \
  --max_tokens_prediction 10

# Cleanup
echo "Stopping trained model router and workers..."
kill $TRAINED_ROUTER_PID || true
for p in "${TRAINED_PIDS[@]}"; do kill $p || true; done

echo "Stopping prediction model router and workers..."
kill $PRED_ROUTER_PID || true
for p in "${PRED_PIDS[@]}"; do kill $p || true; done

# Run F1 analysis as well
F1_ANALYSIS_OUTPUT_DIR="$EXISTING_OUTPUT_PATH/f1_analysis_macro"
python analyze_baseline_results_f1_macro.py \
  --input_path "$EXISTING_OUTPUT_PATH" \
  --output_dir "$F1_ANALYSIS_OUTPUT_DIR"

echo "Job completed successfully!"