#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=baseline_modular
#SBATCH --gres=gpu:L40S:8
#SBATCH --output=logs/baseline_modular_%J.out
#SBATCH --error=logs/baseline_modular_%J.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=220G

# Usage: sbatch run_baselines_modular_sglang.sh [split] [output_folder] [seed] [instruction_model] [prediction_model]
# Example:
# sbatch run_baselines_modular_sglang.sh validation baseline_eval_v3_qwen2_5_7b_validation_sglang 42 Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct
# sbatch run_baselines_modular_sglang.sh validation baseline_eval_v3_llama3_1_8b_validation_sglang 42 meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3.1-8B-Instruct

set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
unset NCCL_P2P_DISABLE

# Args
if [ $# -lt 5 ]; then
  echo "Usage: sbatch $0 [test|validation|train] [output_folder] [seed] [instruction_model] [prediction_model]"
  exit 1
fi
DATASET_SPLIT="$1"
OUTPUT_FOLDER_NAME="$2"
RANDOM_SEED="$3"
INSTRUCTION_MODEL="$4"
PREDICTION_MODEL="$5"

# Ports / Paths
INSTR_ROUTER_PORT=8000
PRED_ROUTER_PORT=8100
INSTR_BASE_PORT=8200   # worker base ports (instruction)
PRED_BASE_PORT=8300    # worker base ports (prediction)
OUTPUT_PATH="/out/${OUTPUT_FOLDER_NAME}"
DATASET_PATH="/data/user_data/emilyx/prompt_gen/v3_3_processed/${DATASET_SPLIT}" # TODO replace with dir that you download from https://huggingface.co/datasets/milli19/promptmii-dataset
EXAMPLES_TO_TEST="5,10,20,50,100"
BASELINES="naive,naive+icl,generated_instruction3,generated_instruction"

MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.9}          # KV cache pool (effective flag)
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-4096}
SCHEDULE_CONSERVATIVENESS=${SCHEDULE_CONSERVATIVENESS:-0.3}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-256}
DTYPE=${SGLANG_DTYPE:-bfloat16}
CTX_LEN=${CTX_LEN:-32767}

echo "Configuration:"
echo "  Dataset split: $DATASET_SPLIT"
echo "  Instruction model: $INSTRUCTION_MODEL"
echo "  Prediction model:  $PREDICTION_MODEL"
echo "  Output path:       $OUTPUT_PATH"
echo "  Dataset path:      $DATASET_PATH"
echo "  Routers:           ${INSTR_ROUTER_PORT} (instr), ${PRED_ROUTER_PORT} (pred)"
echo "  Worker ports:      instr ${INSTR_BASE_PORT}..$(($INSTR_BASE_PORT+3)), pred ${PRED_BASE_PORT}..$(($PRED_BASE_PORT+3))"
echo "  mem-fraction-static: $MEM_FRACTION_STATIC"
echo "  chunked-prefill-size: $CHUNKED_PREFILL_SIZE"
echo "  max-running-requests: $MAX_RUNNING_REQUESTS"
echo "  schedule-conservativeness: $SCHEDULE_CONSERVATIVENESS"
echo "  cuda-graph-max-bs: $CUDA_GRAPH_MAX_BS"
echo "  dtype: $DTYPE, ctx: $CTX_LEN"

mkdir -p logs
mkdir -p "$OUTPUT_PATH"

# Launch 4 workers for Instruction role (GPUs 0–3)
INSTR_PIDS=()
# Prefill workers (GPUs 0–1)
for i in 0 1; do
  GPU=$i
  PORT=$(($INSTR_BASE_PORT + $i))
  echo "Starting INSTRUCTION PREFILL worker gpu=$GPU port=$PORT"
  CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
    --model-path "$INSTRUCTION_MODEL" \
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
    > logs/instr_prefill_worker_${PORT}.log 2>&1 & INSTR_PIDS+=($!)
done
# Decode workers (GPUs 2–3)
for i in 2 3; do
  GPU=$i
  PORT=$(($INSTR_BASE_PORT + $i))
  echo "Starting INSTRUCTION DECODE worker gpu=$GPU port=$PORT"
  CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
    --model-path "$INSTRUCTION_MODEL" \
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
    > logs/instr_decode_worker_${PORT}.log 2>&1 & INSTR_PIDS+=($!)
done

# Launch 4 workers for Prediction role (GPUs 4–7)
PRED_PIDS=()
# Prefill workers (GPUs 4–5)
for idx in 0 1; do
  GPU=$(($idx + 4))
  PORT=$(($PRED_BASE_PORT + $idx))
  echo "Starting PREDICTION PREFILL worker gpu=$GPU port=$PORT"
  CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
    --model-path "$PREDICTION_MODEL" \
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
done
# Decode workers (GPUs 6–7)
for idx in 2 3; do
  GPU=$(($idx + 4))
  PORT=$(($PRED_BASE_PORT + $idx))
  echo "Starting PREDICTION DECODE worker gpu=$GPU port=$PORT"
  CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
    --model-path "$PREDICTION_MODEL" \
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
done

# Launch Routers
echo "Starting INSTRUCTION Router on port $INSTR_ROUTER_PORT"
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prometheus-port 28000 \
  --host 0.0.0.0 \
  --port $INSTR_ROUTER_PORT \
  --policy power_of_two \
  --prefill http://127.0.0.1:$(($INSTR_BASE_PORT+0)) \
  --prefill http://127.0.0.1:$(($INSTR_BASE_PORT+1)) \
  --decode http://127.0.0.1:$(($INSTR_BASE_PORT+2)) \
  --decode http://127.0.0.1:$(($INSTR_BASE_PORT+3)) \
  > logs/instr_router_${INSTR_ROUTER_PORT}.log 2>&1 & INSTR_ROUTER_PID=$!

echo "Starting PREDICTION Router on port $PRED_ROUTER_PORT"
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prometheus-port 29000 \
  --host 0.0.0.0 \
  --port $PRED_ROUTER_PORT \
  --policy power_of_two \
  --prefill http://127.0.0.1:$(($PRED_BASE_PORT+0)) \
  --prefill http://127.0.0.1:$(($PRED_BASE_PORT+1)) \
  --decode http://127.0.0.1:$(($PRED_BASE_PORT+2)) \
  --decode http://127.0.0.1:$(($PRED_BASE_PORT+3)) \
  > logs/pred_router_${PRED_ROUTER_PORT}.log 2>&1 & PRED_ROUTER_PID=$!

# Health checks
echo "Waiting for routers to be ready..."
until curl -sSf http://127.0.0.1:${INSTR_ROUTER_PORT}/v1/models >/dev/null; do
  echo "  waiting instr router..."
  sleep 5
done
until curl -sSf http://127.0.0.1:${PRED_ROUTER_PORT}/v1/models >/dev/null; do
  echo "  waiting pred router..."
  sleep 5
done
echo "Routers are up."

# Run evaluation
python baseline_eval_v3_modular.py \
  --instruction_model "$INSTRUCTION_MODEL" \
  --prediction_model "$PREDICTION_MODEL" \
  --vllm_url "http://localhost:${INSTR_ROUTER_PORT}/v1/chat/completions" \
  --prediction_vllm_url "http://localhost:${PRED_ROUTER_PORT}/v1/chat/completions" \
  --output_path "$OUTPUT_PATH" \
  --dataset_path "$DATASET_PATH" \
  --max_datasets 100 \
  --baselines "$BASELINES" \
  --examples_to_test "$EXAMPLES_TO_TEST" \
  --mode "overwrite" \
  --random_seed "$RANDOM_SEED" \
  --max_tokens_instruction 2048 \
  --max_tokens_prediction 10

# Cleanup & analysis
echo "Stopping routers and workers..."
kill $INSTR_ROUTER_PID || true
kill $PRED_ROUTER_PID || true
for p in "${INSTR_PIDS[@]}"; do kill $p || true; done
for p in "${PRED_PIDS[@]}"; do kill $p || true; done

ANALYSIS_OUTPUT_DIR_F1="$OUTPUT_PATH/analysis_f1_macro"

python analyze_baseline_results_f1_macro.py \
  --input_path "$OUTPUT_PATH" \
  --output_dir "$ANALYSIS_OUTPUT_DIR_F1"

echo "Done."
