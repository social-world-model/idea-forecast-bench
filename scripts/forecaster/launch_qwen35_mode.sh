#!/usr/bin/env bash
# Launch ONE GRPO mode against the qwen35-train env (Qwen3.5-9B + cuda-compat).
#
# Args:
#   $1 mode         soft | novelty | coverage
#   $2 trainer_gpu  e.g. 6
#   $3 rollout_gpu  e.g. 7
#   $4 rollout_port e.g. 9101
#   $5 group_port   e.g. 51216
#   $6 judge_gpus   optional comma-list for soft, e.g. "8,9"
#   $7 judge_port   optional, e.g. 9201
#
# Writes pids and logs under outputs/grpo_<mode>_<TS>/

set -uo pipefail
cd "$(dirname "$0")/../.."

MODE="$1"
TRAINER_GPU="$2"
ROLLOUT_GPU="$3"
ROLLOUT_PORT="$4"
GROUP_PORT="$5"
JUDGE_GPUS="${6:-}"
JUDGE_PORT="${7:-}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/grpo_${MODE}_${TS}"
mkdir -p "$OUT_DIR"
echo "OUT_DIR=$OUT_DIR" > "/tmp/grpo_${MODE}_dir.txt"

PYTHON_BIN=/data/haofeiy2/anaconda3/envs/qwen35-train/bin/python
COMPAT_LIB=/data/haofeiy2/anaconda3/envs/qwen35-train/cuda-compat

# ------- Rollout vLLM (Qwen3.5-9B, TP=1) -------
echo "[$MODE] rollout vLLM GPU=$ROLLOUT_GPU port=$ROLLOUT_PORT"
nohup env CUDA_VISIBLE_DEVICES="$ROLLOUT_GPU" \
  NCCL_P2P_DISABLE=1 \
  LD_LIBRARY_PATH="$COMPAT_LIB:${LD_LIBRARY_PATH:-}" \
  PATH="/data/haofeiy2/anaconda3/envs/qwen35-train/bin:$PATH" \
  "$PYTHON_BIN" scripts/forecaster/_trl_vllm_serve.py \
  --model Qwen/Qwen3.5-9B \
  --port "$ROLLOUT_PORT" --host 127.0.0.1 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.70 \
  --enforce-eager \
  > "$OUT_DIR/rollout_vllm.log" 2>&1 &
echo $! > "$OUT_DIR/rollout_vllm.pid"

# ------- Judge vLLM (only for soft) -------
if [[ "$MODE" == "soft" && -n "$JUDGE_GPUS" && -n "$JUDGE_PORT" ]]; then
  local_tp=$(awk -F, '{print NF}' <<< "$JUDGE_GPUS")
  echo "[$MODE] judge vLLM GPUs=$JUDGE_GPUS port=$JUDGE_PORT TP=$local_tp"
  nohup env CUDA_VISIBLE_DEVICES="$JUDGE_GPUS" \
    NCCL_P2P_DISABLE=1 \
    LD_LIBRARY_PATH="$COMPAT_LIB:${LD_LIBRARY_PATH:-}" \
    PATH="/data/haofeiy2/anaconda3/envs/qwen35-train/bin:$PATH" \
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B \
    --port "$JUDGE_PORT" --host 127.0.0.1 \
    --tensor-parallel-size "$local_tp" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --reasoning-parser qwen3 \
    > "$OUT_DIR/judge_vllm.log" 2>&1 &
  echo $! > "$OUT_DIR/judge_vllm.pid"
fi

# ------- Wait for rollout ready -------
echo "[$MODE] waiting for rollout vLLM ..."
for s in $(seq 0 10 900); do
  if curl -sf -m 2 "http://localhost:${ROLLOUT_PORT}/health/" >/dev/null 2>&1; then
    echo "[$MODE] rollout ready in ${s}s"; break
  fi
  if ! kill -0 "$(cat "$OUT_DIR/rollout_vllm.pid")" 2>/dev/null; then
    echo "[$MODE] rollout died"; exit 1
  fi
  sleep 10
done

# ------- Wait for judge ready (if any) -------
if [[ "$MODE" == "soft" && -n "$JUDGE_PORT" ]]; then
  echo "[$MODE] waiting for judge ..."
  for s in $(seq 0 10 900); do
    if curl -sf -m 2 "http://localhost:${JUDGE_PORT}/v1/models" >/dev/null 2>&1; then
      echo "[$MODE] judge ready in ${s}s"; break
    fi
    sleep 10
  done
fi

# ------- Launch trainer -------
echo "[$MODE] launching trainer on GPU $TRAINER_GPU (group_port=$GROUP_PORT)"
extra_env=()
if [[ "$MODE" == "soft" && -n "$JUDGE_PORT" ]]; then
  extra_env+=("JUDGE_BASE_URL=http://localhost:${JUDGE_PORT}/v1" \
              "JUDGE_API_KEY=EMPTY" "JUDGE_MODEL=Qwen/Qwen3.5-9B")
fi

nohup env CUDA_VISIBLE_DEVICES="$TRAINER_GPU" \
  LIB_VLLM_GROUP_PORT="$GROUP_PORT" \
  NCCL_P2P_DISABLE=1 \
  LD_LIBRARY_PATH="$COMPAT_LIB:${LD_LIBRARY_PATH:-}" \
  PATH="/data/haofeiy2/anaconda3/envs/qwen35-train/bin:$PATH" \
  "${extra_env[@]}" \
  timeout 18000 "$PYTHON_BIN" examples/forecaster/train_grpo_metric.py \
  --model qwen3.5-9b \
  --papers data/csml_v2/raw_markdown \
  --output-dir "$OUT_DIR" \
  --reward-mode "$MODE" \
  --start-month 2023-01 --end-month 2024-09 \
  --num-generations 4 \
  --max-completion-length 768 \
  --max-grpo-rows 480 \
  --use-vllm-server --vllm-server-port "$ROLLOUT_PORT" \
  > "$OUT_DIR/trainer.log" 2>&1 &
echo $! > "$OUT_DIR/trainer.pid"
echo "[$MODE] trainer pid=$(cat $OUT_DIR/trainer.pid)"
