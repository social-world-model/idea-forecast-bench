#!/usr/bin/env bash
# One vLLM replica per GPU, all serving the same model under the same aliases.
#
#   GPUS="0 1 2" BASE_PORT=31000 bash scripts/benchmark/serve_multi.sh Qwen/Qwen3.5-9B
#   bash scripts/benchmark/serve_multi.sh stop
#
# Replica i listens on BASE_PORT+i. Logs go to $LOG_DIR/gpu<i>.log and PIDs to
# $LOG_DIR/pids. The script returns once every replica answers /v1/models, so
# it can be followed directly by a run. A 9B model in bf16 fits one 48 GB card
# with room for a 16k context, which is what the benchmark prompts need.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_DIR="${LOG_DIR:-output/serve}"
PID_FILE="$LOG_DIR/pids"

if [[ "${1:-}" == "stop" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    while read -r pid; do
      [[ -n "$pid" ]] && kill "$pid" 2>/dev/null && echo "stopped $pid" || true
    done < "$PID_FILE"
    rm -f "$PID_FILE"
  else
    echo "no $PID_FILE; nothing to stop"
  fi
  exit 0
fi

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <model-path-or-hf-id> | stop" >&2
  exit 2
fi
MODEL="$1"
GPUS="${GPUS:-0}"
BASE_PORT="${BASE_PORT:-31000}"
SERVED_NAMES="${SERVED_NAMES:-gpt-4o-qwen35 qwen35-judge}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
BACKEND="${BACKEND:-vllm}"          # vllm | sglang
EXTRA_ARGS="${EXTRA_ARGS:-}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"

if [[ "$MAX_MODEL_LEN" -lt 16384 ]]; then
  echo "MAX_MODEL_LEN=$MAX_MODEL_LEN is below 16384; the client asks for 4096 output tokens on ~4k prompts" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
: > "$PID_FILE"
read -ra names <<< "$SERVED_NAMES"
read -ra gpus <<< "$GPUS"
read -ra extra <<< "$EXTRA_ARGS"

ports=()
i=0
for gpu in "${gpus[@]}"; do
  port=$((BASE_PORT + i))
  ports+=("$port")
  log="$LOG_DIR/gpu${gpu}.log"
  echo "replica $i: GPU $gpu -> :$port ($BACKEND, $MODEL as ${names[*]}) log=$log"
  if [[ "$BACKEND" == "vllm" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" nohup python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --served-model-name "${names[@]}" \
      --port "$port" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-model-len "$MAX_MODEL_LEN" \
      "${extra[@]}" > "$log" 2>&1 &
  else
    # SGLang serves one name; the client aliases both to it via --served-model-name.
    CUDA_VISIBLE_DEVICES="$gpu" nohup python -m sglang.launch_server \
      --model-path "$MODEL" \
      --served-model-name "${names[0]}" \
      --port "$port" \
      --context-length "$MAX_MODEL_LEN" \
      --mem-fraction-static "$GPU_MEM_UTIL" \
      "${extra[@]}" > "$log" 2>&1 &
  fi
  echo $! >> "$PID_FILE"
  i=$((i + 1))
done

echo "waiting for ${#ports[@]} replica(s) to answer /v1/models (timeout ${READY_TIMEOUT}s) ..."
deadline=$((SECONDS + READY_TIMEOUT))
for port in "${ports[@]}"; do
  until curl -sf "http://127.0.0.1:${port}/v1/models" > /dev/null; do
    if (( SECONDS > deadline )); then
      echo "replica on :$port did not come up; see $LOG_DIR" >&2
      exit 1
    fi
    sleep 5
  done
  echo "  :$port ready"
done
echo "PORTS=\"${ports[*]}\""
