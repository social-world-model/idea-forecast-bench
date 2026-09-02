#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <model-path-or-hf-id> <port>" >&2
  exit 2
fi
MODEL="$1"
PORT="$2"
GPU="${GPU:-0}"
SERVED_NAMES="${SERVED_NAMES:-gpt-4.1-local local-judge}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # e.g. "--reasoning-parser qwen3"

if [[ "$MAX_MODEL_LEN" -lt 16384 ]]; then
  echo "MAX_MODEL_LEN=$MAX_MODEL_LEN is below 16384; summary_prompting requests will be rejected (see header)" >&2
  exit 1
fi

read -ra names <<< "$SERVED_NAMES"
read -ra extra <<< "$EXTRA_ARGS"

export CUDA_VISIBLE_DEVICES="$GPU"
echo "serving $MODEL on :$PORT as ${names[*]} (GPU $GPU, max_model_len $MAX_MODEL_LEN)"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "${names[@]}" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  "${extra[@]}"
