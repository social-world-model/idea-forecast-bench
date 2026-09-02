#!/usr/bin/env bash
# Serve one model with vLLM as an OpenAI-compatible endpoint for the benchmark.
#
# Two requirements that are easy to get wrong, both of which fail silently:
#
# 1. The served name decides the routing. The benchmark's LLM client only sends
#    a model to OPENAI_BASE_URL when its name starts with gpt-4o / gpt-4.1 /
#    gpt-5 (idea_forecast_bench/llm.py, _is_openai_model), so a local backbone
#    must be served under an alias like `gpt-4o-qwen7b`. The judge is addressed
#    by --judge-model and needs no alias. Serving one model under BOTH names
#    lets a single GPU handle generation and judging without a reload, but if
#    the judge's name is missing the judge gets a 404 and every judgement is
#    stored as zero on all three dimensions -- the run finishes, the numbers
#    look plausible, and nothing errors.
#
# 2. --max-model-len must be at least 16384. The client fixes the output
#    budget at 4096 tokens (llm.py, MAX_NUM_TOKENS) and summary_prompting's
#    prompt is ~4100 tokens, so at 8192 every request is rejected, window by
#    window; the run then ends with total_windows=0 and a clean exit code.
#
# Usage:
#   bash scripts/benchmark/serve_vllm.sh <model-path-or-hf-id> <port>
#
# Environment:
#   GPU             CUDA_VISIBLE_DEVICES value                 0
#   SERVED_NAMES    space-separated served names               gpt-4.1-local local-judge
#   MAX_MODEL_LEN   context length                             16384
#   GPU_MEM_UTIL    --gpu-memory-utilization                   0.85
#   HF_HUB_OFFLINE  set to 1 to forbid downloads               (unset)
#   On Blackwell (sm_120) cards vLLM needed VLLM_USE_FLASHINFER_SAMPLER=0 and
#   TORCH_CUDA_ARCH_LIST=12.0 exported before launch; set them yourself if so.
#
# Examples:
#   bash scripts/benchmark/serve_vllm.sh Qwen/Qwen3.5-9B 31000                  # judge
#   SERVED_NAMES="gpt-4o-qwen7b" bash scripts/benchmark/serve_vllm.sh /models/Qwen2.5-7B-Instruct 31001
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

if [[ "$MAX_MODEL_LEN" -lt 16384 ]]; then
  echo "MAX_MODEL_LEN=$MAX_MODEL_LEN is below 16384; summary_prompting requests will be rejected (see header)" >&2
  exit 1
fi

read -ra names <<< "$SERVED_NAMES"

export CUDA_VISIBLE_DEVICES="$GPU"
echo "serving $MODEL on :$PORT as ${names[*]} (GPU $GPU, max_model_len $MAX_MODEL_LEN)"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "${names[@]}" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN"
