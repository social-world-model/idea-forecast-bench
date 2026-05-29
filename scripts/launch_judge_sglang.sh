#!/usr/bin/env bash
# Launch SGLang serving Qwen3.5-9B-Instruct as a JUDGE-only OpenAI-compatible endpoint.
#
# Differs from scripts/launch_sglang.sh in three ways:
#   1. No LoRA attached — the judge is the base model, not the trained policy.
#   2. Sized for a single inference workload, not concurrent training rollouts.
#   3. Defaults the served model name to `qwen3.5-9b-instruct` to match
#      scripts/phase2_rubric_validation.py's expectation.
#
# Usage:
#   bash scripts/launch_judge_sglang.sh                       # foreground
#   nohup bash scripts/launch_judge_sglang.sh > judge.log 2>&1 &   # background
#
# Override via env: PORT, HOST, MODEL_ALIAS, MEM_FRACTION_STATIC.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/jiayey3/.conda/envs/ideabench/bin/python}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3.5-9b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-instruct}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"

BASE_MODEL_ID=$("$PYTHON_BIN" -c "
from forecaster.realization.model_zoo import resolve_small_model
print(resolve_small_model('${MODEL_ALIAS}').model_id)
")

echo "=============================================="
echo " SGLang JUDGE server"
echo "  Base model:        ${BASE_MODEL_ID}"
echo "  Served model name: ${SERVED_MODEL_NAME}"
echo "  Endpoint:          http://${HOST}:${PORT}/v1"
echo "=============================================="

exec "$PYTHON_BIN" -m sglang.launch_server \
    --model-path "$BASE_MODEL_ID" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype bfloat16 \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --trust-remote-code
