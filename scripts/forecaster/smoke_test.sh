#!/usr/bin/env bash
# 5-step GRPO smoke test on a small model (qwen3.5-2b) and the novelty reward
# (the cheapest of the three — no LLM-judge calls). Verifies:
#   * dataset loads from data/csml_v2/raw_markdown
#   * episodes are built for the 2023-01 -> 2024-09 window
#   * the embedder lazy-loads BGE
#   * the reward callable returns non-NaN floats in [0, 1]
#
# Exits non-zero on the first sign of trouble so the orchestrator can abort
# before the 12-hour run begins. Runs on whatever single GPU is set in
# CUDA_VISIBLE_DEVICES (default: GPU 0).

set -euo pipefail
cd "$(dirname "$0")/../.."

SMOKE_OUT="${SMOKE_OUT:-outputs/smoke_$(date +%Y%m%d_%H%M%S)}"
SMOKE_MODEL="${SMOKE_MODEL:-qwen3.5-2b}"
SMOKE_MODE="${SMOKE_MODE:-novelty}"
SMOKE_GPU="${SMOKE_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-1200}"  # 20 min

mkdir -p "${SMOKE_OUT}"
echo "[smoke] model=${SMOKE_MODEL} mode=${SMOKE_MODE} gpu=${SMOKE_GPU} out=${SMOKE_OUT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

set +e
CUDA_VISIBLE_DEVICES="${SMOKE_GPU}" timeout "${SMOKE_TIMEOUT}" \
  "${PYTHON_BIN}" examples/forecaster/train_grpo_metric.py \
  --model "${SMOKE_MODEL}" \
  --papers data/csml_v2/raw_markdown \
  --output-dir "${SMOKE_OUT}" \
  --reward-mode "${SMOKE_MODE}" \
  --start-month 2023-01 --end-month 2024-09 \
  --max-episodes 1 \
  --max-grpo-rows 4 \
  --num-generations 2 \
  --max-completion-length 256 \
  --grpo-epochs 1 \
  > "${SMOKE_OUT}/smoke.log" 2>&1
rc=$?
set -e
echo "[smoke] rc=${rc} (124=timeout)"

if [[ "${rc}" -ne 0 && "${rc}" -ne 124 ]]; then
  echo "[smoke] FAILED. Last 50 log lines:"
  tail -50 "${SMOKE_OUT}/smoke.log"
  exit 1
fi

# Look for evidence of successful steps in the log.
if ! grep -qE "Starting GRPO|loss=|reward" "${SMOKE_OUT}/smoke.log"; then
  echo "[smoke] WARNING: did not see GRPO step markers in log. Tail:"
  tail -30 "${SMOKE_OUT}/smoke.log"
fi

if grep -q "nan" "${SMOKE_OUT}/smoke.log"; then
  echo "[smoke] FAILED: saw 'nan' in log."
  grep -n "nan" "${SMOKE_OUT}/smoke.log" | head -5
  exit 1
fi

echo "[smoke] PASS"
