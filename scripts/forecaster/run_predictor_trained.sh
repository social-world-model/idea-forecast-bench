#!/usr/bin/env bash
# Run the predictor_llm strategy with a TRAINED Qwen3.5-9B LoRA adapter.
# Serves base Qwen3.5-9B + the adapter via vLLM (--enable-lora, no merge),
# routes the predictor to it through OPENAI_BASE_URL + a gpt-5-* alias
# (which also disables thinking), runs run_domain_backtest.py, then tears
# the server down.
#
# Usage:
#   run_predictor_trained.sh <mode> <adapter_path> <gpu> <port>
#     mode         : label, e.g. novelty | coverage | soft  (alias becomes gpt-5-<mode>)
#     adapter_path : .../final_checkpoint   (dir with adapter_config.json)
#     gpu          : GPU id to serve on
#     port         : vLLM HTTP port
#
# Env overrides:
#   PYTHON_BIN   /home/max7/.conda/envs/idea-grpo/bin/python
#   BASE_MODEL   Qwen/Qwen3.5-9B
#   PAPERS       data/csml_v2/raw_markdown
#   EVAL_START   2024-10   EVAL_END 2025-03   (held-out window; train was 2023-01..2024-09)
#   TOP_K 5   HORIZON 3   WORKERS 4
#   OUT          outputs/predict_<mode>_<ts>.json
set -uo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:?usage: run_predictor_trained.sh <mode> <adapter_path> <gpu> <port>}"
ADAPTER="${2:?need adapter_path}"
GPU="${3:?need gpu}"
PORT="${4:?need port}"

PYTHON_BIN="${PYTHON_BIN:-/home/max7/.conda/envs/idea-grpo/bin/python}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
PAPERS="${PAPERS:-data/csml_v2/raw_markdown}"
# Load papers from 2024-06 so the first test cutoff has prior-month reading
# context, but only EVALUATE cutoffs >= MIN_CUTOFF. With MIN_CUTOFF=2024-09 the
# evaluated cutoffs are 2024-09/10/11/12, whose 3-month futures span the full
# test window 2024-10..2025-03 (fixes the earlier 2-window run that started
# loading at 2024-10 and dropped the first cutoff for an empty train set).
EVAL_START="${EVAL_START:-2024-06}"
EVAL_END="${EVAL_END:-2025-03}"
MIN_CUTOFF="${MIN_CUTOFF:-2024-09}"
TOP_K="${TOP_K:-5}"
HORIZON="${HORIZON:-3}"
WORKERS="${WORKERS:-4}"
ALIAS="gpt-5-${MODE}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-outputs/predict_${MODE}_${TS}.json}"
LOG_DIR="$(dirname "${OUT}")"; mkdir -p "${LOG_DIR}"
SRV_LOG="${LOG_DIR}/vllm_${MODE}_${PORT}.log"

if [[ ! -f "${ADAPTER}/adapter_config.json" ]]; then
  echo "ERROR: no adapter_config.json under ${ADAPTER}" >&2; exit 2
fi

echo "============================================================"
echo "[predict] mode=${MODE} alias=${ALIAS}"
echo "  adapter : ${ADAPTER}"
echo "  serve   : ${BASE_MODEL} + LoRA on GPU ${GPU} :${PORT}"
echo "  window  : ${EVAL_START}..${EVAL_END} top_k=${TOP_K} horizon=${HORIZON} workers=${WORKERS}"
echo "  output  : ${OUT}"
echo "============================================================"

SRV_PID=""
cleanup() {
  [[ -n "${SRV_PID}" ]] && { pkill -TERM -P "${SRV_PID}" 2>/dev/null || true; kill -TERM "${SRV_PID}" 2>/dev/null || true; }
  sleep 2
  pkill -KILL -f "api_server .*--port ${PORT}\b" 2>/dev/null || true
  # sweep our EngineCore left on this GPU
  local uuid; uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="${GPU}" '$1==g{print $2}')"
  if [[ -n "$uuid" ]]; then
    local envroot; envroot="$(dirname "$(dirname "${PYTHON_BIN}")")"
    for pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$uuid" '$2==u{print $1}'); do
      readlink -f "/proc/$pid/exe" 2>/dev/null | grep -q "$envroot" && kill -KILL "$pid" 2>/dev/null || true
    done
  fi
}
trap cleanup EXIT

# ---- serve adapter ----
nohup env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL}" --enable-lora --max-lora-rank 16 \
  --lora-modules "${ALIAS}=${ADAPTER}" \
  --port "${PORT}" --host 0.0.0.0 \
  --max-model-len 16384 --gpu-memory-utilization 0.85 --dtype bfloat16 --enforce-eager \
  > "${SRV_LOG}" 2>&1 &
SRV_PID=$!
echo "[predict] vLLM pid=${SRV_PID} log=${SRV_LOG}"

echo "[predict] waiting for :${PORT}/v1/models ..."
ok=0
for ((s=0; s<900; s+=10)); do
  if curl -sf -m 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then echo "[predict] server ready (${s}s)"; ok=1; break; fi
  kill -0 "${SRV_PID}" 2>/dev/null || { echo "[predict] server died"; tail -30 "${SRV_LOG}" >&2; exit 1; }
  sleep 10
done
[[ "$ok" -eq 1 ]] || { echo "[predict] server never came up"; tail -30 "${SRV_LOG}" >&2; exit 1; }

# ---- run predictor backtest ----
export OPENAI_BASE_URL="http://localhost:${PORT}/v1"
export OPENAI_API_KEY="EMPTY"
echo "[predict] running run_domain_backtest.py ..."
"${PYTHON_BIN}" examples/benchmark/run_domain_backtest.py \
  --strategy predictor_llm \
  --model-name "${ALIAS}" \
  --input-dir "${PAPERS}" \
  --start-month "${EVAL_START}" --end-month "${EVAL_END}" \
  --min-cutoff-month "${MIN_CUTOFF}" \
  --top-k "${TOP_K}" --horizon-months "${HORIZON}" \
  --similarity-engine heuristic --workers "${WORKERS}" \
  --output "${OUT}"
RC=$?
echo "[predict] mode=${MODE} rc=${RC} -> ${OUT}"
exit "${RC}"
