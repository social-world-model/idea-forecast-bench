#!/usr/bin/env bash
# Run ONE single-metric GRPO mode in the foreground on explicitly chosen GPUs.
# Built for the max7 env (Qwen3.5-9B) under tight/fluctuating GPU availability:
# the caller picks free cards and runs modes sequentially / in small batches so
# we never exceed the 4-card budget.
#
# Usage:
#   run_one_grpo.sh <mode> <trainer_gpu> <rollout_gpu> [judge_gpu]
#     mode        : soft | coverage | novelty
#     trainer_gpu : GPU id for the Unsloth trainer
#     rollout_gpu : GPU id for the rollout vLLM server
#     judge_gpu   : GPU id for the judge vLLM server (required only for soft)
#
# Env overrides (with smoke-friendly defaults):
#   PYTHON_BIN  /home/max7/.conda/envs/idea-grpo/bin/python
#   MODEL_ALIAS qwen3.5-9b
#   PAPERS_DIR  data/csml_v2/raw_markdown
#   START_MONTH 2023-01   END_MONTH 2024-09
#   MAX_GRPO_ROWS 8        (smoke; set 500 for the real run)
#   NUM_GENERATIONS 4
#   MAX_COMPLETION_LENGTH 1024
#   PER_RUN_TIMEOUT 3600
#   ROLLOUT_PORT 8765   JUDGE_PORT 8766
#   JUDGE_MODEL Qwen/Qwen2.5-7B-Instruct
#   OUT_DIR     (defaults to outputs/grpo_<mode>_<ts>)
set -uo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:?usage: run_one_grpo.sh <mode> <trainer_gpu> <rollout_gpu> [judge_gpu]}"
TRAINER_GPU="${2:?need trainer_gpu}"
ROLLOUT_GPU="${3:?need rollout_gpu}"
JUDGE_GPU="${4:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/max7/.conda/envs/idea-grpo/bin/python}"
# Put the env's bin on PATH so subprocesses (flashinfer JIT -> ninja, nvcc, etc.)
# are found even though we invoke python directly instead of `conda activate`.
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3.5-9b}"
PAPERS_DIR="${PAPERS_DIR:-data/csml_v2/raw_markdown}"
START_MONTH="${START_MONTH:-2023-01}"
END_MONTH="${END_MONTH:-2024-09}"
MAX_GRPO_ROWS="${MAX_GRPO_ROWS:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-3600}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8765}"
JUDGE_PORT="${JUDGE_PORT:-8766}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-outputs/grpo_${MODE}_${TS}}"
mkdir -p "${OUT_DIR}"

if [[ "${MODE}" == "soft" && -z "${JUDGE_GPU}" ]]; then
  echo "ERROR: soft mode requires a judge_gpu (4th arg)." >&2; exit 2
fi

MODEL_ID="$("${PYTHON_BIN}" -c "from forecaster.realization.model_zoo import resolve_small_model; print(resolve_small_model('${MODEL_ALIAS}').model_id)")"
echo "============================================================"
echo "[one_grpo] mode=${MODE} model=${MODEL_ID}"
echo "  trainer GPU=${TRAINER_GPU} | rollout GPU=${ROLLOUT_GPU} -> :${ROLLOUT_PORT}"
[[ -n "${JUDGE_GPU}" ]] && echo "  judge GPU=${JUDGE_GPU} -> :${JUDGE_PORT} (${JUDGE_MODEL})"
echo "  rows=${MAX_GRPO_ROWS} gen=${NUM_GENERATIONS} comp=${MAX_COMPLETION_LENGTH} timeout=${PER_RUN_TIMEOUT}s"
echo "  out=${OUT_DIR}"
echo "============================================================"

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do
    [[ -n "$p" ]] || continue
    pkill -TERM -P "$p" 2>/dev/null || true
    kill -TERM "$p" 2>/dev/null || true
  done
  sleep 2
  for p in "${PIDS[@]:-}"; do
    [[ -n "$p" ]] || continue
    pkill -KILL -P "$p" 2>/dev/null || true
    kill -KILL "$p" 2>/dev/null || true
  done
  pkill -KILL -f "_trl_vllm_serve.py .*--port ${ROLLOUT_PORT}\b" 2>/dev/null || true
  [[ -n "${JUDGE_GPU}" ]] && pkill -KILL -f "api_server .*--port ${JUDGE_PORT}\b" 2>/dev/null || true
  # vLLM spawns EngineCore via multiprocessing; that child orphans (renamed
  # "VLLM::EngineCore") when the uvicorn parent dies. Sweep compute procs on
  # OUR rollout/judge GPUs whose exe is this env's python (so we never touch
  # the co-tenant project running a different interpreter).
  local envroot; envroot="$(dirname "$(dirname "${PYTHON_BIN}")")"
  for gpu in "${ROLLOUT_GPU}" "${JUDGE_GPU}"; do
    [[ -n "$gpu" ]] || continue
    local uuid; uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$gpu" '$1==g{print $2}')"
    [[ -n "$uuid" ]] || continue
    for pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$uuid" '$2==u{print $1}'); do
      if readlink -f "/proc/$pid/exe" 2>/dev/null | grep -q "$envroot"; then
        kill -KILL "$pid" 2>/dev/null || true
      fi
    done
  done
}
trap cleanup EXIT

wait_health() {  # wait_health <url> <pid> <secs>
  local url="$1" pid="$2" secs="$3" s=0
  while (( s < secs )); do
    if curl -sf -m 2 "${url}" >/dev/null 2>&1; then echo "[one_grpo] ready: ${url} (${s}s)"; return 0; fi
    if ! kill -0 "${pid}" 2>/dev/null; then echo "[one_grpo] server pid ${pid} died waiting on ${url}" >&2; return 1; fi
    sleep 10; s=$((s+10))
  done
  echo "[one_grpo] timeout waiting on ${url}" >&2; return 1
}

NO_VLLM_SERVER="${NO_VLLM_SERVER:-0}"

# ---- rollout vLLM (TP=1) ----
# Skipped in NO_VLLM_SERVER mode: TRL's external-server NCCL weight-sync
# deadlocks with the Unsloth + Qwen3.5 hybrid stack, so we let the trainer
# generate in-process (use_vllm=False, fast_inference=False).
if [[ "${NO_VLLM_SERVER}" != "1" ]]; then
  ROLLOUT_LOG="${OUT_DIR}/rollout_vllm.log"
  nohup env CUDA_VISIBLE_DEVICES="${ROLLOUT_GPU}" "${PYTHON_BIN}" scripts/forecaster/_trl_vllm_serve.py \
    --model "${MODEL_ID}" --port "${ROLLOUT_PORT}" --host 0.0.0.0 \
    --tensor-parallel-size 1 --max-model-len 6144 --gpu-memory-utilization 0.85 --enforce-eager \
    > "${ROLLOUT_LOG}" 2>&1 &
  ROLLOUT_PID=$!; PIDS+=("${ROLLOUT_PID}")
  echo "[one_grpo] rollout vLLM pid=${ROLLOUT_PID} log=${ROLLOUT_LOG}"
else
  echo "[one_grpo] NO_VLLM_SERVER=1 -> trainer generates in-process (no rollout server)"
fi

# ---- judge vLLM (soft only) ----
JUDGE_EXTRA=()
if [[ -n "${JUDGE_GPU}" ]]; then
  JUDGE_LOG="${OUT_DIR}/judge_vllm.log"
  nohup env CUDA_VISIBLE_DEVICES="${JUDGE_GPU}" "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model "${JUDGE_MODEL}" --port "${JUDGE_PORT}" --host 0.0.0.0 \
    --tensor-parallel-size 1 --max-model-len 8192 --gpu-memory-utilization 0.85 --dtype bfloat16 --enforce-eager \
    > "${JUDGE_LOG}" 2>&1 &
  JUDGE_PID=$!; PIDS+=("${JUDGE_PID}")
  echo "[one_grpo] judge vLLM pid=${JUDGE_PID} log=${JUDGE_LOG}"
fi

if [[ "${NO_VLLM_SERVER}" != "1" ]]; then
  wait_health "http://localhost:${ROLLOUT_PORT}/health/" "${ROLLOUT_PID}" 1200 || { tail -30 "${ROLLOUT_LOG}" >&2; exit 1; }
fi
if [[ -n "${JUDGE_GPU}" ]]; then
  wait_health "http://localhost:${JUDGE_PORT}/v1/models" "${JUDGE_PID}" 900 || { tail -30 "${JUDGE_LOG}" >&2; exit 1; }
  export JUDGE_BASE_URL="http://localhost:${JUDGE_PORT}/v1"
  export JUDGE_API_KEY="EMPTY"
  export JUDGE_MODEL="${JUDGE_MODEL}"
fi

# ---- trainer ----
TRAINER_LOG="${OUT_DIR}/trainer.log"
echo "[one_grpo] launching trainer -> ${TRAINER_LOG}"
export CUDA_VISIBLE_DEVICES="${TRAINER_GPU}"
EP_ARG=()
[[ -n "${MAX_EPISODES:-}" ]] && EP_ARG=(--max-episodes "${MAX_EPISODES}")
VLLM_ARG=(--use-vllm-server --vllm-server-port "${ROLLOUT_PORT}")
[[ "${NO_VLLM_SERVER}" == "1" ]] && VLLM_ARG=()
timeout "${PER_RUN_TIMEOUT}" "${PYTHON_BIN}" examples/forecaster/train_grpo_metric.py \
  --model "${MODEL_ALIAS}" --papers "${PAPERS_DIR}" --output-dir "${OUT_DIR}" \
  --reward-mode "${MODE}" --start-month "${START_MONTH}" --end-month "${END_MONTH}" \
  --num-generations "${NUM_GENERATIONS}" --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  --max-grpo-rows "${MAX_GRPO_ROWS}" "${EP_ARG[@]}" "${VLLM_ARG[@]}" \
  > "${TRAINER_LOG}" 2>&1
RC=$?
echo "[one_grpo] trainer rc=${RC} (124=timeout/partial-ok) out=${OUT_DIR}"
exit "${RC}"
