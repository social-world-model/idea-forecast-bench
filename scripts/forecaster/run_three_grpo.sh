#!/usr/bin/env bash
# Sequentially run three GRPO experiments on Qwen3.5-9B, each optimizing
# ONE eval metric (soft, coverage, novelty) over the train window
# 2023-01 → 2024-09. Total wall-clock budget ~12 hours.
#
# GPU layout (8x A6000, 48 GB each):
#   Soft     : trainer 4 GPUs DDP | rollout vLLM 2 GPUs TP=2 | judge vLLM 2 GPUs TP=2
#   Coverage : trainer 4 GPUs DDP | rollout vLLM 4 GPUs TP=4
#   Novelty  : trainer 4 GPUs DDP | rollout vLLM 4 GPUs TP=4
#
# Env overrides:
#   MODEL_ALIAS=qwen3.5-9b
#   PAPERS_DIR=data/csml/raw_markdown
#   START_MONTH=2023-01  END_MONTH=2024-09
#   PER_RUN_TIMEOUT=14400   (seconds; 4h default)
#   GPUS=0,1,2,3,4,5,6,7
#   NUM_GENERATIONS=4   (override grpo_train.yaml; lower if 4h is tight)
#   MAX_COMPLETION_LENGTH=1024
#   JUDGE_PORT=8766  ROLLOUT_PORT=8765
#   SMOKE=1                 (run smoke test first, abort if it fails)
#   HEALTH_CHECK=1          (start a 15-min health probe loop)
#   MODES="soft coverage novelty"  (space-separated subset)
#
# Outputs:
#   outputs/grpo_<mode>_<ts>/

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL_ALIAS="${MODEL_ALIAS:-qwen3.5-9b}"
PAPERS_DIR="${PAPERS_DIR:-data/csml/raw_markdown}"
START_MONTH="${START_MONTH:-2023-01}"
END_MONTH="${END_MONTH:-2024-09}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-14400}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
JUDGE_PORT="${JUDGE_PORT:-8766}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8765}"
SMOKE="${SMOKE:-1}"
HEALTH_CHECK="${HEALTH_CHECK:-1}"
MODES="${MODES:-soft coverage novelty}"

TS="$(date +%Y%m%d_%H%M%S)"
ROOT_OUT="outputs"
mkdir -p "${ROOT_OUT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NUM_GPUS=${#GPU_ARR[@]}
if [[ "${NUM_GPUS}" -lt 4 ]]; then
  echo "ERROR: run_three_grpo expects >=4 GPUs (got ${NUM_GPUS}: ${GPUS})." >&2
  exit 1
fi

# Helper: pick a GPU subset
slice_gpus() {  # slice_gpus <start> <count>
  local start="$1" count="$2"
  local out=""
  for ((i=0; i<count; i++)); do
    [[ -n "$out" ]] && out+=","
    out+="${GPU_ARR[$((start+i))]}"
  done
  echo "${out}"
}

run_mode() {
  local mode="$1"
  local out_dir="${ROOT_OUT}/grpo_${mode}_${TS}"
  mkdir -p "${out_dir}"

  local trainer_gpus rollout_gpus judge_gpus rollout_tp judge_tp
  # NCCL on multi-GPU vLLM deadlocks on this stack (vllm cpp extensions
  # skipped because torch 2.8 < 2.11). Standalone TP=1 works. So we pin
  # all vLLM servers to TP=1 and give the rest of the GPUs to the trainer.
  if [[ "${mode}" == "soft" ]]; then
    trainer_gpus=$(slice_gpus 0 6)
    rollout_gpus=$(slice_gpus 6 1)
    judge_gpus=$(slice_gpus 7 1)
    rollout_tp=1
    judge_tp=1
  else
    trainer_gpus=$(slice_gpus 0 7)
    rollout_gpus=$(slice_gpus 7 1)
    judge_gpus=""
    rollout_tp=1
    judge_tp=0
  fi

  echo "============================================================"
  echo "[run_three_grpo] mode=${mode}"
  echo "  trainer GPUs : ${trainer_gpus}"
  echo "  rollout GPUs : ${rollout_gpus} (tp=${rollout_tp}) :${ROLLOUT_PORT}"
  if [[ -n "${judge_gpus}" ]]; then
    echo "  judge GPUs   : ${judge_gpus} (tp=${judge_tp}) :${JUDGE_PORT}"
  fi
  echo "  output       : ${out_dir}"
  echo "============================================================"

  # Judge server (only for soft).
  if [[ "${mode}" == "soft" ]]; then
    JUDGE_CUDA="${judge_gpus}" JUDGE_TP="${judge_tp}" JUDGE_PORT="${JUDGE_PORT}" \
      bash scripts/forecaster/serve_judge.sh
    export JUDGE_BASE_URL="http://localhost:${JUDGE_PORT}/v1"
    export JUDGE_API_KEY="EMPTY"
    export JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
  fi

  # Health check (15-min loop) in background.
  local hc_pid=""
  if [[ "${HEALTH_CHECK}" == "1" ]]; then
    bash scripts/forecaster/health_check.sh "${out_dir}" "${ROLLOUT_PORT}" "${JUDGE_PORT}" "${mode}" &
    hc_pid=$!
    echo "[run_three_grpo] health check pid=${hc_pid}"
  fi

  # Resolved model id (needed for the rollout vLLM server).
  local MODEL_ID
  MODEL_ID=$("${PYTHON_BIN}" -c "from forecaster.realization.model_zoo import resolve_small_model; print(resolve_small_model('${MODEL_ALIAS}').model_id)")
  echo "[run_three_grpo] rollout model: ${MODEL_ID}"

  # Rollout vLLM server.
  local rollout_log="${out_dir}/rollout_vllm.log"
  local rollout_pid_file="${out_dir}/rollout_vllm.pid"
  nohup env CUDA_VISIBLE_DEVICES="${rollout_gpus}" "${PYTHON_BIN}" examples/forecaster/_trl_vllm_serve.py \
    --model "${MODEL_ID}" \
    --port "${ROLLOUT_PORT}" \
    --host 0.0.0.0 \
    --tensor-parallel-size "${rollout_tp}" \
    --max-model-len 6144 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    > "${rollout_log}" 2>&1 &
  echo $! > "${rollout_pid_file}"
  echo "[run_three_grpo] rollout vLLM pid=$(cat "${rollout_pid_file}") log=${rollout_log}"

  # Wait for rollout server (model load + KV cache init can take 2-4 min).
  echo "[run_three_grpo] Waiting for rollout vLLM on :${ROLLOUT_PORT} ..."
  for ((s=0; s<900; s+=10)); do
    if curl -sf -m 2 "http://localhost:${ROLLOUT_PORT}/health/" > /dev/null 2>&1; then
      echo "[run_three_grpo] rollout ready after ${s}s"
      break
    fi
    if [[ -f "${rollout_pid_file}" ]] && ! kill -0 "$(cat "${rollout_pid_file}")" 2>/dev/null; then
      echo "[run_three_grpo] rollout vLLM process died during startup."
      break
    fi
    sleep 10
  done
  if ! curl -sf -m 2 "http://localhost:${ROLLOUT_PORT}/health/" > /dev/null 2>&1; then
    echo "ERROR: rollout vLLM did not come up. Tail of log:" >&2
    tail -30 "${rollout_log}" >&2 || true
    [[ -n "${hc_pid}" ]] && kill "${hc_pid}" 2>/dev/null || true
    exit 1
  fi

  # Trainer.
  local TORCHRUN_BIN
  TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"
  if [[ ! -x "${TORCHRUN_BIN}" ]]; then
    TORCHRUN_BIN="torchrun"
  fi

  local trainer_log="${out_dir}/trainer.log"
  local trainer_nproc
  trainer_nproc=$(awk -F, '{print NF}' <<< "${trainer_gpus}")
  echo "[run_three_grpo] Launching trainer nproc=${trainer_nproc} (timeout ${PER_RUN_TIMEOUT}s) -> ${trainer_log}"
  set +e
  CUDA_VISIBLE_DEVICES="${trainer_gpus}" timeout "${PER_RUN_TIMEOUT}" \
    "${TORCHRUN_BIN}" --standalone --nproc_per_node="${trainer_nproc}" \
    examples/forecaster/train_grpo_metric.py \
    --model "${MODEL_ALIAS}" \
    --papers "${PAPERS_DIR}" \
    --output-dir "${out_dir}" \
    --reward-mode "${mode}" \
    --start-month "${START_MONTH}" \
    --end-month "${END_MONTH}" \
    --num-generations "${NUM_GENERATIONS}" \
    --max-completion-length "${MAX_COMPLETION_LENGTH}" \
    --use-vllm-server \
    --vllm-server-port "${ROLLOUT_PORT}" \
    > "${trainer_log}" 2>&1
  local rc=$?
  set -e
  echo "[run_three_grpo] trainer rc=${rc}"

  # Tear down servers.
  if [[ -f "${rollout_pid_file}" ]]; then
    local rp; rp=$(cat "${rollout_pid_file}")
    pkill -TERM -P "${rp}" 2>/dev/null || true
    kill -TERM "${rp}" 2>/dev/null || true
    sleep 3
    pkill -KILL -P "${rp}" 2>/dev/null || true
    kill -KILL "${rp}" 2>/dev/null || true
    rm -f "${rollout_pid_file}"
  fi
  pkill -KILL -f "trl.scripts.vllm_serve.*--port ${ROLLOUT_PORT}" 2>/dev/null || true

  if [[ "${mode}" == "soft" ]]; then
    JUDGE_PORT="${JUDGE_PORT}" bash scripts/forecaster/serve_judge.sh stop || true
  fi

  if [[ -n "${hc_pid}" ]]; then
    kill "${hc_pid}" 2>/dev/null || true
  fi

  if [[ "${rc}" -ne 0 && "${rc}" -ne 124 ]]; then
    # 124 = timeout; we accept that as "ran out of time, partial checkpoint ok"
    echo "ERROR: trainer for mode=${mode} exited with rc=${rc}." >&2
    return "${rc}"
  fi
  return 0
}

# Optional smoke test before any real run.
if [[ "${SMOKE}" == "1" ]]; then
  echo "[run_three_grpo] Smoke test ..."
  bash scripts/forecaster/smoke_test.sh
fi

for mode in ${MODES}; do
  run_mode "${mode}"
done

echo "[run_three_grpo] All runs done. Outputs in ${ROOT_OUT}/"
