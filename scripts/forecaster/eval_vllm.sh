#!/usr/bin/env bash
# ============================================================================
#  Multi-GPU vLLM-accelerated forecaster eval.
#
#  Mirrors scripts/forecaster/train_vllm.sh: starts vLLM server(s) before
#  invoking the Python entry point. The new entry point is
#  examples/forecaster/eval.py — same shape as examples/forecaster/train.py.
#
#  Eval needs TWO LM endpoints:
#    1. Prior LoRA       (p_θ(z|M_t))   — used by sample_innovations
#    2. Realization LoRA (p_ψ(y|z,X))   — used by generate_proposals_batch
#
#  We launch ONE vLLM server per LoRA, each on its own GPU subset, both
#  serving the same base model with --enable-lora --lora-modules <name>=<path>.
#  vLLM applies the LoRA at inference time — no merge step needed.
#
#  The existing helpers in forecaster/prior/sampler.py:_sglang_prior_url
#  and forecaster/realization/proposal_generator.py:_sglang_api_url speak
#  plain OpenAI /v1/completions and /v1/chat/completions, so the env-var
#  names (SGLANG_PRIOR_URL / SGLANG_URL) are historical — vLLM works fine.
#
#  REQUIREMENTS:
#    * 2+ GPUs (one per server — vLLM server can't share a device with
#      another vLLM instance loading a separate model)
#    * vLLM installed (handled by scripts/forecaster/setup_env.sh)
#
#  TOPOLOGY:
#    2 GPUs:  prior=1, realization=1
#    4 GPUs:  prior=2 (tp=2), realization=2 (tp=2)
#    8 GPUs:  prior=4 (tp=4), realization=4 (tp=4)
#    Others:  prior=floor(N/2), realization=N-floor(N/2)
#
#  GPU SELECTION:
#    Set CUDA_VISIBLE_DEVICES=1,2 *before* invoking the script to pin to
#    specific physical cards on a multi-tenant box. The script honors the
#    list and assigns the first half to the prior server, the rest to the
#    realization server, passing PHYSICAL ids per subprocess.
#
#  OVERRIDES (env vars):
#    CUDA_VISIBLE_DEVICES=1,2     restrict to specific physical GPUs
#    VLLM_PRIOR_PORT=8770         HTTP port for the prior server
#    VLLM_REAL_PORT=8771          HTTP port for the realization server
#    VLLM_GPU_MEM_UTIL=0.85       per-server --gpu-memory-utilization
#    VLLM_MAX_MODEL_LEN=6144      per-server --max-model-len
#    VLLM_BOOT_TIMEOUT=360        seconds to wait for /v1/models per server
#    PYTHON_BIN=$CONDA_PREFIX/bin/python   force a specific interpreter
#
#  Activate the env first:
#      conda activate live-idea-bench-unsloth
#
#  Examples
#  --------
#  Default test window (2024-10 ~ 2025-03):
#      bash scripts/forecaster/eval_vllm.sh \
#          --model qwen3.5-2b \
#          --output-dir output/forecaster_qwen3.5-2b
#
#  Pin to physical GPUs 6,7 on a multi-tenant box:
#      CUDA_VISIBLE_DEVICES=6,7 bash scripts/forecaster/eval_vllm.sh \
#          --model qwen3.5-2b \
#          --output-dir output/forecaster_qwen3.5-2b
#
#  Smoke (1 topic only):
#      CUDA_VISIBLE_DEVICES=6,7 bash scripts/forecaster/eval_vllm.sh \
#          --model qwen3.5-2b \
#          --output-dir output/forecaster_qwen3.5-2b \
#          --max-topics 1
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- Defaults ----
VLLM_PRIOR_PORT="${VLLM_PRIOR_PORT:-8770}"
VLLM_REAL_PORT="${VLLM_REAL_PORT:-8771}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-6144}"
VLLM_BOOT_TIMEOUT="${VLLM_BOOT_TIMEOUT:-360}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ---- Parse --model and --output-dir from $@ ----
MODEL_ALIAS=""
OUTPUT_DIR=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    --model)        MODEL_ALIAS="${ARGS[i+1]:-}" ;;
    --model=*)      MODEL_ALIAS="${ARGS[i]#*=}" ;;
    --output-dir)   OUTPUT_DIR="${ARGS[i+1]:-}" ;;
    --output-dir=*) OUTPUT_DIR="${ARGS[i]#*=}" ;;
  esac
done
if [[ -z "${MODEL_ALIAS}" ]]; then
  MODEL_ALIAS="qwen3.5-2b"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "ERROR: --output-dir is required (the dir train.py wrote to)." >&2
  exit 1
fi

# ---- Resolve checkpoints ----
PRIOR_CKPT="${OUTPUT_DIR}/prior_sft/final_checkpoint"
REAL_CKPT="${OUTPUT_DIR}/realization_grpo/grpo/checkpoints/final_checkpoint"
if [[ ! -f "${PRIOR_CKPT}/adapter_config.json" ]]; then
  echo "ERROR: prior LoRA missing: ${PRIOR_CKPT}/adapter_config.json" >&2
  echo "       Did training finish? Override with --prior-checkpoint." >&2
  exit 1
fi
if [[ ! -f "${REAL_CKPT}/adapter_config.json" ]]; then
  echo "ERROR: realization LoRA missing: ${REAL_CKPT}/adapter_config.json" >&2
  echo "       Did GRPO finish? Override with --realization-checkpoint." >&2
  exit 1
fi

# ---- Resolve base model id ----
# Prefer the LoRA's adapter_config.json:base_model_name_or_path so the served
# base matches what training was warm-started against. Fall back to the model_zoo
# alias mapping if the adapter doesn't record it.
BASE_MODEL_ID=$("${PYTHON_BIN}" -c "
from forecaster.prior.sampler import _detect_base_model
from forecaster.realization.model_zoo import resolve_small_model
detected = _detect_base_model('${PRIOR_CKPT}')
print(detected if detected else resolve_small_model('${MODEL_ALIAS}').model_id)
")
if [[ -z "${BASE_MODEL_ID}" ]]; then
  echo "ERROR: could not resolve base model id from ${PRIOR_CKPT} or alias ${MODEL_ALIAS}" >&2
  exit 1
fi

# ---- Detect available GPUs ----
# Honor a parent CUDA_VISIBLE_DEVICES if set; otherwise enumerate all physical
# GPUs from nvidia-smi. Either way we end up with an array of PHYSICAL gpu ids.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a AVAILABLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  for i in "${!AVAILABLE_GPUS[@]}"; do
    AVAILABLE_GPUS[i]="${AVAILABLE_GPUS[i]// /}"
  done
  unset CUDA_VISIBLE_DEVICES
else
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. This script requires NVIDIA GPUs." >&2
    exit 1
  fi
  mapfile -t AVAILABLE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
fi

NUM_GPUS=${#AVAILABLE_GPUS[@]}
if [[ "${NUM_GPUS}" -lt 2 ]]; then
  echo "ERROR: eval_vllm.sh requires at least 2 GPUs (one per LoRA server)." >&2
  echo "       Found ${NUM_GPUS}: [${AVAILABLE_GPUS[*]}]" >&2
  echo "" >&2
  echo "       For single-GPU eval, use HF generate fallback (no vLLM):" >&2
  echo "         ${PYTHON_BIN} examples/forecaster/eval.py \\" >&2
  echo "             --model ${MODEL_ALIAS} \\" >&2
  echo "             --output-dir ${OUTPUT_DIR} \\" >&2
  echo "             --no-vllm-server" >&2
  exit 1
fi

# ---- Decide split ----
HALF=$(( NUM_GPUS / 2 ))
PRIOR_GPUS=("${AVAILABLE_GPUS[@]:0:${HALF}}")
REAL_GPUS=("${AVAILABLE_GPUS[@]:${HALF}}")
PRIOR_CUDA=$(IFS=','; echo "${PRIOR_GPUS[*]}")
REAL_CUDA=$(IFS=','; echo "${REAL_GPUS[*]}")
PRIOR_TP=${#PRIOR_GPUS[@]}
REAL_TP=${#REAL_GPUS[@]}

PRIOR_LOG="/tmp/vllm_eval_prior_${VLLM_PRIOR_PORT}.log"
REAL_LOG="/tmp/vllm_eval_real_${VLLM_REAL_PORT}.log"

echo "============================================================"
echo " vLLM eval"
echo "  Available GPUs    : [${AVAILABLE_GPUS[*]}] (count=${NUM_GPUS})"
echo "  Base model        : ${BASE_MODEL_ID}"
echo "  Prior LoRA        : ${PRIOR_CKPT}"
echo "  Realization LoRA  : ${REAL_CKPT}"
echo "  Prior server      : GPUs [${PRIOR_CUDA}] tp=${PRIOR_TP} port=${VLLM_PRIOR_PORT}"
echo "  Realization server: GPUs [${REAL_CUDA}]  tp=${REAL_TP} port=${VLLM_REAL_PORT}"
echo "  Output dir        : ${OUTPUT_DIR}"
echo "  Prior log         : ${PRIOR_LOG}"
echo "  Real  log         : ${REAL_LOG}"
echo "============================================================"

# ---- Stale port check ----
for port in "${VLLM_PRIOR_PORT}" "${VLLM_REAL_PORT}"; do
  if curl -sf -m 2 "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
    echo "ERROR: a server is already running on port ${port}." >&2
    echo "       Kill it first or set a different VLLM_*_PORT." >&2
    exit 1
  fi
done

# ---- Launch prior server ----
# We invoke vLLM via `${PYTHON_BIN} -m vllm.entrypoints.openai.api_server` (NOT
# bare `vllm serve`) so the Python from the active conda env is used, even when
# ~/.local/bin/vllm exists from a previous `pip install --user` and would
# otherwise be picked up first by PATH lookup. Same fix as train_vllm.sh uses
# for ~/.local/bin/trl.
nohup env CUDA_VISIBLE_DEVICES="${PRIOR_CUDA}" "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL_ID}" \
  --enable-lora \
  --lora-modules "prior=${PRIOR_CKPT}" \
  --port "${VLLM_PRIOR_PORT}" --host "0.0.0.0" \
  --tensor-parallel-size "${PRIOR_TP}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
  > "${PRIOR_LOG}" 2>&1 &
PRIOR_PID=$!
echo "Started prior vLLM server pid=${PRIOR_PID}"

# ---- Launch realization server ----
nohup env CUDA_VISIBLE_DEVICES="${REAL_CUDA}" "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL_ID}" \
  --enable-lora \
  --lora-modules "realization=${REAL_CKPT}" \
  --port "${VLLM_REAL_PORT}" --host "0.0.0.0" \
  --tensor-parallel-size "${REAL_TP}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
  > "${REAL_LOG}" 2>&1 &
REAL_PID=$!
echo "Started realization vLLM server pid=${REAL_PID}"

# ---- Cleanup trap ----
# vLLM spawns EngineCore worker processes that survive a SIGTERM on the parent.
# Kill the whole tree, then sweep stray workers by name. Same pattern as
# train_vllm.sh:211-227.
cleanup() {
  echo ""
  echo "[wrapper] Cleaning up vLLM servers..."
  for pid in "${PRIOR_PID:-}" "${REAL_PID:-}"; do
    [[ -z "${pid}" ]] && continue
    if kill -0 "${pid}" 2>/dev/null; then
      pkill -TERM -P "${pid}" 2>/dev/null || true
      kill -TERM "${pid}" 2>/dev/null || true
      sleep 2
      if kill -0 "${pid}" 2>/dev/null; then
        pkill -KILL -P "${pid}" 2>/dev/null || true
        kill -KILL "${pid}" 2>/dev/null || true
      fi
    fi
  done
  # Scoped to OUR ports. An unscoped `pkill -f "VLLM::EngineCore"` here would
  # kill every vLLM process on the box -- including other users' runs on a
  # shared GPU machine -- every time this script exits.
  for port in "${VLLM_PRIOR_PORT}" "${VLLM_REAL_PORT}"; do
    pkill -KILL -f "api_server.*--port ${port}\b" 2>/dev/null || true
    pkill -KILL -f "VLLM::EngineCore.*${port}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

# ---- Wait for both /v1/models endpoints ----
echo "Waiting for both vLLM servers (max ${VLLM_BOOT_TIMEOUT}s)..."
P_READY=0
R_READY=0
for ((s=0; s<VLLM_BOOT_TIMEOUT; s+=5)); do
  if [[ "${P_READY}" -eq 0 ]] && curl -sf -m 2 "http://localhost:${VLLM_PRIOR_PORT}/v1/models" > /dev/null 2>&1; then
    echo "[wrapper] prior server ready after ${s}s"
    P_READY=1
  fi
  if [[ "${R_READY}" -eq 0 ]] && curl -sf -m 2 "http://localhost:${VLLM_REAL_PORT}/v1/models" > /dev/null 2>&1; then
    echo "[wrapper] realization server ready after ${s}s"
    R_READY=1
  fi
  if [[ "${P_READY}" -eq 1 && "${R_READY}" -eq 1 ]]; then
    break
  fi
  if ! kill -0 "${PRIOR_PID}" 2>/dev/null; then
    echo "ERROR: prior vLLM server died during startup. Last 30 log lines:" >&2
    tail -30 "${PRIOR_LOG}" >&2 || true
    exit 1
  fi
  if ! kill -0 "${REAL_PID}" 2>/dev/null; then
    echo "ERROR: realization vLLM server died during startup. Last 30 log lines:" >&2
    tail -30 "${REAL_LOG}" >&2 || true
    exit 1
  fi
  sleep 5
done
if [[ "${P_READY}" -eq 0 || "${R_READY}" -eq 0 ]]; then
  echo "ERROR: vLLM servers did not become ready within ${VLLM_BOOT_TIMEOUT}s." >&2
  echo "  prior ready=${P_READY}  realization ready=${R_READY}" >&2
  echo "  prior log tail:" >&2
  tail -30 "${PRIOR_LOG}" >&2 || true
  echo "  realization log tail:" >&2
  tail -30 "${REAL_LOG}" >&2 || true
  exit 1
fi

# ---- Export env vars and launch eval ----
# These names are historical (originally for SGLang) but the helpers in
# forecaster/prior/sampler.py:_sglang_prior_url and
# forecaster/realization/proposal_generator.py:_sglang_api_url speak plain
# OpenAI HTTP and work against any compatible server.
export SGLANG_PRIOR_URL="http://localhost:${VLLM_PRIOR_PORT}"
export SGLANG_URL="http://localhost:${VLLM_REAL_PORT}"

echo ""
echo "[wrapper] Launching eval.py against vLLM servers..."
"${PYTHON_BIN}" examples/forecaster/eval.py "$@"
