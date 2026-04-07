#!/usr/bin/env bash
# ============================================================================
#  Multi-GPU vLLM-accelerated training.
#
#  Runs Phase 1 (Prior SFT, METHOD §3.2) → Phase 2 (dataset prep) →
#  Phase 3 (Realization GRPO, METHOD §3.3) using Unsloth + TRL with vLLM
#  server-mode generation. Generation throughput goes from ~80 tok/s (HF
#  generate) to ~500-1000 tok/s (vLLM colocated KV-cache) — ~5-10x faster
#  on the generation phase, ~3-5x faster overall per training step.
#
#  REQUIREMENTS:
#    * 2+ GPUs (TRL refuses to put trainer + vLLM server on the same GPU
#      because it leads to NCCL hangs / incorrect collective ops)
#    * vLLM nightly installed (handled by scripts/forecaster/setup_env.sh)
#    * trl ≥ 1.0 (provides `trl vllm-serve` with /init_communicator,
#      /update_named_param, etc. — plain `vllm serve` does NOT expose these)
#
#  TOPOLOGY (auto-picked unless overridden):
#    2 GPUs:  vLLM=1                 + trainer=1
#    4 GPUs:  vLLM=2 (tp=2)          + trainer=2
#    8 GPUs:  vLLM=4 (tp=4)          + trainer=4
#    Others:  vLLM=floor(N/2)        + trainer=N-floor(N/2)
#
#  GPU SELECTION:
#    By default, all physical GPUs reported by nvidia-smi are used. To pin
#    the script to specific physical cards (e.g. GPUs 1 and 2 on an 8-GPU
#    box), set CUDA_VISIBLE_DEVICES *before* invoking the script:
#
#      CUDA_VISIBLE_DEVICES=1,2 bash scripts/forecaster/train_vllm.sh ...
#
#    The script honors the comma-separated list and assigns the first GPU(s)
#    in it to vLLM and the rest to the trainer, passing PHYSICAL ids
#    (not re-indexed) to each subprocess via its own CUDA_VISIBLE_DEVICES.
#
#  OVERRIDES (env vars):
#    CUDA_VISIBLE_DEVICES=1,2  restrict to specific physical GPUs (see above)
#    VLLM_NUM_GPUS=N      number of GPUs reserved for the vLLM server
#                         (becomes its --tensor_parallel_size)
#    VLLM_PORT=8765       HTTP port for trl vllm-serve
#    VLLM_GPU_MEM_UTIL=0.8 vLLM's --gpu_memory_utilization
#                         (raise this on dedicated cards — they're not
#                         sharing with the trainer anymore)
#    VLLM_MAX_MODEL_LEN=6144 vLLM's --max_model_len
#    VLLM_BOOT_TIMEOUT=360   seconds to wait for /health/ to come up
#
#  Activate the env first:
#      conda activate live-idea-bench-unsloth
#
#  Examples
#  --------
#  Auto split across all visible GPUs:
#      bash scripts/forecaster/train_vllm.sh \
#          --model qwen3.5-2b \
#          --hindsight output/hindsight_samples.jsonl \
#          --output-dir output/forecaster_qwen3.5-2b
#
#  Pin to physical GPUs 1 and 2 only (e.g. 8-GPU box, others busy):
#      CUDA_VISIBLE_DEVICES=1,2 bash scripts/forecaster/train_vllm.sh \
#          --model qwen3.5-2b \
#          --hindsight output/hindsight_samples.jsonl \
#          --output-dir output/forecaster_qwen3.5-2b
#
#  4 GPUs with a custom split (3 for vLLM tp=3, 1 for trainer):
#      CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_NUM_GPUS=3 \
#      bash scripts/forecaster/train_vllm.sh \
#          --model qwen3.5-9b \
#          --hindsight output/hindsight_samples.jsonl \
#          --output-dir output/forecaster_qwen3.5-9b
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- Defaults ----
VLLM_PORT="${VLLM_PORT:-8765}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.8}"  # higher than the colocate path because the GPU is dedicated
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-6144}"
VLLM_BOOT_TIMEOUT="${VLLM_BOOT_TIMEOUT:-360}"

# ---- Detect available GPUs ----
# Honor a parent CUDA_VISIBLE_DEVICES if set (lets the user pin to specific
# physical cards on a multi-tenant box). Otherwise enumerate all physical GPUs
# from nvidia-smi. Either way we end up with an array of PHYSICAL gpu ids.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a AVAILABLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  # Strip any whitespace around entries.
  for i in "${!AVAILABLE_GPUS[@]}"; do
    AVAILABLE_GPUS[i]="${AVAILABLE_GPUS[i]// /}"
  done
  # Unset the inherited CUDA_VISIBLE_DEVICES so subprocesses get a clean slate
  # — we'll set it explicitly per subprocess below with the physical id(s)
  # we want them to see.
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
  echo "ERROR: train_vllm.sh requires at least 2 GPUs (found ${NUM_GPUS}: [${AVAILABLE_GPUS[*]}])." >&2
  echo "       TRL refuses to put trainer + vLLM server on the same device." >&2
  echo "       Use scripts/forecaster/train.sh for single-GPU training." >&2
  exit 1
fi

# ---- Decide GPU split ----
if [[ -z "${VLLM_NUM_GPUS:-}" ]]; then
  case "${NUM_GPUS}" in
    2)  VLLM_NUM_GPUS=1 ;;
    4)  VLLM_NUM_GPUS=2 ;;
    8)  VLLM_NUM_GPUS=4 ;;
    *)  VLLM_NUM_GPUS=$(( NUM_GPUS / 2 )) ;;
  esac
fi
TRAINER_NUM_GPUS=$(( NUM_GPUS - VLLM_NUM_GPUS ))
if [[ "${TRAINER_NUM_GPUS}" -lt 1 ]]; then
  echo "ERROR: VLLM_NUM_GPUS=${VLLM_NUM_GPUS} leaves no GPUs for the trainer." >&2
  exit 1
fi
if [[ "${VLLM_NUM_GPUS}" -lt 1 ]]; then
  echo "ERROR: VLLM_NUM_GPUS=${VLLM_NUM_GPUS} must be >= 1." >&2
  exit 1
fi

# Slice the AVAILABLE_GPUS array into vLLM and trainer halves. The first
# VLLM_NUM_GPUS physical ids go to vLLM, the rest go to the trainer.
VLLM_GPUS=("${AVAILABLE_GPUS[@]:0:${VLLM_NUM_GPUS}}")
TRAINER_GPUS=("${AVAILABLE_GPUS[@]:${VLLM_NUM_GPUS}}")
VLLM_CUDA=$(IFS=','; echo "${VLLM_GPUS[*]}")
TRAINER_CUDA=$(IFS=','; echo "${TRAINER_GPUS[*]}")

# ---- Resolve --model alias to HF model id ----
MODEL_ALIAS=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    --model)
      MODEL_ALIAS="${ARGS[i+1]:-}"
      ;;
    --model=*)
      MODEL_ALIAS="${ARGS[i]#*=}"
      ;;
  esac
done
if [[ -z "${MODEL_ALIAS}" ]]; then
  MODEL_ALIAS="qwen3.5-2b"
fi

# Resolve PYTHON_BIN early so every Python invocation in this script honors
# the same interpreter. Default: bare `python` from PATH (assumes the user has
# `conda activate <env>`-ed and PATH puts the env's bin first). On boxes with
# multiple anaconda installations or odd ~/.bashrc PATH manipulation, the
# user can override with PYTHON_BIN=$CONDA_PREFIX/bin/python (or any absolute
# path) and every step below will use exactly that interpreter.
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_ID=$("${PYTHON_BIN}" -c "
from forecaster.realization.model_zoo import resolve_small_model
print(resolve_small_model('${MODEL_ALIAS}').model_id)
")
if [[ -z "${MODEL_ID}" ]]; then
  echo "ERROR: could not resolve --model ${MODEL_ALIAS} to an HF model id." >&2
  exit 1
fi

# ---- Pretty-print plan ----
VLLM_LOG="/tmp/vllm_serve_${VLLM_PORT}.log"
echo "============================================================"
echo " Multi-GPU vLLM training"
echo "  Available GPUs   : [${AVAILABLE_GPUS[*]}] (count=${NUM_GPUS})"
echo "  vLLM GPUs        : ${VLLM_NUM_GPUS} → physical [${VLLM_CUDA}] (tp=${VLLM_NUM_GPUS})"
echo "  Trainer GPUs     : ${TRAINER_NUM_GPUS} → physical [${TRAINER_CUDA}]"
echo "  Model            : ${MODEL_ALIAS} → ${MODEL_ID}"
echo "  vLLM port        : ${VLLM_PORT}"
echo "  vLLM gpu-mem-util: ${VLLM_GPU_MEM_UTIL}"
echo "  vLLM max-model-len: ${VLLM_MAX_MODEL_LEN}"
echo "  vLLM log         : ${VLLM_LOG}"
echo "============================================================"

# ---- Stale server check ----
if curl -sf -m 2 "http://localhost:${VLLM_PORT}/health/" > /dev/null 2>&1; then
  echo "WARNING: a server is already running on port ${VLLM_PORT}." >&2
  echo "         Kill it first or set VLLM_PORT=<different>." >&2
  exit 1
fi

# ---- Start trl vllm-serve on the dedicated vLLM GPUs ----
# We invoke the launcher via `${PYTHON_BIN} -m trl.scripts.vllm_serve` (NOT
# bare `trl vllm-serve`) so the Python from the active conda env is used,
# even when ~/.local/bin/trl exists from a previous `pip install --user` and
# would otherwise be picked up first by PATH lookup, OR when the user has
# multiple anaconda installations and `which python` resolves to the wrong
# one despite `conda activate` updating the prompt.
nohup env CUDA_VISIBLE_DEVICES="${VLLM_CUDA}" "${PYTHON_BIN}" -m trl.scripts.vllm_serve \
  --model "${MODEL_ID}" \
  --port "${VLLM_PORT}" \
  --host "0.0.0.0" \
  --tensor-parallel-size "${VLLM_NUM_GPUS}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "Started trl vllm-serve via ${PYTHON_BIN} -m trl.scripts.vllm_serve (pid=${VLLM_PID})"

# ---- Cleanup trap ----
# vLLM spawns EngineCore worker processes that survive a SIGTERM on the
# parent. Kill the whole process tree, then sweep any stray workers by name.
cleanup() {
  echo ""
  echo "[wrapper] Cleaning up vLLM server (pid=${VLLM_PID})..."
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    pkill -TERM -P "${VLLM_PID}" 2>/dev/null || true
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
    sleep 2
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
      pkill -KILL -P "${VLLM_PID}" 2>/dev/null || true
      kill -KILL "${VLLM_PID}" 2>/dev/null || true
    fi
  fi
  pkill -KILL -f "VLLM::EngineCore" 2>/dev/null || true
  pkill -KILL -f "trl vllm-serve" 2>/dev/null || true
  pkill -KILL -f "trl.scripts.vllm_serve" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- Wait for /health/ to come up ----
echo "Waiting for vLLM to be ready (max ${VLLM_BOOT_TIMEOUT}s)..."
for ((s=0; s<VLLM_BOOT_TIMEOUT; s+=5)); do
  if curl -sf -m 2 "http://localhost:${VLLM_PORT}/health/" > /dev/null 2>&1; then
    echo "[wrapper] vLLM ready after ${s}s."
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "ERROR: vLLM server died during startup. Last 30 log lines:" >&2
    tail -30 "${VLLM_LOG}" >&2 || true
    exit 1
  fi
  sleep 5
done
if ! curl -sf -m 2 "http://localhost:${VLLM_PORT}/health/" > /dev/null 2>&1; then
  echo "ERROR: vLLM server did not become ready within ${VLLM_BOOT_TIMEOUT}s." >&2
  tail -30 "${VLLM_LOG}" >&2 || true
  exit 1
fi

# ---- Launch trainer on the dedicated trainer GPUs ----
echo ""
echo "[wrapper] Launching trainer on CUDA_VISIBLE_DEVICES=${TRAINER_CUDA}..."

if [[ "${TRAINER_NUM_GPUS}" -gt 1 ]]; then
  # Multi-rank trainer: use torchrun for DDP / FSDP. Unsloth + TRL GRPO
  # supports this via accelerate's process group setup. Each rank loads its
  # own copy of the model; LoRA grads are all-reduced across ranks.
  # Resolve torchrun from the same Python's bin/ to keep all subprocesses
  # on the env's interpreter (avoids /usr/local vs ~/.conda PATH confusion).
  TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"
  if [[ ! -x "${TORCHRUN_BIN}" ]]; then
    TORCHRUN_BIN="torchrun"  # fall back to PATH lookup
  fi
  echo "[wrapper] Multi-rank trainer (nproc_per_node=${TRAINER_NUM_GPUS}) via ${TORCHRUN_BIN}"
  CUDA_VISIBLE_DEVICES="${TRAINER_CUDA}" "${TORCHRUN_BIN}" \
    --standalone \
    --nproc_per_node="${TRAINER_NUM_GPUS}" \
    examples/forecaster/train.py \
    "$@" \
    --use-vllm-server \
    --vllm-server-port "${VLLM_PORT}"
else
  # Single-rank trainer: use the same PYTHON_BIN as the vLLM launcher so the
  # trainer and the server agree on which interpreter / site-packages they're
  # loading from.
  CUDA_VISIBLE_DEVICES="${TRAINER_CUDA}" "${PYTHON_BIN}" \
    examples/forecaster/train.py \
    "$@" \
    --use-vllm-server \
    --vllm-server-port "${VLLM_PORT}"
fi
