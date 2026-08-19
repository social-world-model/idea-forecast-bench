#!/usr/bin/env bash
# Start (or check) a local vLLM OpenAI-compatible server for the soft-reward
# judge model. Picks 1-2 GPUs from CUDA_VISIBLE_DEVICES and runs Qwen2.5-7B
# Instruct as the judge. Output: /tmp/judge_vllm_${JUDGE_PORT}.log
#
# The training scripts call this at startup of the soft run and shut it down
# when soft completes (coverage/novelty don't need it and reclaim the GPUs).
#
# Env overrides:
#   JUDGE_MODEL          HF id of the judge (default Qwen/Qwen2.5-7B-Instruct)
#   JUDGE_PORT           HTTP port (default 8766; rollout vLLM owns 8765)
#   JUDGE_TP             tensor parallel degree (default 1)
#   JUDGE_GPU_MEM_UTIL   vLLM gpu memory utilization (default 0.85)
#   JUDGE_MAX_MODEL_LEN  default 8192
#   JUDGE_CUDA           comma-list of physical GPU ids to give the judge

set -euo pipefail
cd "$(dirname "$0")/../.."

JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
JUDGE_PORT="${JUDGE_PORT:-8766}"
JUDGE_TP="${JUDGE_TP:-1}"
JUDGE_GPU_MEM_UTIL="${JUDGE_GPU_MEM_UTIL:-0.85}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-8192}"
JUDGE_BOOT_TIMEOUT="${JUDGE_BOOT_TIMEOUT:-360}"
JUDGE_CUDA="${JUDGE_CUDA:-}"
JUDGE_LOG="/tmp/judge_vllm_${JUDGE_PORT}.log"
JUDGE_PID_FILE="/tmp/judge_vllm_${JUDGE_PORT}.pid"

if [[ "${1:-}" == "stop" ]]; then
  if [[ -f "${JUDGE_PID_FILE}" ]]; then
    pid=$(cat "${JUDGE_PID_FILE}")
    echo "[serve_judge] stop pid=${pid}"
    if kill -0 "${pid}" 2>/dev/null; then
      pkill -TERM -P "${pid}" 2>/dev/null || true
      kill -TERM "${pid}" 2>/dev/null || true
      sleep 2
      pkill -KILL -P "${pid}" 2>/dev/null || true
      kill -KILL "${pid}" 2>/dev/null || true
    fi
    rm -f "${JUDGE_PID_FILE}"
  fi
  pkill -KILL -f "vllm.entrypoints.openai.api_server.*--port ${JUDGE_PORT}" 2>/dev/null || true
  pkill -KILL -f "vllm serve.*--port ${JUDGE_PORT}" 2>/dev/null || true
  pkill -KILL -f "VLLM::EngineCore.*${JUDGE_PORT}" 2>/dev/null || true
  exit 0
fi

if curl -sf -m 2 "http://localhost:${JUDGE_PORT}/v1/models" > /dev/null 2>&1; then
  echo "[serve_judge] Judge already up on :${JUDGE_PORT}."
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

env_args=()
if [[ -n "${JUDGE_CUDA}" ]]; then
  env_args+=(CUDA_VISIBLE_DEVICES="${JUDGE_CUDA}")
fi

echo "[serve_judge] Starting ${JUDGE_MODEL} on :${JUDGE_PORT} (tp=${JUDGE_TP}, CUDA=${JUDGE_CUDA:-inherit})"
nohup env "${env_args[@]}" "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${JUDGE_MODEL}" \
  --port "${JUDGE_PORT}" \
  --host 0.0.0.0 \
  --tensor-parallel-size "${JUDGE_TP}" \
  --max-model-len "${JUDGE_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${JUDGE_GPU_MEM_UTIL}" \
  --dtype bfloat16 \
  --enforce-eager \
  > "${JUDGE_LOG}" 2>&1 &
JUDGE_PID=$!
echo "${JUDGE_PID}" > "${JUDGE_PID_FILE}"
echo "[serve_judge] pid=${JUDGE_PID} log=${JUDGE_LOG}"

echo "[serve_judge] Waiting up to ${JUDGE_BOOT_TIMEOUT}s for /v1/models ..."
for ((s=0; s<JUDGE_BOOT_TIMEOUT; s+=5)); do
  if curl -sf -m 2 "http://localhost:${JUDGE_PORT}/v1/models" > /dev/null 2>&1; then
    echo "[serve_judge] Ready after ${s}s."
    exit 0
  fi
  if ! kill -0 "${JUDGE_PID}" 2>/dev/null; then
    echo "ERROR: judge server died during startup. Tail of log:" >&2
    tail -30 "${JUDGE_LOG}" >&2 || true
    exit 1
  fi
  sleep 5
done
echo "ERROR: judge server did not come up within ${JUDGE_BOOT_TIMEOUT}s." >&2
tail -30 "${JUDGE_LOG}" >&2 || true
exit 1
