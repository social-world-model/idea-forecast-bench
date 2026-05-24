#!/usr/bin/env bash
# Launch all three single-metric GRPO runs IN PARALLEL.
#
# Background: Unsloth's loader is single-GPU only — torchrun + DDP either
# OOMs (every rank loads to cuda:0) or breaks accelerate's barrier when we
# restrict CUDA_VISIBLE_DEVICES per rank. The clean path is to run each
# mode as its own single-GPU process, all three in parallel.
#
# GPU layout (8 A6000 / 49 GB each):
#   GPU 0  trainer (novelty)
#   GPU 1  rollout vLLM (novelty)
#   GPU 2  trainer (coverage)
#   GPU 3  rollout vLLM (coverage)
#   GPU 4  trainer (soft)
#   GPU 5  rollout vLLM (soft)
#   GPU 6  judge vLLM (soft only)
#   GPU 7  spare
#
# Each (mode, rollout, judge) trio uses a unique port so the three runs
# don't collide. Trainer is launched plain python (no torchrun).

set -uo pipefail
cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/data/haofeiy2/miniconda3/envs/ideabench-unsloth/bin/python}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3-8b}"
PAPERS_DIR="${PAPERS_DIR:-data/csml_v2/raw_markdown}"
START_MONTH="${START_MONTH:-2023-01}"
END_MONTH="${END_MONTH:-2024-09}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-10800}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-768}"
MAX_GRPO_ROWS="${MAX_GRPO_ROWS:-120}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
HEALTH_CHECK="${HEALTH_CHECK:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
ROOT_OUT="outputs"
mkdir -p "${ROOT_OUT}"

MODEL_ID=$("${PYTHON_BIN}" -c "from forecaster.realization.model_zoo import resolve_small_model; print(resolve_small_model('${MODEL_ALIAS}').model_id)")
echo "[parallel] model_id=${MODEL_ID}"

# ---------------------------------------------------------------------------
# Function to launch one mode in the background.
# Args: mode trainer_gpu rollout_gpu rollout_port [judge_gpu] [judge_port]
# ---------------------------------------------------------------------------
launch_mode() {
  local mode="$1"
  local trainer_gpu="$2"
  local rollout_gpu="$3"
  local rollout_port="$4"
  local judge_gpu="${5:-}"
  local judge_port="${6:-}"
  local out_dir="${ROOT_OUT}/grpo_${mode}_${TS}"
  mkdir -p "${out_dir}"

  echo "============================================================"
  echo "[parallel] mode=${mode}"
  echo "  trainer GPU  : ${trainer_gpu}"
  echo "  rollout GPU  : ${rollout_gpu} -> :${rollout_port}"
  [[ -n "${judge_gpu}" ]] && echo "  judge GPU    : ${judge_gpu} -> :${judge_port}"
  echo "  output       : ${out_dir}"
  echo "============================================================"

  # ----- Rollout vLLM (TP=1) -----
  local rollout_log="${out_dir}/rollout_vllm.log"
  nohup env CUDA_VISIBLE_DEVICES="${rollout_gpu}" "${PYTHON_BIN}" scripts/forecaster/_trl_vllm_serve.py \
    --model "${MODEL_ID}" \
    --port "${rollout_port}" \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --max-model-len 6144 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    > "${rollout_log}" 2>&1 &
  echo $! > "${out_dir}/rollout_vllm.pid"

  # ----- Judge vLLM (soft only) -----
  if [[ -n "${judge_gpu}" ]]; then
    local judge_log="${out_dir}/judge_vllm.log"
    nohup env CUDA_VISIBLE_DEVICES="${judge_gpu}" "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
      --model "${JUDGE_MODEL}" \
      --port "${judge_port}" \
      --host 0.0.0.0 \
      --tensor-parallel-size 1 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.85 \
      --dtype bfloat16 \
      --enforce-eager \
      > "${judge_log}" 2>&1 &
    echo $! > "${out_dir}/judge_vllm.pid"
  fi

  # ----- Wait for rollout vLLM ready -----
  echo "[parallel:${mode}] Waiting up to 900s for rollout :${rollout_port} ..."
  for ((s=0; s<900; s+=10)); do
    if curl -sf -m 2 "http://localhost:${rollout_port}/health/" > /dev/null 2>&1; then
      echo "[parallel:${mode}] rollout ready in ${s}s"
      break
    fi
    if ! kill -0 "$(cat "${out_dir}/rollout_vllm.pid")" 2>/dev/null; then
      echo "[parallel:${mode}] rollout died — see ${rollout_log}"
      return 1
    fi
    sleep 10
  done
  if ! curl -sf -m 2 "http://localhost:${rollout_port}/health/" > /dev/null 2>&1; then
    echo "[parallel:${mode}] rollout never became healthy — bailing"
    return 1
  fi

  # ----- Wait for judge vLLM (soft only) -----
  if [[ -n "${judge_gpu}" ]]; then
    echo "[parallel:${mode}] Waiting up to 600s for judge :${judge_port} ..."
    for ((s=0; s<600; s+=10)); do
      if curl -sf -m 2 "http://localhost:${judge_port}/v1/models" > /dev/null 2>&1; then
        echo "[parallel:${mode}] judge ready in ${s}s"
        break
      fi
      if ! kill -0 "$(cat "${out_dir}/judge_vllm.pid")" 2>/dev/null; then
        echo "[parallel:${mode}] judge died"
        return 1
      fi
      sleep 10
    done
  fi

  # ----- Launch trainer -----
  local trainer_log="${out_dir}/trainer.log"
  local extra=()
  if [[ -n "${judge_gpu}" ]]; then
    extra+=("JUDGE_BASE_URL=http://localhost:${judge_port}/v1" \
            "JUDGE_API_KEY=EMPTY" "JUDGE_MODEL=${JUDGE_MODEL}")
  fi
  echo "[parallel:${mode}] Launching trainer (timeout ${PER_RUN_TIMEOUT}s) -> ${trainer_log}"
  nohup env CUDA_VISIBLE_DEVICES="${trainer_gpu}" "${extra[@]}" \
    timeout "${PER_RUN_TIMEOUT}" "${PYTHON_BIN}" \
    examples/forecaster/train_grpo_metric.py \
    --model "${MODEL_ALIAS}" \
    --papers "${PAPERS_DIR}" \
    --output-dir "${out_dir}" \
    --reward-mode "${mode}" \
    --start-month "${START_MONTH}" \
    --end-month "${END_MONTH}" \
    --num-generations "${NUM_GENERATIONS}" \
    --max-completion-length "${MAX_COMPLETION_LENGTH}" \
    --max-grpo-rows "${MAX_GRPO_ROWS}" \
    --use-vllm-server \
    --vllm-server-port "${rollout_port}" \
    > "${trainer_log}" 2>&1 &
  echo $! > "${out_dir}/trainer.pid"
}

# Launch all three in parallel — each in its own background subshell so
# `launch_mode`'s vLLM-wait doesn't block the others.
# GPUs 0 and 6 are reserved by another project (paper-graph vLLM); we use
# the free GPUs 1-5,7,8 and leave 9 spare.
#   novelty : trainer 1 | rollout 2
#   coverage: trainer 3 | rollout 4
#   soft    : trainer 5 | rollout 7 | judge 8
(launch_mode novelty  1 2 8765                  > "${ROOT_OUT}/launch_novelty_${TS}.log"  2>&1) &
(launch_mode coverage 3 4 8775                  > "${ROOT_OUT}/launch_coverage_${TS}.log" 2>&1) &
(launch_mode soft     5 7 8785 8 8786           > "${ROOT_OUT}/launch_soft_${TS}.log"     2>&1) &

wait
echo "[parallel] all launchers returned. trainer processes are running in background."
echo "[parallel] check status with:"
echo "  tail -f ${ROOT_OUT}/grpo_*_${TS}/trainer.log"
