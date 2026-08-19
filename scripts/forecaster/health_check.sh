#!/usr/bin/env bash
set -euo pipefail
# 15-minute health probe for an active GRPO run. Loops every 900 s and writes
# a one-line status record to <out_dir>/health.log. On a critical condition
# also writes <out_dir>/ALERT_<reason>.txt so the orchestrator (or a human)
# can spot it without grepping the log.
#
# Designed to be launched in the background by run_three_grpo.sh and killed
# when the trainer for that mode exits.
#
# Usage:
#   bash scripts/forecaster/health_check.sh <out_dir> <rollout_port> <judge_port> <mode>

set -uo pipefail

OUT_DIR="${1:-outputs/health}"
ROLLOUT_PORT="${2:-8765}"
JUDGE_PORT="${3:-8766}"
MODE="${4:-unknown}"
INTERVAL="${INTERVAL:-900}"     # 15 minutes
WARMUP="${WARMUP:-180}"         # first tick after ~3 min so vLLM is up

mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/health.log"
TRAINER_LOG="${OUT_DIR}/trainer.log"

stall_ticks=0
prev_step=""

sleep "${WARMUP}"

while true; do
  ts="$(date +%Y-%m-%dT%H:%M:%S)"

  # Trainer log tail: pull last logged step, loss, reward, kl if present.
  step="-"; loss="-"; reward="-"; kl="-"
  if [[ -s "${TRAINER_LOG}" ]]; then
    last_line=$(grep -E "loss|reward" "${TRAINER_LOG}" | tail -1 || true)
    step=$(echo "${last_line}"   | grep -oE "step[^a-zA-Z0-9]*[0-9]+" | grep -oE "[0-9]+" | head -1 || echo "-")
    loss=$(echo "${last_line}"   | grep -oE "loss[^0-9-]*[-0-9.eE+]+" | grep -oE "[-0-9.eE+]+" | tail -1 || echo "-")
    reward=$(echo "${last_line}" | grep -oE "reward[^0-9-]*[-0-9.eE+]+" | grep -oE "[-0-9.eE+]+" | tail -1 || echo "-")
    kl=$(echo "${last_line}"     | grep -oE "kl[^0-9-]*[-0-9.eE+]+"     | grep -oE "[-0-9.eE+]+" | tail -1 || echo "-")
  fi

  # GPU snapshot.
  gpu_csv=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' || echo "nvidia-smi-unavailable")

  # vLLM health.
  rollout_ok="-"; judge_ok="-"
  if curl -sf -m 2 "http://localhost:${ROLLOUT_PORT}/health/" > /dev/null 2>&1; then
    rollout_ok="ok"
  else
    rollout_ok="DOWN"
  fi
  if [[ "${MODE}" == "soft" ]]; then
    if curl -sf -m 2 "http://localhost:${JUDGE_PORT}/v1/models" > /dev/null 2>&1; then
      judge_ok="ok"
    else
      judge_ok="DOWN"
    fi
  fi

  # Disk in OUT_DIR.
  du_size=$(du -sh "${OUT_DIR}" 2>/dev/null | awk '{print $1}')

  printf "%s mode=%s step=%s loss=%s reward=%s kl=%s rollout=%s judge=%s out_size=%s gpu=%s\n" \
    "${ts}" "${MODE}" "${step}" "${loss}" "${reward}" "${kl}" "${rollout_ok}" "${judge_ok}" "${du_size}" "${gpu_csv}" \
    >> "${LOG}"

  # Alarms.
  alert=""
  if [[ "${loss}" != "-" && ( "${loss}" == "nan" || "${loss}" == "NaN" || "${loss}" == "inf" ) ]]; then
    alert="loss_${loss}"
  fi
  if [[ "${rollout_ok}" == "DOWN" ]]; then
    alert="rollout_vllm_down"
  fi
  if [[ "${MODE}" == "soft" && "${judge_ok}" == "DOWN" ]]; then
    alert="judge_vllm_down"
  fi
  if [[ -n "${step}" && "${step}" == "${prev_step}" && "${step}" != "-" ]]; then
    stall_ticks=$((stall_ticks + 1))
  else
    stall_ticks=0
  fi
  if [[ "${stall_ticks}" -ge 3 ]]; then
    alert="trainer_stalled_at_step_${step}"
  fi
  prev_step="${step}"

  if [[ -n "${alert}" ]]; then
    echo "${ts} ALERT ${alert}" >> "${LOG}"
    echo "${ts} ${alert}" > "${OUT_DIR}/ALERT_${alert}.txt"
  fi

  sleep "${INTERVAL}"
done
