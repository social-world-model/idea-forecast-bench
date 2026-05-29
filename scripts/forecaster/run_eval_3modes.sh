#!/usr/bin/env bash
# Single-GPU re-eval of the three single-metric GRPO adapters
# (soft / coverage / novelty): predictor generation + 9B-judge LLM eval, with
# cutoffs covering the FULL test window 2024-10..2025-03 (4 windows:
# cutoffs 2024-09/10/11/12). Runs one 9B model at a time, so one GPU is enough.
#
# Usage:  PYTHON_BIN=/path/to/env/bin/python scripts/forecaster/run_eval_3modes.sh <gpu>
#
# Prereqs:
#   - env built (vllm/transformers/trl/peft/sentence-transformers/openai)
#   - data/csml_v2/raw_markdown present (the paper corpus)
#   - the three adapters under outputs/grpo_{soft,coverage,novelty}_*/...
#   - VOYAGE_API_KEY exported (else the eval falls back to a local embedder)
set -uo pipefail
cd "$(dirname "$0")/../.."

GPU="${1:?usage: run_eval_3modes.sh <gpu>}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
GEN_PORT="${GEN_PORT:-8800}"
JUDGE_PORT="${JUDGE_PORT:-8767}"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs outputs
MODES=(soft coverage novelty)

adapter_for() { ls -dt outputs/grpo_"$1"_*/realization_grpo/grpo/checkpoints/final_checkpoint 2>/dev/null | head -1; }

echo "=== Phase 1: predictor generation (one adapter at a time) ==="
for mode in "${MODES[@]}"; do
  adp="$(adapter_for "$mode")"
  [[ -d "$adp" ]] || { echo "ERROR: no adapter for $mode (outputs/grpo_${mode}_*/.../final_checkpoint)"; exit 2; }
  echo "--- gen $mode  adapter=$adp ---"
  OUT="outputs/predict_${mode}_${TS}.json" PYTHON_BIN="$PYTHON_BIN" BASE_MODEL="$BASE_MODEL" \
    scripts/forecaster/run_predictor_trained.sh "$mode" "$adp" "$GPU" "$GEN_PORT" \
    || { echo "gen failed for $mode"; exit 1; }
done

echo "=== Phase 2: serve 9B judge on GPU $GPU, eval all three ==="
CUDA_VISIBLE_DEVICES="$GPU" nohup "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" --served-model-name qwen3.5-9b-instruct \
  --host 0.0.0.0 --port "$JUDGE_PORT" --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 > "logs/judge_${JUDGE_PORT}.log" 2>&1 &
JUDGE_PID=$!
trap '[[ -n "${JUDGE_PID:-}" ]] && kill "$JUDGE_PID" 2>/dev/null || true' EXIT
echo "judge pid=$JUDGE_PID — waiting for :$JUDGE_PORT ..."
ok=0
for i in $(seq 1 120); do
  curl -sf -m3 "http://localhost:${JUDGE_PORT}/v1/models" >/dev/null 2>&1 && { ok=1; echo "judge ready (${i}0s)"; break; }
  kill -0 "$JUDGE_PID" 2>/dev/null || { echo "judge died"; tail -20 "logs/judge_${JUDGE_PORT}.log"; exit 1; }
  sleep 5
done
[[ "$ok" -eq 1 ]] || { echo "judge never came up"; exit 1; }

for mode in "${MODES[@]}"; do
  echo "--- eval $mode ---"
  JUDGE_BASE_URL="http://localhost:${JUDGE_PORT}/v1" JUDGE_API_KEY=EMPTY \
  "$PYTHON_BIN" examples/benchmark/llm_judge_eval.py \
    --input-json "outputs/predict_${mode}_${TS}.json" \
    --papers-dir data/csml_v2/raw_markdown \
    --output "outputs/llm_judge_${mode}9b_${TS}.json" \
    --judge-model qwen3.5-9b-instruct --judge-base-url "http://localhost:${JUDGE_PORT}/v1" \
    --workers 8 --topic-workers 2 \
    || { echo "eval failed for $mode"; exit 1; }
done

echo
echo "=== DONE.  ts=${TS}  results: ==="
for mode in "${MODES[@]}"; do
  "$PYTHON_BIN" - "$mode" "$TS" <<'PY'
import json, sys
mode, ts = sys.argv[1], sys.argv[2]
d = json.load(open(f"outputs/llm_judge_{mode}9b_{ts}.json"))
a = d.get("aggregate_summary", {})
print(f"  {mode:9s} windows={d.get('total_windows')} soft={a.get('avg_soft_score')} "
      f"hit@k={a.get('avg_hit_at_k')} mrr={a.get('avg_mrr')} novelty={a.get('avg_novelty')}")
PY
done
