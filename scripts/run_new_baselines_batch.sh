#!/usr/bin/env bash
# Run the two new paper baselines (Summary Prompting + Retrieval-Augmented Prompting)
# against the same rolling-window configuration as the existing four baselines in
# data/baselines/*_raw.json — so the numbers are directly comparable.
#
# Existing baseline config (matches data/baselines/*_raw.json):
#   start_month=2023-01  end_month=2025-06  horizon_months=3
#   top_k=5  min_train_papers=5  monthly step  similarity_engine=hybrid
#
# Required environment:
#   OPENAI_API_KEY    — for the OpenAI Batch API
#
# Optional environment:
#   MODEL_NAME           default: gpt-5.4
#   REASONING_EFFORT     default: medium
#   SIMILARITY_ENGINE    default: hybrid
#   INPUT_DIR            default: data/arxiv_csml/raw_markdown
#   OUTPUT_DIR           default: data/baselines
#   BATCH_ROOT           default: logs/batch
#   POLL_INTERVAL        default: 60
#   MAX_WAIT_HOURS       default: 24
#   ONLY                 optional: "summary" or "retrieval" to run just one
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   bash scripts/run_new_baselines_batch.sh
#
# Resume support: if a run is interrupted, just re-run with the same BATCH_ROOT
# (or the auto-generated batch dir printed on the first run). Completed phases
# are skipped via state.json checkpoints.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-gpt-5.4}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
SIMILARITY_ENGINE="${SIMILARITY_ENGINE:-hybrid}"
INPUT_DIR="${INPUT_DIR:-data/arxiv_csml/raw_markdown}"
OUTPUT_DIR="${OUTPUT_DIR:-data/baselines}"
BATCH_ROOT="${BATCH_ROOT:-logs/batch}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-24}"
ONLY="${ONLY:-}"

# Window config — must match existing baselines in data/baselines/*_raw.json
START_MONTH="2023-01"
END_MONTH="2025-06"
HORIZON_MONTHS=3
TOP_K=5
MIN_TRAIN_PAPERS=5

mkdir -p "$OUTPUT_DIR" "$BATCH_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

run_strategy() {
  local strategy="$1"
  local output_file="$OUTPUT_DIR/${strategy}_raw.json"
  local batch_dir="$BATCH_ROOT/${strategy}"

  echo "========================================================================"
  echo "Running baseline: $strategy"
  echo "  model            = $MODEL_NAME (reasoning=$REASONING_EFFORT)"
  echo "  window           = $START_MONTH .. $END_MONTH  horizon=${HORIZON_MONTHS}m  top_k=$TOP_K"
  echo "  similarity       = $SIMILARITY_ENGINE"
  echo "  output           = $output_file"
  echo "  batch_dir        = $batch_dir"
  echo "========================================================================"

  "$PYTHON_BIN" examples/run_batch_backtest.py \
    --input-dir "$INPUT_DIR" \
    --strategy "$strategy" \
    --model-name "$MODEL_NAME" \
    --reasoning-effort "$REASONING_EFFORT" \
    --start-month "$START_MONTH" \
    --end-month "$END_MONTH" \
    --horizon-months "$HORIZON_MONTHS" \
    --top-k "$TOP_K" \
    --min-train-papers "$MIN_TRAIN_PAPERS" \
    --similarity-engine "$SIMILARITY_ENGINE" \
    --output "$output_file" \
    --batch-dir "$batch_dir" \
    --poll-interval "$POLL_INTERVAL" \
    --max-wait-hours "$MAX_WAIT_HOURS"
}

case "$ONLY" in
  summary)
    run_strategy summary_prompting
    ;;
  retrieval)
    run_strategy retrieval_prompting
    ;;
  "")
    run_strategy summary_prompting
    run_strategy retrieval_prompting
    ;;
  *)
    echo "ERROR: ONLY must be 'summary', 'retrieval', or unset." >&2
    exit 2
    ;;
esac

echo
echo "Done. Outputs:"
ls -l "$OUTPUT_DIR"/summary_prompting_raw.json "$OUTPUT_DIR"/retrieval_prompting_raw.json 2>/dev/null || true
