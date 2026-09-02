#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT_DIR="${INPUT_DIR:-data/csml/raw_markdown}"
STRATEGY="${STRATEGY:-summary_prompting}"
MODEL="${MODEL:-gpt-4.1}"
OUTPUT="${OUTPUT:-output/backtest/${STRATEGY}.json}"
START_MONTH="${START_MONTH:-2024-04}"
END_MONTH="${END_MONTH:-2025-09}"
MIN_CUTOFF_MONTH="${MIN_CUTOFF_MONTH:-2024-07}"
HORIZON_MONTHS="${HORIZON_MONTHS:-3}"
TOP_K="${TOP_K:-5}"
MIN_TRAIN_PAPERS="${MIN_TRAIN_PAPERS:-2}"
WORKERS="${WORKERS:-8}"
SKIP_MATCHING="${SKIP_MATCHING:-1}"
TOPICS="${TOPICS:-}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-}"
REALIZATION_CHECKPOINT="${REALIZATION_CHECKPOINT:-}"

extra=()
[[ "$SKIP_MATCHING" == "1" ]] && extra+=(--skip-matching)
[[ -n "$TOPICS" ]] && extra+=(--topics "$TOPICS")
[[ -n "$PRIOR_CHECKPOINT" ]] && extra+=(--prior-checkpoint "$PRIOR_CHECKPOINT")
[[ -n "$REALIZATION_CHECKPOINT" ]] && extra+=(--realization-checkpoint "$REALIZATION_CHECKPOINT")

python examples/benchmark.py \
  --input-dir "$INPUT_DIR" \
  --strategy "$STRATEGY" \
  --model-name "$MODEL" \
  --start-month "$START_MONTH" \
  --end-month "$END_MONTH" \
  --min-cutoff-month "$MIN_CUTOFF_MONTH" \
  --horizon-months "$HORIZON_MONTHS" \
  --top-k "$TOP_K" \
  --min-train-papers "$MIN_TRAIN_PAPERS" \
  --workers "$WORKERS" \
  --output "$OUTPUT" \
  "${extra[@]}" \
  "$@"
