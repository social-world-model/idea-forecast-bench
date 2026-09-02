#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT_DIR="${INPUT_DIR:-data/csml/raw_markdown}"
OUTPUT_DIR="${OUTPUT_DIR:-output/baselines}"
MODEL="${MODEL:-gpt-4.1}"
START_MONTH="${START_MONTH:-2024-01}"
END_MONTH="${END_MONTH:-2025-06}"
TOP_K="${TOP_K:-5}"
WORKERS="${WORKERS:-4}"
ONLY="${ONLY:-}"

extra=()
[[ -n "$ONLY" ]] && extra+=(--only "$ONLY")

python examples/baselines.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL" \
  --start-month "$START_MONTH" \
  --end-month "$END_MONTH" \
  --top-k "$TOP_K" \
  --workers "$WORKERS" \
  "${extra[@]}" \
  "$@"
