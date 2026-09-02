#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT_DIR="${INPUT_DIR:-data/csml/raw_markdown}"
OUTPUT_DIR="${OUTPUT_DIR:-data/topic_hindsight}"
MODE="${MODE:-full}"
HINDSIGHT_MODEL="${HINDSIGHT_MODEL:-}"

extra=()
[[ -n "$HINDSIGHT_MODEL" ]] && extra+=(--model "$HINDSIGHT_MODEL")

python examples/hindsight.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mode "$MODE" \
  "${extra[@]}" \
  "$@"
