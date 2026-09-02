#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

INPUT_DIR="${INPUT_DIR:-data/csml/raw_markdown}"
START_MONTH="${START_MONTH:-2024-04}"
END_MONTH="${END_MONTH:-2025-09}"
CACHE_DIR="${CACHE_DIR:-.cache/elements}"
MODEL="${MODEL:-gpt-4o-qwen35}"
WORKERS="${WORKERS:-16}"
TOPICS="${TOPICS:-}"
BASE_URLS="${BASE_URLS:-}"

extra=()
[[ -n "$TOPICS" ]] && extra+=(--topics "$TOPICS")
[[ -n "$BASE_URLS" ]] && extra+=(--base-urls "$BASE_URLS")

python examples/benchmark/extract_elements.py \
  --input-dir "$INPUT_DIR" \
  --start-month "$START_MONTH" \
  --end-month "$END_MONTH" \
  --cache-dir "$CACHE_DIR" \
  --model-name "$MODEL" \
  --workers "$WORKERS" \
  "${extra[@]}" \
  "$@"
