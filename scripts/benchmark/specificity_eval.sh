#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

INPUT_JSON="${INPUT_JSON:-output/backtest/*.json}"
OUTPUT="${OUTPUT:-output/specificity.json}"
MODEL="${MODEL:-gpt-4o-qwen35}"
WORKERS="${WORKERS:-16}"
BASE_URLS="${BASE_URLS:-}"

extra=()
[[ -n "$BASE_URLS" ]] && extra+=(--base-urls "$BASE_URLS")

python examples/benchmark/specificity_eval.py \
  --input-json "$INPUT_JSON" \
  --output "$OUTPUT" \
  --model-name "$MODEL" \
  --workers "$WORKERS" \
  "${extra[@]}" \
  "$@"
