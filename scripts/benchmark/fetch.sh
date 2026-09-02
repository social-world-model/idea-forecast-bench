#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

OUT_DIR="${OUT_DIR:-data/csml/raw_markdown}"
QUERY="${QUERY:-cat:cs.LG}"
MAX_RESULTS="${MAX_RESULTS:-2000}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-540}"

python examples/benchmark/fetch.py \
  --out-dir "$OUT_DIR" \
  --query "$QUERY" \
  --max-results "$MAX_RESULTS" \
  --lookback-days "$LOOKBACK_DAYS" \
  "$@"
