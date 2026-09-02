#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

INPUT="${INPUT:-output/backtest/summary_prompting.json}"
OUTPUT="${OUTPUT:-output/analysis/leakage.json}"

# shellcheck disable=SC2086  # INPUT may list several files
python examples/benchmark/analysis_leakage.py \
  --input $INPUT \
  --output "$OUTPUT" \
  "$@"
