#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT_DIR="${INPUT_DIR:-data/csml/raw_markdown}"
SHARDS="${SHARDS:-4}"
START_MONTH="${START_MONTH:-2024-04}"
END_MONTH="${END_MONTH:-2025-09}"

python examples/split_topics.py \
  --input-dir "$INPUT_DIR" \
  --shards "$SHARDS" \
  --start-month "$START_MONTH" \
  --end-month "$END_MONTH" \
  "$@"
