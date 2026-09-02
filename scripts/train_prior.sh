#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="${MODEL:-qwen2.5-7b-instruct}"
HINDSIGHT="${HINDSIGHT:-data/topic_hindsight/hindsight_samples.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/mdf/prior_sft}"

python examples/train_prior.py \
  --model "$MODEL" \
  --hindsight "$HINDSIGHT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
