#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-output/mdf/prior_sft/final_checkpoint}"
REALIZATION_CHECKPOINT="${REALIZATION_CHECKPOINT:-output/mdf/realization_grpo/grpo/checkpoints/final_checkpoint}"
HINDSIGHT="${HINDSIGHT:-data/topic_hindsight/hindsight_samples.jsonl}"
INPUT_DIR="${INPUT_DIR:-data/csml/raw_markdown}"
OUTPUT_DIR="${OUTPUT_DIR:-output/mdf/inference}"

python examples/forecaster/infer.py \
  --prior-checkpoint "$PRIOR_CHECKPOINT" \
  --realization-checkpoint "$REALIZATION_CHECKPOINT" \
  --hindsight "$HINDSIGHT" \
  --papers-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
