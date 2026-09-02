#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL="${MODEL:-qwen2.5-7b-instruct}"
INPUT_DIR="${INPUT_DIR:-data/csml/raw_markdown}"
HINDSIGHT="${HINDSIGHT:-data/topic_hindsight/hindsight_samples.jsonl}"
INIT_POLICY="${INIT_POLICY:-output/mdf/prior_sft/final_checkpoint}"
TRAINER_CONFIG="${TRAINER_CONFIG:-grpo_train.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-output/mdf/realization_grpo}"
export USE_VLLM="${USE_VLLM:-1}"

python examples/forecaster/train.py \
  --model-preset "$MODEL" \
  --input-dir "$INPUT_DIR" \
  --hindsight "$HINDSIGHT" \
  --init-policy-path "$INIT_POLICY" \
  --trainer grpo \
  --trainer-config "$TRAINER_CONFIG" \
  --skip-alignment-check \
  --output-dir "$OUTPUT_DIR" \
  "$@"
