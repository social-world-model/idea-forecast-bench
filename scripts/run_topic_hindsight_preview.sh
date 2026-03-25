#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${INPUT_DIR:-md_mineru}"
OUTPUT_DIR="${OUTPUT_DIR:-data/topic_hindsight}"
MANIFEST_PATH="${MANIFEST_PATH:-$OUTPUT_DIR/manifest.json}"
TOPICS_CONFIG="${TOPICS_CONFIG:-config/topics_v2.yaml}"
MODEL="${MODEL:-gpt-5.4}"
PREVIEW_COUNT="${PREVIEW_COUNT:-10}"

python "$ROOT_DIR/examples/run_topic_hindsight_preview.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --manifest-path "$MANIFEST_PATH" \
  --topics-config "$TOPICS_CONFIG" \
  --model "$MODEL" \
  --preview-count "$PREVIEW_COUNT" \
  "$@"
