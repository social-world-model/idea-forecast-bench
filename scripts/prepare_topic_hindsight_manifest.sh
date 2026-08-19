#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${INPUT_DIR:-md_mineru}"
OUTPUT_DIR="${OUTPUT_DIR:-data/topic_hindsight}"
TOPICS_CONFIG="${TOPICS_CONFIG:-config/topics_v2.yaml}"
PER_TOPIC_PER_EPISODE="${PER_TOPIC_PER_EPISODE:-2}"

python "$ROOT_DIR/examples/data/prepare_topic_hindsight_manifest.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --topics-config "$TOPICS_CONFIG" \
  --per-topic-per-episode "$PER_TOPIC_PER_EPISODE" \
  "$@"
