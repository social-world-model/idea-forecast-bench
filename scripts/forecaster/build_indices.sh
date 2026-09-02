#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

INPUT_DIR="${INPUT_DIR:-data/csml/raw_markdown}"
HINDSIGHT="${HINDSIGHT:-data/topic_hindsight/hindsight_samples.jsonl}"
DZ="${DZ:-data/topic_hindsight/dz.jsonl}"
ARTIFACT_DIR="${ARTIFACT_DIR:-output/foresight_artifacts}"

python examples/forecaster/build_indices.py \
  --papers-dir "$INPUT_DIR" \
  --hindsight "$HINDSIGHT" \
  --dz "$DZ" \
  --art "$ARTIFACT_DIR" \
  "$@"
