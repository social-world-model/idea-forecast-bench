#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DZ="${DZ:-data/topic_hindsight/dz.jsonl}"
ARTIFACT_DIR="${ARTIFACT_DIR:-output/foresight_artifacts}"
MODE="${MODE:-live}"
JUDGE_MODEL_FLAG="${JUDGE_MODEL:-}"
JUDGE_BASE_URL_FLAG="${JUDGE_BASE_URL:-}"

extra=()
[[ -n "$JUDGE_MODEL_FLAG" ]] && extra+=(--model "$JUDGE_MODEL_FLAG")
[[ -n "$JUDGE_BASE_URL_FLAG" ]] && extra+=(--judge-base-url "$JUDGE_BASE_URL_FLAG")

python examples/forecaster/build_rubrics.py \
  --dz "$DZ" \
  --rubrics-dir "$ARTIFACT_DIR/rubrics" \
  --report "$ARTIFACT_DIR/rubric_validation.md" \
  --leakage-report "$ARTIFACT_DIR/leakage.md" \
  --mode "$MODE" \
  "${extra[@]}" \
  "$@"
