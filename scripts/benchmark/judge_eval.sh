#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

INPUT_JSON="${INPUT_JSON:-output/backtest/summary_prompting.json}"
INPUT_DIR="${INPUT_DIR:-data/csml/raw_markdown}"
OUTPUT="${OUTPUT:-${INPUT_JSON%.json}.judged.json}"
STATE_FILE="${STATE_FILE:-${OUTPUT%.json}.state.json}"
WORKERS="${WORKERS:-8}"
TOPIC_WORKERS="${TOPIC_WORKERS:-4}"
TOPICS="${TOPICS:-}"
JUDGE_MODEL_FLAG="${JUDGE_MODEL:-}"
JUDGE_BASE_URL_FLAG="${JUDGE_BASE_URL:-}"

# The judge is chosen by flag only. Any of these left exported silently
# redirects scoring to another model without an error.
unset JUDGE_BASE_URL JUDGE_MODEL JUDGE_API_KEY OPENAI_BASE_URL VOYAGE_BASE_URL

extra=()
[[ -n "$TOPICS" ]] && extra+=(--topics "$TOPICS")
[[ -n "$JUDGE_MODEL_FLAG" ]] && extra+=(--judge-model "$JUDGE_MODEL_FLAG")
[[ -n "$JUDGE_BASE_URL_FLAG" ]] && extra+=(--judge-base-url "$JUDGE_BASE_URL_FLAG")

python examples/benchmark/judge_eval.py \
  --input-json "$INPUT_JSON" \
  --papers-dir "$INPUT_DIR" \
  --output "$OUTPUT" \
  --state-file "$STATE_FILE" \
  --workers "$WORKERS" \
  --topic-workers "$TOPIC_WORKERS" \
  "${extra[@]}" \
  "$@"
