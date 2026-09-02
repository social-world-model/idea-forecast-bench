#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${INPUT_DIR:-data/csml/raw_markdown}"
OUTPUT_DIR="${OUTPUT_DIR:-output/sharded/backtest}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
MODEL="${MODEL:-gpt-4.1}"
STRATEGIES="${STRATEGIES:-topic_trend predictor_llm summary_prompting retrieval_prompting memory_prompting}"
SHARDS="${SHARDS:-4}"
WORKERS="${WORKERS:-8}"
START_MONTH="${START_MONTH:-2024-04}"
END_MONTH="${END_MONTH:-2025-09}"
MIN_CUTOFF_MONTH="${MIN_CUTOFF_MONTH:-2024-07}"
HORIZON_MONTHS="${HORIZON_MONTHS:-3}"
TOP_K="${TOP_K:-5}"
MIN_TRAIN_PAPERS="${MIN_TRAIN_PAPERS:-2}"
SKIP_MATCHING="${SKIP_MATCHING:-1}"

if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  : "${OPENAI_API_KEY:?OPENAI_API_KEY is not set and OPENAI_BASE_URL is empty -- generation cannot run}"
fi
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "INPUT_DIR does not exist: $INPUT_DIR" >&2
  exit 1
fi

# The fingerprint is one stat per paper; over NFS on a 108k-paper corpus that
# is minutes, paid once per process. Derive it once and hand it to all of them.
if [[ -z "${IDEA_FORECAST_CORPUS_FINGERPRINT:-}" ]]; then
  IDEA_FORECAST_CORPUS_FINGERPRINT="$("$PYTHON_BIN" -c \
    "from idea_forecast_bench.papers import corpus_fingerprint; print(corpus_fingerprint('$INPUT_DIR'))")"
  export IDEA_FORECAST_CORPUS_FINGERPRINT
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

mapfile -t SHARD_TOPICS < <("$PYTHON_BIN" examples/benchmark/split_topics.py \
  --input-dir "$INPUT_DIR" --shards "$SHARDS" \
  --start-month "$START_MONTH" --end-month "$END_MONTH")
if [[ "${#SHARD_TOPICS[@]}" -ne "$SHARDS" ]]; then
  echo "expected $SHARDS shards, split_topics.py produced ${#SHARD_TOPICS[@]}" >&2
  exit 1
fi

extra=()
if [[ "$SKIP_MATCHING" == "1" ]]; then
  extra+=(--skip-matching)
fi

pids=()
labels=()
for strategy in $STRATEGIES; do
  for ((i = 0; i < SHARDS; i++)); do
    label="${strategy}.s${i}"
    "$PYTHON_BIN" examples/benchmark/run_domain_backtest.py \
      --input-dir "$INPUT_DIR" \
      --strategy "$strategy" --model-name "$MODEL" \
      --start-month "$START_MONTH" --end-month "$END_MONTH" \
      --min-cutoff-month "$MIN_CUTOFF_MONTH" \
      --horizon-months "$HORIZON_MONTHS" --top-k "$TOP_K" \
      --min-train-papers "$MIN_TRAIN_PAPERS" \
      --workers "$WORKERS" \
      --topics "${SHARD_TOPICS[$i]}" \
      --output "$OUTPUT_DIR/${label}.json" \
      "${extra[@]}" \
      > "$LOG_DIR/${label}.log" 2>&1 &
    pids+=("$!")
    labels+=("$label")
  done
done

echo "launched ${#pids[@]} processes (${SHARDS} shards x strategies: ${STRATEGIES})"
echo "  model:  $MODEL"
echo "  output: $OUTPUT_DIR"
echo "  logs:   $LOG_DIR"

failed=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "FAILED: ${labels[$idx]} (see $LOG_DIR/${labels[$idx]}.log)" >&2
    failed=$((failed + 1))
  fi
done
if [[ "$failed" -gt 0 ]]; then
  echo "$failed process(es) failed" >&2
  exit 1
fi
echo "all ${#pids[@]} processes finished"
