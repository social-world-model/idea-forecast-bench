#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The paper's sweep: generate (sharded by topic) -> judge -> table.
INPUT_DIR="${INPUT_DIR:-data/csml/raw_markdown}"
OUTPUT_DIR="${OUTPUT_DIR:-output/sweep}"
MODEL="${MODEL:-gpt-4.1}"
LABEL="${LABEL:-$MODEL}"
STRATEGIES="${STRATEGIES:-topic_trend predictor_llm summary_prompting retrieval_prompting memory_prompting}"
SHARDS="${SHARDS:-4}"
export START_MONTH="${START_MONTH:-2024-04}"
export END_MONTH="${END_MONTH:-2025-09}"
export INPUT_DIR MODEL

if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  : "${OPENAI_API_KEY:?OPENAI_API_KEY is not set and OPENAI_BASE_URL is empty}"
fi
: "${VOYAGE_API_KEY:?VOYAGE_API_KEY is not set (judging needs it)}"

# Every process would otherwise walk the corpus to compute this.
if [[ -z "${IDEA_FORECAST_CORPUS_FINGERPRINT:-}" ]]; then
  IDEA_FORECAST_CORPUS_FINGERPRINT="$(python -c \
    "from idea_forecast_bench.papers import corpus_fingerprint; print(corpus_fingerprint('$INPUT_DIR'))")"
  export IDEA_FORECAST_CORPUS_FINGERPRINT
fi

GEN_DIR="$OUTPUT_DIR/backtest"
JUDGED_DIR="$OUTPUT_DIR/judged"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$GEN_DIR" "$JUDGED_DIR" "$LOG_DIR"

wait_all() {
  local failed=0 pid
  for pid in "$@"; do
    wait "$pid" || failed=$((failed + 1))
  done
  if [[ "$failed" -gt 0 ]]; then
    echo "$failed process(es) failed; see $LOG_DIR" >&2
    exit 1
  fi
}

# 1. generate ---------------------------------------------------------------
SHARD_TOPICS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && SHARD_TOPICS+=("$line")
done < <(SHARDS="$SHARDS" bash scripts/benchmark/split_topics.sh)
echo "generating: ${#SHARD_TOPICS[@]} shards x strategies: $STRATEGIES"
pids=()
for strategy in $STRATEGIES; do
  for i in "${!SHARD_TOPICS[@]}"; do
    label="${strategy}.s${i}"
    STRATEGY="$strategy" TOPICS="${SHARD_TOPICS[$i]}" OUTPUT="$GEN_DIR/$label.json" \
      bash scripts/benchmark/benchmark.sh > "$LOG_DIR/$label.log" 2>&1 &
    pids+=("$!")
  done
done
wait_all "${pids[@]}"

# 2. judge ------------------------------------------------------------------
# One process per artifact, each with its own state file: the judge's
# checkpoint is safe within a process, not across processes.
artifacts=("$GEN_DIR"/*.json)
echo "judging: ${#artifacts[@]} artifacts"
pids=()
for artifact in "${artifacts[@]}"; do
  label="$(basename "$artifact" .json)"
  topics="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("topics_shard") or ",".join(d.get("topic_results",{})))' "$artifact")"
  INPUT_JSON="$artifact" TOPICS="$topics" OUTPUT="$JUDGED_DIR/$label.judged.json" \
    bash scripts/benchmark/judge_eval.sh > "$LOG_DIR/$label.judge.log" 2>&1 &
  pids+=("$!")
done
wait_all "${pids[@]}"

# A judge request that errored is stored as a zero score, which is
# indistinguishable from "no match" by score alone.
for judged in "$JUDGED_DIR"/*.judged.json; do
  n="$(grep -c "Error code" "$judged" || true)"
  [[ "$n" != "0" ]] && echo "WARNING: $judged has $n stored judge errors (scored 0, not 'no match')" >&2
done

# 3. table ------------------------------------------------------------------
SOURCES="$LABEL=$JUDGED_DIR/*.judged.json" bash scripts/benchmark/main_table.sh
