#!/usr/bin/env bash
# Generate predictions for several strategies, sharded across processes by topic.
#
# This is how the paper's sweep was produced: STRATEGIES x SHARDS processes, each
# running `benchmark` over a disjoint --topics set with its own --output. Sharding
# is by topic because a single interpreter saturates on GIL-bound work long
# before the LLM endpoint does; the shards are balanced by paper count
# (examples/benchmark/split_topics.py) because the cutoffs inside a shard run
# serially and an unbalanced split leaves every other process waiting on one.
#
# By default the run skips the embedding match (--skip-matching): the reported
# numbers come from `judge-eval`, which re-embeds and re-retrieves everything
# itself, so matching here would be ~1M wasted Voyage calls. Every metric in the
# generation artifacts is therefore NaN on purpose; score them with
# scripts/benchmark/run_sharded_judge.sh.
#
# Defaults reproduce the paper's window: 12 monthly cutoffs 2024-07..2025-06,
# 3-month horizon, 52 topics = 624 windows per strategy.
#
# Environment (every variable has a default):
#   INPUT_DIR         corpus of markdown papers        data/csml/raw_markdown
#   OUTPUT_DIR        where <strategy>.s<N>.json land   output/sharded/backtest
#   LOG_DIR           one log per process               $OUTPUT_DIR/logs
#   MODEL             generation model                  gpt-4.1
#   STRATEGIES        space-separated                   all five baselines
#   SHARDS            processes per strategy            4
#   WORKERS           threads per process               8
#   START_MONTH / END_MONTH / MIN_CUTOFF_MONTH / HORIZON_MONTHS / TOP_K / MIN_TRAIN_PAPERS
#   SKIP_MATCHING     1 to skip the embedding match     1
#   PYTHON_BIN        interpreter to use                python
#   OPENAI_BASE_URL   point generation at a local OpenAI-compatible server; see
#                     scripts/benchmark/serve_vllm.sh for the naming requirement
#   IDEA_FORECAST_CORPUS_FINGERPRINT  derived once here if unset, so the
#                     processes do not each walk the corpus to compute it
#
# Usage:
#   OPENAI_API_KEY=... bash scripts/benchmark/run_sharded_backtest.sh
#   MODEL=gpt-4o-qwen7b OPENAI_BASE_URL=http://127.0.0.1:31000/v1 OPENAI_API_KEY=EMPTY \
#     bash scripts/benchmark/run_sharded_backtest.sh
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
