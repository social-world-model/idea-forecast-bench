#!/usr/bin/env bash
# Score sharded generation artifacts with the retrieve-then-judge protocol.
#
# One `judge-eval` process per artifact in GEN_DIR, each with its OWN
# --state-file. judge/state.py's RunState guards its checkpoint with a
# threading.Lock only: two processes sharing a state file each load it, then
# replace it over the other's writes, and the tmp-file rename races. Never
# point two processes at one state file.
#
# Concurrency is three layers multiplied: processes x --workers x --topic-workers.
# Against a hosted API (the default gpt-4.1-mini judge) 20 x 8 x 4 = 640 is fine;
# the API queues. Against a self-hosted vLLM endpoint the same numbers collapse
# it -- thousands of timeouts, throughput down to a third -- so keep the product
# near 240 there (e.g. WORKERS=4 TOPIC_WORKERS=1 over 60 processes). The signal
# to watch is the endpoint's num_requests_waiting, not GPU utilisation: waiting
# above zero means back off; waiting at zero with high utilisation means the
# clients are not feeding it and you can add processes.
#
# Environment (every variable has a default):
#   GEN_DIR           artifacts from run_sharded_backtest.sh   output/sharded/backtest
#   OUTPUT_DIR        <name>.judged.json + .state.json          output/sharded/judged
#   LOG_DIR           one log per process                      $OUTPUT_DIR/logs
#   INPUT_DIR         the same corpus the generation run used  data/csml/raw_markdown
#   WORKERS           judge threads per window                 8
#   TOPIC_WORKERS     topics in flight per process             4
#   JUDGE_MODEL       passed as --judge-model                  (judge-eval's default)
#   JUDGE_BASE_URL    passed as --judge-base-url               (hosted API)
#   PYTHON_BIN        interpreter to use                       python
#   IDEA_FORECAST_CORPUS_FINGERPRINT  derived once here if unset
#
# Usage:
#   OPENAI_API_KEY=... VOYAGE_API_KEY=... bash scripts/benchmark/run_sharded_judge.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GEN_DIR="${GEN_DIR:-output/sharded/backtest}"
OUTPUT_DIR="${OUTPUT_DIR:-output/sharded/judged}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
INPUT_DIR="${INPUT_DIR:-data/csml/raw_markdown}"
WORKERS="${WORKERS:-8}"
TOPIC_WORKERS="${TOPIC_WORKERS:-4}"
JUDGE_MODEL_FLAG="${JUDGE_MODEL:-}"
JUDGE_BASE_URL_FLAG="${JUDGE_BASE_URL:-}"

# The judge and the embedder are chosen HERE, by flag, and only here. Any of
# these left exported in the shell silently redirects scoring somewhere else
# (a GRPO run sets JUDGE_BASE_URL to a local vLLM, for example) and nothing
# errors: the run finishes and produces numbers that were never judged by the
# model you think you used. JUDGE_MODEL / JUDGE_BASE_URL were captured into
# the *_FLAG variables above and are passed explicitly as flags below.
unset JUDGE_BASE_URL JUDGE_MODEL JUDGE_API_KEY OPENAI_BASE_URL VOYAGE_BASE_URL

: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set (the judge needs it)}"
: "${VOYAGE_API_KEY:?VOYAGE_API_KEY is not set (retrieval embeddings need it)}"

if pgrep -f "[e]xamples/benchmark/llm_judge_eval.py" > /dev/null 2>&1; then
  echo "a judge-eval process is already running; refusing to start a second sweep" >&2
  echo "(two processes on one state file corrupt each other -- see the header)" >&2
  exit 1
fi

shopt -s nullglob
artifacts=("$GEN_DIR"/*.json)
if [[ "${#artifacts[@]}" -eq 0 ]]; then
  echo "no artifacts in $GEN_DIR" >&2
  exit 1
fi

if [[ -z "${IDEA_FORECAST_CORPUS_FINGERPRINT:-}" ]]; then
  IDEA_FORECAST_CORPUS_FINGERPRINT="$("$PYTHON_BIN" -c \
    "from idea_forecast_bench.papers import corpus_fingerprint; print(corpus_fingerprint('$INPUT_DIR'))")"
  export IDEA_FORECAST_CORPUS_FINGERPRINT
fi

judge_flags=()
[[ -n "$JUDGE_MODEL_FLAG" ]] && judge_flags+=(--judge-model "$JUDGE_MODEL_FLAG")
[[ -n "$JUDGE_BASE_URL_FLAG" ]] && judge_flags+=(--judge-base-url "$JUDGE_BASE_URL_FLAG")

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

pids=()
labels=()
for artifact in "${artifacts[@]}"; do
  label="$(basename "$artifact" .json)"
  # Restrict each process to its shard's topics; otherwise the judge walks all
  # 52 and no-ops on the ones the artifact does not contain.
  topics="$("$PYTHON_BIN" -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("topics_shard") or ",".join(d.get("topic_results", {})))
' "$artifact")"
  "$PYTHON_BIN" examples/benchmark/llm_judge_eval.py \
    --input-json "$artifact" \
    --papers-dir "$INPUT_DIR" \
    --output "$OUTPUT_DIR/${label}.judged.json" \
    --state-file "$OUTPUT_DIR/${label}.state.json" \
    --topics "$topics" \
    --workers "$WORKERS" --topic-workers "$TOPIC_WORKERS" \
    "${judge_flags[@]}" \
    > "$LOG_DIR/${label}.log" 2>&1 &
  pids+=("$!")
  labels+=("$label")
done

echo "launched ${#pids[@]} judge processes (${WORKERS} x ${TOPIC_WORKERS} threads each)"
echo "  input:  $GEN_DIR"
echo "  output: $OUTPUT_DIR"

failed=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "FAILED: ${labels[$idx]} (see $LOG_DIR/${labels[$idx]}.log)" >&2
    failed=$((failed + 1))
  fi
done

# A judge request that errors (a 404 from a mis-named endpoint, say) is stored
# as a judgement of zero on every dimension, which is indistinguishable from
# "genuinely no match" by score alone. Count the stored errors before trusting
# the numbers.
for judged in "$OUTPUT_DIR"/*.judged.json; do
  n="$(grep -c "Error code" "$judged" || true)"
  if [[ "$n" != "0" ]]; then
    echo "WARNING: $judged contains $n stored judge errors -- those windows scored 0, not 'no match'" >&2
  fi
done

if [[ "$failed" -gt 0 ]]; then
  echo "$failed process(es) failed" >&2
  exit 1
fi
echo "all ${#pids[@]} judge processes finished"
