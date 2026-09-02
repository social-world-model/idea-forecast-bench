#!/usr/bin/env bash
# The combinatorial (Ramon-Llull) experiment, end to end, against local
# OpenAI-compatible replicas (see scripts/benchmark/serve_multi.sh):
#
#   extract elements -> embed labels -> generate (4 sampler arms, sharded by
#   topic) -> judge -> main table -> specificity rating
#
# Every step is idempotent: extraction and judging resume from their own
# caches, generation skips completed topics inside an artifact.
#
#   PORTS="31000 31001" TOPICS="llm_pretraining,moe" OUTPUT_DIR=output/pilot \
#     bash scripts/run_combinatorial.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT_DIR="${INPUT_DIR:-data/hf_full/raw_markdown}"
OUTPUT_DIR="${OUTPUT_DIR:-output/sweep_combi}"
PORTS="${PORTS:-31000}"
HOST="${HOST:-127.0.0.1}"
MODEL="${MODEL:-gpt-4o-qwen35}"          # served alias; must start with gpt-4o/gpt-4.1/gpt-5
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-qwen35-judge}"
LABEL="${LABEL:-Qwen3.5-9B}"
STRATEGIES="${STRATEGIES:-combinatorial combinatorial_frequency combinatorial_independent combinatorial_random}"
TOPICS="${TOPICS:-}"                     # comma-separated; empty = all 52
ELEMENT_CACHE="${ELEMENT_CACHE:-$OUTPUT_DIR/elements}"
EXTRACT_WORKERS="${EXTRACT_WORKERS:-64}"
GEN_WORKERS="${GEN_WORKERS:-8}"
JUDGE_WORKERS="${JUDGE_WORKERS:-16}"
export START_MONTH="${START_MONTH:-2024-04}"
export END_MONTH="${END_MONTH:-2025-09}"
export MIN_CUTOFF_MONTH="${MIN_CUTOFF_MONTH:-2024-07}"
export INPUT_DIR

: "${VOYAGE_API_KEY:?VOYAGE_API_KEY is not set (element merging and judging embed with Voyage)}"
# Anything left over from another run silently redirects calls elsewhere.
unset OPENAI_BASE_URL JUDGE_BASE_URL JUDGE_MODEL JUDGE_API_KEY VOYAGE_BASE_URL TOGETHER_API_KEY DEEPSEEK_API_KEY
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

read -ra ports <<< "$PORTS"
n_ports=${#ports[@]}
base_urls=""
for p in "${ports[@]}"; do
  base_urls+="${base_urls:+,}http://${HOST}:${p}/v1"
done
port_for() { echo "${ports[$(( $1 % n_ports ))]}"; }

if [[ -z "${IDEA_FORECAST_CORPUS_FINGERPRINT:-}" ]]; then
  IDEA_FORECAST_CORPUS_FINGERPRINT="$(python -c \
    "from idea_forecast_bench.papers import corpus_fingerprint; print(corpus_fingerprint('$INPUT_DIR'))")"
  export IDEA_FORECAST_CORPUS_FINGERPRINT
fi

GEN_DIR="$OUTPUT_DIR/backtest"
JUDGED_DIR="$OUTPUT_DIR/judged"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$GEN_DIR" "$JUDGED_DIR" "$LOG_DIR" "$ELEMENT_CACHE"

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

topic_args=()
[[ -n "$TOPICS" ]] && topic_args+=(--topics "$TOPICS")

# 1. extract + embed --------------------------------------------------------
echo "== 1/5 extract elements ($MODEL via $n_ports replica(s)) =="
python examples/benchmark/extract_elements.py \
  --input-dir "$INPUT_DIR" --start-month "$START_MONTH" --end-month "$END_MONTH" \
  --cache-dir "$ELEMENT_CACHE" --model-name "$MODEL" --base-urls "$base_urls" \
  --workers "$EXTRACT_WORKERS" "${topic_args[@]}" 2>&1 | tee "$LOG_DIR/extract.log"
echo "== 2/5 embed element labels (Voyage) =="
python examples/benchmark/extract_elements.py \
  --input-dir "$INPUT_DIR" --start-month "$START_MONTH" --end-month "$END_MONTH" \
  --cache-dir "$ELEMENT_CACHE" --model-name "$MODEL" --base-urls "$base_urls" \
  --embed --dump-clusters 30 "${topic_args[@]}" 2>&1 | tee "$LOG_DIR/embed.log"

# 2. generate (shard = one topic group per replica) ---------------------------
echo "== 3/5 generate: $STRATEGIES =="
if [[ -n "$TOPICS" ]]; then
  IFS=',' read -ra all_topics <<< "$TOPICS"
else
  mapfile -t all_topics < <(python -c "from idea_forecast_bench.config import load_topics; print('\n'.join(t.id for t in load_topics()))")
fi
shards=()
for ((i = 0; i < n_ports; i++)); do shards+=(""); done
for idx in "${!all_topics[@]}"; do
  s=$((idx % n_ports))
  shards[$s]="${shards[$s]:+${shards[$s]},}${all_topics[$idx]}"
done
pids=()
for strategy in $STRATEGIES; do
  for i in "${!shards[@]}"; do
    [[ -z "${shards[$i]}" ]] && continue
    label="${strategy}.s${i}"
    STRATEGY="$strategy" TOPICS="${shards[$i]}" OUTPUT="$GEN_DIR/$label.json" \
      MODEL="$MODEL" ELEMENT_CACHE="$ELEMENT_CACHE" WORKERS="$GEN_WORKERS" \
      OPENAI_BASE_URL="http://${HOST}:$(port_for "$i")/v1" \
      bash scripts/benchmark/benchmark.sh > "$LOG_DIR/$label.log" 2>&1 &
    pids+=("$!")
  done
done
wait_all "${pids[@]}"

# 3. judge -------------------------------------------------------------------
echo "== 4/5 judge ($JUDGE_MODEL_NAME) =="
artifacts=("$GEN_DIR"/*.json)
pids=()
i=0
for artifact in "${artifacts[@]}"; do
  label="$(basename "$artifact" .json)"
  topics="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("topics_shard") or ",".join(d.get("topic_results",{})))' "$artifact")"
  INPUT_JSON="$artifact" INPUT_DIR="$INPUT_DIR" TOPICS="$topics" \
    OUTPUT="$JUDGED_DIR/$label.judged.json" WORKERS="$JUDGE_WORKERS" \
    JUDGE_MODEL="$JUDGE_MODEL_NAME" JUDGE_BASE_URL="http://${HOST}:$(port_for "$i")/v1" \
    bash scripts/benchmark/judge_eval.sh > "$LOG_DIR/$label.judge.log" 2>&1 &
  pids+=("$!")
  i=$((i + 1))
done
wait_all "${pids[@]}"
for judged in "$JUDGED_DIR"/*.judged.json; do
  n="$(grep -c "Error code" "$judged" || true)"
  [[ "$n" != "0" ]] && echo "WARNING: $judged has $n stored judge errors (scored 0, not 'no match')" >&2
done

# 4. table + specificity -----------------------------------------------------
echo "== 5/5 table + specificity =="
expected_topics=${#all_topics[@]}
expected_windows=$((expected_topics * 12))
SOURCES="$LABEL=$JUDGED_DIR/*.judged.json" EXPECTED_TOPICS="$expected_topics" EXPECTED_WINDOWS="$expected_windows" \
  bash scripts/benchmark/main_table.sh | tee "$OUTPUT_DIR/table.txt"
python examples/benchmark/specificity_eval.py \
  --input-json "$GEN_DIR/*.json" --output "$OUTPUT_DIR/specificity.json" \
  --model-name "$MODEL" --base-urls "$base_urls" 2>&1 | tee "$LOG_DIR/specificity.log"
echo "done: $OUTPUT_DIR/table.txt, $OUTPUT_DIR/specificity.json"
