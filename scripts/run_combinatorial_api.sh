#!/usr/bin/env bash
# The combinatorial experiment against a hosted OpenAI-compatible API
# (DashScope / 百炼 by default), for machines without a GPU.
#
# Two steps, because the cheap one answers most of the risk:
#
#   JUDGE=0  extract elements -> generate ideas, NOT scored.        (~minutes)
#            Read the ideas. If the elements are junk or the ideas are
#            generic, stop here: no amount of judging fixes that.
#   JUDGE=1  the same, then score with the frozen retrieve-then-judge
#            protocol and print the arm table.                      (~hours)
#
#   DASHSCOPE_API_KEY=... VOYAGE_API_KEY=... \
#     TOPICS=llm_alignment_rlhf,rag_retrieval MIN_CUTOFF_MONTH=2025-03 \
#     OUTPUT_DIR=output/pilot bash scripts/run_combinatorial_api.sh
#
# Thinking is off by default at every stage. vanchin/deepseek-v4-pro-0813
# enables it unless told otherwise, does not support thinking_budget, and
# promotes reasoning_effort low/medium to high -- so it is either off or
# unbounded, and an unbounded trace can eat the completion budget before the
# JSON appears. REALIZE_THINKING=high turns it on for realisation only, which
# is the one stage cheap enough (hundreds of calls) to experiment with.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT_DIR="${INPUT_DIR:-data/hf_full/raw_markdown}"
OUTPUT_DIR="${OUTPUT_DIR:-output/combi_api}"
# Element extraction is a high-volume tagging task, so it runs on a fast, cheap
# model; realisation is a few hundred calls where quality matters. Marketplace
# ids (vendor/model) can carry a far lower RPM quota than first-party ones --
# check before pointing a bulk stage at one.
MODEL="${MODEL:-vanchin/deepseek-v4-pro-0813}"
EXTRACT_MODEL="${EXTRACT_MODEL:-deepseek-v4-flash}"
REALIZE_MODEL="${REALIZE_MODEL:-$MODEL}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-$EXTRACT_MODEL}"
JUDGE_BASE_URL_VALUE="${JUDGE_BASE_URL_VALUE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
LABEL="${LABEL:-DeepSeek-V4-Pro}"
ARMS="${ARMS:-combinatorial combinatorial_independent combinatorial_random}"
TOPICS="${TOPICS:-}"
JUDGE="${JUDGE:-0}"
REALIZE_THINKING="${REALIZE_THINKING:-off}"   # off | high | max
ELEMENT_CACHE="${ELEMENT_CACHE:-$OUTPUT_DIR/elements}"
EXTRACT_WORKERS="${EXTRACT_WORKERS:-16}"
EXTRACT_RPM="${EXTRACT_RPM:-0}"   # 0 = uncapped; set it when the model has a low RPM quota
GEN_WORKERS="${GEN_WORKERS:-4}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
JUDGE_TOPIC_WORKERS="${JUDGE_TOPIC_WORKERS:-4}"
export START_MONTH="${START_MONTH:-2024-04}"
export END_MONTH="${END_MONTH:-2025-09}"
MIN_CUTOFF_MONTH="${MIN_CUTOFF_MONTH:-2024-07}"

: "${DASHSCOPE_API_KEY:?DASHSCOPE_API_KEY is not set}"
# Judge retrieval is Voyage-only by design, so scoring cannot start without
# the key. Element merging can fall back to offline hashing, which is enough
# to eyeball a JUDGE=0 pass -- near-synonyms just stay separate.
# NOTE: no apostrophes inside ${VAR:?...} -- bash 3.2 (macOS) mis-parses them.
EMBED_BACKEND="${EMBED_BACKEND:-}"
if [[ "$JUDGE" == "1" ]]; then
  : "${VOYAGE_API_KEY:?VOYAGE_API_KEY is not set: the judge retrieval embeds with Voyage and has no fallback}"
elif [[ -z "${VOYAGE_API_KEY:-}" && -z "$EMBED_BACKEND" ]]; then
  EMBED_BACKEND=hash
  echo "WARNING: no VOYAGE_API_KEY; merging elements with offline hashing." >&2
  echo "         Fine for reading ideas, NOT for reported numbers." >&2
fi
# Any of these left over from another run silently redirects calls elsewhere.
unset OPENAI_BASE_URL VOYAGE_BASE_URL TOGETHER_API_KEY JUDGE_BASE_URL JUDGE_MODEL

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "corpus not found at $INPUT_DIR" >&2
  echo "  idea-forecast-bench fetch --from-hf --out-dir $INPUT_DIR" >&2
  exit 1
fi

LOG_DIR="$OUTPUT_DIR/logs"
GEN_DIR="$OUTPUT_DIR/backtest"
JUDGED_DIR="$OUTPUT_DIR/judged"
mkdir -p "$LOG_DIR" "$GEN_DIR" "$JUDGED_DIR" "$ELEMENT_CACHE"
export IDEA_FORECAST_USAGE_LOG="${IDEA_FORECAST_USAGE_LOG:-$OUTPUT_DIR/usage.jsonl}"

topic_args=()
[[ -n "$TOPICS" ]] && topic_args+=(--topics "$TOPICS")

echo "== 1/3 extract elements with $EXTRACT_MODEL (thinking off) =="
IDEA_FORECAST_USAGE_STAGE=extract DASHSCOPE_THINKING=off \
  python examples/benchmark/extract_elements.py \
    --input-dir "$INPUT_DIR" --start-month "$START_MONTH" --end-month "$END_MONTH" \
    --cache-dir "$ELEMENT_CACHE" --model-name "$EXTRACT_MODEL" \
    --workers "$EXTRACT_WORKERS" --rpm "$EXTRACT_RPM" --embed --dump-clusters 30 \
    ${EMBED_BACKEND:+--embed-backend "$EMBED_BACKEND"} \
    "${topic_args[@]}" 2>&1 | tee "$LOG_DIR/extract.log"

echo "== 2/3 generate with $REALIZE_MODEL: $ARMS (thinking $REALIZE_THINKING) =="
for arm in $ARMS; do
  echo "  -- $arm"
  IDEA_FORECAST_USAGE_STAGE="realize:$arm" DASHSCOPE_THINKING="$REALIZE_THINKING" \
    python examples/benchmark/benchmark.py \
      --strategy "$arm" --model-name "$REALIZE_MODEL" --element-cache "$ELEMENT_CACHE" \
      --input-dir "$INPUT_DIR" --start-month "$START_MONTH" --end-month "$END_MONTH" \
      --min-cutoff-month "$MIN_CUTOFF_MONTH" --skip-matching \
      --workers "$GEN_WORKERS" --output "$GEN_DIR/$arm.json" \
      "${topic_args[@]}" 2>&1 | tee "$LOG_DIR/$arm.log"
done

python - "$GEN_DIR" <<'PY'
import glob, json, sys
print("\n== generated ideas ==")
for path in sorted(glob.glob(f"{sys.argv[1]}/*.json")):
    d = json.loads(open(path).read())
    windows = [w for t in d["topic_results"].values()
               for w in (t.get("backtest") or {}).get("windows", [])]
    preds = [p for w in windows for p in w["predictions"]]
    short = sum(1 for w in windows if len(w["predictions"]) < 5)
    fb = sum(1 for p in preds if p["metadata"].get("fallback"))
    print(f"{d['strategy']:28s} windows={len(windows):3d} ideas={len(preds):4d} "
          f"short={short} template_fallback={fb}")
PY

if [[ "$JUDGE" != "1" ]]; then
  cat <<EOF

Stopped before judging (JUDGE=0). Read a few ideas first:

  python -c "import json,glob;d=json.load(open(glob.glob('$GEN_DIR/combinatorial.json')[0]));\\
w=[w for t in d['topic_results'].values() for w in (t.get('backtest') or {})['windows']][0];\\
[print('-',p['title'],'|',[e['label'] for e in p['metadata']['elements']],'|',p['metadata']['move']) for p in w['predictions']]"

Then re-run with JUDGE=1 to score them.
EOF
  exit 0
fi

echo "== 3/3 judge (thinking off) + table =="
for arm in $ARMS; do
  artifact="$GEN_DIR/$arm.json"
  topics="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("topics_shard") or ",".join(d.get("topic_results",{})))' "$artifact")"
  echo "  -- judging $arm"
  JUDGE_API_KEY="$DASHSCOPE_API_KEY" \
    python examples/benchmark/judge_eval.py \
      --input-json "$artifact" --papers-dir "$INPUT_DIR" \
      --output "$JUDGED_DIR/$arm.judged.json" \
      --judge-model "$JUDGE_MODEL_NAME" --judge-base-url "$JUDGE_BASE_URL_VALUE" \
      --topics "$topics" --workers "$JUDGE_WORKERS" \
      --topic-workers "$JUDGE_TOPIC_WORKERS" 2>&1 | tee "$LOG_DIR/$arm.judge.log"
done

n_topics="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d.get("topic_results",{})))' "$GEN_DIR/$(echo "$ARMS" | awk '{print $1}').json")"
n_windows="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sum(len((t.get("backtest") or {}).get("windows",[])) for t in d["topic_results"].values()))' "$GEN_DIR/$(echo "$ARMS" | awk '{print $1}').json")"
SOURCES="$LABEL=$JUDGED_DIR/*.judged.json" EXPECTED_TOPICS="$n_topics" EXPECTED_WINDOWS="$n_windows" \
  bash scripts/benchmark/main_table.sh | tee "$OUTPUT_DIR/table.txt"

python - "$IDEA_FORECAST_USAGE_LOG" <<'PY'
import collections, json, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except OSError:
    sys.exit(0)
agg = collections.defaultdict(lambda: [0, 0, 0, 0])
for r in rows:
    a = agg[r.get("stage") or "?"]
    a[0] += 1
    a[1] += r.get("prompt_tokens") or 0
    a[2] += r.get("completion_tokens") or 0
    a[3] += r.get("reasoning_tokens") or 0
print(f"\n== token usage ==\n{'stage':22s}{'calls':>8}{'input':>14}{'output':>12}{'reasoning':>12}")
for stage, (n, i, o, t) in sorted(agg.items()):
    print(f"{stage:22s}{n:>8}{i:>14,}{o:>12,}{t:>12,}")
PY
echo "done: $OUTPUT_DIR/table.txt"
