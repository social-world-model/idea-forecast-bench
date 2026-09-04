#!/usr/bin/env bash
# Forward vs backward judge on the same saved predictions, for the v2
# vocabulary arm (generated here) and the v1 pilot arm (reused from
# output/pilot/backtest/combinatorial.json). Same judge, same windows.
#
#   DASHSCOPE_API_KEY=... VOYAGE_API_KEY=... bash scripts/run_forward_backward.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DASHSCOPE_API_KEY:?}"; : "${VOYAGE_API_KEY:?}"
unset OPENAI_BASE_URL VOYAGE_BASE_URL TOGETHER_API_KEY JUDGE_BASE_URL JUDGE_MODEL

INPUT_DIR="${INPUT_DIR:-data/hf_full/raw_markdown}"
OUT="${OUT:-output/fb}"
TOPICS="${TOPICS:-llm_long_context,quantization,moe}"
START_MONTH="${START_MONTH:-2024-04}"; END_MONTH="${END_MONTH:-2025-09}"
MIN_CUTOFF_MONTH="${MIN_CUTOFF_MONTH:-2024-07}"
REALIZE_MODEL="${REALIZE_MODEL:-vanchin/deepseek-v4-pro-0813}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-deepseek-v4-flash}"
JUDGE_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
JUDGE_WORKERS="${JUDGE_WORKERS:-24}"; JUDGE_TOPIC_WORKERS="${JUDGE_TOPIC_WORKERS:-3}"
VOCAB_STORE="${VOCAB_STORE:-output/vocab/cache/b493410c0021}"
V1_ARTIFACT="${V1_ARTIFACT:-output/pilot/backtest/combinatorial.json}"
STAGES="${STAGES:-gen judge table}"
mkdir -p "$OUT/backtest" "$OUT/judged" "$OUT/logs"
export IDEA_FORECAST_USAGE_LOG="$OUT/usage.jsonl"

if [[ " $STAGES " == *" gen "* ]]; then
  echo "== generate v2 arm"
  .venv/bin/python examples/benchmark/benchmark.py \
    --strategy vocab_combinatorial --model-name "$REALIZE_MODEL" \
    --vocab-store "$VOCAB_STORE" --vocab-config config/vocab.yaml \
    --input-dir "$INPUT_DIR" --start-month "$START_MONTH" --end-month "$END_MONTH" \
    --min-cutoff-month "$MIN_CUTOFF_MONTH" --skip-matching --workers 4 \
    --topics "$TOPICS" --output "$OUT/backtest/vocab_combinatorial.json" \
    2>&1 | tee "$OUT/logs/gen.log"
fi

judge() {  # artifact tag direction extra...
  local artifact="$1" tag="$2" direction="$3"; shift 3
  echo "== judge $tag $direction $*"
  JUDGE_API_KEY="$DASHSCOPE_API_KEY" .venv/bin/python examples/benchmark/judge_eval.py \
    --input-json "$artifact" --papers-dir "$INPUT_DIR" \
    --output "$OUT/judged/$tag.$direction.judged.json" \
    --judge-model "$JUDGE_MODEL_NAME" --judge-base-url "$JUDGE_BASE" \
    --topics "$TOPICS" --workers "$JUDGE_WORKERS" --topic-workers "$JUDGE_TOPIC_WORKERS" \
    --direction "$direction" "$@" 2>&1 | tee "$OUT/logs/$tag.$direction.judge.log"
}

if [[ " $STAGES " == *" judge "* ]]; then
  judge "$OUT/backtest/vocab_combinatorial.json" v2 forward
  judge "$OUT/backtest/vocab_combinatorial.json" v2 backward --exclude-evidence
  judge "$V1_ARTIFACT" v1 forward
  judge "$V1_ARTIFACT" v1 backward --exclude-evidence
fi

if [[ " $STAGES " == *" table "* ]]; then
  .venv/bin/python - "$OUT/judged" <<'PY'
import json, sys, glob, os, statistics as st, random
from scipy.stats import binomtest
d = sys.argv[1]
def load(path):
    j = json.load(open(path)); out = {}
    for tid, t in j["topic_results"].items():
        for w in (t.get("backtest") or t).get("windows", []):
            ev = w.get("evaluation") or {}
            out[(tid, w["cutoff_month"])] = (ev.get("precision_at_k"), ev.get("hit_at_k"))
    return out
rows = {}
for p in sorted(glob.glob(f"{d}/*.judged.json")):
    tag, direction = os.path.basename(p).split(".")[:2]
    rows.setdefault(tag, {})[direction] = load(p)
print(f"{'arm':6s} {'n':>3s} {'fwd P@5':>8s} {'bwd P@5':>8s} {'delta':>7s} {'p(sign)':>8s} {'95% CI':>16s} {'fwd Hit@5':>9s} {'bwd Hit@5':>9s}")
for tag, dirs in rows.items():
    if "forward" not in dirs or "backward" not in dirs: continue
    keys = sorted(set(dirs["forward"]) & set(dirs["backward"]))
    f = [dirs["forward"][k][0] for k in keys]; b = [dirs["backward"][k][0] for k in keys]
    fh = [dirs["forward"][k][1] for k in keys]; bh = [dirs["backward"][k][1] for k in keys]
    deltas = [x - y for x, y in zip(f, b)]
    nz = [x for x in deltas if abs(x) > 1e-9]
    p = binomtest(sum(1 for x in nz if x > 0), len(nz)).pvalue if nz else 1.0
    rng = random.Random(0); boots = sorted(st.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
    print(f"{tag:6s} {len(keys):3d} {st.mean(f):8.3f} {st.mean(b):8.3f} {st.mean(deltas):+7.3f} {p:8.3f} [{boots[50]:+.3f}, {boots[1949]:+.3f}] {st.mean(fh):9.3f} {st.mean(bh):9.3f}")
PY
fi
