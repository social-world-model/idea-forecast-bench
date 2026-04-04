#!/usr/bin/env bash
# Train Qwen3.5-2B prior, eval trained vs untrained with voyage-4-large.
#
# Usage:
#   conda activate live-idea-bench
#   VOYAGE_API_KEY=your-key bash scripts/run_prior_sft_and_eval.sh
#
# Override:
#   MODEL=qwen3.5-4b VOYAGE_API_KEY=... bash scripts/run_prior_sft_and_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-qwen3.5-2b}"
HINDSIGHT="${HINDSIGHT:-output/hindsight_samples.jsonl}"
PAPERS="${PAPERS:-data/csml_v2/raw_markdown}"
OUT="output/prior_sft_${MODEL}"
THRESHOLD="${THRESHOLD:-0.80}"

BASE_MODEL_ID=$(python3 -c "
from forecaster.realization.model_zoo import resolve_small_model
print(resolve_small_model('${MODEL}').model_id)
")

echo "Model: ${MODEL} (${BASE_MODEL_ID})"
echo "Output: ${OUT}"

# ---- 1. Train ----
echo ""; echo "===== 1/3 Train ====="
python3 examples/run_prior_sft.py \
  --hindsight "$HINDSIGHT" \
  --output-dir "${OUT}/trained" \
  --model "$MODEL"

CHECKPOINT=$(python3 -c "import json; print(json.load(open('${OUT}/trained/train_result.json'))['checkpoint_path'])")

# ---- 2. Eval trained ----
echo ""; echo "===== 2/3 Eval trained ====="
python3 examples/run_prior_eval.py \
  --model-path "$CHECKPOINT" \
  --hindsight "$HINDSIGHT" \
  --papers-dir "$PAPERS" \
  --output-dir "${OUT}/trained"

if [ -n "${VOYAGE_API_KEY:-}" ]; then
  python3 examples/reeval_voyage.py \
    --input-json "${OUT}/trained/predictions_for_eval.json" \
    --papers-dir "$PAPERS" \
    --output "${OUT}/trained/eval_voyage.json" \
    --threshold "$THRESHOLD"
fi

# ---- 3. Eval untrained baseline ----
echo ""; echo "===== 3/3 Eval untrained baseline ====="
python3 examples/run_prior_eval.py \
  --model-path "$BASE_MODEL_ID" \
  --hindsight "$HINDSIGHT" \
  --papers-dir "$PAPERS" \
  --output-dir "${OUT}/untrained"

if [ -n "${VOYAGE_API_KEY:-}" ]; then
  python3 examples/reeval_voyage.py \
    --input-json "${OUT}/untrained/predictions_for_eval.json" \
    --papers-dir "$PAPERS" \
    --output "${OUT}/untrained/eval_voyage.json" \
    --threshold "$THRESHOLD"
fi

# ---- Summary ----
echo ""; echo "===== Done ====="
if [ -f "${OUT}/trained/eval_voyage.json" ] && [ -f "${OUT}/untrained/eval_voyage.json" ]; then
  python3 -c "
import json
t = json.load(open('${OUT}/trained/eval_voyage.json')).get('aggregate_summary', {})
u = json.load(open('${OUT}/untrained/eval_voyage.json')).get('aggregate_summary', {})
print(f\"{'Metric':<25} {'Trained':>10} {'Untrained':>10} {'Delta':>10}\")
print('-' * 58)
for k in ('avg_hit_at_k', 'avg_recall_at_k', 'avg_precision_at_k', 'avg_mrr'):
    tv, uv = t.get(k, 0.0), u.get(k, 0.0)
    print(f'{k:<25} {tv:>10.4f} {uv:>10.4f} {\"+\" if tv-uv>=0 else \"\"}{tv-uv:>9.4f}')
"
fi
