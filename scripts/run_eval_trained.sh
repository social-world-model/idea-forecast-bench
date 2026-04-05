#!/usr/bin/env bash
# ==========================================================================
#  Evaluate trained forecaster models using the benchmark backtest.
#  Skips training — expects checkpoints to already exist.
#
#  Usage:
#    MODEL=qwen3.5-0.8b PAPERS=/path/to/papers bash scripts/run_eval_trained.sh
#
#  Evaluate multiple models:
#    for m in qwen3.5-0.8b qwen3.5-2b; do
#      MODEL=$m bash scripts/run_eval_trained.sh
#    done
#
#  Settings match other baselines (memory_prompting, keyword_trend, etc.)
#  for fair comparison via run_domain_backtest.py.
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-qwen3.5-0.8b}"
PAPERS="${PAPERS:-data/csml_v2/raw_markdown}"
OUT="output/forecaster_${MODEL}"
EVAL_START="${EVAL_START:-2024-10}"
EVAL_END="${EVAL_END:-2025-03}"

# Resolve model ID
BASE_MODEL_ID=$(python3 -c "
from forecaster.realization.model_zoo import resolve_small_model
print(resolve_small_model('${MODEL}').model_id)
")

# Locate checkpoints
PRIOR_CKPT="${OUT}/prior_sft/final_checkpoint"
GRPO_CKPT=$(python3 -c "
from pathlib import Path
g = Path('${OUT}/realization_grpo/grpo')
t = g / 'checkpoints' / 'final_checkpoint'
if t.exists(): print(t); exit()
for p in g.rglob('adapter_config.json'): print(p.parent); exit()
print(g)
")

# Validate checkpoints exist
if [ ! -f "${PRIOR_CKPT}/adapter_config.json" ]; then
  echo "ERROR: Prior checkpoint not found: ${PRIOR_CKPT}" >&2; exit 1
fi
if [ ! -f "${GRPO_CKPT}/adapter_config.json" ]; then
  echo "ERROR: GRPO checkpoint not found: ${GRPO_CKPT}" >&2; exit 1
fi

echo "=============================================="
echo " Eval: ${MODEL} (${BASE_MODEL_ID})"
echo "  Prior:       ${PRIOR_CKPT}"
echo "  Realization: ${GRPO_CKPT}"
echo "  Range:       ${EVAL_START} ~ ${EVAL_END}"
echo "=============================================="

TRAINED_EVAL="${OUT}/eval_trained.json"
if [ -f "$TRAINED_EVAL" ]; then
  echo ""; echo "Eval already exists: $TRAINED_EVAL"
  echo "Delete it to re-run: rm $TRAINED_EVAL"
else
  python3 examples/run_domain_backtest.py \
    --strategy forecaster \
    --model-name "$BASE_MODEL_ID" \
    --prior-checkpoint "$PRIOR_CKPT" \
    --realization-checkpoint "$GRPO_CKPT" \
    --input-dir "$PAPERS" \
    --start-month "$EVAL_START" --end-month "$EVAL_END" \
    --top-k 5 --horizon-months 3 \
    --similarity-engine heuristic --workers 1 \
    --output "$TRAINED_EVAL"
fi

echo ""
echo "===== Results: ${MODEL} ====="
python3 -c "
import json
data = json.load(open('${TRAINED_EVAL}'))
s = data.get('aggregate_summary', {})
print(f'  hit@k={s.get(\"avg_hit_at_k\", 0):.4f}  mrr={s.get(\"avg_mrr\", 0):.4f}  '
      f'novelty={s.get(\"avg_novelty\", 0):.4f}  diversity={s.get(\"avg_diversity\", 0):.4f}')
"
