#!/usr/bin/env bash
# ==========================================================================
#  Full paper pipeline: Prior SFT → Realization GRPO → Joint Inference → Eval
#  Skips SFT/GRPO if checkpoints already exist.
#
#  Usage:
#    conda activate live-idea-bench
#    VOYAGE_API_KEY=your-key bash scripts/run_prior_sft_and_eval.sh
#
#  Override:
#    MODEL=qwen3.5-4b bash scripts/run_prior_sft_and_eval.sh
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-qwen3.5-2b}"
HINDSIGHT="${HINDSIGHT:-output/hindsight_samples.jsonl}"
PAPERS="${PAPERS:-data/csml_v2/raw_markdown}"
OUT="output/forecaster_${MODEL}"
THRESHOLD="${THRESHOLD:-0.80}"

BASE_MODEL_ID=$(python3 -c "
from forecaster.realization.model_zoo import resolve_small_model
print(resolve_small_model('${MODEL}').model_id)
")

echo "=============================================="
echo " Full Pipeline (Paper §3.2–3.4)"
echo "  Model:  ${MODEL} (${BASE_MODEL_ID})"
echo "  Output: ${OUT}"
echo "=============================================="

# ---- Phase 2: Prior SFT (§3.2) ----
PRIOR_CKPT="${OUT}/prior_sft/final_checkpoint"

if [ -d "$PRIOR_CKPT" ] && [ -f "${OUT}/prior_sft/train_result.json" ]; then
  echo ""; echo "===== Phase 2: Prior SFT — SKIPPED (checkpoint exists) ====="
else
  echo ""; echo "===== Phase 2: Prior SFT ====="
  python3 examples/run_prior_sft.py \
    --hindsight "$HINDSIGHT" \
    --output-dir "${OUT}/prior_sft" \
    --model "$MODEL"
fi

PRIOR_CKPT=$(python3 -c "import json; print(json.load(open('${OUT}/prior_sft/train_result.json'))['checkpoint_path'])")

# ---- Phase 3: Realization GRPO (§3.3) ----
REAL_CKPT="${OUT}/realization_grpo/grpo"

if [ -d "$REAL_CKPT" ]; then
  echo ""; echo "===== Phase 3: Realization GRPO — SKIPPED (checkpoint exists) ====="
else
  echo ""; echo "===== Phase 3: Realization GRPO ====="
  python3 examples/run_policy_rl_training.py \
    --input-dir "$PAPERS" \
    --output-dir "${OUT}/realization_grpo" \
    --model-preset "$MODEL" \
    --trainer grpo \
    --skip-alignment-check
fi

# ---- Phase 4: Joint Inference (§3.4, Algorithm 1) ----
echo ""; echo "===== Phase 4: Joint Inference ====="
python3 examples/run_joint_inference.py \
  --prior-checkpoint "$PRIOR_CKPT" \
  --realization-checkpoint "$REAL_CKPT" \
  --hindsight "$HINDSIGHT" \
  --papers-dir "$PAPERS" \
  --output-dir "${OUT}/inference"

# ---- Evaluation with Voyage ----
echo ""; echo "===== Evaluation ====="
if [ -n "${VOYAGE_API_KEY:-}" ]; then
  python3 examples/reeval_voyage.py \
    --input-json "${OUT}/inference/predictions_for_eval.json" \
    --papers-dir "$PAPERS" \
    --output "${OUT}/inference/eval_voyage.json" \
    --threshold "$THRESHOLD"
  echo ""; echo "Results: ${OUT}/inference/eval_voyage.json"
else
  echo "VOYAGE_API_KEY not set. Run manually:"
  echo "  VOYAGE_API_KEY=... python3 examples/reeval_voyage.py \\"
  echo "    --input-json ${OUT}/inference/predictions_for_eval.json \\"
  echo "    --papers-dir $PAPERS --output ${OUT}/inference/eval_voyage.json"
fi

echo ""; echo "===== Done ====="
