#!/usr/bin/env bash
# ==========================================================================
#  Full paper pipeline: Prior SFT → Realization GRPO → Joint Inference → Eval
#  Skips SFT/GRPO if checkpoints already exist.
#
#  Works with both Qwen3 (vLLM-enabled) and Qwen3.5 (transformers 5.x) envs.
#  Auto-detects model family and validates the current conda environment.
#
#  Usage:
#    MODEL=qwen3-1.7b bash scripts/run_train_and_eval.sh
#    MODEL=qwen3.5-2b bash scripts/run_train_and_eval.sh
#
#  Environment setup (run once per model family):
#    bash scripts/setup_rl_env.sh qwen3      # vLLM + transformers <5
#    bash scripts/setup_rl_env.sh qwen3.5    # transformers >=5, no vLLM
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-qwen3-1.7b}"
HINDSIGHT="${HINDSIGHT:-output/hindsight_samples.jsonl}"
PAPERS="${PAPERS:-data/csml_v2/raw_markdown}"
OUT="output/forecaster_${MODEL}"
THRESHOLD="${THRESHOLD:-0.80}"
START_MONTH="${START_MONTH:-2023-01}"
END_MONTH="${END_MONTH:-2025-03}"

# ---- Auto-detect model family and validate environment ----
ENV_INFO=$(python3 -c "
from forecaster.realization.model_zoo import resolve_small_model
import transformers
spec = resolve_small_model('${MODEL}')
tv = int(transformers.__version__.split('.')[0])
family = spec.family
vllm_ok = False
try:
    import vllm
    from trl.import_utils import is_vllm_available
    vllm_ok = is_vllm_available()
except Exception:
    pass

# Validate env compatibility
if family == 'qwen3.5' and tv < 5:
    print(f'ERROR: {spec.alias} requires transformers >=5.x (found {transformers.__version__})')
    print(f'  Fix: conda activate <qwen35-env> or bash scripts/setup_rl_env.sh qwen3.5')
    import sys; sys.exit(1)
if family == 'qwen3' and tv >= 5:
    print(f'WARNING: {spec.alias} works best with transformers <5 + vLLM (found {transformers.__version__})')
    print(f'  Hint: conda activate <qwen3-env> or bash scripts/setup_rl_env.sh qwen3')

print(f'{spec.model_id}|{family}|tv={transformers.__version__}|vllm={vllm_ok}')
")

# Check if validation failed
if echo "$ENV_INFO" | grep -q "^ERROR:"; then
  echo "$ENV_INFO" >&2; exit 1
fi
if echo "$ENV_INFO" | grep -q "^WARNING:"; then
  echo "$ENV_INFO" >&2
fi

BASE_MODEL_ID=$(echo "$ENV_INFO" | tail -1 | cut -d'|' -f1)
MODEL_FAMILY=$(echo "$ENV_INFO" | tail -1 | cut -d'|' -f2)
ENV_DETAIL=$(echo "$ENV_INFO" | tail -1 | cut -d'|' -f3-)

echo "=============================================="
echo " Full Pipeline (Paper §3.2–3.4)"
echo "  Model:  ${MODEL} (${BASE_MODEL_ID})"
echo "  Family: ${MODEL_FAMILY} | ${ENV_DETAIL}"
echo "  Output: ${OUT}"
echo "  Dates:  ${START_MONTH} ~ ${END_MONTH}"
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
GRPO_DIR="${OUT}/realization_grpo"
GRPO_MANIFEST="${GRPO_DIR}/grpo/policy_manifest.json"

if [ -f "$GRPO_MANIFEST" ]; then
  echo ""; echo "===== Phase 3: Realization GRPO — SKIPPED (manifest exists) ====="
else
  echo ""; echo "===== Phase 3: Realization GRPO ====="
  python3 examples/run_policy_rl_training.py \
    --input-dir "$PAPERS" \
    --output-dir "$GRPO_DIR" \
    --model-preset "$MODEL" \
    --trainer grpo \
    --hindsight "$HINDSIGHT" \
    --start-month "$START_MONTH" \
    --end-month "$END_MONTH" \
    --skip-alignment-check
fi

# Resolve the actual checkpoint path (supports both TRL and veRL layouts)
REAL_CKPT=$(python3 -c "
from pathlib import Path
import glob

grpo_dir = Path('${GRPO_DIR}/grpo')

# TRL: checkpoints/final_checkpoint/
trl_ckpt = grpo_dir / 'checkpoints' / 'final_checkpoint'
if trl_ckpt.exists():
    print(str(trl_ckpt)); exit()

# veRL: artifacts/default/global_step_*/actor/
pattern = str(grpo_dir / 'artifacts' / 'default' / 'global_step_*' / 'actor')
candidates = sorted(glob.glob(pattern))
if candidates:
    print(candidates[-1]); exit()

# Fallback: find any adapter_config.json
for p in grpo_dir.rglob('adapter_config.json'):
    print(str(p.parent)); exit()
print(str(grpo_dir))
")
echo "Realization checkpoint: ${REAL_CKPT}"

# ---- Phase 4: Joint Inference (§3.4, Algorithm 1) ----
echo ""; echo "===== Phase 4: Joint Inference ====="
python3 examples/run_joint_inference.py \
  --prior-checkpoint "$PRIOR_CKPT" \
  --realization-checkpoint "$REAL_CKPT" \
  --hindsight "$HINDSIGHT" \
  --papers-dir "$PAPERS" \
  --output-dir "${OUT}/inference" \
  --start-month "$START_MONTH" \
  --end-month "$END_MONTH"

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
