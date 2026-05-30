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
#  Reward (GRPO Phase 3):
#    REWARD_MODE=foresight (default) — the gated, future-grounded reward used
#      for the reported results. Needs a prebuilt artifact dir (per-cutoff
#      future/history indices + validated rubrics). Build it first:
#        1. provide the paper corpus at $PAPERS
#        2. run the hindsight pipeline to produce data/topic_hindsight/dz.jsonl
#        3. python build_indices.py --papers-dir "$PAPERS" \
#               --dz data/topic_hindsight/dz.jsonl --art output/foresight_artifacts
#        4. generate validated rubrics (see forecaster/foresight/README.md)
#    REWARD_MODE=legacy — fixed-weight composite reward, NO artifacts required.
#      Use it to run the whole pipeline end to end on a fresh clone, or while
#      the foresight artifacts are still being built:
#        REWARD_MODE=legacy bash scripts/run_train_and_eval.sh
#
#  Environment setup (run once per model family):
#    bash scripts/setup_rl_env.sh qwen3      # vLLM + transformers <5
#    bash scripts/setup_rl_env.sh qwen3.5    # transformers >=5, no vLLM
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-qwen3-1.7b}"
HINDSIGHT="${HINDSIGHT:-output/hindsight_samples.jsonl}"
PAPERS="${PAPERS:-data/csml/raw_markdown}"
OUT="output/forecaster_${MODEL}"
THRESHOLD="${THRESHOLD:-0.80}"
START_MONTH="${START_MONTH:-2023-01}"
END_MONTH="${END_MONTH:-2025-03}"
REWARD_MODE="${REWARD_MODE:-foresight}"
FORESIGHT_ARTIFACT_DIR="${FORESIGHT_ARTIFACT_DIR:-output/foresight_artifacts}"

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
echo " Full Pipeline (prior SFT -> realization GRPO -> eval)"
echo "  Model:  ${MODEL} (${BASE_MODEL_ID})"
echo "  Family: ${MODEL_FAMILY} | ${ENV_DETAIL}"
echo "  Output: ${OUT}"
echo "  Dates:  ${START_MONTH} ~ ${END_MONTH}"
echo "=============================================="

# ---- Phase 2: Prior SFT ----
PRIOR_CKPT="${OUT}/prior_sft/final_checkpoint"

if [ -d "$PRIOR_CKPT" ] && [ -f "${OUT}/prior_sft/train_result.json" ]; then
  echo ""; echo "===== Phase 2: Prior SFT — SKIPPED (checkpoint exists) ====="
else
  echo ""; echo "===== Phase 2: Prior SFT ====="
  python3 examples/forecaster/run_prior_sft.py \
    --hindsight "$HINDSIGHT" \
    --output-dir "${OUT}/prior_sft" \
    --model "$MODEL"
fi

PRIOR_CKPT=$(python3 -c "import json; print(json.load(open('${OUT}/prior_sft/train_result.json'))['checkpoint_path'])")

# ---- Phase 3: Realization GRPO ----
GRPO_DIR="${OUT}/realization_grpo"
GRPO_MANIFEST="${GRPO_DIR}/grpo/policy_manifest.json"

# Pick the trainer config by reward mode. The foresight reward (default, paper
# results) needs prebuilt artifacts; the legacy composite reward needs none.
case "$REWARD_MODE" in
  foresight)
    GRPO_CONFIG="grpo_train.yaml"
    if [ ! -d "${FORESIGHT_ARTIFACT_DIR}/indices" ] || [ ! -d "${FORESIGHT_ARTIFACT_DIR}/rubrics" ]; then
      echo "ERROR: REWARD_MODE=foresight but the artifact dir is missing or incomplete:" >&2
      echo "         ${FORESIGHT_ARTIFACT_DIR}/{indices,rubrics}" >&2
      echo "       The gated foresight reward needs per-cutoff indices + validated rubrics." >&2
      echo "       Build them first:" >&2
      echo "         python build_indices.py --papers-dir \"${PAPERS}\" \\" >&2
      echo "             --dz data/topic_hindsight/dz.jsonl --art \"${FORESIGHT_ARTIFACT_DIR}\"" >&2
      echo "         (then generate validated rubrics — see forecaster/foresight/README.md)" >&2
      echo "       Or run the whole pipeline with the no-artifacts reward:" >&2
      echo "         REWARD_MODE=legacy bash scripts/run_train_and_eval.sh" >&2
      exit 1
    fi
    ;;
  legacy)
    GRPO_CONFIG="grpo_train_legacy.yaml"
    ;;
  *)
    echo "ERROR: REWARD_MODE must be 'foresight' or 'legacy' (got: ${REWARD_MODE})" >&2
    exit 1
    ;;
esac

if [ -f "$GRPO_MANIFEST" ]; then
  echo ""; echo "===== Phase 3: Realization GRPO — SKIPPED (manifest exists) ====="
else
  echo ""; echo "===== Phase 3: Realization GRPO (reward=${REWARD_MODE}, config=${GRPO_CONFIG}) ====="
  python3 examples/forecaster/run_policy_rl_training.py \
    --input-dir "$PAPERS" \
    --output-dir "$GRPO_DIR" \
    --model-preset "$MODEL" \
    --trainer grpo \
    --trainer-config "$GRPO_CONFIG" \
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

# ---- Phase 4: Evaluation via Benchmark Backtest ----
# Uses run_domain_backtest.py (same as all other strategies) for fair comparison.
# Eval range: 2024-10 to 2025-03 (3 windows with horizon=3).

EVAL_START="${EVAL_START:-2024-10}"
EVAL_END="${EVAL_END:-2025-03}"

TRAINED_EVAL="${OUT}/eval_trained.json"
if [ -f "$TRAINED_EVAL" ]; then
  echo ""; echo "===== Phase 4: Eval (trained) — SKIPPED (exists) ====="
else
  echo ""; echo "===== Phase 4: Eval (trained forecaster) ====="
  python3 examples/benchmark/run_domain_backtest.py \
    --strategy forecaster \
    --model-name "$BASE_MODEL_ID" \
    --prior-checkpoint "$PRIOR_CKPT" \
    --realization-checkpoint "$REAL_CKPT" \
    --input-dir "$PAPERS" \
    --start-month "$EVAL_START" --end-month "$EVAL_END" \
    --top-k 5 --horizon-months 3 \
    --similarity-engine heuristic --workers 1 \
    --output "$TRAINED_EVAL"
fi

# ---- Voyage re-eval (paper-comparable scores) ----
VOYAGE_EVAL="${OUT}/eval_voyage.json"
if [ -n "${VOYAGE_API_KEY:-}" ] && [ -f "$TRAINED_EVAL" ]; then
  if [ -f "$VOYAGE_EVAL" ]; then
    echo ""; echo "===== Voyage Re-eval — SKIPPED (exists) ====="
  else
    echo ""; echo "===== Voyage Re-eval (threshold=0.80) ====="
    VOYAGE_API_KEY="$VOYAGE_API_KEY" python3 examples/benchmark/reeval_voyage.py \
      --input-json "$TRAINED_EVAL" \
      --papers-dir "$PAPERS" \
      --output "$VOYAGE_EVAL" \
      --threshold 0.80
  fi
fi

echo ""
echo "===== Results ====="
if [ -f "$VOYAGE_EVAL" ]; then
  python3 -c "
import json
data = json.load(open('${VOYAGE_EVAL}'))
s = data.get('aggregate_summary', {})
print(f'  hit@5={s.get(\"avg_hit_at_k\", 0):.4f}  P@5={s.get(\"avg_precision_at_k\", 0):.4f}  '
      f'R@5={s.get(\"avg_recall_at_k\", 0):.4f}  MRR={s.get(\"avg_mrr\", 0):.4f}  '
      f'Nov={s.get(\"avg_novelty\", 0):.4f}  Div={s.get(\"avg_diversity\", 0):.4f}')
"
elif [ -f "$TRAINED_EVAL" ]; then
  echo "  (Heuristic only — set VOYAGE_API_KEY for paper-comparable scores)"
fi

echo ""; echo "===== Done ====="
