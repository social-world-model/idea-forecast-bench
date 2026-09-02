#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-qwen2.5-7b-instruct}"
HINDSIGHT="${HINDSIGHT:-data/topic_hindsight/hindsight_samples.jsonl}"
PAPERS="${PAPERS:-data/csml/raw_markdown}"
OUT="output/forecaster_${MODEL}"
THRESHOLD="${THRESHOLD:-0.80}"
START_MONTH="${START_MONTH:-2023-01}"
END_MONTH="${END_MONTH:-2025-03}"
REWARD_MODE="${REWARD_MODE:-foresight}"
FORESIGHT_ARTIFACT_DIR="${FORESIGHT_ARTIFACT_DIR:-output/foresight_artifacts}"

BASE_MODEL_ID=$(python3 -c "
from forecaster.realization.model_zoo import resolve_small_model
print(resolve_small_model('${MODEL}').model_id)")

echo "=============================================="
echo " Full Pipeline (prior SFT -> realization GRPO -> eval)"
echo "  Model:  ${MODEL} (${BASE_MODEL_ID})"
echo "  Output: ${OUT}"
echo "  Dates:  ${START_MONTH} ~ ${END_MONTH}"
echo "=============================================="

# ---- Prior SFT ----
PRIOR_CKPT="${OUT}/prior_sft/final_checkpoint"

if [ -d "$PRIOR_CKPT" ] && [ -f "${OUT}/prior_sft/train_result.json" ]; then
  echo ""; echo "===== Prior SFT — SKIPPED (checkpoint exists) ====="
else
  echo ""; echo "===== Prior SFT ====="
  python3 examples/forecaster/run_prior_sft.py \
    --hindsight "$HINDSIGHT" \
    --output-dir "${OUT}/prior_sft" \
    --model "$MODEL"
fi

PRIOR_CKPT=$(python3 -c "import json; print(json.load(open('${OUT}/prior_sft/train_result.json'))['checkpoint_path'])")

# ---- Realization GRPO ----
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
      echo "         python examples/forecaster/build_indices.py --papers-dir \"${PAPERS}\" \\" >&2
      echo "             --hindsight \"${HINDSIGHT}\" --art \"${FORESIGHT_ARTIFACT_DIR}\"" >&2
      echo "         python examples/forecaster/build_rubrics.py --mode live \\" >&2
      echo "             --rubrics-dir \"${FORESIGHT_ARTIFACT_DIR}/rubrics\"" >&2
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
  echo ""; echo "===== Realization GRPO — SKIPPED (manifest exists) ====="
else
  echo ""; echo "===== Realization GRPO (reward=${REWARD_MODE}, config=${GRPO_CONFIG}) ====="
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

# ---- Generate: benchmark backtest with the trained forecaster ----
# Uses run_domain_backtest.py (same as all other strategies) for fair comparison,
# then scores the predictions with the retrieve-then-judge protocol, which is
# where the reported numbers come from. Generation skips the embedding match
# (--skip-matching) because judge-eval re-embeds and re-retrieves itself.
# Eval range: 2024-10 to 2025-03 (3 windows with horizon=3).

EVAL_START="${EVAL_START:-2024-10}"
EVAL_END="${EVAL_END:-2025-03}"

TRAINED_EVAL="${OUT}/eval_trained.json"
if [ -f "$TRAINED_EVAL" ]; then
  echo ""; echo "===== Generate — SKIPPED (exists) ====="
else
  echo ""; echo "===== Generate (trained forecaster) ====="
  python3 examples/benchmark/run_domain_backtest.py \
    --strategy forecaster \
    --model-name "$BASE_MODEL_ID" \
    --prior-checkpoint "$PRIOR_CKPT" \
    --realization-checkpoint "$REAL_CKPT" \
    --input-dir "$PAPERS" \
    --start-month "$EVAL_START" --end-month "$EVAL_END" \
    --top-k 5 --horizon-months 3 \
    --skip-matching --workers 1 \
    --output "$TRAINED_EVAL"
fi

# ---- Judge: retrieve-then-judge scoring ----
# Needs OPENAI_API_KEY (gpt-4.1-mini judge by default) and VOYAGE_API_KEY
# (retrieval embeddings). The judge is chosen by flag; none of the *_BASE_URL
# variables may be left exported, or scoring is silently redirected.
unset JUDGE_BASE_URL JUDGE_MODEL JUDGE_API_KEY OPENAI_BASE_URL VOYAGE_BASE_URL
JUDGED="${OUT}/eval_trained.judged.json"
if [ -z "${OPENAI_API_KEY:-}" ] || [ -z "${VOYAGE_API_KEY:-}" ]; then
  echo ""; echo "===== Judge — SKIPPED (set OPENAI_API_KEY and VOYAGE_API_KEY) ====="
elif [ -f "$JUDGED" ]; then
  echo ""; echo "===== Judge — SKIPPED (exists) ====="
else
  echo ""; echo "===== Judge (retrieve-then-judge) ====="
  python3 examples/benchmark/llm_judge_eval.py \
    --input-json "$TRAINED_EVAL" \
    --papers-dir "$PAPERS" \
    --output "$JUDGED" \
    --state-file "${JUDGED%.json}.state.json"
fi

echo ""
echo "===== Results ====="
echo "  predictions: $TRAINED_EVAL"
if [ -f "$JUDGED" ]; then
  echo "  judged:      $JUDGED"
  echo "  table:       idea-forecast-bench main-table --source \"${MODEL}=${JUDGED}\" \\"
  echo "                 --expected-topics <n> --expected-windows <n>   # counts for this eval range"
else
  echo "  (metrics in the predictions file are NaN by design; run the judge for scores)"
fi
