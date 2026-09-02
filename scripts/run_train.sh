#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# MDF end to end: labels -> reward artifacts -> prior SFT -> GRPO -> generate -> judge.
# Stages whose output already exists are skipped, so the script can be re-run.
export INPUT_DIR="${INPUT_DIR:-data/csml/raw_markdown}"
export MODEL="${MODEL:-qwen2.5-7b-instruct}"
export HINDSIGHT="${HINDSIGHT:-data/topic_hindsight/hindsight_samples.jsonl}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-output/foresight_artifacts}"
OUT="${OUT:-output/mdf}"
REWARD_MODE="${REWARD_MODE:-foresight}"   # or legacy: no artifacts needed

PRIOR_CKPT="$OUT/prior_sft/final_checkpoint"
GRPO_DIR="$OUT/realization_grpo"
REAL_CKPT="$GRPO_DIR/grpo/checkpoints/final_checkpoint"
GEN="$OUT/eval/forecaster.json"

case "$REWARD_MODE" in
  foresight) TRAINER_CONFIG="grpo_train.yaml" ;;
  legacy)    TRAINER_CONFIG="grpo_train_legacy.yaml" ;;
  *) echo "REWARD_MODE must be foresight or legacy (got $REWARD_MODE)" >&2; exit 1 ;;
esac

stage() { echo ""; echo "===== $1 ====="; }

stage "hindsight labels"
if [[ -s "$HINDSIGHT" ]]; then echo "exists: $HINDSIGHT"; else
  OUTPUT_DIR="$(dirname "$HINDSIGHT")" bash scripts/hindsight.sh
fi

if [[ "$REWARD_MODE" == "foresight" ]]; then
  stage "foresight indices"
  if [[ -d "$ARTIFACT_DIR/indices" ]]; then echo "exists: $ARTIFACT_DIR/indices"; else
    bash scripts/build_indices.sh
  fi
  stage "foresight rubrics"
  if [[ -d "$ARTIFACT_DIR/rubrics" ]]; then echo "exists: $ARTIFACT_DIR/rubrics"; else
    bash scripts/build_rubrics.sh
  fi
fi

stage "prior SFT"
if [[ -d "$PRIOR_CKPT" ]]; then echo "exists: $PRIOR_CKPT"; else
  OUTPUT_DIR="$OUT/prior_sft" bash scripts/train_prior.sh
fi

stage "realization GRPO ($REWARD_MODE reward)"
if [[ -d "$REAL_CKPT" ]]; then echo "exists: $REAL_CKPT"; else
  INIT_POLICY="$PRIOR_CKPT" TRAINER_CONFIG="$TRAINER_CONFIG" OUTPUT_DIR="$GRPO_DIR" bash scripts/train.sh
fi

stage "generate"
BASE_MODEL_ID="$(python -c "from forecaster.realization.model_zoo import resolve_small_model; print(resolve_small_model('$MODEL').model_id)")"
if [[ -s "$GEN" ]]; then echo "exists: $GEN"; else
  STRATEGY=forecaster MODEL="$BASE_MODEL_ID" OUTPUT="$GEN" \
    PRIOR_CHECKPOINT="$PRIOR_CKPT" REALIZATION_CHECKPOINT="$REAL_CKPT" WORKERS=1 \
    bash scripts/benchmark.sh
fi

stage "judge"
if [[ -z "${OPENAI_API_KEY:-}" || -z "${VOYAGE_API_KEY:-}" ]]; then
  echo "skipped: set OPENAI_API_KEY and VOYAGE_API_KEY to score $GEN"
else
  INPUT_JSON="$GEN" bash scripts/judge_eval.sh
  echo "judged: ${GEN%.json}.judged.json"
fi
