#!/usr/bin/env bash
# ==========================================================================
#  Evaluate trained forecaster models using the benchmark backtest.
#  Skips training — expects checkpoints to already exist.
#
#  Fast path (SGLang):
#    Merges LoRA adapters into base model, wraps as VLM with original
#    vision weights, serves via SGLang (~1000 tok/s). Falls back to
#    HF generate (~50 tok/s) if SGLang is unavailable.
#
#  Usage:
#    MODEL=qwen3.5-0.8b bash scripts/run_eval_trained.sh
#
#  Evaluate multiple models:
#    for m in qwen3.5-0.8b qwen3.5-2b; do
#      MODEL=$m bash scripts/run_eval_trained.sh
#    done
#
#  Force HF generate (skip SGLang):
#    USE_SGLANG=0 MODEL=qwen3.5-0.8b bash scripts/run_eval_trained.sh
#
#  Settings match other baselines (memory_prompting, topic_trend, etc.)
#  for fair comparison via run_domain_backtest.py.
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-qwen3.5-0.8b}"
PAPERS="${PAPERS:-data/csml/raw_markdown}"
OUT="output/forecaster_${MODEL}"
EVAL_START="${EVAL_START:-2024-10}"
EVAL_END="${EVAL_END:-2025-03}"
USE_SGLANG="${USE_SGLANG:-1}"
SGLANG_MEM="${SGLANG_MEM:-0.8}"
PRIOR_PORT="${PRIOR_PORT:-30000}"
REAL_PORT="${REAL_PORT:-30001}"

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

# ---- SGLang fast path: merge LoRA + build VLM + serve ----
SGLANG_OK=0
cleanup_sglang() {
  if [ "$SGLANG_OK" -eq 1 ]; then
    echo "Stopping SGLang servers..."
    kill $(cat /tmp/sglang_prior_${MODEL}.pid 2>/dev/null) 2>/dev/null || true
    kill $(cat /tmp/sglang_real_${MODEL}.pid 2>/dev/null) 2>/dev/null || true
    rm -f /tmp/sglang_prior_${MODEL}.pid /tmp/sglang_real_${MODEL}.pid
  fi
}
trap cleanup_sglang EXIT

if [ "$USE_SGLANG" = "1" ] && python3 -c "import sglang" 2>/dev/null; then
  echo ""
  echo "===== Setting up SGLang fast serving ====="

  VLM_PRIOR="${OUT}/vlm_prior"
  VLM_REAL="${OUT}/vlm_realization"

  # Step 1: Merge LoRA adapters + build VLM models (if not already done)
  if [ ! -f "${VLM_PRIOR}/model.safetensors" ] || [ ! -f "${VLM_REAL}/model.safetensors" ]; then
    echo "Merging LoRA adapters and building VLM models..."
    python3 -c "
import torch, json, os, shutil
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download

base_id = '${BASE_MODEL_ID}'
prior_lora = '${PRIOR_CKPT}'
real_lora = '${GRPO_CKPT}'

# Download original VLM (for vision weights + config)
print('Downloading original VLM...')
orig_dir = snapshot_download(base_id, allow_patterns=['*.safetensors', '*.json', '*.txt', '*.jinja', 'merges.txt'])

# Load vision + MTP weights from original
st_files = [f for f in os.listdir(orig_dir) if f.endswith('.safetensors') and 'index' not in f]
vision_mtp = {}
for sf in st_files:
    from safetensors import safe_open
    f = safe_open(os.path.join(orig_dir, sf), framework='pt')
    for k in f.keys():
        if 'language_model' not in k:
            vision_mtp[k] = f.get_tensor(k)
print(f'Vision+MTP weights: {len(vision_mtp)} keys')

for lora_path, vlm_dir, label in [
    (prior_lora, '${VLM_PRIOR}', 'prior'),
    (real_lora, '${VLM_REAL}', 'realization'),
]:
    print(f'Merging {label} LoRA...')
    os.makedirs(vlm_dir, exist_ok=True)

    # Merge LoRA into base model
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map='cpu')
    model = PeftModel.from_pretrained(base, lora_path, torch_dtype=torch.bfloat16)
    merged = model.merge_and_unload()

    # Save merged weights temporarily
    merged.save_pretrained(vlm_dir)
    del base, model, merged

    # Load merged weights (have model.language_model.X prefix)
    trained = load_file(os.path.join(vlm_dir, 'model.safetensors'))

    # Ensure weights have model.language_model. prefix (VLM layout)
    first_key = list(trained.keys())[0]
    if not first_key.startswith('model.language_model.'):
        trained = {k.replace('model.', 'model.language_model.', 1): v for k, v in trained.items()}

    # Combine vision + trained language model
    combined = dict(vision_mtp)
    combined.update(trained)
    save_file(combined, os.path.join(vlm_dir, 'model.safetensors'))

    # Write index file
    index = {
        'metadata': {'total_size': sum(v.numel() * v.element_size() for v in combined.values())},
        'weight_map': {k: 'model.safetensors' for k in combined.keys()}
    }
    json.dump(index, open(os.path.join(vlm_dir, 'model.safetensors.index.json'), 'w'), indent=2)

    # Copy VLM config + tokenizer + processor files from original
    for fname in os.listdir(orig_dir):
        src = os.path.join(orig_dir, fname)
        if os.path.isfile(src) and (fname.endswith('.json') or fname.endswith('.jinja') or fname.endswith('.txt') or fname == 'vocab.json'):
            shutil.copy2(src, os.path.join(vlm_dir, fname))

    print(f'  {label}: {len(combined)} keys → {vlm_dir}')

print('VLM models ready.')
"
  else
    echo "VLM models already exist, skipping merge."
  fi

  # Step 2: Launch SGLang servers
  echo "Starting SGLang servers (prior=:${PRIOR_PORT}, realization=:${REAL_PORT})..."

  # Detect LD_LIBRARY_PATH for nvidia cufile (needed by torch 2.9+)
  NVIDIA_LD=$(python3 -c "
import site, glob, os
dirs = []
for base in [site.getusersitepackages(), *site.getsitepackages()]:
    dirs.extend(glob.glob(os.path.join(base, 'nvidia', '*', 'lib')))
    dirs.extend(glob.glob(os.path.join(base, 'nvidia', 'cufile', 'lib')))
print(':'.join(dirs))
" 2>/dev/null)
  export LD_LIBRARY_PATH="${NVIDIA_LD}:${LD_LIBRARY_PATH:-}"

  python3 -m sglang.launch_server \
    --model-path "${VLM_PRIOR}" \
    --dtype bfloat16 \
    --mem-fraction-static "${SGLANG_MEM}" \
    --port "${PRIOR_PORT}" \
    --max-total-tokens 4096 \
    > /tmp/sglang_prior_${MODEL}.log 2>&1 &
  echo $! > /tmp/sglang_prior_${MODEL}.pid

  python3 -m sglang.launch_server \
    --model-path "${VLM_REAL}" \
    --dtype bfloat16 \
    --mem-fraction-static "${SGLANG_MEM}" \
    --port "${REAL_PORT}" \
    --max-total-tokens 4096 \
    > /tmp/sglang_real_${MODEL}.log 2>&1 &
  echo $! > /tmp/sglang_real_${MODEL}.pid

  # Wait for both servers to come up
  echo "Waiting for servers..."
  for i in $(seq 1 30); do
    sleep 10
    P=$(curl -s "http://localhost:${PRIOR_PORT}/v1/models" 2>/dev/null | head -1)
    R=$(curl -s "http://localhost:${REAL_PORT}/v1/models" 2>/dev/null | head -1)
    if [ -n "$P" ] && [ -n "$R" ]; then
      echo "Both servers UP after $((i*10))s"
      SGLANG_OK=1
      break
    fi
    # Check if either process died
    if ! kill -0 $(cat /tmp/sglang_prior_${MODEL}.pid 2>/dev/null) 2>/dev/null; then
      echo "Prior server died. Falling back to HF generate."
      echo "Log: /tmp/sglang_prior_${MODEL}.log"
      break
    fi
    if ! kill -0 $(cat /tmp/sglang_real_${MODEL}.pid 2>/dev/null) 2>/dev/null; then
      echo "Realization server died. Falling back to HF generate."
      echo "Log: /tmp/sglang_real_${MODEL}.log"
      break
    fi
    echo "  ${i}0s..."
  done

  if [ "$SGLANG_OK" -eq 1 ]; then
    export SGLANG_PRIOR_URL="http://localhost:${PRIOR_PORT}"
    export SGLANG_URL="http://localhost:${REAL_PORT}"
    echo "SGLang serving active: prior=:${PRIOR_PORT} realization=:${REAL_PORT}"
  else
    echo "SGLang setup failed. Using HF generate (slower)."
    kill $(cat /tmp/sglang_prior_${MODEL}.pid 2>/dev/null) 2>/dev/null || true
    kill $(cat /tmp/sglang_real_${MODEL}.pid 2>/dev/null) 2>/dev/null || true
  fi
else
  if [ "$USE_SGLANG" = "1" ]; then
    echo "SGLang not installed. Using HF generate. Install with: pip install 'sglang[all]'"
  else
    echo "SGLang disabled (USE_SGLANG=0). Using HF generate."
  fi
fi

# ---- Run evaluation ----
TRAINED_EVAL="${OUT}/eval_trained.json"
if [ -f "$TRAINED_EVAL" ]; then
  echo ""; echo "Eval already exists: $TRAINED_EVAL"
  echo "Delete it to re-run: rm $TRAINED_EVAL"
else
  echo ""; echo "===== Running evaluation ====="
  python3 examples/benchmark/run_domain_backtest.py \
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

# ---- Voyage re-eval (produces paper-comparable scores) ----
VOYAGE_EVAL="${OUT}/eval_voyage.json"
if [ -n "${VOYAGE_API_KEY:-}" ]; then
  if [ -f "$VOYAGE_EVAL" ]; then
    echo ""; echo "Voyage re-eval already exists: $VOYAGE_EVAL"
  else
    echo ""; echo "===== Voyage Re-eval (threshold=0.80) ====="
    VOYAGE_API_KEY="$VOYAGE_API_KEY" python3 examples/benchmark/reeval_voyage.py \
      --input-json "$TRAINED_EVAL" \
      --papers-dir "$PAPERS" \
      --output "$VOYAGE_EVAL" \
      --threshold 0.80
  fi
else
  echo ""
  echo "VOYAGE_API_KEY not set. Run Voyage re-eval manually for paper-comparable scores:"
  echo "  VOYAGE_API_KEY=... python3 examples/benchmark/reeval_voyage.py \\"
  echo "    --input-json $TRAINED_EVAL --papers-dir $PAPERS \\"
  echo "    --output $VOYAGE_EVAL --threshold 0.80"
fi

echo ""
echo "===== Results: ${MODEL} ====="
if [ -f "$VOYAGE_EVAL" ]; then
  echo "  (Voyage, threshold=0.80 — comparable to paper baselines)"
  python3 -c "
import json
data = json.load(open('${VOYAGE_EVAL}'))
s = data.get('aggregate_summary', {})
print(f'  hit@5={s.get(\"avg_hit_at_k\", 0):.4f}  P@5={s.get(\"avg_precision_at_k\", 0):.4f}  '
      f'R@5={s.get(\"avg_recall_at_k\", 0):.4f}  MRR={s.get(\"avg_mrr\", 0):.4f}  '
      f'Nov={s.get(\"avg_novelty\", 0):.4f}  Div={s.get(\"avg_diversity\", 0):.4f}')
"
elif [ -f "$TRAINED_EVAL" ]; then
  echo "  (Heuristic — not comparable to paper baselines)"
  python3 -c "
import json
data = json.load(open('${TRAINED_EVAL}'))
s = data.get('aggregate_summary', {})
print(f'  hit@5={s.get(\"avg_hit_at_k\", 0):.4f}  P@5={s.get(\"avg_precision_at_k\", 0):.4f}  '
      f'R@5={s.get(\"avg_recall_at_k\", 0):.4f}  MRR={s.get(\"avg_mrr\", 0):.4f}  '
      f'Nov={s.get(\"avg_novelty\", 0):.4f}  Div={s.get(\"avg_diversity\", 0):.4f}')
"
fi

echo ""; echo "===== Done ====="
