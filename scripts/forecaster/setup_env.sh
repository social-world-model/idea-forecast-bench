#!/usr/bin/env bash
# ============================================================================
#  Forecaster Unsloth environment setup.
#
#  Installs Unsloth (FastLanguageModel + cached Qwen3.5 modules), the latest
#  TRL (GRPOTrainer + SFTTrainer), and the rest of the project's training
#  dependencies.
#
#  Usage:
#      conda create -n live-idea-bench-unsloth python=3.11 -y
#      conda activate live-idea-bench-unsloth
#      bash scripts/forecaster/setup_env.sh
# ============================================================================
set -uo pipefail
trap 'echo "ERROR: setup failed at line $LINENO (exit code $?)" >&2' ERR
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-python3}"

echo "=============================================="
echo " Forecaster Unsloth env setup"
echo "=============================================="
echo ""

# ---- Pre-flight checks ----
echo "=== Pre-flight checks ==="

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: Linux required." >&2; exit 1
fi
echo "  OS: Linux OK"

PY_VERSION=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PY" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PY" -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -ne 3 ] || [ "$PY_MINOR" -lt 10 ] || [ "$PY_MINOR" -gt 11 ]; then
  echo "ERROR: Python 3.10 or 3.11 required (found $PY_VERSION)." >&2; exit 1
fi
echo "  Python: $PY_VERSION OK"

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
  echo "  GPU: ${GPU_NAME}"
else
  echo "  GPU: not detected (ok for install — needed at runtime)"
fi
echo ""

# ---- Install dependencies ----
echo "=== Installing dependencies ==="
"$PY" -m pip install --upgrade pip

# 1. PyTorch with CUDA 12.4
echo "  [1/5] Installing PyTorch (CUDA 12.4)..."
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. Unsloth + Unsloth-Zoo (pulls compatible transformers / peft / trl /
#    accelerate / bitsandbytes / datasets via its own dependency graph).
echo "  [2/5] Installing Unsloth + latest TRL stack..."
"$PY" -m pip install "unsloth" "unsloth_zoo"
"$PY" -m pip install --upgrade "trl" "transformers" "peft" "accelerate" "datasets"

# 3. Reward + similarity stack
echo "  [3/5] Installing reward / similarity dependencies..."
"$PY" -m pip install \
  "sentence-transformers>=3.0.0" \
  "scikit-learn>=1.4.0" \
  "sentencepiece>=0.2.0" \
  "bitsandbytes>=0.43.0"

# 4. Project backend deps
echo "  [4/5] Installing project backend deps..."
"$PY" -m pip install \
  "pydantic>=2.8.2" \
  "pyyaml>=6.0" \
  "requests>=2.28.0" \
  "beartype" \
  "backoff" \
  "rich" \
  "pandas>=2.2.0"
"$PY" -m pip install "openai>=1.0.0" "anthropic" 2>/dev/null || true

# 5. vLLM nightly — required for Qwen3.5 GRPO generation speedup
#    The stable vLLM (0.19.0) has a torch.fx compilation bug on Qwen3.5 +
#    torch>=2.10. The nightly fixes it. We install vLLM and then immediately
#    re-upgrade transformers because vLLM's stale package metadata pins
#    transformers<5 even though its runtime supports v5 (per Unsloth's docs:
#    https://huggingface.co/unsloth/Qwen3.5-2B#vllm).
#    Set DISABLE_VLLM=1 to skip — training will fall back to HF generate
#    (~5x slower per step on 2B+).
if [[ "${DISABLE_VLLM:-0}" == "1" ]]; then
  echo "  [5/6] Skipping vLLM (DISABLE_VLLM=1). Generation will use HF generate (slow)."
else
  echo "  [5/6] Installing vLLM nightly + restoring transformers>=5.5..."
  "$PY" -m pip install --pre -U vllm --extra-index-url https://wheels.vllm.ai/nightly 2>&1 | tail -5
  # vLLM's metadata downgrades transformers; force it back so Qwen3.5 still
  # loads (Unsloth requires transformers>=5.2 for Qwen3.5).
  "$PY" -m pip install --upgrade "transformers>=5.5.0" 2>&1 | tail -5
fi

# 6. Optional: FlashAttention 2/3 — SKIPPED BY DEFAULT
#    Unsloth ships its own optimized attention kernels, so flash-attn is
#    redundant for the A100 path. On Hopper (H100/H200/B200) you can install
#    flash-attn 3 for an additional ~30-50% on the attention path.
#    Building from source takes 30-60 minutes. Set INSTALL_FLASH_ATTN=1 to
#    install anyway.
if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  echo "  [6/6] Installing FlashAttention 2 (this will take 30-60 minutes)..."
  "$PY" -m pip install flash-attn --no-build-isolation || \
    echo "  WARNING: flash-attn build failed. Unsloth will fall back to its own kernels."
else
  echo "  [6/6] Skipping FlashAttention 2 (Unsloth has its own kernels). Set INSTALL_FLASH_ATTN=1 to install."
fi

echo ""

# ---- Validation ----
echo "=== Validating installation ==="
cd "$ROOT_DIR"
"$PY" - <<'PY'
import sys
errors = []

try:
    import torch
    if not torch.cuda.is_available():
        print("  WARNING: torch installed but CUDA not available — training requires a GPU.")
    else:
        print(f"  torch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    errors.append(f"torch: {e}")

try:
    from unsloth import FastLanguageModel  # noqa: F401
    import unsloth
    print(f"  unsloth {unsloth.__version__} OK")
except Exception as e:
    errors.append(f"unsloth: {e}")

try:
    import trl, transformers, peft
    print(f"  trl {trl.__version__}, transformers {transformers.__version__}, peft {peft.__version__}")
    from trl import GRPOTrainer, SFTTrainer  # noqa: F401
    # Qwen3.5 needs transformers >= 5.2 — fail loudly if vLLM downgraded it.
    tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if tv < (5, 2):
        errors.append(
            f"transformers {transformers.__version__} is < 5.2 — Qwen3.5 will not load. "
            f"Re-run: pip install --upgrade 'transformers>=5.5.0'"
        )
except Exception as e:
    errors.append(f"trl: {e}")

try:
    import vllm
    print(f"  vllm {vllm.__version__} OK")
except ImportError:
    print("  vllm: NOT installed (training will use slow HF generate fallback)")

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    print("  sentence-transformers OK")
except Exception as e:
    errors.append(f"sentence-transformers: {e}")

try:
    from forecaster.realization.trainers import create_trainer_runner  # noqa: F401
    from forecaster.realization.trl.runner import prepare_trl_artifacts, train_with_trl  # noqa: F401
    from forecaster.realization.verl.reward_fn import compute_score  # noqa: F401
    from forecaster.realization.model_zoo import resolve_small_model
    from forecaster.prior.trainer import train_prior  # noqa: F401
    spec = resolve_small_model("qwen3.5-2b")
    print(f"  project imports OK ({spec.alias} -> {spec.model_id})")
except Exception as e:
    errors.append(f"project: {e}")

if errors:
    print("\nFAILED:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
print("\nAll checks passed. Ready for training.")
PY

echo ""
echo "=== Setup complete ==="
echo ""
echo "To train:"
echo "  bash scripts/forecaster/train.sh \\"
echo "      --model qwen3.5-2b \\"
echo "      --hindsight output/hindsight_samples.jsonl \\"
echo "      --papers data/csml/raw_markdown \\"
echo "      --output-dir output/forecaster_qwen3.5-2b"
