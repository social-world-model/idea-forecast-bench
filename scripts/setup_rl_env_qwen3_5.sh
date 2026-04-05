#!/usr/bin/env bash
# ==========================================================================
#  Environment setup for Qwen3.5 models (VLM architecture, no vLLM)
#
#  Key stack: transformers >=5.x + TRL 1.0 + torch >=2.4 (NO vLLM)
#
#  Why Qwen3.5 needs its own env:
#    - Qwen3.5 registers as ForConditionalGeneration (VLM) in HF
#    - model_type "qwen3_5" only exists in transformers >=5.x
#    - vLLM 0.17.x does not support Qwen3.5 and pins transformers <5
#    - Generation uses HF generate() instead of vLLM (slower but compatible)
#    - We force AutoModelForCausalLM to use the text-only CausalLM head
#
#  Usage:
#    conda create -n live-idea-bench-qwen35 python=3.11 -y
#    conda activate live-idea-bench-qwen35
#    bash scripts/setup_rl_env_qwen3_5.sh
#
#  Then run training:
#    source scripts/activate_cuda_libs.sh
#    MODEL=qwen3.5-2b bash scripts/run_train_and_eval.sh
#
#  Supported models: qwen3.5-0.8b, qwen3.5-2b, qwen3.5-4b, qwen3.5-9b
#                    (also base variants: qwen3.5-4b-base, qwen3.5-9b-base)
# ==========================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=============================================="
echo " Qwen3.5 Environment Setup (no vLLM)"
echo "=============================================="
echo ""

# ---- Pre-flight checks ----
echo "=== Pre-flight checks ==="

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: Linux required." >&2; exit 1
fi
echo "  OS: Linux OK"

PY_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -ne 3 ] || [ "$PY_MINOR" -lt 10 ] || [ "$PY_MINOR" -gt 11 ]; then
  echo "ERROR: Python 3.10 or 3.11 required (found $PY_VERSION)." >&2; exit 1
fi
echo "  Python: $PY_VERSION OK"

if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. NVIDIA driver required." >&2; exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "  GPU: ${GPU_NAME} (${GPU_MEM} MiB), Driver: ${DRIVER}"

echo ""

# ---- Install dependencies ----
# No vLLM — install torch first, then transformers >=5 for Qwen3.5 support.

echo "=== Installing dependencies ==="
"$PYTHON_BIN" -m pip install --upgrade pip

# 1. PyTorch with CUDA (latest stable)
echo "  [1/5] Installing PyTorch with CUDA..."
"$PYTHON_BIN" -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. Transformers >=5.x (required for Qwen3.5 model_type "qwen3_5")
echo "  [2/5] Installing transformers >=5.x + TRL training stack..."
"$PYTHON_BIN" -m pip install \
  "transformers>=5.0.0" \
  "trl>=1.0.0" \
  "peft>=0.18.0" \
  "accelerate>=1.0.0" \
  "datasets>=2.21.0"

# 3. Similarity / reward computation
echo "  [3/5] Installing similarity + reward dependencies..."
"$PYTHON_BIN" -m pip install \
  "sentence-transformers>=3.0.0" \
  "scikit-learn>=1.4.0" \
  "sentencepiece>=0.2.0" \
  "bitsandbytes>=0.43.0"

# 4. Project backend dependencies
echo "  [4/5] Installing backend requirements..."
"$PYTHON_BIN" -m pip install \
  "pydantic>=2.8.2" \
  "pyyaml>=6.0" \
  "requests>=2.28.0" \
  "beartype" \
  "backoff" \
  "rich" \
  "pandas>=2.2.0"
# Optional API clients (only needed for joint inference / eval, not GRPO)
"$PYTHON_BIN" -m pip install "openai>=1.0.0" "anthropic" 2>/dev/null || true

# 5. FlashAttention 2 (optional, falls back to SDPA)
echo "  [5/5] Installing FlashAttention 2..."
"$PYTHON_BIN" -m pip install flash-attn --no-build-isolation 2>/dev/null || \
  echo "  WARNING: flash-attn build failed. Using SDPA attention (still fast)."

echo ""

# ---- Set up LD_LIBRARY_PATH ----

echo "=== Configuring CUDA library paths ==="
NVIDIA_LIB_DIRS=$("$PYTHON_BIN" -c "
import site, glob, os
dirs = []
for base in [site.getusersitepackages(), *site.getsitepackages()]:
    dirs.extend(glob.glob(os.path.join(base, 'nvidia', '*', 'lib')))
print(':'.join(dirs))
")

if [ -n "$NVIDIA_LIB_DIRS" ]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
  ACTIVATE_SCRIPT="$ROOT_DIR/scripts/activate_cuda_libs.sh"
  cat > "$ACTIVATE_SCRIPT" <<ACTIVATE
#!/usr/bin/env bash
# Source this before running training: source scripts/activate_cuda_libs.sh
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:\${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH configured."
ACTIVATE
  chmod +x "$ACTIVATE_SCRIPT"
  echo "  Wrote $ACTIVATE_SCRIPT"
fi

echo ""

# ---- Validation ----

echo "=== Validating installation ==="
"$PYTHON_BIN" - <<'PY'
import sys
errors = []

try:
    import torch
    if not torch.cuda.is_available():
        errors.append("PyTorch installed but CUDA not available")
    else:
        print(f"  torch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    errors.append(f"torch: {e}")

# vLLM should NOT be installed in this env
try:
    import vllm
    print(f"  WARNING: vLLM {vllm.__version__} is installed but not used for Qwen3.5")
    print("           (Qwen3.5 is not supported by vLLM, generation uses HF generate)")
except ImportError:
    print("  vLLM: not installed (expected — Qwen3.5 uses HF generate)")

try:
    from trl import GRPOConfig, GRPOTrainer
    import trl
    print(f"  trl {trl.__version__} GRPOTrainer OK")
except Exception as e:
    errors.append(f"trl: {e}")

try:
    import transformers, peft
    tv = transformers.__version__
    print(f"  transformers {tv}, peft {peft.__version__}")
    major = int(tv.split('.')[0])
    if major < 5:
        errors.append(f"transformers {tv} is <5.x — Qwen3.5 requires >=5.x")
except ImportError as e:
    errors.append(f"transformers/peft: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("  sentence-transformers OK")
except ImportError:
    print("  WARNING: sentence-transformers not available (needed for reward)")

try:
    from forecaster.realization.trl.runner import train_with_trl
    from forecaster.realization.model_zoo import resolve_small_model
    spec = resolve_small_model("qwen3.5-2b")
    print(f"  project imports OK (verified: {spec.alias} → {spec.model_id})")
except ImportError as e:
    errors.append(f"project: {e}")

if errors:
    print("\nFAILED:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
print("\nAll checks passed. Ready for Qwen3.5 training (HF generate, no vLLM).")
PY

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run training:"
echo "  source scripts/activate_cuda_libs.sh"
echo "  MODEL=qwen3.5-2b bash scripts/run_train_and_eval.sh"
echo ""
echo "Supported Qwen3.5 models: qwen3.5-0.8b, qwen3.5-2b, qwen3.5-4b, qwen3.5-9b"
echo "                          qwen3.5-4b-base, qwen3.5-9b-base"
echo ""
echo "NOTE: No vLLM — generation is slower than Qwen3 env (~5-10x)."
echo "      Consider Qwen3 models if generation speed is a bottleneck."
