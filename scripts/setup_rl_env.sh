#!/usr/bin/env bash
# ==========================================================================
#  Set up the Python environment for running the full training pipeline:
#    bash scripts/run_train_and_eval.sh
#
#  Prerequisites:
#    - Linux with NVIDIA GPU (CUDA 12.x)
#    - Python 3.10 or 3.11
#    - conda or venv environment activated
#
#  Usage:
#    conda create -n live-idea-bench python=3.11 -y
#    conda activate live-idea-bench
#    bash scripts/setup_rl_env.sh
# ==========================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---- Pre-flight checks ----

echo "=== Pre-flight checks ==="

# OS
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: Linux required." >&2; exit 1
fi
echo "  OS: Linux OK"

# Python version
PY_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -ne 3 ] || [ "$PY_MINOR" -lt 10 ] || [ "$PY_MINOR" -gt 11 ]; then
  echo "ERROR: Python 3.10 or 3.11 required (found $PY_VERSION)." >&2; exit 1
fi
echo "  Python: $PY_VERSION OK"

# NVIDIA GPU
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. NVIDIA driver required." >&2; exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "  GPU: ${GPU_NAME} (${GPU_MEM} MiB)"

# CUDA version
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "  Driver: ${CUDA_VERSION}"

echo ""

# ---- Install dependencies ----

echo "=== Installing dependencies ==="
"$PYTHON_BIN" -m pip install --upgrade pip

# Core project dependencies (LLM clients, config, utilities)
if [ -f "$ROOT_DIR/backend/requirements.txt" ]; then
  echo "  Installing backend requirements..."
  "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/backend/requirements.txt"
fi

# PyTorch with CUDA 12.4
echo "  Installing PyTorch + CUDA..."
"$PYTHON_BIN" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0+cu124"

# TRL GRPO training stack
echo "  Installing TRL training stack..."
"$PYTHON_BIN" -m pip install \
  "trl>=1.0.0" \
  "transformers>=5.3.0" \
  "peft>=0.18.0" \
  "accelerate>=1.0.0" \
  "datasets>=2.21.0"

# Data processing
echo "  Installing data dependencies..."
"$PYTHON_BIN" -m pip install \
  "pandas>=2.2.0" \
  "pyarrow>=15.0.0" \
  "sentencepiece>=0.2.0" \
  "bitsandbytes>=0.43.0"

# Similarity / reward computation
echo "  Installing similarity dependencies..."
"$PYTHON_BIN" -m pip install \
  "sentence-transformers>=3.0.0" \
  "scikit-learn>=1.4.0"

# CUDA runtime libraries (often missing from conda envs)
echo "  Installing CUDA runtime libraries..."
"$PYTHON_BIN" -m pip install \
  "nvidia-cudnn-cu12>=9.0" \
  "nvidia-nccl-cu12>=2.20"

# FlashAttention 2 (optional, falls back to SDPA)
echo "  Installing FlashAttention 2..."
"$PYTHON_BIN" -m pip install flash-attn --no-build-isolation 2>/dev/null || \
  echo "  WARNING: flash-attn build failed (needs CUDA toolkit). Using SDPA attention."

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
  echo "  LD_LIBRARY_PATH set with $(echo "$NVIDIA_LIB_DIRS" | tr ':' '\n' | wc -l) nvidia lib dirs"

  # Write a helper script for future sessions
  ACTIVATE_SCRIPT="$ROOT_DIR/scripts/activate_cuda_libs.sh"
  cat > "$ACTIVATE_SCRIPT" <<ACTIVATE
#!/usr/bin/env bash
# Source this before running training: source scripts/activate_cuda_libs.sh
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:\${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH configured for CUDA libraries."
ACTIVATE
  chmod +x "$ACTIVATE_SCRIPT"
  echo "  Wrote $ACTIVATE_SCRIPT (source this before training)"
fi

echo ""

# ---- Validation ----

echo "=== Validating installation ==="
"$PYTHON_BIN" - <<'PY'
import sys

errors = []

# PyTorch + CUDA
try:
    import torch
    if not torch.cuda.is_available():
        errors.append("PyTorch installed but CUDA not available")
    else:
        print(f"  torch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    errors.append(f"torch: {e}")

# TRL
try:
    from trl import GRPOTrainer, GRPOConfig
    import trl
    print(f"  trl {trl.__version__} GRPOTrainer OK")
except Exception as e:
    errors.append(f"trl GRPOTrainer: {e}")

# transformers
try:
    import transformers
    print(f"  transformers {transformers.__version__}")
except ImportError as e:
    errors.append(f"transformers: {e}")

# peft
try:
    import peft
    print(f"  peft {peft.__version__}")
except ImportError as e:
    errors.append(f"peft: {e}")

# sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    print("  sentence-transformers OK")
except ImportError:
    print("  WARNING: sentence-transformers not available (reward will use hybrid fallback)")

# flash-attn
try:
    import flash_attn
    print(f"  flash-attn {flash_attn.__version__}")
except ImportError:
    print("  INFO: flash-attn not installed (using SDPA attention)")

# Project imports
try:
    from live_idea_bench.papers import load_papers_from_markdown
    from forecaster.realization.pipeline import run_policy_rl_pipeline
    from forecaster.realization.trl.runner import train_with_trl
    print("  project imports OK")
except ImportError as e:
    errors.append(f"project imports: {e}")

if errors:
    print("\nFAILED:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("\nAll checks passed.")
PY

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run training:"
echo "  source scripts/activate_cuda_libs.sh"
echo "  PAPERS=/path/to/papers nohup bash scripts/run_train_and_eval.sh > output/run.log 2>&1 &"
