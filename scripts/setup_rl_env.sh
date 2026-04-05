#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "scripts/setup_rl_env.sh is intended for Linux GPU hosts." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Installing base backend dependencies into the current environment..."
"$PYTHON_BIN" -m pip install --upgrade pip

if [ -f "$ROOT_DIR/backend/requirements.txt" ]; then
  "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/backend/requirements.txt"
fi

echo "Installing CUDA PyTorch and TRL training stack..."
"$PYTHON_BIN" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0+cu124" \
  "transformers>=5.3.0" \
  "datasets>=2.21.0" \
  "pandas>=2.2.0" \
  "pyarrow>=15.0.0" \
  "peft>=0.18.0" \
  "trl>=1.0.0" \
  "accelerate>=1.0.0" \
  "sentencepiece>=0.2.0" \
  "bitsandbytes>=0.43.0" \
  "sentence-transformers>=3.0.0" \
  "scikit-learn>=1.4.0" \
  "mergekit>=0.1.0"

echo "Installing FlashAttention 2 (requires CUDA toolkit)..."
"$PYTHON_BIN" -m pip install flash-attn --no-build-isolation || \
  echo "WARNING: flash-attn installation failed. Training will use SDPA attention."

echo "Installing CUDA runtime libraries..."
"$PYTHON_BIN" -m pip install \
  "nvidia-cudnn-cu12>=9.0" \
  "nvidia-nccl-cu12>=2.20"

echo "Validating TRL training stack..."
"$PYTHON_BIN" - <<'PY'
import sys

try:
    from trl import GRPOTrainer, GRPOConfig
    import trl
    print(f"TRL {trl.__version__} GRPOTrainer OK")
except ImportError as e:
    sys.exit(f"TRL GRPOTrainer import failed: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("sentence-transformers OK")
except ImportError:
    print("WARNING: sentence-transformers not available, GRPO reward will use hybrid fallback")

try:
    import flash_attn
    print(f"flash-attn OK (v{flash_attn.__version__})")
except ImportError:
    print("WARNING: flash-attn not installed, will use SDPA attention")
PY

echo "RL environment setup complete."
