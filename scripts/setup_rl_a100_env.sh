#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "scripts/setup_rl_a100_env.sh is intended for Linux A100 hosts." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Installing base backend dependencies into the current environment..."
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$ROOT_DIR/backend/requirements.txt"

echo "Installing CUDA PyTorch and RL / 2L training stack for A100..."
"$PYTHON_BIN" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0+cu124" \
  "transformers>=4.51.0" \
  "datasets>=2.21.0" \
  "peft>=0.13.0" \
  "trl>=0.25.0" \
  "accelerate>=1.0.0" \
  "sentencepiece>=0.2.0" \
  "bitsandbytes>=0.43.0" \
  "vllm>=0.8.5"

echo "RL / 2L environment setup complete."
