#!/usr/bin/env bash
# ==========================================================================
#  Environment setup dispatcher for the training pipeline.
#
#  Qwen3 and Qwen3.5 require different dependency stacks:
#
#    Qwen3  (text-only):  vLLM 0.17.1 + transformers <5  → FAST generation
#    Qwen3.5 (VLM arch):  transformers >=5.x, no vLLM    → slower generation
#
#  Usage:
#    # Option A: specify model family directly
#    bash scripts/setup_rl_env.sh qwen3      # or: qwen3.5
#
#    # Option B: auto-detect from MODEL env var
#    MODEL=qwen3-1.7b bash scripts/setup_rl_env.sh
#
#    # Option C: use the specific scripts directly
#    bash scripts/setup_rl_env_qwen3.sh
#    bash scripts/setup_rl_env_qwen3_5.sh
#
#  Recommended conda environments:
#    conda create -n idea-forecast-bench-qwen3  python=3.11 -y   # for Qwen3
#    conda create -n idea-forecast-bench-qwen35 python=3.11 -y   # for Qwen3.5
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine model family from argument or MODEL env var
FAMILY="${1:-}"
if [ -z "$FAMILY" ]; then
  MODEL="${MODEL:-qwen3-1.7b}"
  case "$MODEL" in
    qwen3.5*) FAMILY="qwen3.5" ;;
    qwen3*)   FAMILY="qwen3"   ;;
    qwen2.5*) FAMILY="qwen3.5" ;;  # legacy, use qwen3.5 env
    llama*)   FAMILY="qwen3.5" ;;  # no vLLM needed, use generic env
    *)        FAMILY="qwen3"   ;;  # default to vLLM-enabled env
  esac
fi

case "$FAMILY" in
  qwen3.5|qwen35)
    echo "Setting up Qwen3.5 environment (transformers >=5.x, no vLLM)..."
    echo ""
    exec bash "$SCRIPT_DIR/setup_rl_env_qwen3_5.sh"
    ;;
  qwen3|*)
    echo "Setting up Qwen3 environment (vLLM 0.17.1 + transformers <5)..."
    echo ""
    exec bash "$SCRIPT_DIR/setup_rl_env_qwen3.sh"
    ;;
esac
