#!/usr/bin/env bash
# ============================================================================
#  Single-GPU training (no vLLM).
#
#  Runs Phase 1 (Prior SFT, METHOD §3.2) → Phase 2 (dataset prep) →
#  Phase 3 (Realization GRPO, METHOD §3.3) using Unsloth + the latest TRL,
#  with `fast_inference=False` (Unsloth's documented Qwen3.5 GRPO path).
#
#  Generation runs through HF generate (~50-100 tok/s on 2B) — slow but
#  rock-solid on a single GPU. Per-step warm time on A100 80GB at G=4 with
#  max_completion_length=2048: ~120-160s.
#
#  For multi-GPU vLLM-accelerated training (3-5x faster), see
#  scripts/forecaster/train_vllm.sh.
#
#  Activate the env first:
#      conda activate live-idea-bench-unsloth
#
#  Example:
#      bash scripts/forecaster/train.sh \
#          --model qwen3.5-2b \
#          --hindsight output/hindsight_samples.jsonl \
#          --output-dir output/forecaster_qwen3.5-2b
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

# Reject --use-vllm-server here — that path requires multi-GPU and a
# separately-launched vLLM server. The user should use train_vllm.sh instead.
for arg in "$@"; do
  if [[ "$arg" == "--use-vllm-server" ]]; then
    echo "ERROR: --use-vllm-server is not supported in train.sh." >&2
    echo "       It requires 2+ GPUs and a separately-launched vLLM server." >&2
    echo "       Use scripts/forecaster/train_vllm.sh instead." >&2
    exit 1
  fi
done

# Honor PYTHON_BIN if the user set it explicitly (e.g. when there are
# multiple anaconda installs and `which python` is unreliable). Defaults to
# `python` from PATH (assumes `conda activate <env>` was run first).
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "${PYTHON_BIN}" examples/forecaster/train.py "$@"
