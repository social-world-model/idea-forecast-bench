#!/usr/bin/env bash
# One-shot env build for the foresight GRPO pipeline on a NEW machine
# (target: RTX Pro 6000 / Blackwell sm_120, CUDA 13.0 driver).
#
# Replicates the env that worked on the A6000 box. CUDA 13 drivers are
# backward-compatible with the cu128 (CUDA 12.8) torch wheels, and cu128
# wheels ship sm_120 (Blackwell) kernels, so this should run as-is.
# If torch reports "no kernel image for sm_120", switch the torch index
# to the cu130 / nightly channel (see FALLBACK note at the bottom).
set -euo pipefail

ENV_NAME="${ENV_NAME:-idea-grpo}"
PY=python

echo "=== [1/5] create conda env '$ENV_NAME' (python 3.12) ==="
conda create -y -n "$ENV_NAME" python=3.12
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "=== [2/5] torch 2.11.0+cu128 (Blackwell kernels) ==="
pip install --no-cache-dir \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

echo "=== [3/5] vLLM 0.21.0 (cu129 wheel — runs on CUDA13 driver) ==="
pip install --no-cache-dir \
  "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.21.0/vllm-0.21.0+cu129-cp38-abi3-manylinux_2_34_x86_64.whl"

echo "=== [4/5] training + foresight deps (pinned) ==="
pip install --no-cache-dir \
  transformers==5.9.0 trl==1.4.0 peft==0.19.1 accelerate==1.13.0 \
  datasets==4.8.5 sentence-transformers==5.5.1 bitsandbytes==0.49.2 \
  flashinfer-python==0.6.8.post1 triton==3.6.0 numpy==2.3.5 \
  openai==2.38.0 openai-harmony==0.0.8
# Qwen3.5 hybrid-linear-attention kernels (JIT-recompile for sm_120 on first run)
pip install --no-cache-dir flash-linear-attention==0.5.0 tilelang==0.1.9
# unsloth pins torch<2.11 → install without deps so it doesn't downgrade torch
pip install --no-cache-dir --no-deps unsloth==2026.5.6 unsloth_zoo

echo "=== [5/5] verify CUDA + Blackwell ==="
$PY - <<'PYEOF'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda"); print("matmul ok:", (x@x).sum().item() == (x@x).sum().item())
import vllm, transformers, trl, peft
print("vllm", vllm.__version__, "transformers", transformers.__version__, "trl", trl.__version__, "peft", peft.__version__)
PYEOF

echo
echo "=== DONE. Next: transfer artifacts + launch (see docs/new-machine-setup.md) ==="
echo "FALLBACK: if you see 'no kernel image for device sm_120', reinstall torch from"
echo "  --index-url https://download.pytorch.org/whl/nightly/cu130   (then rebuild vllm to match)"
