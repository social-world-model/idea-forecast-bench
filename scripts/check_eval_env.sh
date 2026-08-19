#!/usr/bin/env bash
set -euo pipefail
# ==========================================================================
#  Check if a machine can run the trained forecaster eval.
#  Usage: bash scripts/check_eval_env.sh
# ==========================================================================

echo "=============================================="
echo " Environment Check for Forecaster Eval"
echo "=============================================="

# ---- System RAM ----
echo ""
echo "=== System RAM ==="
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_RAM_GB=$((TOTAL_RAM_KB / 1024 / 1024))
echo "  Total: ${TOTAL_RAM_GB} GB"

# Check cgroup limit (containers/k8s)
CGROUP_LIMIT=""
if [ -f /sys/fs/cgroup/memory.max ]; then
  CGROUP_LIMIT=$(cat /sys/fs/cgroup/memory.max)
elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
  CGROUP_LIMIT=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
fi
if [ -n "$CGROUP_LIMIT" ] && [ "$CGROUP_LIMIT" != "max" ]; then
  CGROUP_GB=$((CGROUP_LIMIT / 1024 / 1024 / 1024))
  echo "  Cgroup limit: ${CGROUP_GB} GB  ← THIS is the real limit"
  if [ "$CGROUP_GB" -lt 48 ]; then
    echo "  WARNING: <48GB cgroup. SGLang servers may get OOM-killed."
    echo "           Use HF generate (USE_SGLANG=0) or request more RAM."
  fi
else
  echo "  Cgroup limit: none (full system RAM available)"
fi

# ---- GPU ----
echo ""
echo "=== GPU ==="
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
  GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo ""
  echo "  Minimum VRAM needed:"
  echo "    0.8B eval (HF generate):   ~3 GB"
  echo "    2B eval (HF generate):     ~6 GB"
  echo "    0.8B eval (SGLang VLM):    ~5 GB"
  echo "    2B eval (SGLang VLM):      ~8 GB"
  echo "    Both SGLang servers:       ~10 GB (0.8B) / ~16 GB (2B)"
  echo ""
  if [ "$GPU_FREE" -lt 6000 ]; then
    echo "  WARNING: <6GB free VRAM. May not fit even 0.8B model."
  else
    echo "  OK: ${GPU_FREE} MiB free — sufficient for eval."
  fi
else
  echo "  ERROR: nvidia-smi not found. GPU required."
fi

# ---- RAM usage estimates ----
echo ""
echo "=== RAM Usage Estimates ==="
echo "  HF generate eval:        ~8-12 GB RAM (model + torch + python)"
echo "  SGLang (2 servers):      ~20-30 GB RAM (scheduler + tokenizer + engine per server)"
echo "  SGLang (1 server):       ~12-18 GB RAM"
echo ""
echo "  Recommendation:"
if [ -n "$CGROUP_LIMIT" ] && [ "$CGROUP_LIMIT" != "max" ] && [ "$CGROUP_GB" -lt 48 ]; then
  echo "    → Cgroup ${CGROUP_GB}GB: use HF generate (USE_SGLANG=0)"
  echo "      Slower (~50 tok/s) but fits in limited RAM."
elif [ "$TOTAL_RAM_GB" -ge 48 ]; then
  echo "    → ${TOTAL_RAM_GB}GB RAM, no tight cgroup: use SGLang (fast, ~1000 tok/s)"
else
  echo "    → ${TOTAL_RAM_GB}GB RAM: use HF generate to be safe"
fi

# ---- Python / CUDA ----
echo ""
echo "=== Software ==="
python3 --version 2>/dev/null || echo "  ERROR: python3 not found"
python3 -c "import torch; print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}')" 2>/dev/null || echo "  torch: not installed"
python3 -c "import transformers; print(f'  transformers {transformers.__version__}')" 2>/dev/null || echo "  transformers: not installed"
python3 -c "import peft; print(f'  peft {peft.__version__}')" 2>/dev/null || echo "  peft: not installed"
python3 -c "import sglang; print(f'  sglang {sglang.__version__} (fast eval available)')" 2>/dev/null || echo "  sglang: not installed (will use HF generate)"

echo ""
echo "=============================================="
