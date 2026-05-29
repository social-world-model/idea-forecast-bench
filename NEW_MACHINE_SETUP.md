# Foresight GRPO — new machine (RTX Pro 6000 / CUDA 13) setup

Target: 4× RTX Pro 6000 (Blackwell, 96 GB), CUDA 13.0. Goal: run the full
foresight GRPO (SFT→GRPO) with **vLLM enabled** → ETA ~3–5h instead of ~26h.

## 0. Get the code
```bash
git clone https://github.com/ulab-uiuc/live-idea-bench.git
cd live-idea-bench
git checkout feature/foresight-judge-soft-mustnot   # has all the fixes (commit 2279092+)
```

## 1. Build the env
```bash
bash scripts/setup_new_machine.sh          # creates conda env 'idea-grpo'
conda activate idea-grpo
```
If `torch` errors with `no kernel image for sm_120` on a generation, see the
FALLBACK note at the end of the script (switch to cu130/nightly torch).

## 2. Transfer the non-git artifacts (papers already present on the new box)
Only the **expensive / paid** artifacts need to move (~1.3 G). Run FROM the old
box (sn4622122392), or pull from the new box — fill in the host/path:
```bash
OLD=max7@sn4622122392:/home/max7/live_idea_bench_fenghai/live-idea-bench/.worktrees/foresight-judge
DST=.                                       # = repo root on the new machine

rsync -avP "$OLD/output/foresight_artifacts"                                 "$DST/output/"
rsync -avP "$OLD/output/forecaster_qwen3.5-9b/prior_sft/final_checkpoint"     "$DST/output/forecaster_qwen3.5-9b/prior_sft/"
rsync -avP "$OLD/output/hindsight_samples.jsonl"                              "$DST/output/"
rsync -avP "$OLD/data/topic_hindsight/dz.jsonl"                              "$DST/data/topic_hindsight/"
```
- `foresight_artifacts/` = indices + gpt-5.4 rubrics (transferring avoids re-paying the OpenAI API and re-encoding indices).
- `prior_sft/final_checkpoint/` = the SFT adapter (transferring avoids hours of SFT retraining).
- **No API keys needed** on the new box: judge is local Qwen3.5-9B, rubrics already generated.
- Qwen3.5-9B base weights: auto-downloaded from HF on first run.

> Alternative if you'd rather regenerate the indices than transfer them:
> edit the hardcoded `/home/max7/...` paths in `build_indices.py`, then run it
> (≈30–60 min on a Pro 6000). You still need the rubrics + SFT adapter
> transferred.

## 3. Start the judge (own GPU)
```bash
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-9B --served-model-name qwen3.5-9b-instruct \
  --host 0.0.0.0 --port 8767 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 > logs/judge_8767.log 2>&1 &
# (drop --enforce-eager on Blackwell — CUDA graphs make the judge faster)
```

## 4. Launch the full foresight GRPO (vLLM ON, SFT-init)
```bash
PAPERS=data/csml_v2/raw_markdown          # adjust to where the corpus is

USE_VLLM=1 VLLM_GPU_MEM_UTIL=0.45 BATCH_SIZE=8 CUDA_VISIBLE_DEVICES=0 \
JUDGE_MODEL=qwen3.5-9b-instruct JUDGE_API_KEY=EMPTY JUDGE_BASE_URL=http://localhost:8767/v1 \
nohup python examples/run_policy_rl_training.py \
  --input-dir "$PAPERS" \
  --output-dir output/forecaster_qwen3.5-9b/realization_grpo \
  --model-preset qwen3.5-9b --trainer grpo \
  --hindsight output/hindsight_samples.jsonl \
  --init-policy-path output/forecaster_qwen3.5-9b/prior_sft/final_checkpoint \
  --start-month 2023-01 --end-month 2024-09 --skip-alignment-check \
  > output/grpo_full.log 2>&1 &
```
- `USE_VLLM=1` → colocate vLLM on card 0 (96 GB fits trainer + vLLM copies of 9B).
- `--init-policy-path …/final_checkpoint` → continues the SFT adapter through GRPO.
- Tune `VLLM_GPU_MEM_UTIL` (0.4–0.6) and `BATCH_SIZE` if OOM/underutilized.
- Watch: `grep -aE "rewards/foresight_reward_fn/mean|s/it" output/grpo_full.log`
  and judge calls in `logs/judge_8767.log`. Reward should be non-zero (~0.5–0.8).

## Notes / caveats to verify on first run
- **vLLM + continued LoRA adapter**: TRL colocate must generate from base+SFT-adapter.
  If generation looks like the base model (adapter not applied in vLLM), fall back to
  `USE_VLLM=0` or sync the adapter into vLLM. Verify reward is non-zero early.
- Dataset rebuilds fresh (840 rows, 6 cutoffs) since `shared/` won't exist.
- First startup ~5–20 min (tilelang/fla recompile for sm_120, then cached).
