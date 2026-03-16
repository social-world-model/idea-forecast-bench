# RL Runbook

This directory documents the end-to-end RL workflow for `live-idea-bench`.

The repository uses a single online-RL backend: **veRL**.
`ppo`, `grpo`, and `rloo` all run through the same top-level script and the same function-based reward pipeline.

## Prerequisites

- Linux + A100 is the supported non-dry-run training environment
- Python 3.11 environment dedicated to RL
- CUDA-compatible PyTorch
- `verl>=0.4.1`

`RLOO` support exists in veRL release history before `0.4.1`; this repo pins `verl>=0.4.1` so `ppo`, `grpo`, and `rloo` all share one verified backend line.

## Environment Setup

From the repo root:

```bash
conda create -n live-idea-bench-rl python=3.11 -y
conda activate live-idea-bench-rl
bash scripts/setup_rl_a100_env.sh
```

The setup script installs:

- `torch==2.6.0+cu124`
- `verl>=0.4.1`
- `pandas>=2.2.0`
- `pyarrow>=15.0.0`
- `transformers`, `datasets`, `peft`, `accelerate`, `sentencepiece`, `bitsandbytes`, `vllm`

## Canonical Entry Script

All RL workflows use the same wrapper:

```bash
bash scripts/run_policy_rl_training.sh ...
```

The underlying CLI is:

```bash
python examples/run_policy_rl_training.py ...
```

Supported trainers:

- `ppo`
- `grpo`
- `rloo`

## Prepare-Only Flow

Prepare shared artifacts plus the veRL dataset without starting training:

```bash
bash scripts/run_policy_rl_training.sh \
  --input-dir data/arxiv_csml/raw_markdown \
  --model-preset qwen3-4b-instruct-2507 \
  --trainer grpo \
  --prepare-only \
  --output-dir data/rl_runs/qwen3_4b_prepare
```

Prepare a specific non-train split for inspection:

```bash
bash scripts/run_policy_rl_training.sh \
  --input-dir data/arxiv_csml/raw_markdown \
  --model-preset qwen3-4b-instruct-2507 \
  --trainer rloo \
  --prepare-only \
  --prepare-split validation \
  --output-dir data/rl_runs/qwen3_4b_rloo_validation
```

## Training Commands

Run PPO:

```bash
bash scripts/run_policy_rl_training.sh \
  --input-dir data/arxiv_csml/raw_markdown \
  --model-preset qwen2.5-3b-instruct \
  --trainer ppo \
  --output-dir data/rl_runs/qwen25_3b_ppo
```

Run GRPO:

```bash
bash scripts/run_policy_rl_training.sh \
  --input-dir data/arxiv_csml/raw_markdown \
  --model-preset qwen3-4b-instruct-2507 \
  --trainer grpo \
  --output-dir data/rl_runs/qwen3_4b_grpo
```

Run RLOO:

```bash
bash scripts/run_policy_rl_training.sh \
  --input-dir data/arxiv_csml/raw_markdown \
  --model-preset qwen3-4b-instruct-2507 \
  --trainer rloo \
  --output-dir data/rl_runs/qwen3_4b_rloo
```

Optional warm start:

```bash
--init-policy-path /abs/path/to/checkpoint
```

## Expected Outputs

Each RL run writes a trainer subdirectory under `--output-dir`, plus shared preparation artifacts.

Important files:

- `shared/episodes.json`
- `shared/prompts.jsonl`
- `<trainer>/trainer_dataset.parquet`
- `<trainer>/trainer_dataset.preview.jsonl`
- `<trainer>/verl_launch_config.json`
- `<trainer>/verl_launch_command.txt`
- `<trainer>/policy_manifest.json`
- `pipeline_manifest.json`

`policy_manifest.json` is the artifact later consumed by the `policy_rl` strategy.

## Dry Run Behavior

Dry run means:

- the veRL dataset is prepared
- launch config and launch command are written
- manifests are written
- veRL itself is not executed

Dry run does **not** require `verl` to be importable at execution time, but non-dry-run training does.

## Reward Model Note

The RL stack still uses a **function-based reward**, not a separately trained reward model.
The veRL custom reward hook calls the existing completion parser and reward evaluator in `live_idea_bench/rl/reward.py`.

## Troubleshooting

`pyarrow` missing:

- install it in the RL environment
- without `pyarrow` or `fastparquet`, the repo can only write the preview JSONL, not the real Parquet dataset veRL needs

`verl` missing:

- rerun `bash scripts/setup_rl_a100_env.sh`
- confirm `python -m verl.trainer.main_ppo --help` works in the RL environment

Alignment-gate failure:

- the CLI runs the online reward alignment gate before non-prepare training
- inspect `<trainer>/alignment_report.json`
- retry with `--skip-alignment-check` only if you intentionally want to bypass the guardrail

Wrong platform:

- non-dry-run RL is intended for Linux + A100 hosts
- local macOS development should use `--prepare-only` or dry-run-style validation
