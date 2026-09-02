<h1 align="center">IdeaForecastBench</h1>

<p align="center">
  <b>Can Large Language Models Forecast What Researchers Study Next?</b>
</p>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2609.00747-b31b1b.svg)](https://arxiv.org/abs/2609.00747)
[![CI](https://github.com/ulab-uiuc/live-idea-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/ulab-uiuc/live-idea-bench/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/downloads/release/python-3109/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-red)](https://github.com/ulab-uiuc/live-idea-bench/pulls)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

</div>

## News

- **Sep 1, 2026** — Our paper is out on arXiv ([2609.00747](https://arxiv.org/abs/2609.00747)) and accepted to **EMNLP 2026** 🎉

## Introduction

**IdeaForecastBench** evaluates research idea *forecasting*. Given a community's
literature up to a cutoff, a system produces up to five ranked ideas, which are then
scored against the papers that actually appeared afterwards. Judging an idea's novelty
at generation time does not establish whether it anticipates subsequent work; this
benchmark does.

- **624 rolling episodes across 52 topics**, with a fixed **retrieve-then-judge**
  protocol and separately reported results from two judges.
- **Five history-compression baselines** (topic trend, direct, summary, retrieval,
  memory) across GPT-4.1, Qwen2.5-7B/14B, and Qwen3.5-9B.
- **MDF**, the *Mode-Decomposition Forecaster*: a learned method that samples a latent
  innovation from a memory-conditioned prior, realizes it into a grounded proposal, and
  trains the realization policy with GRPO against a future-grounded reward.

The Python package is `idea_forecast_bench` and the CLI is `idea-forecast-bench`. The
GitHub repository keeps its original slug, `live-idea-bench`.

## Installation

```bash
# create an environment (conda shown; any Python >=3.10,<3.13 works)
conda create -n idea-forecast-bench python=3.11 -y
conda activate idea-forecast-bench

# install Poetry if you do not have it
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

# install from source
git clone https://github.com/ulab-uiuc/live-idea-bench.git
cd live-idea-bench
poetry install
```

`poetry install` is enough to run the benchmark and every baseline. Training MDF needs
one more group:

| Group | Command | When you need it |
|---|---|---|
| forecaster | `poetry install --with forecaster` | train or run MDF locally (torch, trl, peft) |


## Setup

Baselines write their predictions with an LLM, and matching is embedding-based, so
two keys are needed:

```bash
export VOYAGE_API_KEY="your-voyage-key"     # embeddings (matching and the judge's retrieval)
export OPENAI_API_KEY="your-openai-key"     # generation and the default gpt-4.1-mini judge
```

Other providers work for generation: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
`TOGETHER_API_KEY`, `DEEPSEEK_API_KEY`. To point generation at a local OpenAI-compatible
server (vLLM, SGLang), set `OPENAI_BASE_URL`. The `baselines` command checks both keys
before it starts, so a missing key costs a second rather than five failed runs.

## Get started

### 1. Get a corpus

Use the frozen corpus the paper was run against
([md_files.zip, ~5 GB](https://drive.google.com/file/d/1182o0Teo3G128c3mxeimDQNBZNY-iWCe/view?usp=sharing))
and point `--input-dir` at wherever you unpacked it. Or pull a fresh one from arXiv,
which needs no account:

```bash
idea-forecast-bench fetch                                   # into data/csml/raw_markdown
idea-forecast-bench fetch --query "cat:cs.CL" --max-results 5000
```

A fresh corpus is useful for checking that the pipeline works, but it will not reproduce
the paper's numbers; the frozen corpus will.

### 2. Run the baselines

```bash
idea-forecast-bench baselines --input-dir /path/to/corpus
```

This scores all five baselines under identical windows and an identical matcher, then
prints one row per strategy:

```text
strategy                windows        hit_at_k   precision_at_k             mrr
-------------------------------------------------------------------------------
topic_trend                 ...             ...             ...             ...
summary_prompting           ...             ...             ...             ...
```

Read `windows` first. A strategy that produced none is reported as
`NOT SCORED -- no windows produced` and the command exits non-zero, so "could not run"
is never mistaken for "scored 0.0". If that happens, the corpus does not cover the
window densely enough: widen it with `--start-month` / `--end-month`, lower
`--min-train-papers`, or fetch more.

Every flag has a default, so the commands above are complete as written. Common
variations:

```bash
idea-forecast-bench baselines --only topic_trend,summary_prompting     # a subset
idea-forecast-bench baselines --start-month 2024-06 --end-month 2025-06 --top-k 10
idea-forecast-bench benchmark --strategy summary_prompting             # one strategy
idea-forecast-bench benchmark --strategy summary_prompting --model-name Qwen/Qwen2.5-7B-Instruct
```

### 3. Score with the retrieve-then-judge protocol

The paper's reported numbers come from the LLM judge, which re-embeds the papers,
retrieves candidates for each prediction, and scores each pair on problem, method, and
specificity:

```bash
idea-forecast-bench judge-eval \
  --input-json output/backtest/summary_prompting.json \
  --papers-dir /path/to/corpus \
  --output output/judged/summary_prompting.judged.json
```

The judge defaults to `gpt-4.1-mini`; pass `--judge-model` and `--judge-base-url` to use
another one. When the judge supplies your numbers, run `benchmark` with
`--skip-matching` to skip the embedding match it would otherwise duplicate. Assemble the
final table from judged artifacts with `idea-forecast-bench main-table`.

### Checking the wiring without provider keys

Both generation and matching accept an OpenAI-compatible endpoint, so the whole path can
be exercised against a local server or stub with no provider account:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1  OPENAI_API_KEY=EMPTY
export VOYAGE_BASE_URL=http://127.0.0.1:8000/v1  VOYAGE_API_KEY=EMPTY
idea-forecast-bench baselines --only topic_trend
```

This verifies wiring, not quality: only `windows` reaching a non-zero value tells you the
run was real.

## Baselines

| Strategy | Idea |
|---|---|
| `topic_trend` | rank the 52-topic taxonomy by trend, write ideas for the top clusters |
| `predictor_llm` | prompt an LLM with recent abstracts (*Direct* in the paper) |
| `summary_prompting` | prompt over summarised recent work |
| `retrieval_prompting` | retrieval-augmented prompting |
| `memory_prompting` | prompting with a running memory |

All five need both keys: `topic_trend` picks its clusters arithmetically but still asks a
model to write the predictions. **Matching is embedding-only.** Scores from different
matchers are not comparable, so there is no flag to change the matcher.

## Training MDF

MDF is trained in two stages, a supervised prior and a GRPO-trained realization
policy, then evaluated through the same `benchmark` and `judge-eval` path as every
baseline. The commands below are the full pipeline; `scripts/run_train_and_eval.sh`
chains steps 3 to 5 and skips any stage whose checkpoint already exists.

### Step 0 — GPU environment

```bash
conda create -n idea-forecast-bench-train python=3.11 -y
conda activate idea-forecast-bench-train
poetry install --with forecaster
# Poetry resolves torch from PyPI, which may not match your CUDA driver.
# Reinstall it from the PyTorch index for your CUDA version (cu124 shown):
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
pip install vllm          # optional; USE_VLLM=1 makes GRPO rollouts ~10x faster
python -c "import torch; assert torch.cuda.is_available()"
```

### Step 1 — Hindsight labels

Extract a latent innovation (base direction, operator, gap) from each future paper
in the fixed training episodes. Uses `gpt-4o` by default.

```bash
export OPENAI_API_KEY="your-openai-key"
idea-forecast-bench hindsight --input-dir /path/to/corpus \
  --output-dir data/topic_hindsight --mode full
# -> data/topic_hindsight/hindsight_samples.jsonl
```

### Step 2 — Foresight reward artifacts

The GRPO reward retrieves from a per-cutoff future index and judges against a
per-topic rubric. Build both (one GPU for the embedder; the rubric step calls a
judge, either the OpenAI default or a local endpoint via `--judge-base-url`):

```bash
python examples/forecaster/build_indices.py --papers-dir /path/to/corpus \
  --hindsight data/topic_hindsight/hindsight_samples.jsonl \
  --art output/foresight_artifacts
python examples/forecaster/build_rubrics.py --mode live \
  --rubrics-dir output/foresight_artifacts/rubrics
```

To skip this step, train with `--trainer-config grpo_train_legacy.yaml`, a
fixed-weight composite reward that needs no artifacts.

### Step 3 — Prior SFT

```bash
idea-forecast-bench train-prior --model qwen2.5-7b-instruct \
  --hindsight data/topic_hindsight/hindsight_samples.jsonl \
  --output-dir output/mdf/prior_sft
# -> output/mdf/prior_sft/final_checkpoint (LoRA adapter)
```

### Step 4 — Realization GRPO

Warm-starts from the prior adapter. The foresight judge is `gpt-4o` unless
`JUDGE_BASE_URL` points at a local endpoint (`scripts/benchmark/serve_vllm.sh`);
`FORESIGHT_JUDGE_WORKERS` bounds its concurrency.

```bash
USE_VLLM=1 idea-forecast-bench train --model-preset qwen2.5-7b-instruct \
  --input-dir /path/to/corpus \
  --hindsight data/topic_hindsight/hindsight_samples.jsonl \
  --init-policy-path output/mdf/prior_sft/final_checkpoint \
  --trainer grpo --trainer-config grpo_train.yaml \
  --skip-alignment-check \
  --output-dir output/mdf/realization_grpo
# -> output/mdf/realization_grpo/grpo/checkpoints/final_checkpoint
```

`--skip-alignment-check` is required when the validation and test windows start in
the same month, which is the paper's configuration.

### Step 5 — Evaluate

Unset `JUDGE_BASE_URL` first if step 4 set it, or the judge is silently redirected.

```bash
idea-forecast-bench benchmark --strategy forecaster \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --prior-checkpoint output/mdf/prior_sft/final_checkpoint \
  --realization-checkpoint output/mdf/realization_grpo/grpo/checkpoints/final_checkpoint \
  --input-dir /path/to/corpus --skip-matching \
  --output output/backtest/forecaster.json
idea-forecast-bench judge-eval --input-json output/backtest/forecaster.json \
  --papers-dir /path/to/corpus --output output/judged/forecaster.judged.json
```

## Reproducing the paper's sweep

A full sweep is 624 windows per (strategy, backbone), which is too slow for one
process: the work between API calls is GIL-bound, so the run is sharded by topic
across processes. Two scripts do this with the paper's settings as defaults:

```bash
# 1. generate: 5 strategies x 4 topic shards, predictions only (--skip-matching)
OPENAI_API_KEY=... bash scripts/benchmark/run_sharded_backtest.sh

# 2. judge: one judge-eval process per shard, each with its own state file
OPENAI_API_KEY=... VOYAGE_API_KEY=... bash scripts/benchmark/run_sharded_judge.sh

# 3. table
idea-forecast-bench main-table --source "gpt-4.1=output/sharded/judged/*.judged.json"
```

To run a local backbone or the Qwen3.5-9B judge, serve it with
`scripts/benchmark/serve_vllm.sh` and point `OPENAI_BASE_URL` at it. Two settings
fail silently when wrong. The served model name decides routing: generation only
goes to `OPENAI_BASE_URL` for names starting with `gpt-4o`, `gpt-4.1` or `gpt-5`, so
serve a local backbone under an alias such as `gpt-4o-qwen7b` and pass the judge's
own name with `--judge-model`. The context length must be at least 16384, because
the client requests 4096 output tokens and the longest prompts are about 4100.

```bash
SERVED_NAMES="gpt-4o-qwen7b" bash scripts/benchmark/serve_vllm.sh /models/Qwen2.5-7B-Instruct 31000
```

Before judging, make sure none of `JUDGE_BASE_URL`, `JUDGE_MODEL`, `OPENAI_BASE_URL`
or `VOYAGE_BASE_URL` are left exported from another run. Any of them silently
redirects scoring to a different model, and nothing errors. The judge script unsets
them and takes the judge by flag instead.

Concurrency, corpus-fingerprint caching, and the silent failure modes met while
running the sweep are documented in [docs/running-at-scale.md](docs/running-at-scale.md).

## Package structure

```
idea_forecast_bench/            # the benchmark
├── backtest.py             # rolling-window backtest loop
├── papers.py, paper_cache.py  # corpus loading and caching
├── similarity.py           # embedding matcher
├── strategy/               # the five baselines + the MDF wrapper
├── judge/                  # retrieve-then-judge protocol
└── __main__.py             # CLI entry point (dispatches into examples/)

forecaster/                 # MDF: Mode-Decomposition Forecaster
├── hindsight/              # latent-innovation label extraction
├── prior/                  # memory-conditioned prior (SFT)
├── realization/            # realization policy (GRPO)
└── foresight/              # future-grounded reward, indices, rubrics

examples/                   # Python entry scripts behind each CLI command
scripts/                    # sharded sweep launchers, vLLM serving, MDF pipeline wrapper
config/                     # YAML config, including the 52-topic taxonomy
docs/                       # running-at-scale notes
```

<details>
<summary><b>All commands</b></summary>

`idea-forecast-bench <cmd> --help` shows any command's own flags.

| Command | What it does |
|---|---|
| `fetch` | Download an arXiv corpus the benchmark can read |
| `baselines` | Score every baseline on one corpus, print a comparison table |
| `benchmark` | Backtest a single forecasting strategy |
| `judge-eval` | Score saved predictions with the retrieve-then-judge judge |
| `main-table` | Assemble the main results table from judged artifacts |
| `hindsight` | Extract latent-innovation training labels |
| `train-prior` | SFT the memory-conditioned innovation prior |
| `train` | GRPO-train the realization policy |
| `infer` | Joint inference: prior → realize → select |
| `analysis` | Evaluation-validity analyses (citation / coauthor / leakage) |

</details>

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the CI
gate, and commit conventions.

## Citation

```bibtex
@misc{li2026ideaforecastbench,
  title         = {Can Large Language Models Forecast What Researchers Study Next?},
  author        = {Fenghai Li and Zihan Tang and Haofei Yu and Yining Zhao and Jiaxuan You},
  year          = {2026},
  eprint        = {2609.00747},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.00747}
}
```

## License

MIT — see [LICENSE](LICENSE).
