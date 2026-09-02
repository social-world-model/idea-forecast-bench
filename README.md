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

The Python package and CLI are named `live-idea-bench`, the project's working name.

## Installation

```bash
# create an environment (conda shown; any Python >=3.10,<3.13 works)
conda create -n live-idea-bench python=3.11 -y
conda activate live-idea-bench

# install Poetry if you do not have it
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

# install from source
git clone https://github.com/ulab-uiuc/live-idea-bench.git
cd live-idea-bench
poetry install
```

`poetry install` is enough to run the benchmark and every baseline. Two optional groups
add more:

| Group | Command | When you need it |
|---|---|---|
| forecaster | `poetry install --with forecaster` | train or run MDF locally (torch, trl, peft) |
| webapp | `poetry install --with webapp` | the optional Flask API under `backend/` |

> For GPU training, install a CUDA-matched torch with `scripts/setup_rl_env*.sh`
> rather than through Poetry: the default index can resolve a wheel that leaves
> `torch.cuda.is_available()` False.

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
live-idea-bench fetch                                   # into data/csml/raw_markdown
live-idea-bench fetch --query "cat:cs.CL" --max-results 5000
```

A fresh corpus is useful for checking that the pipeline works, but it will not reproduce
the paper's numbers; the frozen corpus will.

### 2. Run the baselines

```bash
live-idea-bench baselines --input-dir /path/to/corpus
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
live-idea-bench baselines --only topic_trend,summary_prompting     # a subset
live-idea-bench baselines --start-month 2024-06 --end-month 2025-06 --top-k 10
live-idea-bench benchmark --strategy summary_prompting             # one strategy
live-idea-bench benchmark --strategy summary_prompting --model-name Qwen/Qwen2.5-7B-Instruct
```

### 3. Score with the retrieve-then-judge protocol

The paper's reported numbers come from the LLM judge, which re-embeds the papers,
retrieves candidates for each prediction, and scores each pair on problem, method, and
specificity:

```bash
live-idea-bench judge-eval \
  --input-json output/backtest/summary_prompting.json \
  --papers-dir /path/to/corpus \
  --output output/judged/summary_prompting.judged.json
```

The judge defaults to `gpt-4.1-mini`; pass `--judge-model` and `--judge-base-url` to use
another one. When the judge supplies your numbers, run `benchmark` with
`--skip-matching` to skip the embedding match it would otherwise duplicate. Assemble the
final table from judged artifacts with `live-idea-bench main-table`.

### Checking the wiring without provider keys

Both generation and matching accept an OpenAI-compatible endpoint, so the whole path can
be exercised against a local server or stub with no provider account:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1  OPENAI_API_KEY=EMPTY
export VOYAGE_BASE_URL=http://127.0.0.1:8000/v1  VOYAGE_API_KEY=EMPTY
live-idea-bench baselines --only topic_trend
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

MDF needs trained checkpoints, so it is run with `benchmark --strategy forecaster` rather
than through `baselines`. Install the `forecaster` group, then:

```bash
live-idea-bench hindsight     # extract latent-innovation labels from future papers
live-idea-bench train-prior   # SFT the memory-conditioned prior
live-idea-bench train         # GRPO-train the realization policy
live-idea-bench infer         # prior → realize → select
```

`scripts/run_train_and_eval.sh` chains these end to end. The GRPO step defaults to the
gated foresight reward and needs prebuilt artifacts; see
[forecaster/foresight/README.md](forecaster/foresight/README.md) for that sequence, or
set `REWARD_MODE=legacy` to run the pipeline without them.

## Running at scale

A full sweep is 624 windows per (strategy, backbone). Sharding with `--topics`,
concurrency against self-hosted judges, corpus-fingerprint caching, and the silent
failure modes we hit along the way are documented in
[docs/running-at-scale.md](docs/running-at-scale.md).

## Package structure

```
live_idea_bench/            # the benchmark
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
scripts/                    # shell wrappers and GPU environment setup
config/                     # YAML config, including the 52-topic taxonomy
docs/                       # running-at-scale notes
backend/, frontend/         # optional web app
```

<details>
<summary><b>All commands</b></summary>

`live-idea-bench <cmd> --help` shows any command's own flags.

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
| `eval` | Evaluate a trained forecaster on a held-out window |
| `ablate` | Single-metric GRPO (soft / coverage / novelty) |
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
