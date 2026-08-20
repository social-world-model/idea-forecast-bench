<h1 align="center">LiveIdeaBench</h1>

<p align="center">
  <b>Can a model read the literature up to a cutoff and forecast what the field does next?</b>
</p>

<p align="center">
  <a href="https://github.com/ulab-uiuc/live-idea-bench/actions/workflows/ci.yml"><img src="https://github.com/ulab-uiuc/live-idea-bench/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python 3.10-3.12">
</p>

This repo contains two things:

- **A benchmark.** Given only papers published before a cutoff, a system produces a ranked
  list of research ideas. Those are scored against the papers that actually appeared
  afterwards, under a reproducible **retrieve-then-judge** protocol.
- **A reference method — MDF**, the *Mode-Decomposition Forecaster*. It samples a latent
  innovation from a memory-conditioned prior, realizes it into a grounded proposal, and
  trains the realization policy with GRPO against a future-grounded reward.

## Install

```bash
git clone https://github.com/ulab-uiuc/live-idea-bench.git && cd live-idea-bench
poetry install
```

| Extra | Command | When |
|---|---|---|
| — | `poetry install` | run the benchmark and the baselines |
| forecaster | `poetry install --with forecaster` | train or run MDF locally (torch, trl, peft) |
| webapp | `poetry install --with webapp` | the optional Flask API under `backend/` |

> Install CUDA-matched torch from `scripts/setup_rl_env*.sh`, not from
> `poetry install --with forecaster` — the default index can resolve a wheel that leaves
> `torch.cuda.is_available()` False.

## Quick start

Every baseline writes its predictions with an LLM, and matching is
embedding-based, so two keys are needed:

```bash
export VOYAGE_API_KEY=...     # matching (embedding-only)
export OPENAI_API_KEY=...     # generation; ANTHROPIC_/GOOGLE_/TOGETHER_/DEEPSEEK_ also work

live-idea-bench fetch        # pull an arXiv corpus into data/csml/raw_markdown
live-idea-bench baselines    # score every baseline, print a comparison table
```

`baselines` checks both before it starts, so a missing key costs a second
rather than five failed runs. To point generation at a local
OpenAI-compatible server instead of a provider, set `OPENAI_BASE_URL`.

`baselines` echoes the settings every strategy shared, then one row each:

```text
strategy                windows        hit_at_k   precision_at_k             mrr
-------------------------------------------------------------------------------
topic_trend                 ...             ...             ...             ...
summary_prompting           ...             ...             ...             ...
```

Read `windows` first. A strategy that produced none is reported as
`NOT SCORED -- no windows produced` and the command exits non-zero, so "could not run"
is never mistaken for "scored 0.0". Fix it by widening `fetch --lookback-days` or
lowering `--min-train-papers`.

No numbers are quoted here: `fetch` pulls whatever arXiv serves today, which is not the
frozen corpus behind the paper.

**Every flag has a default** — the commands above are complete as written.

```bash
live-idea-bench fetch --query "cat:cs.CL" --max-results 5000
live-idea-bench baselines --start-month 2024-06 --end-month 2025-06 --top-k 10
live-idea-bench benchmark --strategy summary_prompting     # a single strategy
```

## Baselines

`baselines` runs all five under identical windows and an identical matcher, so
the rows are comparable:

| Strategy | Idea |
|---|---|
| `topic_trend` | rank the 52-topic taxonomy by trend, write ideas for the top clusters |
| `predictor_llm` | prompt an LLM with recent abstracts |
| `summary_prompting` | prompt over summarised recent work |
| `retrieval_prompting` | retrieval-augmented prompting |
| `memory_prompting` | prompting with a running memory |

All five need both keys. There is no LLM-free baseline: `topic_trend` picks its
clusters arithmetically but still asks a model to write the predictions.

MDF is not in this table because it needs trained checkpoints; run it with
`live-idea-bench benchmark --strategy forecaster`. Use `--only` for a subset:

```bash
live-idea-bench baselines --only topic_trend,summary_prompting
```

### Checking the pipeline without provider keys

Both sides accept an OpenAI-compatible endpoint, so the whole path -- corpus →
windows → generation → embedding match → metrics -- can be exercised against a
local server (vLLM, SGLang, or any stub) with no provider account:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1  OPENAI_API_KEY=EMPTY
export VOYAGE_BASE_URL=http://127.0.0.1:8000/v1  VOYAGE_API_KEY=EMPTY
live-idea-bench baselines --only topic_trend
```

This verifies wiring, not quality: `hit@k` against a stub is meaningless, and
only `windows` reaching a non-zero value tells you the run was real.

**Matching is embedding-only** — there is no `--similarity-engine`. Scores from
different matchers are not comparable, so the choice was removed rather than
left as a flag a typo could change.

## Training MDF

```bash
live-idea-bench hindsight     # extract latent-innovation labels from future papers
live-idea-bench train-prior   # SFT the memory-conditioned prior
live-idea-bench train         # GRPO-train the realization policy
live-idea-bench infer         # prior → realize → select
```

`scripts/run_train_and_eval.sh` chains these end to end. The GRPO step defaults to the
gated foresight reward and needs prebuilt artifacts — see
[forecaster/foresight/README.md](forecaster/foresight/README.md) for that sequence, or use
`REWARD_MODE=legacy` to run the whole pipeline without them.

<details>
<summary><b>All commands</b></summary>

`live-idea-bench <cmd> --help` shows any command's own flags.

| Command | What it does |
|---|---|
| `fetch` | Download an arXiv corpus the benchmark can read |
| `baselines` | Score every baseline on one corpus, print a comparison table |
| `benchmark` | Backtest a single forecasting strategy |
| `judge-eval` | Score saved predictions with the retrieve-then-judge judge |
| `hindsight` | Extract latent-innovation training labels |
| `train-prior` | SFT the memory-conditioned innovation prior |
| `train` | GRPO-train the realization policy |
| `infer` | Joint inference: prior → realize → select |
| `eval` | Evaluate a trained forecaster on a held-out window |
| `ablate` | Single-metric GRPO (soft / coverage / novelty) |
| `analysis` | Evaluation-validity analyses (citation / coauthor / leakage) |

</details>

## Citation

<!-- TODO: replace with the published reference once the paper is out. -->
The paper is not yet public. If you use LiveIdeaBench before then, please link to this
repository.

## License

MIT — see [LICENSE](LICENSE).
