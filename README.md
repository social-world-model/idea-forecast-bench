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
| eval | `poetry install --with eval` | local embedder instead of a hosted one |
| webapp | `poetry install --with webapp` | the optional Flask API under `backend/` |

> Install CUDA-matched torch from `scripts/setup_rl_env*.sh`, not from
> `poetry install --with forecaster` — the default index can resolve a wheel that leaves
> `torch.cuda.is_available()` False. See [docs/new-machine-setup.md](docs/new-machine-setup.md).

## Quick start

No API key, nothing to download by hand:

```bash
live-idea-bench fetch        # pull an arXiv corpus into data/csml/raw_markdown
live-idea-bench baselines    # score every keyless baseline, print a comparison table
```

`baselines` echoes the settings every strategy shared, then one row each:

```text
strategy                windows        hit_at_k     recall_at_k             mrr
-------------------------------------------------------------------------------
keyword_trend               ...             ...             ...             ...
topic_trend                 ...             ...             ...             ...
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

| Strategy | Key? | Idea |
|---|---|---|
| `keyword_trend` | no | extrapolate rising keywords |
| `topic_trend` | no | same, over the 52-topic taxonomy |
| `predictor_llm` | yes | prompt an LLM with recent abstracts |
| `summary_prompting` | yes | prompt over summarised recent work |
| `retrieval_prompting` | yes | retrieval-augmented prompting |
| `memory_prompting` | yes | prompting with a running memory |
| `forecaster` | yes + checkpoints | the MDF method |

`baselines` runs the keyless two by default. Add the rest once a provider key is set:

```bash
export OPENAI_API_KEY=sk-...
live-idea-bench baselines --include-llm
```

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
<summary><b>Scoring engines</b> — how a prediction is matched to a future paper</summary>

| `--similarity-engine` | Needs | Notes |
|---|---|---|
| `heuristic` | nothing | lexical matcher; the default, for smoke runs |
| `embedding` | `VOYAGE_API_KEY` | Voyage-only by design, no silent fallback |
| `llm` | a judge key | pair with `--eval-model`, e.g. `--eval-model gpt-5.4` |

Scores from different engines are not comparable — pick one per experiment. The reported
results use `embedding`.

</details>

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

## Documentation

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | how the packages fit together, and the two environments |
| [CONTRIBUTING.md](CONTRIBUTING.md) | setup, what CI checks, commit conventions |
| [forecaster/foresight/README.md](forecaster/foresight/README.md) | the foresight reward: indices, rubrics, smoke checks |
| [docs/](docs/) | runbooks and the optional web app's API |

## Citation

<!-- TODO: replace with the published reference once the paper is out. -->
The paper is not yet public. If you use LiveIdeaBench before then, please link to this
repository.

## License

MIT — see [LICENSE](LICENSE).
