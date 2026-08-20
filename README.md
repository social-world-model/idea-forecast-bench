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
> `torch.cuda.is_available()` False.

## Quick start

Matching is embedding-based, so set a Voyage key first:

```bash
export VOYAGE_API_KEY=...

live-idea-bench fetch        # pull an arXiv corpus into data/csml/raw_markdown
live-idea-bench baselines    # score the baselines, print a comparison table
```

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

| Strategy | Key? | Idea |
|---|---|---|
| `topic_trend` | no | extrapolate rising keywords |
| `topic_trend` | no LLM | extrapolates the 52-topic taxonomy |
| `predictor_llm` | yes | prompt an LLM with recent abstracts |
| `summary_prompting` | yes | prompt over summarised recent work |
| `retrieval_prompting` | yes | retrieval-augmented prompting |
| `memory_prompting` | yes | prompting with a running memory |
| `forecaster` | yes + checkpoints | the MDF method |

`baselines` runs `topic_trend` by default — it needs no *LLM* provider, though
scoring still needs `VOYAGE_API_KEY`. Add the LLM baselines with a provider key:

```bash
export OPENAI_API_KEY=sk-...
live-idea-bench baselines --include-llm
```

**Matching is embedding-only.** There is no `--similarity-engine`: scores from
different matchers are not comparable, and making the matcher selectable meant
a typo could silently produce numbers that looked fine but could not be
compared to anything.

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
