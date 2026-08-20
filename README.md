# LiveIdeaBench

**Forecasting emerging research ideas against future literature.**

LiveIdeaBench asks whether a model can read the machine-learning literature up to a
cutoff and forecast the research ideas the community pursues next. It ships two things:

1. **A benchmark** — a temporally grounded evaluation of *research idea forecasting*: given
   only papers before a cutoff, a system produces a ranked list of ideas, scored against the
   papers that actually appear afterward under a reproducible **retrieve-then-judge** protocol.
2. **A reference forecaster (MDF)** — the *Mode-Decomposition Forecaster*: it predicts a latent
   innovation from a memory-conditioned prior and realizes it into a grounded proposal, with the
   realization policy trained by GRPO against a future-grounded reward.

---

## Quick start

Three commands, no API key, nothing to download by hand:

```bash
git clone https://github.com/ulab-uiuc/live-idea-bench.git && cd live-idea-bench
poetry install

live-idea-bench fetch                  # pull an arXiv corpus into data/csml/raw_markdown
live-idea-bench baselines              # run every keyless baseline, print a comparison table
```

It echoes the settings every strategy shared (so the rows are comparable), then
one row per baseline:

```text
Shared settings (identical for every baseline, so the rows compare):
  corpus            data/csml/raw_markdown
  window            2024-01 .. 2025-06
  horizon_months    3
  top_k             5
  similarity_engine heuristic

strategy                windows        hit_at_k     recall_at_k             mrr
-------------------------------------------------------------------------------
keyword_trend               ...             ...             ...             ...
topic_trend                 ...             ...             ...             ...
```

Absolute values depend entirely on the corpus you fetched, so none are quoted
here — `fetch` pulls whatever arXiv returns today, which is not the frozen
corpus behind the paper's numbers.

`windows` is how many scored backtest windows the corpus supported, and it is
the first thing to read. A strategy that produced none is reported as

```text
predictor_llm                 0   NOT SCORED -- no windows produced
```

rather than a row of zeros, and the command exits non-zero — "could not be
scored" must never be mistakable for "scored 0.0". Usual causes are a corpus
too thin for the window (widen `fetch --lookback-days`, or lower
`--min-train-papers`) or a missing API key, which the command also warns about
before it starts. A small keyless smoke corpus of ~800 papers gives
`keyword_trend` a few dozen windows but typically leaves `topic_trend`
unscored, because the 52-topic taxonomy needs denser per-topic coverage.

**Every flag has a default.** The commands above are complete as written; reach
for flags only to change something:

```bash
live-idea-bench fetch --query "cat:cs.CL" --max-results 5000
live-idea-bench baselines --start-month 2024-06 --end-month 2025-06 --top-k 10
live-idea-bench benchmark --strategy summary_prompting      # one strategy, not all
```

### Baselines

| Strategy | Needs a key? | What it does |
|---|---|---|
| `keyword_trend` | no | extrapolates rising keywords |
| `topic_trend` | no | same, over the 52-topic taxonomy |
| `predictor_llm` | yes | prompts an LLM with recent abstracts |
| `summary_prompting` | yes | prompts over summarised recent work |
| `retrieval_prompting` | yes | retrieval-augmented prompting |
| `memory_prompting` | yes | prompting with a running memory |
| `forecaster` | yes + checkpoints | the MDF method |

`baselines` runs the keyless two by default. Add the LLM ones once a provider
key is set:

```bash
export OPENAI_API_KEY=sk-...
live-idea-bench baselines --include-llm
live-idea-bench baselines --only summary_prompting,retrieval_prompting
```

### Scoring engines

How a prediction is matched to a future paper. `heuristic` is the default
because it needs no credentials; the reported results use `embedding`.

| `--similarity-engine` | Needs | Notes |
|---|---|---|
| `heuristic` | nothing | lexical matcher; use for smoke runs |
| `embedding` | `VOYAGE_API_KEY` | Voyage-only by design, no silent fallback |
| `llm` | a judge key | pair with `--eval-model`, e.g. `--eval-model gpt-5.4` |

Scores from different engines are not comparable — pick one per experiment.

### All commands

`live-idea-bench <cmd> --help` shows any command's own flags.

| Area | Command | What it does |
|------|---------|--------------|
| **Benchmark** | `fetch` | Download an arXiv corpus the benchmark can read |
| | `baselines` | Run every baseline on one corpus, print a comparison table |
| | `benchmark` | Backtest a single forecasting strategy |
| | `judge-eval` | Score saved predictions with the retrieve-then-judge LLM judge |
| **MDF forecaster** | `hindsight` | Extract latent-innovation training labels from future papers |
| | `train-prior` | SFT the memory-conditioned innovation prior |
| | `train` | GRPO-train the realization policy |
| | `infer` | Joint inference: prior → realize → select |
| | `eval` | Evaluate a trained forecaster on a held-out window |
| **Ablation** | `ablate` | Single-metric GRPO (soft / coverage / novelty) |
| **Analysis** | `analysis` | Evaluation-validity analyses (citation / coauthor / leakage) |

Optional installs, only when you need them:

```bash
poetry install --with forecaster   # MDF training stack (torch/transformers/trl/peft/...)
poetry install --with eval         # local embedder for retrieve-then-judge
poetry install --with webapp       # the Flask API under backend/
```

## Repository layout

```text
live_idea_bench/      # Core package: benchmark + evaluation protocol
  __main__.py         #   the `live-idea-bench` / `python -m live_idea_bench` CLI
  backtest.py         #   rolling/domain backtest runner
  similarity.py       #   retrieve-then-judge evaluation
  strategy/           #   pluggable forecasting strategies (the baselines + MDF)
  prompt/             #   predictor / similarity prompts
forecaster/           # The MDF forecaster
  hindsight/          #   latent-innovation label extraction
  prior/              #   memory-conditioned innovation prior (SFT)
  realization/        #   GRPO-trained realization policy (trl backend)
  foresight/          #   future-grounded reward, soft must_not judge, rubric, indices
  inference/          #   joint inference (Algorithm 1)
examples/             # Python entry scripts (the CLI dispatches into these)
  benchmark/          #   backtests + retrieve-then-judge + validity analyses
  forecaster/         #   MDF training/inference, index building, phase smoke checks
  data/               #   corpus prep
scripts/              # Shell wrappers for the above, plus environment setup
  benchmark/          #   benchmark run wrappers
  forecaster/         #   training / serving / eval wrappers
config/               # YAML configs (config/, config/forecaster/)
docs/                 # Setup notes and runbooks
backend/  frontend/   # Optional web app (Flask API + React UI), not gated by CI
```

`examples/` holds Python and `scripts/` holds shell — see CONTRIBUTING.md.
For how the two packages relate and why there are two environments, see
[docs/architecture.md](docs/architecture.md).

---

## Installation detail

- **Core** (`poetry install`): runs the benchmark and the LLM-API baselines, and the
  retrieve-then-judge protocol against a hosted judge/embedding API. CI's smoke job
  verifies this configuration on Python 3.10 and 3.12.
  `poetry install` puts `live_idea_bench` and `forecaster` on the path in editable mode,
  so `import live_idea_bench` works from any directory. Editable is required, not
  incidental: several modules resolve `config/` and `examples/` relative to their own
  `__file__` (see the note in `pyproject.toml`).
- **`--with forecaster`**: the local training/inference stack for the MDF method —
  `torch, transformers, trl, peft, datasets, accelerate, sentence-transformers`. Linux + a
  recent NVIDIA GPU recommended for non-dry-run training. **Install torch from the setup
  scripts, not from `poetry install`** — `poetry install --with forecaster` pulls a
  default-index torch whose CUDA build may not match your driver (it can resolve to a
  too-new wheel and leave `torch.cuda.is_available()` False). `scripts/setup_rl_env.sh`,
  `scripts/setup_rl_env_qwen3_5.sh`, and `scripts/setup_new_machine.sh` pin the correct
  `+cuXXX` wheel via `--index-url` for your GPU. See also `docs/new-machine-setup.md`.
- **`--with eval`**: a local sentence-transformer embedder so the retrieve-then-judge step
  works without a hosted embedding API.

API keys (set as needed for the providers you use): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY`, `VOYAGE_API_KEY`.

---

## Development & reproduction

- **Checks:** `pre-commit run --all-files` (lint) and `mypy` (types) — the two
  gating jobs in CI, plus a smoke job. There is no automated test suite; see
  CONTRIBUTING.md.
- **MDF training pipeline:** `scripts/run_train_and_eval.sh` runs prior SFT → GRPO →
  eval end to end. The GRPO step defaults to the gated foresight reward used for the
  reported results, which needs a prebuilt artifact dir (`output/foresight_artifacts/{indices,rubrics}`):
  provide a paper corpus, run the hindsight pipeline to produce `data/topic_hindsight/dz.jsonl`,
  build the indices with `examples/forecaster/build_indices.py`, then generate validated rubrics
  (`forecaster/foresight/README.md` has the full sequence). If those artifacts are
  missing the script stops before training with the build instructions. To run the
  whole pipeline end to end on a fresh clone without that prerequisite, use the
  fixed-weight composite reward instead: `REWARD_MODE=legacy bash scripts/run_train_and_eval.sh`.
- **Single-metric ablation:** `scripts/forecaster/run_three_grpo.sh` drives the
  single-metric GRPO runs (soft / coverage / novelty); it uses its own reward and
  needs no foresight artifacts. Phase-by-phase smoke checks for the foresight method
  are in `examples/forecaster/phase*_*.py` (documented in `forecaster/foresight/README.md`).
- **Web app (optional):** `poetry install --with webapp && python backend/app.py` (API) and
  `cd frontend && npm install && npm start` (UI). Not part of the CI gate; see `backend/README.md`.

## Citation

If you use LiveIdeaBench, please cite the paper (see the manuscript repository).
