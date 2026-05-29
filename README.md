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

```bash
# 1. Clone
git clone https://github.com/ulab-uiuc/live-idea-bench.git
cd live-idea-bench

# 2. Install (core = everything the benchmark needs)
poetry install                       # deps for the benchmark + test suite
#   ...add extras only when you train/run the forecaster locally:
poetry install --with forecaster     # MDF forecaster training stack (torch/transformers/trl/peft/...)
poetry install --with eval           # local-embedder for retrieve-then-judge scoring
#   The repo runs in-place from the root via `python -m live_idea_bench` — no
#   editable install needed (the package is poetry-managed, package-mode=false).

# 3. Run — one front door:
python -m live_idea_bench --help
```

The CLI is the single entrypoint. Every command forwards its flags to the underlying
script, so `python -m live_idea_bench <cmd> --help` shows that command's options.

| Area | Command | What it does |
|------|---------|--------------|
| **Benchmark** | `python -m live_idea_bench benchmark`   | Run a domain-separated backtest of a forecasting strategy |
|               | `python -m live_idea_bench judge-eval`  | Score saved predictions with the retrieve-then-judge LLM judge |
| **MDF forecaster** | `python -m live_idea_bench hindsight`   | Extract latent-innovation training labels from future papers |
|                    | `python -m live_idea_bench train-prior` | SFT the memory-conditioned innovation prior |
|                    | `python -m live_idea_bench train`       | GRPO-train the realization policy |
|                    | `python -m live_idea_bench infer`       | Joint inference: sample from the prior → realize → select |
|                    | `python -m live_idea_bench eval`        | Evaluate a trained forecaster on a held-out window |
| **Single-metric ablation** | `python -m live_idea_bench ablate` | Single-metric GRPO (soft / coverage / novelty) |
| **Analysis** | `python -m live_idea_bench analysis`    | Evaluation-validity analyses (citation / coauthor / leakage) |

### Minimal example — run the benchmark

First get the arXiv CS.ML corpus (downloads to `data/csml_v2/raw_markdown`):

```bash
bash scripts/forecaster/download_dataset.sh
```

Then run a backtest against it:

```bash
python -m live_idea_bench benchmark \
  --input-dir data/csml_v2/raw_markdown \
  --strategy summary_prompting \
  --eval-model gpt-5.4 \
  --start-month 2024-10 --end-month 2025-03 \
  --output /tmp/backtest.json
```

Available baseline strategies: `predictor_llm` (raw recent-abstract prompting),
`summary_prompting`, `retrieval_prompting`, `memory_prompting`, and `keyword_trend` /
`topic_trend`. The MDF forecaster is the `forecaster` strategy.

---

## Repository layout

```text
live_idea_bench/      # Core package: benchmark + evaluation protocol
  __main__.py         #   the `python -m live_idea_bench` CLI front door
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
examples/             # Entrypoint scripts (the CLI dispatches to these)
  benchmark/  forecaster/  analysis/  data/
scripts/              # Shell wrappers + dev/reproduction helpers
config/               # YAML configs (config/, config/forecaster/)
tests/                # Test suite (pytest) — green out of the box
backend/  frontend/   # Optional web app (Flask API + React UI)
deploy/  docs/        # Deployment manifests + ops notes
```

---

## Installation detail

- **Core** (`poetry install`): runs the benchmark and the LLM-API baselines, and the
  retrieve-then-judge protocol against a hosted judge/embedding API. The full test suite
  (`pytest`) passes on a core install.
- **`--with forecaster`**: the local training/inference stack for the MDF method —
  `torch, transformers, trl, peft, datasets, accelerate, sentence-transformers`. Linux + a
  recent NVIDIA GPU recommended for non-dry-run training. See `NEW_MACHINE_SETUP.md` and
  `scripts/setup_rl_env.sh` for a turnkey training environment.
- **`--with eval`**: a local sentence-transformer embedder so the retrieve-then-judge step
  works without a hosted embedding API.

API keys (set as needed for the providers you use): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY`, `VOYAGE_API_KEY`.

---

## Development & reproduction

- **Tests:** `pytest` (green on a core install).
- **MDF training pipeline:** `scripts/run_train_and_eval.sh` runs prior SFT → GRPO →
  eval end to end; `scripts/forecaster/run_three_grpo.sh` drives the single-metric
  ablation (soft / coverage / novelty). Phase-by-phase smoke checks for the foresight method
  are in `scripts/phase*_*.py` (documented in `forecaster/foresight/README.md`).
- **Web app:** `python backend/app.py` (API) and `cd frontend && npm install && npm start` (UI).

## Citation

If you use LiveIdeaBench, please cite the paper (see the manuscript repository).
