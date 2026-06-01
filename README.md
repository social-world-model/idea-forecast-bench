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

The benchmark runs over the arXiv CS.ML corpus. Place the paper markdown under
`data/csml/raw_markdown/` in the layout `load_papers_from_markdown` expects
(`<paper_id>/auto/<paper_id>.md`), then point `--input-dir` at it:

```bash
python -m live_idea_bench benchmark \
  --input-dir data/csml/raw_markdown \
  --strategy summary_prompting \
  --start-month 2024-10 --end-month 2025-03 \
  --output /tmp/backtest.json
```

This uses the default `heuristic` matcher to decide whether a forecast hit a
future paper — no API key needed. To score matches with an LLM judge instead,
add `--similarity-engine llm` and name the judge with `--eval-model` (the
`--eval-model` value is ignored unless the engine is `llm`):

```bash
python -m live_idea_bench benchmark \
  --input-dir data/csml/raw_markdown \
  --strategy summary_prompting \
  --similarity-engine llm --eval-model gpt-5.4 \
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
  recent NVIDIA GPU recommended for non-dry-run training. **Install torch from the setup
  scripts, not from `poetry install`** — `poetry install --with forecaster` pulls a
  default-index torch whose CUDA build may not match your driver (it can resolve to a
  too-new wheel and leave `torch.cuda.is_available()` False). `scripts/setup_rl_env.sh`,
  `scripts/setup_rl_env_qwen3_5.sh`, and `scripts/setup_new_machine.sh` pin the correct
  `+cuXXX` wheel via `--index-url` for your GPU. See also `NEW_MACHINE_SETUP.md`.
- **`--with eval`**: a local sentence-transformer embedder so the retrieve-then-judge step
  works without a hosted embedding API.

API keys (set as needed for the providers you use): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY`, `VOYAGE_API_KEY`.

---

## Development & reproduction

- **Tests:** `pytest` (green on a core install).
- **MDF training pipeline:** `scripts/run_train_and_eval.sh` runs prior SFT → GRPO →
  eval end to end. The GRPO step defaults to the gated foresight reward used for the
  reported results, which needs a prebuilt artifact dir (`output/foresight_artifacts/{indices,rubrics}`):
  provide a paper corpus, run the hindsight pipeline to produce `data/topic_hindsight/dz.jsonl`,
  build the indices with `build_indices.py`, then generate validated rubrics
  (`forecaster/foresight/README.md` has the full sequence). If those artifacts are
  missing the script stops before training with the build instructions. To run the
  whole pipeline end to end on a fresh clone without that prerequisite, use the
  fixed-weight composite reward instead: `REWARD_MODE=legacy bash scripts/run_train_and_eval.sh`.
- **Single-metric ablation:** `scripts/forecaster/run_three_grpo.sh` drives the
  single-metric GRPO runs (soft / coverage / novelty); it uses its own reward and
  needs no foresight artifacts. Phase-by-phase smoke checks for the foresight method
  are in `scripts/phase*_*.py` (documented in `forecaster/foresight/README.md`).
- **Web app:** `python backend/app.py` (API) and `cd frontend && npm install && npm start` (UI).

## Citation

If you use LiveIdeaBench, please cite the paper (see the manuscript repository).
