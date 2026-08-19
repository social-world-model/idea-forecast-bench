# Architecture

## The two things in this repo

```
                    ┌─────────────────────────────────────┐
                    │  live_idea_bench/   THE BENCHMARK   │
                    │                                     │
   papers ────────► │  papers.py    corpus loading        │ ────► scores
   (markdown)       │  backtest.py  rolling/domain runner │
                    │  similarity.py retrieve-then-judge  │
                    │  strategy/    IdeaStrategy interface│
                    └──────────────▲──────────────────────┘
                                   │ implements IdeaStrategy
                    ┌──────────────┴──────────────────────┐
                    │  forecaster/        THE METHOD (MDF)│
                    │                                     │
                    │  hindsight/   extract latent z      │
                    │  prior/       p(z | memory)   (SFT) │
                    │  realization/ p(y | z, X)    (GRPO) │
                    │  foresight/   future-grounded reward│
                    │  inference/   Algorithm 1: joint    │
                    └─────────────────────────────────────┘
```

`live_idea_bench` is the benchmark: it defines what a forecast is, how it is
scored, and the `IdeaStrategy` interface a forecaster must implement.
`forecaster` is one such forecaster — the Mode-Decomposition Forecaster the
paper proposes. The benchmark evaluates baselines and MDF through the same
interface.

## Dependency direction

**`forecaster` → `live_idea_bench`, one way.** The method depends on the
benchmark's data model and scoring; the benchmark does not depend on the
method.

The one place that looks like an exception is
`live_idea_bench/strategy/{forecaster,policy_rl}.py` — the adapters that let
MDF be selected as a strategy. They import `forecaster` **inside functions**,
never at module scope, so the packages do not form an import cycle. Keep it
that way: a module-scope `import forecaster` anywhere under
`live_idea_bench/` reintroduces the cycle.

```
backend/  ──►  live_idea_bench/  ◄──  forecaster/
                                 ◄──  examples/
```

`backend/` (the optional Flask API) depends only on `live_idea_bench`.
Nothing depends on `backend/` or `frontend/`.

## Where code lives

| Path | Role | Importable? |
|---|---|---|
| `live_idea_bench/` | benchmark package | yes |
| `forecaster/` | MDF method package | yes |
| `backend/` | optional Flask API | yes |
| `examples/` | **Python** entry scripts | no — run by path or via the CLI |
| `scripts/` | **shell** wrappers + env setup | no |
| `config/` | YAML: runtime defaults, taxonomy, MDF hyperparameters | — |

`examples/` and `scripts/` are entry points, not libraries. Nothing under
`live_idea_bench/` or `forecaster/` may import from them. The split is strict:
Python goes in `examples/`, shell goes in `scripts/`.

## The CLI

`live-idea-bench <command>` (or `python -m live_idea_bench <command>`) is the
single front door. `live_idea_bench/__main__.py` holds a dispatch table
mapping each command to a script under `examples/`, and executes it with
`runpy` after rewriting `sys.argv`, so each script's own `argparse` keeps
working.

Consequence: the CLI resolves `examples/` relative to its own `__file__`.
That works because the packages are installed **editable** (see below). Add a
command by adding a script under `examples/` and an entry to `_COMMANDS`; the
CI smoke job verifies every entry resolves.

## Environments

Two, for one reason: CUDA.

**1. Poetry — the default, and the only one CI uses.**

```bash
poetry install                       # benchmark + evaluation protocol
poetry install --with forecaster     # + the training stack
poetry install --with eval           # + local sentence-transformer embedder
poetry install --with webapp         # + Flask, for backend/
poetry install --with dev,test       # + tooling
```

`pyproject.toml` and `poetry.lock` are the single source of truth for
dependencies. There is no `requirements.txt`, no `setup.py`, no `uv.lock`.

`poetry install` installs both packages in **editable** mode, and that is
load-bearing rather than incidental: eight call sites resolve repo-root assets
that live outside the packages (`config/`, `data/`, `examples/`) via
`Path(__file__).resolve().parents[N]`. A built wheel would relocate the
packages into site-packages and every one of those paths would silently point
at nothing. Do not `poetry build` and ship this without first moving those
assets inside the packages.

**2. conda + pip, via `scripts/setup_*.sh` — GPU training boxes only.**

Poetry cannot pin a CUDA-specific torch build (`torch==2.x+cu124` comes from
`--index-url https://download.pytorch.org/whl/cu124`, not PyPI), and the
training stack additionally needs vllm, flash-attn and unsloth wheels matched
to the driver. Those scripts create a conda env and pip-install that stack
directly.

**These two do not agree, by design and by accident.** `poetry.lock` and the
setup scripts resolve different versions of torch, transformers, datasets and
numpy. Treat the setup scripts as authoritative on a GPU box and poetry as
authoritative everywhere else. If you change a training dependency, change it
in both.

## Configuration

| File | Read by |
|---|---|
| `config/config.yaml` | `live_idea_bench.config.load_runtime_config` — model/embedding defaults |
| `config/topics_v2.yaml` | the 52-topic taxonomy, **the only copy**; `config.yaml` points at it via `topics_file:` |
| `config/operators.yaml` | `forecaster.foresight.operators` |
| `config/forecaster/*.yaml` | one file per MDF stage (hindsight, prior, realization, reward, selection, inference, grpo_train) |

Credentials are read from the environment only — `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `VOYAGE_API_KEY`,
`LIVE_IDEA_ADMIN_TOKEN`. No config file has a slot for a key.

## Known rough edges

Honest list; none of these are hidden behind a passing check.

- **No automated test suite.** CI's `smoke` job proves the import graph, the
  CLI and its dispatch table, and config loading. Nothing verifies that the
  benchmark computes correct numbers.
- **`config/forecaster/reward.yaml`'s `weights:` block is inert.** It parses
  into `RewardConfig.weights` and no field is ever read.
- **Two 800+ line modules**: `forecaster/realization/pipeline.py` and
  `forecaster/orchestrator.py`.
- **Four overlapping reward modules** under `forecaster/`: `realization/reward.py`,
  `realization/realization_reward.py`, `realization/judge_rewards.py`,
  `foresight/reward.py`.
- **`forecaster/realization/verl/`** is named after a training backend the
  project does not use; the trl runner imports its dataset helper.
- **`frontend/`** is on deprecated `react-scripts` and is not a CI gate.
