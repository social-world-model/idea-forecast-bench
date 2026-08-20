# Contributing

## Setup

```bash
git clone https://github.com/ulab-uiuc/live-idea-bench.git
cd live-idea-bench
poetry install --with dev          # core + tooling
pre-commit install                      # lint on commit
pre-commit install --hook-type commit-msg
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

`poetry install` installs `live_idea_bench` and `forecaster` in **editable**
mode. That is deliberate: several modules resolve `config/`, `data/` and
`examples/` relative to their own `__file__`, which only works while the
packages live inside the repo. Do not publish this as a wheel without moving
those assets into the packages first — `pyproject.toml` has the details.

Optional groups:

| Group | Adds | When |
|---|---|---|
| `--with forecaster` | torch, transformers, trl, peft, datasets, accelerate | training/running MDF locally |
| `--with eval` | sentence-transformers | local embedder for retrieve-then-judge |
| `--with webapp` | flask, flask-cors | running `backend/app.py` |

Install CUDA-correct torch from `scripts/setup_rl_env*.sh`, not from
`poetry install --with forecaster` — the default index can resolve a wheel
that leaves `torch.cuda.is_available()` False.

## What CI checks

One workflow, `CI`, with three jobs. Reproduce all of them locally:

```bash
pre-commit run --all-files    # lint: ruff check, ruff format --check, codespell, file hygiene
mypy                          # typecheck: strict, over live_idea_bench/ forecaster/ backend/
python -c "import live_idea_bench, forecaster, backend.app"   # smoke
live-idea-bench --help
```

Two rules the tooling enforces that are easy to trip over:

- **Lint never rewrites your code.** `[tool.ruff]` deliberately has no
  `fix = true`; use `ruff check --fix` yourself when you want it. (CI used to
  auto-fix its own checkout and then report success.)
- **CI must not need network or API keys.** Nothing in the gate may reach a
  paid endpoint.

**There is no automated test suite.** It was removed deliberately; the CI
`smoke` job only proves the import graph, the CLI entry point, its dispatch
table, and config loading still work. Nothing verifies that the benchmark
computes the right numbers — so changes to evaluation logic need careful
manual review, and `git log` before commit 79b434b has the deleted suite if
you want to bring parts of it back.

## Layout

| Directory | Contains |
|---|---|
| `live_idea_bench/` | the benchmark package + evaluation protocol |
| `forecaster/` | the MDF forecaster |
| `examples/` | **Python** entry scripts (the CLI dispatches into these) |
| `scripts/` | **shell** wrappers for those entry scripts |
| `config/` | YAML config |
| `backend/`, `frontend/` | optional web app, not part of the CI gate |

`examples/` holds Python, `scripts/` holds bash. Keep it that way — a shell
wrapper belongs next to its siblings in `scripts/`, and the Python it invokes
belongs in `examples/`.

## Commits and PRs

Conventional commits, enforced by the `commitizen` pre-commit hook:

```
<type>: <description>

<body explaining WHY, not what the diff already shows>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`,
`build`, `style`.

Branch names: `type/description`, e.g. `feature/add-llm-agents`.

Keep a PR to one type of change. Several small PRs beat one large one.

## Changes that move published numbers

`live_idea_bench/similarity.py`, `live_idea_bench/backtest.py` and `config/`
determine the reported results. A change to any of them makes previously
reported numbers non-reproducible. Say so explicitly in the PR description —
`CODEOWNERS` routes these for review.
