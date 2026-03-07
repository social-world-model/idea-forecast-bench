# LiveIdeaBench

LiveIdeaBench is a research-idea forecasting benchmark. The clean mainline is:

`ingest -> predictor.yaml -> strategy generation -> similarity.yaml evaluation -> monthly backtest -> daily pipeline -> backend API -> frontend`

## Repo Layout

```text
assets/          # README/site assets
backend/         # Flask API, JSON persistence, startup bootstrap
config/          # Runtime defaults
data/            # Paper markdown, strategy JSON, daily artifacts
deploy/          # Deployment manifests
docs/            # Ops and API notes
examples/        # Python example entrypoints
frontend/        # React UI
live_idea_bench/   # Core package
scripts/         # Bash wrappers for examples
tests/           # Automated tests
```

## Core Package

```text
live_idea_bench/
  config.py      # config/config.yaml + live_idea_bench/prompt/*.yaml loading
  models.py      # PaperRecord / IdeaPrediction / EvaluationResult / MatchResult
  papers.py      # markdown parsing, dates, JSON/file helpers
  llm.py         # model client and request adapters
  predictor.py   # predictor.yaml-driven idea generation
  similarity.py  # similarity.yaml-driven evaluation
  backtest.py    # month-level backtest and window runner
  daily.py       # daily validation + next-generation helpers
  ingest.py      # arXiv ingest helpers
  strategy/
    base.py
    predictor_llm.py
    keyword_trend.py
    execution.py
    registry.py
  prompt/
    predictor.yaml
    similarity.yaml
```

## Main Workflows

Generate one cutoff:

```bash
bash scripts/research_idea_engine.sh \
  --input-dir data/arxiv_csml/raw_markdown \
  --strategy predictor_llm \
  --model-name gpt-4o \
  generate \
  --cutoff-month 2025-06 \
  --output /tmp/idea_generate.json
```

Run monthly backtest:

```bash
bash scripts/research_idea_engine.sh \
  --input-dir data/arxiv_csml/raw_markdown \
  --strategy predictor_llm \
  --model-name gpt-4o \
  backtest \
  --horizon-months 3 \
  --output /tmp/idea_backtest.json
```

Start backend:

```bash
python backend/app.py
```

Start frontend:

```bash
cd frontend
npm install
npm start
```

## Runtime Semantics

- Historical `backtest` is month-based.
- `daily pipeline` evaluates yesterday's generation against newly ingested papers, updates leaderboard state, then generates the next cutoff.
- Backend startup only auto-runs backtest if no local strategy file contains a non-empty `backtest_result`.
