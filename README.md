# Live Idea Bench

A strategy-driven framework for research idea generation and rolling backtest.

## `src` Structure

```text
src/
├── __init__.py              # Unified package exports
├── configs.py               # Config loading
├── matching.py              # Similarity engines and matcher
├── prompting.py             # LLM prompting helpers
├── types.py                 # Shared core data types
├── utils.py                 # File/JSON/common utilities
├── strategy/                # Strategy abstraction + implementations
│   ├── base.py
│   ├── keyword_trend.py
│   └── registry.py
└── backtest/                # Data loading, evaluation, rolling backtest
    ├── data.py
    ├── evaluator.py
    ├── models.py
    └── runner.py
```

## Simplest Usage

Generate ideas at one cutoff month:

```bash
python scripts/research_idea_engine.py \
  --input-dir data/arxiv_csml/raw_markdown \
  --strategy keyword_trend \
  --top-k 5 \
  generate \
  --cutoff-month 2025-06 \
  --output /tmp/idea_generate.json
```

Run rolling backtest:

```bash
python scripts/research_idea_engine.py \
  --input-dir data/arxiv_csml/raw_markdown \
  --strategy keyword_trend \
  --top-k 5 \
  backtest \
  --horizon-months 3 \
  --min-train-papers 4 \
  --output /tmp/idea_backtest.json
```

## Backend / Frontend Run

Backend (Flask):

```bash
python backend/app.py
```

Frontend (React):

```bash
cd frontend
npm install
npm start
```
