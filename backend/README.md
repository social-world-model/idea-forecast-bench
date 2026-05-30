# Backend Runbook

The Flask API for `live-idea-bench` — serves stored forecasting strategies, runs
backtests/generation on request, and the daily ingest/eval pipeline. This is a
**deployment/serving** component only; model training lives in the main README
(`python -m live_idea_bench`) and `forecaster/`, not here.

## What lives here

- `backend/app.py` — Flask API entrypoint
- `backend/strategy_store.py` — JSON-backed strategy persistence under `backend/strategies/`
- `backend/services/daily_pipeline.py` — daily ingest/eval/generate pipeline
- `backend/services/arxiv_ingest.py` — arXiv ingestion service
- `backend/requirements.txt` — backend-only dependencies (no `torch` / ML stack)
- `backend/Dockerfile` — lightweight API image

## 1. Environment setup

```bash
conda create -n live-idea-bench python=3.11 -y
conda activate live-idea-bench
pip install -r backend/requirements.txt
```

`backend/requirements.txt` is intentionally backend-only — it does not install
`torch` or the local training stack, so the API runtime (and Docker image) stays
lightweight.

## 2. Environment variables

```bash
export LIVE_IDEA_ADMIN_TOKEN="change-me"
export LIVE_IDEA_BENCH_DATA_DIR="$(pwd)/data/csml/raw_markdown"
export PORT=5000
export FLASK_DEBUG=0
```

Optional:

- `LIVE_IDEA_CORS_ORIGINS` — comma-separated frontend origins (defaults include `localhost:3000`, `localhost:5173`)
- `LIVE_IDEA_BOOTSTRAP_BACKTEST=0` — disable the startup backtest bootstrap
- `LIVE_IDEA_PIPELINE_LOCK_TTL_SECONDS` — lock timeout for the daily pipeline

Model API keys (only for API-backed strategies):

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
```

## 3. Start the backend

```bash
python backend/app.py        # listens on http://localhost:5000 by default
```

Quick checks:

```bash
curl http://localhost:5000/healthz
curl http://localhost:5000/metrics
curl http://localhost:5000/api/strategies
```

## 4. Strategy API basics

Write endpoints require the admin token in non-test environments:
`-H "X-Live-Idea-Admin-Token: $LIVE_IDEA_ADMIN_TOKEN"`.

Create a keyword baseline:

```bash
curl -X POST http://localhost:5000/api/strategies \
  -H "Content-Type: application/json" \
  -H "X-Live-Idea-Admin-Token: $LIVE_IDEA_ADMIN_TOKEN" \
  -d '{
    "strategy_name": "keyword_trend",
    "config": {"top_k": 5, "horizon_months": 3, "end_month": "2024-09"}
  }'
```

Create an LLM predictor strategy:

```bash
curl -X POST http://localhost:5000/api/strategies \
  -H "Content-Type: application/json" \
  -H "X-Live-Idea-Admin-Token: $LIVE_IDEA_ADMIN_TOKEN" \
  -d '{
    "strategy_name": "predictor_llm",
    "params": {"model_name": "gpt-4o-mini", "predictor_config": "predictor.yaml", "similarity_config": "similarity.yaml", "temperature": 0.7},
    "config": {"top_k": 5, "horizon_months": 3, "end_month": "2024-09"}
  }'
```

Run a backtest / generation for one strategy:

```bash
curl -X POST http://localhost:5000/api/strategies/<strategy_id>/backtest \
  -H "X-Live-Idea-Admin-Token: $LIVE_IDEA_ADMIN_TOKEN"

curl -X POST http://localhost:5000/api/strategies/<strategy_id>/generate \
  -H "Content-Type: application/json" \
  -H "X-Live-Idea-Admin-Token: $LIVE_IDEA_ADMIN_TOKEN" \
  -d '{"cutoff_date": "2024-06-01"}'
```

Check status:

```bash
curl http://localhost:5000/api/strategies/<strategy_id>/status
curl http://localhost:5000/api/strategies/<strategy_id>
```

## 5. Daily pipeline

```bash
python -c "from backend.services.daily_pipeline import run_daily_pipeline; print(run_daily_pipeline())"
```

Ingests fresh arXiv papers, scores yesterday's generation, updates leaderboard
state, and generates the next cutoff.

## 6. Docker

```bash
docker build -t live-idea-bench-backend -f backend/Dockerfile .
docker run --rm -p 5000:5000 \
  -e LIVE_IDEA_ADMIN_TOKEN=change-me \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  live-idea-bench-backend
```

## 7. Default paths

- Paper directory: `data/csml/raw_markdown`
- Strategy store: `backend/strategies/`
- Backend port: `5000`

> Training a forecaster (SFT prior + GRPO realization) is **not** a backend
> concern — see the top-level `README.md` (`python -m live_idea_bench train`) and
> `forecaster/`.
