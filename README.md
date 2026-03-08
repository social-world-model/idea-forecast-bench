# LiveIdeaBench

LiveIdeaBench is a website for comparing research-idea generation strategies on a shared paper stream.

The runtime mainline is:

`bootstrap backtest once if missing -> daily ingest -> daily evaluation -> next generation -> backend API -> frontend`

## Official Deployment Target

This repo is organized for a single AWS EC2 host running Docker Compose:

- `frontend`: Nginx serving the React build and proxying `/api`, `/healthz`, and `/metrics`
- `backend`: Gunicorn serving the Flask API plus bootstrap and generation logic
- `host cron`: triggers the daily pipeline inside the backend container
- `data/`: mounted into the backend container so strategy state, views, and daily artifacts persist across restarts

Detailed EC2 operations live in [docs/ops.md](docs/ops.md).

## Supported Model Families

The backend can execute strategies against these model families:

- `gpt-4o*`, `gpt-5*` via OpenAI
- `claude-*` via Anthropic
- `*gemini*` via Google
- `deepseek*` via Together's OpenAI-compatible API
- `qwen*` via DashScope's OpenAI-compatible API
- `kimi*`, `moonshot*` via Moonshot's OpenAI-compatible API
- `grok*` via xAI's OpenAI-compatible API
- `llama*`, `meta-llama/*` via Together's OpenAI-compatible API

For DeepSeek on Together, prefer Together model IDs such as `deepseek-ai/DeepSeek-V3.1` and `deepseek-ai/DeepSeek-R1`. The backend also maps `deepseek-chat` and `deepseek-reasoner` onto those Together IDs. Export only the provider API keys that match the strategies you plan to run. The full environment variable list lives in [docs/ops.md](docs/ops.md).

## Repo Layout

```text
backend/           Flask API and file-backed strategy store
config/            Runtime defaults
data/              Mounted runtime state and local paper corpus
docs/              Operating notes
examples/          Python entrypoints for backtest and daily jobs
frontend/          React app and Nginx image
live_idea_bench/   Core ingest, prediction, similarity, and backtest package
scripts/           Bash wrappers around the supported examples
tests/             Automated tests
```

## Runtime Semantics

- Historical backtests are month-based and primarily used to establish a baseline leaderboard.
- Backend startup only auto-runs backtest when no local strategy record has a non-empty `backtest_result`.
- The daily pipeline ingests the latest arXiv papers, evaluates the previous generation against new papers, updates leaderboard state, and generates the next cutoff.
- Runtime state is stored under `data/`:
  - `data/strategies/` for persisted strategy records
  - `data/views.json` for the landing-page counter
  - `data/daily_runs/` for daily pipeline reports

## Local Development

### Backend

```bash
poetry install --with test
python backend/app.py
```

Backend default URL: `http://localhost:5000`

Production Docker runs the backend under Gunicorn with a single worker by default because runtime state is still file-backed.

### Frontend

```bash
cd frontend
npm install
npm start
```

The frontend uses same-origin `/api` calls by default. In local development, the CRA dev server proxies them to `http://localhost:5000`.

## Supported CLI Workflows

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

Run a monthly backtest:

```bash
bash scripts/research_idea_engine.sh \
  --input-dir data/arxiv_csml/raw_markdown \
  --strategy predictor_llm \
  --model-name gpt-4o \
  backtest \
  --horizon-months 3 \
  --output /tmp/idea_backtest.json
```

Run the daily pipeline manually:

```bash
bash scripts/run_daily_pipeline.sh
```

## API Surface

- `GET /healthz`
- `GET /metrics`
- `GET /api/strategies`
- `POST /api/strategies`
- `GET /api/strategies/<strategy_id>`
- `GET /api/strategies/<strategy_id>/status`
- `POST /api/strategies/<strategy_id>/backtest`
- `POST /api/strategies/<strategy_id>/generate`
- `GET /api/views`
- `POST /api/views`
