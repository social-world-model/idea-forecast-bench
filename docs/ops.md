# Operations

## Deployment Model

LiveIdeaBench is intended to run on a single EC2 host with Docker Compose.

- Public entrypoint: Nginx in the `frontend` container on port `80`
- Internal API: Gunicorn serving the Flask app on port `5000`
- Daily scheduler: host `cron` invoking the backend container
- Persistent state: host `data/` mounted into `/app/data`

## First-Time Setup

1. Install Docker Engine and the Docker Compose plugin on the EC2 host.
2. Clone the repo onto the instance.
3. Export the required environment variables in the shell, systemd unit, or secret manager.
4. Ensure `data/` exists on the host and is writable by Docker.

## Environment Variables

Required:

- `OPENAI_API_KEY` when using OpenAI-backed strategies
- `GOOGLE_API_KEY` when using Gemini-backed strategies
- `LIVE_IDEA_ADMIN_TOKEN` to allow protected write endpoints in production

Optional:

- `ANTHROPIC_API_KEY`
- `LIVE_IDEA_BOOTSTRAP_BACKTEST`
- `LIVE_IDEA_ARXIV_QUERY`
- `LIVE_IDEA_ARXIV_MAX_RESULTS`
- `LIVE_IDEA_ARXIV_LOOKBACK_DAYS`
- `LIVE_IDEA_CORS_ORIGINS`
- `GUNICORN_WORKERS`
- `GUNICORN_THREADS`
- `GUNICORN_TIMEOUT`
- `GUNICORN_GRACEFUL_TIMEOUT`
- `GUNICORN_KEEPALIVE`
- `GUNICORN_LOG_LEVEL`

## Build And Start

```bash
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
export LIVE_IDEA_ADMIN_TOKEN=...
docker compose build
docker compose up -d
docker compose ps
```

Verify:

```bash
curl http://localhost/healthz
curl http://localhost/metrics
```

Notes:

- The backend container now runs behind Gunicorn.
- Keep `GUNICORN_WORKERS=1` unless you first move strategy state out of local files.
- `frontend` waits for `backend` health before accepting traffic.

## Daily Pipeline

Run once manually:

```bash
docker compose exec -T backend python examples/run_daily_pipeline.py
```

Recommended host cron:

```cron
0 0 * * * cd /path/to/live-idea-bench && docker compose exec -T backend python examples/run_daily_pipeline.py >> /var/log/live-idea-bench-daily.log 2>&1
```

## Runtime State

- `data/strategies/`: persisted strategy JSON records
- `data/views.json`: UI view counter
- `data/daily_runs/latest.json`: latest daily pipeline report
- `data/daily_runs/*.json`: historical daily run reports

Back up the entire `data/` directory if you need to preserve state.

## Recovery

- Restart services: `docker compose restart`
- Rebuild after dependency or image changes: `docker compose up -d --build`
- Re-run one daily cycle manually with the command above
- If the backend container is rebuilt, mounted files in `data/` remain the source of truth
