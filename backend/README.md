# Backend Runbook

This directory contains the Flask API for `live-idea-bench`, plus the backend dependency file and the separate RL setup flow.

## What lives here

- `backend/app.py`: Flask API entrypoint
- `backend/strategy_store.py`: JSON-backed strategy persistence under `backend/strategies/`
- `backend/requirements.txt`: backend-only dependencies
- `scripts/setup_rl_a100_env.sh`: installs the separate RL stack for Linux + A100
- `backend/services/daily_pipeline.py`: daily ingest/eval/generate pipeline logic

## 1. Environment setup

Backend environment from the repo root:

```bash
conda create -n live-idea-bench python=3.11 -y
conda activate live-idea-bench
pip install -r backend/requirements.txt
```

Notes:

- `backend/requirements.txt` is intentionally backend-only; it does not install `torch`, `verl`, `trl`, or the local RL stack
- The backend Docker image uses this file, so the default API runtime stays lightweight

Separate RL environment for Linux + A100:

```bash
conda create -n live-idea-bench-rl python=3.11 -y
conda activate live-idea-bench-rl
bash scripts/setup_rl_a100_env.sh
```

RL notes:

- `scripts/setup_rl_a100_env.sh` installs the dedicated veRL / TRL training stack for Linux + A100
- It installs `torch==2.6.0+cu124` by default
- `bitsandbytes` and `vllm` stay Linux-only

## 2. Required and useful environment variables

Common variables:

```bash
export LIVE_IDEA_ADMIN_TOKEN="change-me"
export LIVE_IDEA_BENCH_DATA_DIR="$(pwd)/data/arxiv_csml/raw_markdown"
export PORT=5000
export FLASK_DEBUG=0
```

Optional variables:

- `LIVE_IDEA_CORS_ORIGINS`: comma-separated frontend origins; defaults include `localhost:3000` and `localhost:5173`
- `LIVE_IDEA_BOOTSTRAP_BACKTEST=0`: disables startup backtest bootstrap
- `LIVE_IDEA_PIPELINE_LOCK_TTL_SECONDS`: lock timeout for the daily pipeline

Model API keys are only needed for API-backed strategies:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
```

## 3. Start the backend

From the repo root:

```bash
python backend/app.py
```

The backend listens on `http://localhost:5000` by default.

Quick checks:

```bash
curl http://localhost:5000/healthz
curl http://localhost:5000/metrics
curl http://localhost:5000/api/strategies
```

## 4. Strategy API basics

All write endpoints require the admin token in non-test environments:

```bash
-H "X-Live-Idea-Admin-Token: $LIVE_IDEA_ADMIN_TOKEN"
```

Create a simple keyword baseline:

```bash
curl -X POST http://localhost:5000/api/strategies \
  -H "Content-Type: application/json" \
  -H "X-Live-Idea-Admin-Token: $LIVE_IDEA_ADMIN_TOKEN" \
  -d '{
    "strategy_name": "keyword_trend",
    "config": {
      "top_k": 5,
      "horizon_months": 3,
      "end_month": "2024-09"
    }
  }'
```

Create an LLM predictor strategy:

```bash
curl -X POST http://localhost:5000/api/strategies \
  -H "Content-Type: application/json" \
  -H "X-Live-Idea-Admin-Token: $LIVE_IDEA_ADMIN_TOKEN" \
  -d '{
    "strategy_name": "predictor_llm",
    "params": {
      "model_name": "gpt-4o-mini",
      "predictor_config": "predictor.yaml",
      "similarity_config": "similarity.yaml",
      "temperature": 0.7
    },
    "config": {
      "top_k": 5,
      "horizon_months": 3,
      "end_month": "2024-09"
    }
  }'
```

Create an RL policy strategy from a trained manifest:

```bash
curl -X POST http://localhost:5000/api/strategies \
  -H "Content-Type: application/json" \
  -H "X-Live-Idea-Admin-Token: $LIVE_IDEA_ADMIN_TOKEN" \
  -d '{
    "strategy_name": "policy_rl",
    "params": {
      "policy_manifest_path": "/absolute/path/to/policy_manifest.json",
      "predictor_config": "predictor.yaml",
      "similarity_config": "similarity.yaml"
    },
    "config": {
      "top_k": 5,
      "horizon_months": 3,
      "end_month": "2024-09"
    }
  }'
```

Run backtest for one strategy:

```bash
curl -X POST http://localhost:5000/api/strategies/<strategy_id>/backtest \
  -H "X-Live-Idea-Admin-Token: $LIVE_IDEA_ADMIN_TOKEN"
```

Run generation for one cutoff:

```bash
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

## 5. Offline CLI commands

The repo also includes two useful offline commands.

Generate/backtest without the API:

```bash
bash scripts/research_idea_engine.sh \
  --input-dir data/arxiv_csml/raw_markdown \
  --strategy predictor_llm \
  --model-name gpt-4o-mini \
  backtest \
  --horizon-months 3 \
  --output /tmp/backtest.json
```

Use a trained RL policy manifest with the same CLI:

```bash
bash scripts/research_idea_engine.sh \
  --input-dir data/arxiv_csml/raw_markdown \
  --strategy policy_rl \
  --policy-manifest-path /absolute/path/to/policy_manifest.json \
  generate \
  --cutoff-month 2024-06 \
  --output /tmp/policy_rl_generate.json
```

## 6. RL training workflow

Run the RL workflow inside the separate RL environment.

List the built-in small-model presets:

```bash
python examples/run_policy_rl_training.py --list-model-presets
```

Recommended first choices:

- `qwen2.5-3b-instruct`
- `qwen3-4b-instruct-2507`
- `llama3.2-3b-instruct` as a comparison baseline

Prepare RL artifacts only:

```bash
bash scripts/run_policy_rl_training.sh \
  --input-dir data/arxiv_csml/raw_markdown \
  --model-preset qwen3-4b-instruct-2507 \
  --trainer grpo \
  --prepare-only \
  --output-dir data/rl_runs/qwen3_4b_prepare
```

Run PPO training:

```bash
bash scripts/run_policy_rl_training.sh \
  --input-dir data/arxiv_csml/raw_markdown \
  --model-preset qwen2.5-3b-instruct \
  --trainer ppo \
  --output-dir data/rl_runs/qwen25_3b_ppo
```

Run GRPO training:

```bash
bash scripts/run_policy_rl_training.sh \
  --input-dir data/arxiv_csml/raw_markdown \
  --model-preset qwen3-4b-instruct-2507 \
  --trainer grpo \
  --output-dir data/rl_runs/qwen3_4b_grpo
```

Important notes:

- Training CLI no longer exposes `--split`; non-prepare runs always train on the `train` split
- Use `--prepare-only --prepare-split validation|test|all` if you need non-train artifacts for inspection
- Default RL episode config keeps all data through `2025-12` in the training split and assigns `2026-01` onward to validation
- `ppo` and `grpo` share the veRL online-RL backend and train against the existing rule-based reward callback
- `rloo` remains on the legacy TRL path for now; its alignment gate still reads validation episodes automatically
- `policy_manifest.json` under the trainer output is what the `policy_rl` strategy consumes later

## 7. Daily pipeline

If you want to run the daily ingest/eval/generate flow directly:

```bash
python -c "from backend.services.daily_pipeline import run_daily_pipeline; print(run_daily_pipeline())"
```

This will ingest fresh arXiv papers, score yesterday's generation, update leaderboard state, and generate the next cutoff.

## 8. Docker

Build and run:

```bash
docker build -t live-idea-bench-backend -f backend/Dockerfile .
docker run --rm -p 5000:5000 \
  -e LIVE_IDEA_ADMIN_TOKEN=change-me \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  live-idea-bench-backend
```

## 9. Default paths

- Default paper directory: `data/arxiv_csml/raw_markdown`
- Default strategy store: `backend/strategies/`
- Default backend port: `5000`
- Default RL output example: `data/rl_runs/...`
