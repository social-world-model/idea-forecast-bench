# Live Idea Bench

Live Idea Bench is a strategy-driven research idea system with three connected capabilities:

1. Define strategy variants.
2. Run backtests over time windows.
3. Generate ideas and evaluate quality/performance.

The goal is to compare strategies side-by-side and expose their metrics in backend APIs and frontend dashboards.

## What Is In Place

- Backtest orchestrator:
  - [/Users/yuhaofei/Downloads/live-idea-bench-backtest-orchestrator/src/runner.py](/Users/yuhaofei/Downloads/live-idea-bench-backtest-orchestrator/src/runner.py)
  - [/Users/yuhaofei/Downloads/live-idea-bench-backtest-orchestrator/scripts/run_backtest.py](/Users/yuhaofei/Downloads/live-idea-bench-backtest-orchestrator/scripts/run_backtest.py)
- Idea generation backend endpoints:
  - `/api/generate-ideas`
  - `/api/research-ideas`
- Frontend idea leaderboard/dashboard.

## Strategy-First Workflow

Use this lifecycle for each strategy (`baseline`, `novelty_boost`, `cost_aware`, etc.):

1. Strategy Definition
- Define strategy id/name and its config knobs (model, prompt mode, filters, ranking weights, budget caps).

2. Backtest Execution
- Run strategy across rolling windows.
- Persist per-window artifacts and resumable state.

3. Generation
- Produce ideas per window (or for latest window).

4. Evaluation
- Score output quality and operational performance.
- Aggregate metrics by strategy.

5. Product Surface
- Backend returns strategy-level metrics and idea-level records.
- Frontend compares strategies and shows idea overview.

## Backtest Orchestrator

Run rolling windows with resumable artifacts.

```bash
python scripts/run_backtest.py \
  --start 2401 \
  --end 2406 \
  --window-months 3 \
  --step-months 1 \
  --command-template "python scripts/predict_ideas.py --start {window_start_yymm} --end {window_end_yymm} --output {window_dir}/predictions.json" \
  --artifacts-dir data/backtests/baseline \
  --resume
```

Supported time inputs:
- `YYMM` (example: `2401`)
- `YYYY-MM` (example: `2024-01`)

Command template placeholders:
- `{window_start}` / `{window_end}` (`YYYY-MM`)
- `{window_start_yymm}` / `{window_end_yymm}` (`YYMM`)
- `{window_id}` / `{window_index}`
- `{window_dir}` / `{artifacts_dir}`

Artifacts:
- `manifest.json`: run-level setup
- `state.json`: resumable window state
- `windows/<window_id>/metadata.json`
- `windows/<window_id>/stdout.log`
- `windows/<window_id>/stderr.log`

## Strategy Metrics Contract (Recommended)

To support “multi-strategy comparison” in backend and frontend, use a standard output contract per strategy run:

- `strategy_summary.json`
- `idea_overview.json`

Example `strategy_summary.json` fields:

```json
{
  "strategy_id": "baseline",
  "run_id": "2026-02-25_baseline",
  "windows_total": 12,
  "windows_completed": 12,
  "success_rate": 1.0,
  "avg_runtime_sec": 18.4,
  "ideas_generated": 240,
  "avg_idea_score": 7.8,
  "avg_novelty": 7.4,
  "avg_feasibility": 8.1,
  "api_cost_usd": 42.15
}
```

Example `idea_overview.json` fields:

```json
{
  "strategy_id": "baseline",
  "top_ideas": [
    {
      "id": "idea_001",
      "title": "...",
      "impact_score": 8.9,
      "novelty": 8.7,
      "feasibility": 8.3,
      "window": "2024-01_to_2024-03",
      "tags": ["LLM", "Evaluation"]
    }
  ]
}
```

## Backend/Frontend Display Targets

Backend should provide:
- Strategy comparison endpoint: per-strategy KPI table.
- Idea overview endpoint: top ideas, trend over windows, score distribution.

Frontend should show:
- Strategy comparison panel:
  - success rate, avg score, avg novelty, avg feasibility, runtime, cost
- Strategy trend charts:
  - per-window score/cost/runtime trends
- Idea overview board:
  - top ideas by strategy and global top ideas

## Existing API Notes

Current backend already serves generated ideas:
- `GET /api/generate-ideas`
- `POST /api/generate-ideas`
- `GET /api/research-ideas`

Current frontend already renders idea list and ranking via `ResearchIdea`.
The strategy comparison APIs/views can be added on top of the metrics contract above.

## Dev Commands

Run tests:

```bash
pytest -q
```

Run backend:

```bash
python backend/app.py
```

Run frontend:

```bash
cd frontend
npm install
npm start
```

## Next Integration Steps

1. Add strategy config loader (YAML/JSON) and strategy registry.
2. Make generation script emit standardized evaluation metrics per window.
3. Add backend endpoints for strategy summaries and trend series.
4. Extend frontend dashboard with multi-strategy comparison + idea overview tabs.
