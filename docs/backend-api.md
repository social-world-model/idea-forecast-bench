# Backend API

Base URL: `http://localhost:5000`

## Health and Monitoring

- `GET /healthz`
  - Returns service status.
- `GET /metrics`
  - Returns strategy, backtest, and generation counters.

## Strategies

- `GET /api/strategies`
  - Returns persisted strategy configurations plus any topic-split backtest, generation, and daily evaluation payloads.
- `POST /api/strategies`
  - Creates a new strategy configuration.
- `GET /api/strategies/<strategy_id>`
  - Returns one strategy record.
- `GET /api/strategies/<strategy_id>/status`
  - Returns current `backtest_status` and `generation_status`.
- `POST /api/strategies/<strategy_id>/backtest`
  - Runs historical month-level backtest for that strategy.
- `POST /api/strategies/<strategy_id>/generate`
  - Runs one forward-looking generation at the provided cutoff date.

Startup bootstrap rule:
- If any local strategy already has non-empty topic backtest data in `topic_runs` or legacy `backtest_result`, backend startup skips automatic backtesting.
- If no strategy has backtest data yet, startup may run one automatic backtest pass.

## Topic Runs

Strategies now persist topic-aware execution under `topic_runs`.

```json
{
  "id": "abcd1234",
  "topic_runs": [
    {
      "topic_id": "optimizer",
      "topic_name": "Optimizer",
      "matched_paper_count": 42,
      "generation_status": "done",
      "backtest_status": "done",
      "generation": {
        "cutoff_date": "2026-03-01",
        "cutoff_month": "2026-03",
        "predictions": []
      },
      "backtest_result": {
        "summary": {
          "windows": 3,
          "avg_hit_at_k": 0.42
        },
        "windows": []
      }
    }
  ]
}
```

Notes:
- Topic taxonomy comes from `config/config.yaml`.
- Each run fans out across all configured topics.
- Top-level `generation` and `backtest_result` remain for legacy compatibility but topic-aware clients should read `topic_runs`.

## Views

- `GET /api/views`
  - Returns current page view count.
- `POST /api/views`
  - Increments and returns page view count.

## Error Model

All API errors return:

```json
{
  "error": {
    "code": "bad_request",
    "message": "n must be a positive integer",
    "details": null
  },
  "timestamp": "2026-02-25T00:00:00+00:00"
}
```

Common HTTP statuses: `400`, `404`, `500`.
