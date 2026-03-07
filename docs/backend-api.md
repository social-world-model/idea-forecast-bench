# Backend API

Base URL: `http://localhost:5000`

## Health and Monitoring

- `GET /healthz`
  - Returns service status.
- `GET /metrics`
  - Returns strategy, backtest, and generation counters.

## Strategies

- `GET /api/strategies`
  - Returns persisted strategy configurations plus any backtest, generation, and daily evaluation payloads.
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
- If any local strategy already has a non-empty `backtest_result`, backend startup skips automatic backtesting.
- If no strategy has backtest data yet, startup may run one automatic backtest pass.

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
