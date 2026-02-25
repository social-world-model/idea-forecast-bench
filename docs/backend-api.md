# Backend API

Base URL: `http://localhost:5000`

## Health and Monitoring

- `GET /healthz`
  - Returns service status.
- `GET /metrics`
  - Returns run counters and success rate.

## Ideas

- `GET /api/generate-ideas`
  - Returns cached generated ideas if present.
- `POST /api/generate-ideas`
  - Body:
    - `keywords: string[]` (required)
    - `n: number` (optional)
  - Generates ideas synchronously and stores them.
- `GET /api/research-ideas`
  - Returns transformed ideas for frontend leaderboard.

## Runs

- `POST /api/runs/start`
  - Body:
    - `keywords: string[]` (optional, defaults to config)
    - `n: number` (optional, defaults to config)
  - Response `202 Accepted` with:
    - `run.run_id`
    - `run.status` (`pending` -> `running` -> `success` | `failed`)

- `GET /api/runs` or `GET /api/runs/list`
  - Query:
    - `limit` (1-500, default 50)
  - Returns latest runs.

- `GET /api/runs/<run_id>`
- `GET /api/runs/detail/<run_id>`
  - Query:
    - `includeIdeas=true|false` (default false)
  - Returns one run and optional generated ideas payload.

- `GET /api/runs/report`
  - Returns aggregated report with:
    - `summary` counters
    - `keyword_frequency`
    - `score_trend`
    - `comparison` (top runs)

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
