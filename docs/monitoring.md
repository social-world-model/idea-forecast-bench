# Monitoring

## Signals

- Availability: `GET /healthz`
- Job throughput and quality: `GET /metrics`, `GET /api/runs/report`
- Error budget source: failed runs / total runs

## Recommended Alerts

- Backend down: `/healthz` non-200 for 2 minutes.
- High failure rate: `runs_failed / runs_total > 0.2` over 15 minutes.
- Stuck jobs: running runs > 0 for more than 20 minutes.

## Dashboards

- Run summary: total/running/success/failed
- Success rate trend
- Average score trend (`score_trend`)
- Average duration and ideas per run
