# Monitoring

## Signals

- Availability: `GET /healthz`
- Strategy throughput and quality: `GET /metrics`, `GET /api/strategies`
- Error budget source: failed generations / total strategies

## Recommended Alerts

- Backend down: `/healthz` non-200 for 2 minutes.
- High failure rate: generation failures / strategy count > 0.2 over 15 minutes.
- Stuck jobs: running backtests or generations > 0 for more than 20 minutes.

## Dashboards

- Strategy summary: total/backtest-done/generation-done
- Backtest completion trend
- Generation completion trend
