# Frontend Notes

The frontend no longer carries legacy run pages or market/trading views. It is intentionally narrow:

- fetch strategy records from `/api/strategies`
- render leaderboard metrics from `backtest_result` and `daily_evaluation`
- render current generated ideas from `generation.predictions`

Key files:

- `frontend/src/App.tsx`
- `frontend/src/components/Dashboard.tsx`
- `frontend/src/components/GeneratedIdeasList.tsx`
- `frontend/src/components/About.tsx`
- `frontend/src/types.ts`
- `frontend/src/config.ts`
