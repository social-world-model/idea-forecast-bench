# Frontend

A React app that is a thin UI over the strategy mainline. It is **optional** —
the benchmark and the MDF forecaster do not depend on it.

- `Leaderboard` — backtest summaries, daily evaluation, strategy metadata
- `Generated Ideas` — latest `strategy.generation.predictions`
- `About` — project and benchmark framing

Deliberately narrow: it fetches strategy records from `/api/strategies` and
renders them. There are no run pages and no market/trading views.

## Run it

```bash
cd frontend
npm install
npm start          # dev server
npm run build      # production bundle into frontend/build/
```

The API defaults to `http://localhost:5000` (see `backend/README.md` for
starting it). Override with:

```bash
REACT_APP_API_BASE_URL=http://localhost:5000
REACT_APP_REFRESH_INTERVAL=30000
```

## Key files

| Path | Role |
|------|------|
| `src/App.tsx` | routes |
| `src/components/Dashboard.tsx` | leaderboard |
| `src/components/GeneratedIdeasList.tsx` | generated ideas |
| `src/components/About.tsx` | about page |
| `src/types.ts` | API response types |
| `src/config.ts` | API base URL / refresh interval |

## API surface it depends on

- `GET  /api/strategies`
- `GET  /api/strategies/:id`
- `GET  /api/strategies/:id/status`
- `POST /api/strategies/:id/backtest`
- `POST /api/strategies/:id/generate`
- `GET  /api/views`
- `POST /api/views`

## Note on the toolchain

This app is still on `react-scripts` (Create React App), which the React team
has deprecated. That is why the app is not part of the CI gate. Migrating to
Vite is tracked as follow-up work; until then treat the frontend as
best-effort.
