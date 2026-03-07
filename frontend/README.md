# Frontend

This React app is a thin UI over the strategy mainline:

- `Leaderboard`: backtest summaries, daily evaluation, strategy metadata
- `Generated Ideas`: latest `strategy.generation.predictions`
- `About`: project and benchmark framing

## Start

```bash
cd frontend
npm install
npm start
```

Default backend: `http://localhost:5000`

Override with:

```bash
REACT_APP_API_BASE_URL=http://localhost:5000
REACT_APP_REFRESH_INTERVAL=30000
```

## Required API

- `GET /api/strategies`
- `GET /api/strategies/:id`
- `GET /api/strategies/:id/status`
- `POST /api/strategies/:id/backtest`
- `POST /api/strategies/:id/generate`
- `GET /api/views`
- `POST /api/views`
