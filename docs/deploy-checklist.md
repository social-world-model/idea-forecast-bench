# Deploy Checklist

## Pre-Deploy

- [ ] `pytest -q` passes.
- [ ] `npm test -- --watch=false` passes in `frontend/`.
- [ ] `npm run build` passes in `frontend/`.
- [ ] Docker images build successfully.
- [ ] Required secrets are set (`OPENAI_API_KEY`, `GOOGLE_API_KEY`).

## Release

- [ ] Deploy backend image.
- [ ] Deploy frontend image.
- [ ] Run DB/data migration checks (if any).
- [ ] Confirm `/healthz` is healthy.
- [ ] Confirm `/metrics` returns strategy stats.

## Post-Deploy Validation

- [ ] Open the leaderboard and verify strategies load.
- [ ] Open Generated Ideas and verify strategy generation cards render.
- [ ] Trigger one strategy generation and verify it appears in the UI.
- [ ] Verify logs and alerts are clean for 15 minutes.
