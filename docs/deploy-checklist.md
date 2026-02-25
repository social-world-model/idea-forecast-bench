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
- [ ] Confirm `/metrics` returns run stats.

## Post-Deploy Validation

- [ ] Start one run from `/runs`.
- [ ] Verify run appears in `/runs/history`.
- [ ] Verify run detail page loads and export buttons work.
- [ ] Verify report charts render.
- [ ] Verify logs and alerts are clean for 15 minutes.
