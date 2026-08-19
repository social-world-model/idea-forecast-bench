# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue for security problems.

Report privately through GitHub's
[private vulnerability reporting](https://github.com/ulab-uiuc/live-idea-bench/security/advisories/new)
(Security → Report a vulnerability). We aim to acknowledge within 5 working
days.

## Scope

This is a research benchmark, not a hosted service. The parts worth reporting
against are:

- **`backend/`** — the optional Flask API. Write endpoints under
  `/api/strategies` are gated by `LIVE_IDEA_ADMIN_TOKEN`. Auth bypass,
  injection, or path traversal here is in scope.
- **Supply chain** — anything in `pyproject.toml` / `poetry.lock` /
  `frontend/package-lock.json`.
- **Credential handling** — see below.

Out of scope: running the benchmark against models that return unsafe text,
and denial of service against your own local instance.

## Secrets

The project reads all credentials from the environment and never from a
committed file:

    OPENAI_API_KEY  ANTHROPIC_API_KEY  GOOGLE_API_KEY  VOYAGE_API_KEY
    LIVE_IDEA_ADMIN_TOKEN

Never commit a key. `detect-private-key` runs as a pre-commit hook, and
`.env` is gitignored, but neither catches every shape of secret. If a key is
exposed, rotate it first and report second.

## Deploying the web app

`backend/app.py` runs Flask's development server and has no rate limiting.
Put it behind a real WSGI server and a reverse proxy, and set
`LIVE_IDEA_ADMIN_TOKEN` to a high-entropy value, before exposing it.

CORS defaults to a localhost allowlist (ports 3000 and 5173). Widen it only
through `LIVE_IDEA_CORS_ORIGINS`, and list explicit origins rather than `*`.
