"""Semantic Scholar lookups for the evaluation-validity analyses.

Both analysis scripts had a byte-identical `_s2_fetch`, and both re-raised
every HTTPError that was not a 404 -- including 429. Semantic Scholar rate
limits unauthenticated traffic aggressively and `--s2-key` is documented as
optional, so 429 is the *expected* response on the supported keyless path, not
an exceptional one. Hitting it killed the run with a raw traceback and threw
away every lookup already made.

Rate limits and transient server errors are retried with exponential backoff
here, honouring Retry-After when the server sends it. A request that still
fails is reported and skipped, which degrades the analysis rather than ending
it -- the same treatment connection errors already got.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"

#: Retried with backoff: 429 is rate limiting, 5xx is the server having a
#: moment. Everything else is a bad request and retrying cannot help.
_RETRY_CODES = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 5
_MAX_BACKOFF = 60.0


def _retry_after(exc: urllib.error.HTTPError, attempt: int) -> float:
    """Seconds to wait: the server's Retry-After if sane, else exponential."""
    raw = exc.headers.get("Retry-After") if exc.headers else None
    if raw:
        try:
            return min(float(raw), _MAX_BACKOFF)
        except ValueError:
            pass
    return min(2.0**attempt, _MAX_BACKOFF)


def fetch_paper(
    arxiv_id: str,
    fields: str,
    api_key: str | None = None,
    *,
    max_attempts: int = _MAX_ATTEMPTS,
) -> dict[str, Any] | None:
    """Fetch one paper by arXiv id. Returns None when it cannot be retrieved.

    None means "no data for this paper" in every failure mode the caller can do
    nothing about: not indexed (404), rate limited past the retry budget, or a
    network error. Callers treat a missing paper as a gap in coverage, so the
    analysis reports on what it could reach instead of dying part-way.
    """
    url = f"{BASE_URL}/arXiv:{arxiv_id}?fields={fields}"
    req = urllib.request.Request(url)  # noqa: S310 - fixed https host built above
    if api_key:
        req.add_header("x-api-key", api_key)

    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                payload: dict[str, Any] = json.loads(resp.read().decode())
                return payload
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code not in _RETRY_CODES:
                raise
            if attempt == max_attempts - 1:
                print(
                    f"  [s2] {arxiv_id}: HTTP {exc.code} after {max_attempts} "
                    "attempts — skipping. Pass --s2-key for a higher rate limit."
                )
                return None
            wait = _retry_after(exc, attempt)
            print(f"  [s2] HTTP {exc.code}; retrying in {wait:.0f}s")
            time.sleep(wait)
        except Exception as exc:  # noqa: BLE001 — best-effort network lookup
            print(f"  [s2 error] {arxiv_id}: {exc}")
            return None
    return None
