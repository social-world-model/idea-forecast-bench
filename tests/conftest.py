"""Shared pytest configuration.

Currently this exists for one reason: the Flask API disables admin auth on
write endpoints only when ``app.testing`` is set. ``app.test_client()`` does
not set that flag by itself, so declare it here once rather than in each of
the ~20 places a test builds a client.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _flask_testing_mode() -> Iterator[None]:
    """Mark the Flask app as under test for the duration of each test."""
    try:
        from backend.app import app
    except Exception:  # pragma: no cover - backend deps not installed
        yield
        return

    previous = app.testing
    app.testing = True
    try:
        yield
    finally:
        app.testing = previous
