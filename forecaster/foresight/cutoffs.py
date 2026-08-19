"""Temporal window constants and leakage assertions for the Foresight plan.

Locked design decision (see memory: foresight-rl-plan-decisions):
- train cutoffs:        t <= 2024-06
- train future window:  ends at  <= 2024-09 (3-month horizon from the latest train cutoff)
- rubric held-out:      cutoff <= 2024-06, positives drawn from 2024-07..2024-09
- test cutoffs:         t >= 2024-10
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

# These are the load-bearing dates. Changing them invalidates the
# train/test disjointness invariant.
TRAIN_CUTOFF_MAX: str = "2024-06-30"
TEST_CUTOFF_MIN: str = "2024-10-01"
# No paper published on or after this date may appear in any *training*
# future index — it would leak the test window into training.
FUTURE_WINDOW_HARD_LIMIT: str = "2024-10-01"


def _to_date(s: str) -> date:
    """Parse YYYY-MM or YYYY-MM-DD to a date (first-of-month for YYYY-MM)."""
    s = s.strip()
    if len(s) == 7:  # YYYY-MM
        return date.fromisoformat(s + "-01")
    return date.fromisoformat(s)


def assert_train_test_disjoint(
    train_cutoffs: Iterable[str],
    test_cutoffs: Iterable[str],
) -> None:
    """Hard invariant: max(train_cutoffs) < min(test_cutoffs), both < 2024-10-01.

    Raises:
        AssertionError: if either window is empty, the windows overlap, or
            any train cutoff is on/after FUTURE_WINDOW_HARD_LIMIT.
    """
    train = sorted(_to_date(c) for c in train_cutoffs)
    test = sorted(_to_date(c) for c in test_cutoffs)
    if not train:
        raise AssertionError("train_cutoffs is empty")
    if not test:
        raise AssertionError("test_cutoffs is empty")
    hard_limit = _to_date(FUTURE_WINDOW_HARD_LIMIT)
    if train[-1] >= hard_limit:
        raise AssertionError(
            f"max(train_cutoffs)={train[-1]} must be < {hard_limit} "
            f"(FUTURE_WINDOW_HARD_LIMIT)"
        )
    if test[0] < hard_limit:
        raise AssertionError(
            f"min(test_cutoffs)={test[0]} must be >= {hard_limit} "
            f"to avoid contiguity with the train future window"
        )
    if train[-1] >= test[0]:
        raise AssertionError(
            f"train/test cutoffs overlap: max(train)={train[-1]} vs min(test)={test[0]}"
        )


def assert_no_test_window_leakage(
    paper_published_dates: Iterable[str],
    *,
    context: str = "future index",
) -> None:
    """Assert that no paper crosses into the test window.

    Used after building any training-time future index to confirm that
    no Y_{t+1} paper has published_date >= FUTURE_WINDOW_HARD_LIMIT.
    """
    limit = _to_date(FUTURE_WINDOW_HARD_LIMIT)
    leaked = [d for d in paper_published_dates if _to_date(d) >= limit]
    if leaked:
        raise AssertionError(
            f"{context}: {len(leaked)} paper(s) have published_date >= {limit} "
            f"(test window). First few: {leaked[:5]}"
        )
