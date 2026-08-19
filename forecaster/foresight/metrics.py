"""Phase-8 non-optimized distribution metrics.

These metrics deliberately do NOT use the rubric-conditioned judge. They
sit alongside the standard bench scorer to rebut the "you just overfit
your own judge" objection.

  * mmd_rbf(P, Q)     — squared MMD between two embedding sets, RBF kernel.
  * wasserstein_1d(p, q) — 1-D Wasserstein between two scalar distributions.
  * impact_stratified_breakdown(rows, bucket_fn) — slice metrics by
    citation count / impact bucket.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- MMD


def _pairwise_squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return shape (|A|, |B|) squared euclidean distance matrix."""
    A2 = (A * A).sum(axis=1, keepdims=True)
    B2 = (B * B).sum(axis=1, keepdims=True).T
    cross = A @ B.T
    return np.maximum(A2 + B2 - 2.0 * cross, 0.0)


def mmd_rbf(
    P: np.ndarray,
    Q: np.ndarray,
    *,
    bandwidth: float | None = None,
) -> float:
    """Unbiased squared MMD with an RBF kernel.

    Args:
        P, Q: (n_p, d) and (n_q, d) embedding matrices (preferably L2-normalized).
        bandwidth: kernel σ. When None, falls back to the median heuristic over
            the union of P and Q (the standard non-optimized choice).
    """
    P = np.asarray(P, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    if P.size == 0 or Q.size == 0:
        return 0.0
    PP = _pairwise_squared_distances(P, P)
    QQ = _pairwise_squared_distances(Q, Q)
    PQ = _pairwise_squared_distances(P, Q)
    if bandwidth is None:
        union = np.concatenate([P, Q], axis=0)
        D = _pairwise_squared_distances(union, union)
        sigma2 = max(float(np.median(D)) / 2.0, 1e-8)
    else:
        sigma2 = float(bandwidth) ** 2
    Kpp = np.exp(-PP / (2.0 * sigma2))
    Kqq = np.exp(-QQ / (2.0 * sigma2))
    Kpq = np.exp(-PQ / (2.0 * sigma2))
    np_, nq_ = P.shape[0], Q.shape[0]
    # Unbiased estimator (drop diagonal).
    sum_pp = float(Kpp.sum() - np.trace(Kpp)) / max(np_ * (np_ - 1), 1)
    sum_qq = float(Kqq.sum() - np.trace(Kqq)) / max(nq_ * (nq_ - 1), 1)
    sum_pq = float(Kpq.sum()) / max(np_ * nq_, 1)
    return float(sum_pp + sum_qq - 2.0 * sum_pq)


# --------------------------------------------------------------------------- Wasserstein-1


def wasserstein_1d(p: Sequence[float], q: Sequence[float]) -> float:
    """1-D Wasserstein-1 between two empirical CDFs.

    Uses the closed-form integral of |F_p - F_q| computed via sorted-merge.
    Returns 0.0 if either input is empty.
    """
    a = np.asarray(sorted(float(x) for x in p), dtype=np.float64)
    b = np.asarray(sorted(float(x) for x in q), dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    # Build a common grid; integrate |F_p - F_q| via trapezoid.
    grid = np.unique(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    # |F_a - F_b| integrated over the support [grid[0], grid[-1]].
    diffs = np.abs(cdf_a - cdf_b)
    widths = np.diff(grid)
    if widths.size == 0:
        return 0.0
    return float((diffs[:-1] * widths).sum())


# --------------------------------------------------------------------------- impact stratification


def impact_stratified_breakdown(
    rows: Iterable[dict],
    *,
    bucket_fn: Callable[[dict], str] | None = None,
    metric_keys: Sequence[str] = ("hit_at_k", "mrr"),
) -> dict[str, dict[str, float]]:
    """Group `rows` by bucket and average each metric within.

    Args:
        rows: iterable of dicts; each must contain `metric_keys`.
        bucket_fn: row -> bucket name. Defaults to citation_count quantile
            buckets {low|mid|high} when a `citation_count` key is present;
            falls back to "all".
        metric_keys: names of numeric keys to aggregate.
    """
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    if bucket_fn is None:
        bucket_fn = _default_impact_bucket
    for row in rows:
        by_bucket[bucket_fn(row)].append(row)
    out: dict[str, dict[str, float]] = {}
    for bucket, items in by_bucket.items():
        agg: dict[str, float] = {"count": float(len(items))}
        for k in metric_keys:
            vals = [float(item.get(k, 0.0)) for item in items if k in item]
            agg[k] = float(np.mean(vals)) if vals else 0.0
        out[bucket] = agg
    return out


def _default_impact_bucket(row: dict) -> str:
    c = row.get("citation_count")
    if c is None:
        return "all"
    try:
        c = float(c)
    except (TypeError, ValueError):
        return "all"
    if c < 5:
        return "low"
    if c < 50:
        return "mid"
    return "high"


__all__ = [
    "mmd_rbf",
    "wasserstein_1d",
    "impact_stratified_breakdown",
]
