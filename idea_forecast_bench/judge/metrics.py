"""Diversity (cluster coverage) and novelty, computed from embeddings."""

from __future__ import annotations

from idea_forecast_bench.judge.embeddings import cosine


def cluster_coverage(
    future_vecs: list[list[float]],
    matched_ids: set[str],
    future_ids: list[str],
    k: int,
) -> float:
    """Fraction of future-paper clusters covered by at least one matched prediction."""
    n = len(future_vecs)
    if n == 0:
        return 0.0
    k = min(k, n)

    # No silent fallback. The old one put every paper in its own cluster, which
    # turns cluster_coverage from "fraction of topical clusters hit" into
    # "fraction of papers matched" -- a different quantity reported under the
    # same name, differing between machines depending on whether scikit-learn
    # happened to be installed.
    import numpy as np
    from sklearn.cluster import KMeans

    X = np.array(future_vecs)
    labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(X)

    matched_clusters = {
        labels[i] for i, pid in enumerate(future_ids) if pid in matched_ids
    }
    return round(len(matched_clusters) / k, 4)


def novelty_score(pred_vec: list[float], train_vecs: list[list[float]]) -> float:
    """1 - max cosine similarity to any training paper. Higher = more novel."""
    if not train_vecs:
        return 1.0
    return round(1.0 - max(cosine(pred_vec, tv) for tv in train_vecs), 4)
