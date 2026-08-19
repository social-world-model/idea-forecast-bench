"""Deduplication for joint inference candidates."""
from __future__ import annotations

from forecaster.models import JointCandidate


def _jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts based on word tokens."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def deduplicate_proposals(
    candidates: list[JointCandidate],
    threshold: float = 0.8,
) -> list[JointCandidate]:
    """Remove near-duplicate proposals using Jaccard similarity.

    Processes candidates in order (caller should sort by score before calling).
    For each candidate, keeps it if it is not similar (above threshold) to any
    already-kept candidate.

    Args:
        candidates: List of JointCandidate objects (should be sorted by score).
        threshold: Similarity threshold above which proposals are considered duplicates.

    Returns:
        Deduplicated list preserving order.
    """
    kept: list[JointCandidate] = []
    for candidate in candidates:
        is_duplicate = any(
            _jaccard_similarity(candidate.proposal_text, kept_candidate.proposal_text) > threshold
            for kept_candidate in kept
        )
        if not is_duplicate:
            kept.append(candidate)
    return kept
