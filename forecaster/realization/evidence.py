"""Embedding-based evidence retrieval for the realization module."""
from __future__ import annotations

import logging
from typing import Any

from live_idea_bench.models import PaperRecord
from live_idea_bench.similarity import compute_similarity, paper_text
from live_idea_bench.config import SimilarityConfig

from forecaster.models import Innovation

logger = logging.getLogger(__name__)


def build_innovation_query(innovation: Innovation) -> str:
    """Build a query string from an Innovation for embedding-based retrieval.

    Combines base_direction, operator, and gap into a natural language query.
    """
    return f"{innovation.base_direction} {innovation.operator}: {innovation.gap}"


def retrieve_evidence(
    innovation: Innovation,
    papers: list[PaperRecord],
    *,
    top_k: int = 5,
    similarity_threshold: float = 0.3,
    similarity_config: SimilarityConfig | None = None,
) -> list[PaperRecord]:
    """Retrieve the most relevant historical papers given an innovation.

    Uses hybrid text similarity to find papers relevant to the innovation.

    Args:
        innovation: The innovation triple to use as query.
        papers: Historical papers to search.
        top_k: Maximum number of papers to return.
        similarity_threshold: Minimum similarity score threshold.
        similarity_config: Optional SimilarityConfig (creates default if None).

    Returns:
        List of top-k relevant papers above threshold, sorted by relevance.
    """
    if not papers:
        return []

    query = build_innovation_query(innovation)
    config = similarity_config or SimilarityConfig(engine="hybrid")

    scored: list[tuple[float, PaperRecord]] = []
    for paper in papers:
        try:
            context = paper_text(paper)
            result = compute_similarity(query, context, config)
            scored.append((result.score, paper))
        except Exception as exc:
            logger.warning(
                "Failed to score paper %s during evidence retrieval: %s",
                paper.paper_id,
                exc,
            )

    scored.sort(key=lambda pair: pair[0], reverse=True)

    filtered = [
        paper
        for score, paper in scored
        if score >= similarity_threshold
    ]

    return filtered[:top_k]
