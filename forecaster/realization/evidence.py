"""Embedding-based evidence retrieval for the realization module."""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

from live_idea_bench.models import PaperRecord
from live_idea_bench.similarity import compute_similarity, paper_text
from live_idea_bench.config import SimilarityConfig

from forecaster.models import Innovation

logger = logging.getLogger(__name__)

_TOKENIZE_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKENIZE_RE.findall(text.lower()))


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

    Uses two-stage retrieval for hybrid engine: fast Jaccard/keyword pre-filter
    on tokenized text, then expensive SequenceMatcher only on top candidates.
    Memory-efficient: tokenizes on-the-fly, no bulk pre-computation.

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
    use_hybrid = config.engine.lower().strip() in ("hybrid", "")

    # For non-hybrid engines, fall back to original per-paper scoring
    if not use_hybrid:
        scored: list[tuple[float, PaperRecord]] = []
        for paper in papers:
            try:
                result = compute_similarity(query, paper_text(paper), config)
                scored.append((result.score, paper))
            except Exception as exc:
                logger.warning("Failed to score paper %s: %s", paper.paper_id, exc)
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [p for s, p in scored if s >= similarity_threshold][:top_k]

    # --- Two-stage hybrid retrieval (memory-efficient) ---
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    query_len = len(query_tokens)

    # Stage 1: Fast Jaccard + keyword overlap pre-filter (no SequenceMatcher)
    PREFILTER_K = max(top_k * 10, 50)
    jaccard_scores: list[tuple[float, int, PaperRecord]] = []
    for idx, paper in enumerate(papers):
        try:
            paper_tokens = _tokenize(paper_text(paper))
            if not paper_tokens:
                continue
            intersection = len(query_tokens & paper_tokens)
            if intersection == 0:
                continue
            union = len(query_tokens | paper_tokens)
            jac = intersection / union
            kw = intersection / min(query_len, len(paper_tokens))
            jaccard_scores.append((max(jac, kw), idx, paper))
        except Exception:
            pass

    jaccard_scores.sort(key=lambda t: t[0], reverse=True)
    candidates = jaccard_scores[:PREFILTER_K]

    # Stage 2: Full hybrid scoring on top candidates only
    query_lower = query.lower()
    scored_final: list[tuple[float, PaperRecord]] = []
    for jac_score, _idx, paper in candidates:
        try:
            context = paper_text(paper)
            seq = SequenceMatcher(None, query_lower, context.lower()).ratio()
            hybrid = (0.65 * jac_score) + (0.35 * seq)
            score = max(hybrid, jac_score)  # jac_score already includes kw
            scored_final.append((score, paper))
        except Exception as exc:
            logger.warning("Failed to score paper %s: %s", paper.paper_id, exc)

    scored_final.sort(key=lambda pair: pair[0], reverse=True)
    return [p for s, p in scored_final if s >= similarity_threshold][:top_k]
