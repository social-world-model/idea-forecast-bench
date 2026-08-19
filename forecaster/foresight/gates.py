"""Hard gates for the Phase-4 reward.

Each gate is a small, side-effect-free predicate that the new
`compute_reward` consults before doing the expensive judge call.

  * format_ok(rollout_text, prompt_mode, innovation) -> bool
  * grounded(rollout_text, history_index, embedder, threshold) -> bool
  * operator_consistent(rollout_text, z_operator, threshold) -> bool
"""
from __future__ import annotations

import logging
import re

from forecaster.foresight.indices import Embedder, HistoryIndex
from forecaster.foresight.operators import (
    CLOSED_OPERATORS,
    UNMAPPABLE_BUCKET,
    OperatorInventory,
)
from forecaster.models import Innovation
from forecaster.realization.realization_reward import compute_operator_adherence
from forecaster.realization.reward import coerce_reward_prediction

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- format gate


def format_ok(
    rollout_text: str,
    *,
    prompt_mode: str = "z_conditioned_realization",
    innovation: Innovation | None = None,
) -> bool:
    """Pass-through of the existing parser. False iff parsing produces None."""
    prediction, _proposal = coerce_reward_prediction(
        rollout_text, prompt_mode=prompt_mode, innovation=innovation,
    )
    return prediction is not None


# --------------------------------------------------------------------------- grounding gate


# Matches arxiv-style ids (legacy or new), plus generic bracket-id refs and DOI-ish patterns.
_CITATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\barxiv[:\s\-]*([0-9]{4}\.[0-9]{4,6})\b", re.IGNORECASE),
    re.compile(r"\b([0-9]{4}\.[0-9]{4,6})\b"),
    re.compile(r"\[([A-Za-z0-9_]{4,40})\]"),
    re.compile(r"\b(10\.\d{4,9}/[\w\.\-]+)\b"),                # DOI
)


def extract_citation_candidates(rollout_text: str) -> list[str]:
    """Return de-duplicated, order-preserving citation-like substrings."""
    seen: dict[str, None] = {}
    for pat in _CITATION_PATTERNS:
        for m in pat.finditer(rollout_text or ""):
            tok = m.group(1).strip()
            if tok and tok not in seen:
                seen[tok] = None
    return list(seen.keys())


def grounded(
    rollout_text: str,
    history_index: HistoryIndex,
    embedder: Embedder,
    *,
    threshold: float = 0.45,
    top_k: int = 5,
    require_citations: bool = False,
) -> bool:
    """Each citation in the rollout must retrieve a history paper above `threshold`.

    Args:
        rollout_text: the policy completion.
        history_index: index over X_{<=t}.
        embedder: same embedder the index was built with.
        threshold: minimum cosine similarity for "exists in history".
        top_k: how many candidates to consider per citation.
        require_citations: when True, a rollout with no citations is REJECTED;
            when False, an absence of explicit citations is treated as a soft pass
            (the realized idea may not cite explicitly).

    Returns:
        True iff every extracted citation finds a near-match in history.
    """
    citations = extract_citation_candidates(rollout_text)
    if not citations:
        return not require_citations
    if history_index.size == 0:
        return False
    known_ids = {pid.lower() for pid in history_index.paper_ids}
    # First pass: literal paper-id match. Arxiv ids / DOIs that already live
    # in the history corpus don't need semantic retrieval.
    unresolved: list[str] = []
    for cit in citations:
        if cit.lower() in known_ids:
            continue
        unresolved.append(cit)
    if not unresolved:
        return True
    # Second pass: semantic retrieval for the rest.
    embs = embedder.encode(unresolved)
    for cit, q in zip(unresolved, embs, strict=False):
        hits = history_index.search(q, top_k=top_k)
        if not hits:
            return False
        best = hits[0][1]
        if best < threshold:
            logger.debug("grounding miss: citation=%s best=%.3f thresh=%.3f", cit, best, threshold)
            return False
    return True


# --------------------------------------------------------------------------- operator gate


def operator_consistent(
    rollout_text: str,
    expected_operator: str,
    *,
    inventory: OperatorInventory | None = None,
    threshold: float = 0.10,
) -> bool:
    """Compare the rollout's apparent operator usage against `expected_operator`.

    Uses forecaster.realization.realization_reward.compute_operator_adherence,
    which is the same keyword-driven score the existing reward already trusts.
    The expected operator may be a free-text token (extend, transfer, ...) or
    a closed id (limitation_extension, ...); we normalize before scoring.
    """
    if not expected_operator:
        return False
    # Closed → free-text reverse mapping isn't 1-1, so use the inventory's
    # closed id directly when passed (it's a valid keyword in the existing
    # operator vocabulary used by compute_operator_adherence).
    inv = inventory  # noqa: F841 (currently unused; kept for forward compatibility)
    op_for_adherence = expected_operator.strip().lower()
    if op_for_adherence in set(CLOSED_OPERATORS):
        # compute_operator_adherence expects the free-text operator. Map back:
        reverse = {
            "limitation_extension": "extend",
            "cross_domain_transfer": "transfer",
            "method_composition": "compose",
            "benchmark_proposal": "benchmark",
        }
        op_for_adherence = reverse.get(op_for_adherence, op_for_adherence)
    if op_for_adherence == UNMAPPABLE_BUCKET:
        # `other` operators are not constrained — pass the gate.
        return True
    innovation = Innovation(
        base_direction="(placeholder)",
        operator=op_for_adherence,
        gap="(placeholder)",
    )
    score = compute_operator_adherence(rollout_text or "", innovation)
    return score >= threshold


__all__ = [
    "format_ok",
    "grounded",
    "operator_consistent",
    "extract_citation_candidates",
]
