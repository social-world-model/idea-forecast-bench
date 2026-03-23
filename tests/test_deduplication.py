"""Tests for forecaster/inference/deduplication.py"""
from __future__ import annotations

import pytest

from forecaster.models import Innovation, JointCandidate
from forecaster.inference.deduplication import deduplicate_proposals, _jaccard_similarity


def _make_innovation(base: str = "transformer", op: str = "extend", gap: str = "efficiency") -> Innovation:
    return Innovation(base_direction=base, operator=op, gap=gap)


def _make_candidate(proposal_text: str, joint_score: float = 0.5) -> JointCandidate:
    return JointCandidate(
        innovation=_make_innovation(),
        prior_score=0.0,
        evidence_paper_ids=(),
        proposal_text=proposal_text,
        realization_score=0.0,
    )


class TestJaccardSimilarity:
    def test_jaccard_similarity_identical(self) -> None:
        """Identical texts return 1.0."""
        text = "the quick brown fox jumps over the lazy dog"
        assert _jaccard_similarity(text, text) == 1.0

    def test_jaccard_similarity_disjoint(self) -> None:
        """Completely different texts return 0.0."""
        text1 = "apple banana cherry"
        text2 = "dog elephant frog"
        assert _jaccard_similarity(text1, text2) == 0.0

    def test_jaccard_similarity_partial(self) -> None:
        """Partial overlap returns correct ratio."""
        text1 = "a b c d"
        text2 = "c d e f"
        # tokens1 = {a, b, c, d}, tokens2 = {c, d, e, f}
        # intersection = {c, d} = 2, union = {a, b, c, d, e, f} = 6
        result = _jaccard_similarity(text1, text2)
        assert abs(result - 2 / 6) < 1e-9

    def test_jaccard_similarity_empty_both(self) -> None:
        """Both empty texts return 1.0."""
        assert _jaccard_similarity("", "") == 1.0

    def test_jaccard_similarity_one_empty(self) -> None:
        """One empty text returns 0.0."""
        assert _jaccard_similarity("some words here", "") == 0.0
        assert _jaccard_similarity("", "some words here") == 0.0

    def test_jaccard_similarity_case_insensitive(self) -> None:
        """Comparison is case-insensitive."""
        result = _jaccard_similarity("Hello World", "hello world")
        assert result == 1.0


class TestDeduplicateProposals:
    def test_deduplicate_empty_list(self) -> None:
        """Empty list returns empty list."""
        result = deduplicate_proposals([])
        assert result == []

    def test_deduplicate_no_duplicates(self) -> None:
        """All distinct proposals kept."""
        candidates = [
            _make_candidate("transformer attention mechanism for long documents"),
            _make_candidate("graph neural network node classification tasks"),
            _make_candidate("reinforcement learning reward shaping exploration"),
        ]
        result = deduplicate_proposals(candidates, threshold=0.8)
        assert len(result) == 3

    def test_deduplicate_exact_duplicate(self) -> None:
        """Only one of identical proposals kept (first one)."""
        text = "exact same proposal text about transformers and attention"
        candidates = [
            _make_candidate(text),
            _make_candidate(text),
        ]
        result = deduplicate_proposals(candidates, threshold=0.8)
        assert len(result) == 1
        assert result[0].proposal_text == text

    def test_deduplicate_near_duplicate(self) -> None:
        """Proposals with similarity above threshold are deduplicated."""
        # Very similar texts — use words with known high overlap
        # text1 tokens: {a, b, c, d, e, f, g, h, i} (9 tokens)
        # text2 tokens: {a, b, c, d, e, f, g, h, j} (9 tokens)
        # intersection = 8, union = 10, jaccard = 0.8 → above 0.79 threshold
        text1 = "alpha bravo charlie delta echo foxtrot golf hotel india"
        text2 = "alpha bravo charlie delta echo foxtrot golf hotel juliet"
        sim = _jaccard_similarity(text1, text2)
        # sim = 8/10 = 0.8; threshold at 0.79 → deduplicated
        assert sim == pytest.approx(8 / 10)

        candidates = [
            _make_candidate(text1),
            _make_candidate(text2),
        ]
        result = deduplicate_proposals(candidates, threshold=0.79)
        assert len(result) == 1

    def test_deduplicate_preserves_order(self) -> None:
        """Earlier (higher-scored) proposals are kept over later ones."""
        text = "transformer attention long sequence efficiency model"
        candidates = [
            _make_candidate(text + " first"),
            _make_candidate(text + " first"),  # same as first
        ]
        # Manually create near-identical texts where first should be kept
        text_a = "apple banana cherry delta echo foxtrot golf"
        text_b = "apple banana cherry delta echo foxtrot golf"  # identical
        c1 = _make_candidate(text_a)
        c2 = _make_candidate(text_b)

        result = deduplicate_proposals([c1, c2], threshold=0.8)
        assert len(result) == 1
        assert result[0] is c1  # first one is kept

    def test_deduplicate_threshold_boundary(self) -> None:
        """At threshold → kept. Above threshold → removed."""
        # Construct two texts with known Jaccard similarity
        # text1: {a, b, c, d, e} (5 tokens)
        # text2: {a, b, c, d, f} (5 tokens)
        # intersection = {a, b, c, d} = 4, union = {a, b, c, d, e, f} = 6
        # jaccard = 4/6 ≈ 0.667
        text1 = "alpha beta gamma delta epsilon"
        text2 = "alpha beta gamma delta zeta"
        sim = _jaccard_similarity(text1, text2)

        c1 = _make_candidate(text1)
        c2 = _make_candidate(text2)

        # threshold just below sim → second is removed (above threshold)
        result_above = deduplicate_proposals([c1, c2], threshold=sim - 0.01)
        assert len(result_above) == 1

        # threshold just above sim → second is kept (below threshold)
        result_below = deduplicate_proposals([c1, c2], threshold=sim + 0.01)
        assert len(result_below) == 2

    def test_deduplicate_single_candidate(self) -> None:
        """Single candidate always kept."""
        candidates = [_make_candidate("single proposal text here")]
        result = deduplicate_proposals(candidates, threshold=0.8)
        assert len(result) == 1

    def test_deduplicate_keeps_distinct_after_removing_duplicate(self) -> None:
        """After removing a duplicate, distinct third entry is still kept."""
        text = "alpha beta gamma delta epsilon zeta eta theta"
        different = "completely different topic unrelated to previous content"
        candidates = [
            _make_candidate(text),
            _make_candidate(text),  # duplicate of first → removed
            _make_candidate(different),
        ]
        result = deduplicate_proposals(candidates, threshold=0.8)
        assert len(result) == 2
