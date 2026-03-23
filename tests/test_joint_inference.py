"""Tests for forecaster/inference/algorithm.py"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from live_idea_bench.models import PaperRecord

from forecaster.models import Innovation, MemoryEntry, MemoryInventory, ScoredProposal
from forecaster.config import InferenceConfig, RealizationConfig
from forecaster.prior.memory import MemoryStore
from forecaster.inference.algorithm import run_joint_inference


def _make_innovation(
    base_direction: str = "transformer attention",
    operator: str = "extend",
    gap: str = "long sequence efficiency",
) -> Innovation:
    return Innovation(base_direction=base_direction, operator=operator, gap=gap)


def _make_paper(paper_id: str, summary: str) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=summary[:40],
        month="2024-01",
        summary=summary,
        keywords=[],
        source_path="",
    )


def _make_memory_store() -> MemoryStore:
    entry = MemoryEntry(
        innovation=_make_innovation(),
        source_paper_id="paper_1",
        timestamp_month="2024-01",
        frequency=2,
        recency_score=0.9,
    )
    inventory = MemoryInventory(entries=(entry,), last_updated_month="2024-01")
    return MemoryStore(inventory)


_MOCK_LLM_RESPONSE = ("Test Proposal Title\nThis is the proposal body.", [])

_PATCH_TARGET = "forecaster.realization.proposal_generator.get_response_from_llm"


class TestRunJointInference:
    def test_run_joint_inference_returns_scored_proposals(self) -> None:
        """Function returns a list of ScoredProposal objects."""
        innovations = [_make_innovation()]
        papers = [_make_paper("p1", "attention mechanism for long document sequences training")]
        memory_store = _make_memory_store()
        llm_client = MagicMock()
        inference_config = InferenceConfig(top_k=5)
        realization_config = RealizationConfig()

        with patch(_PATCH_TARGET, return_value=_MOCK_LLM_RESPONSE):
            result = run_joint_inference(
                innovations=innovations,
                papers=papers,
                memory_store=memory_store,
                llm_client=llm_client,
                model="gpt-4o",
                inference_config=inference_config,
                realization_config=realization_config,
            )

        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(p, ScoredProposal) for p in result)

    def test_run_joint_inference_top_k_limit(self) -> None:
        """Returns at most top_k proposals."""
        innovations = [_make_innovation(gap=f"gap {i}") for i in range(10)]
        papers = [_make_paper("p1", "some paper content about attention")]
        memory_store = MemoryStore.empty("2024-01")
        llm_client = MagicMock()
        inference_config = InferenceConfig(top_k=3)
        realization_config = RealizationConfig()

        with patch(_PATCH_TARGET, return_value=_MOCK_LLM_RESPONSE):
            result = run_joint_inference(
                innovations=innovations,
                papers=papers,
                memory_store=memory_store,
                llm_client=llm_client,
                model="gpt-4o",
                inference_config=inference_config,
                realization_config=realization_config,
            )

        assert len(result) <= 3

    def test_run_joint_inference_ranks_assigned(self) -> None:
        """ScoredProposals have 1-indexed ranks assigned."""
        innovations = [_make_innovation(gap=f"gap {i}") for i in range(3)]
        papers = [_make_paper("p1", "attention model training")]
        memory_store = MemoryStore.empty("2024-01")
        llm_client = MagicMock()
        inference_config = InferenceConfig(top_k=5)
        realization_config = RealizationConfig()

        with patch(_PATCH_TARGET, return_value=_MOCK_LLM_RESPONSE):
            result = run_joint_inference(
                innovations=innovations,
                papers=papers,
                memory_store=memory_store,
                llm_client=llm_client,
                model="gpt-4o",
                inference_config=inference_config,
                realization_config=realization_config,
            )

        ranks = [p.rank for p in result]
        assert ranks == list(range(1, len(result) + 1))

    def test_run_joint_inference_empty_innovations(self) -> None:
        """Empty innovations list returns empty result."""
        papers = [_make_paper("p1", "some paper")]
        memory_store = MemoryStore.empty("2024-01")
        llm_client = MagicMock()
        inference_config = InferenceConfig(top_k=5)
        realization_config = RealizationConfig()

        with patch(_PATCH_TARGET, return_value=_MOCK_LLM_RESPONSE):
            result = run_joint_inference(
                innovations=[],
                papers=papers,
                memory_store=memory_store,
                llm_client=llm_client,
                model="gpt-4o",
                inference_config=inference_config,
                realization_config=realization_config,
            )

        assert result == []

    def test_run_joint_inference_skips_failed_proposals(self) -> None:
        """If LLM fails for some innovations, skips them and continues."""
        innovations = [
            _make_innovation(gap="gap 0"),
            _make_innovation(gap="gap 1"),
            _make_innovation(gap="gap 2"),
        ]
        papers = [_make_paper("p1", "attention model training")]
        memory_store = MemoryStore.empty("2024-01")
        llm_client = MagicMock()
        # Use high dedup threshold so distinct proposals are not collapsed
        inference_config = InferenceConfig(top_k=5, dedup_threshold=0.99)
        realization_config = RealizationConfig()

        call_count = 0
        responses = [
            ("Proposal Alpha\nFirst unique proposal about attention and training.", []),
            None,  # will raise
            ("Proposal Gamma\nThird unique proposal about model evaluation.", []),
        ]

        def side_effect(*args, **kwargs):
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            if resp is None:
                raise RuntimeError("LLM failure")
            return resp

        with patch(_PATCH_TARGET, side_effect=side_effect):
            result = run_joint_inference(
                innovations=innovations,
                papers=papers,
                memory_store=memory_store,
                llm_client=llm_client,
                model="gpt-4o",
                inference_config=inference_config,
                realization_config=realization_config,
            )

        # Should have 2 proposals (skipped the failing one)
        assert len(result) == 2
        assert all(isinstance(p, ScoredProposal) for p in result)

    def test_run_joint_inference_deduplicates(self) -> None:
        """Identical proposals are deduplicated in the output."""
        # All innovations produce the same proposal text → should be deduped
        innovations = [_make_innovation(gap=f"gap {i}") for i in range(5)]
        papers = [_make_paper("p1", "attention model training")]
        memory_store = MemoryStore.empty("2024-01")
        llm_client = MagicMock()
        # Use very low threshold to force deduplication
        inference_config = InferenceConfig(top_k=10, dedup_threshold=0.5)
        realization_config = RealizationConfig()

        # All LLM calls return identical text
        with patch(_PATCH_TARGET, return_value=("Identical Proposal\nSame body text.", [])):
            result = run_joint_inference(
                innovations=innovations,
                papers=papers,
                memory_store=memory_store,
                llm_client=llm_client,
                model="gpt-4o",
                inference_config=inference_config,
                realization_config=realization_config,
            )

        # After deduplication, all identical proposals collapse to 1
        assert len(result) == 1

    def test_run_joint_inference_sorted_by_joint_score(self) -> None:
        """Proposals in result are sorted by joint score descending."""
        innovations = [_make_innovation(gap=f"gap {i}") for i in range(4)]
        papers = [_make_paper("p1", "attention training model")]
        memory_store = MemoryStore.empty("2024-01")
        llm_client = MagicMock()
        inference_config = InferenceConfig(top_k=10)
        realization_config = RealizationConfig()

        with patch(_PATCH_TARGET, return_value=_MOCK_LLM_RESPONSE):
            result = run_joint_inference(
                innovations=innovations,
                papers=papers,
                memory_store=memory_store,
                llm_client=llm_client,
                model="gpt-4o",
                inference_config=inference_config,
                realization_config=realization_config,
            )

        joint_scores = [p.joint_score for p in result]
        assert joint_scores == sorted(joint_scores, reverse=True)
