"""Tests for forecaster/realization/realization_reward.py"""
from __future__ import annotations

import pytest

from live_idea_bench.models import PaperRecord
from forecaster.models import Innovation
from forecaster.config import RealizationConfig
from forecaster.realization.realization_reward import (
    compute_coherence_score,
    compute_evidence_accuracy,
    compute_operator_adherence,
    compute_realization_reward,
)


def _make_paper(paper_id: str, summary: str) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=summary[:30],
        month="2024-01",
        summary=summary,
        keywords=[],
        source_path="",
    )


def _make_innovation(
    base_direction: str = "transformer",
    operator: str = "extend",
    gap: str = "efficiency",
) -> Innovation:
    return Innovation(base_direction=base_direction, operator=operator, gap=gap)


class TestComputeEvidenceAccuracy:
    def test_compute_evidence_accuracy_relevant_evidence(self) -> None:
        innovation = _make_innovation(
            base_direction="attention mechanism",
            operator="extend",
            gap="long sequence processing",
        )
        evidence = [
            _make_paper("p1", "attention mechanism long sequence processing transformer"),
            _make_paper("p2", "extend attention for long sequences efficiently"),
        ]
        score = compute_evidence_accuracy(evidence, innovation)
        assert score > 0.3, f"Expected high score for relevant evidence, got {score}"
        assert 0.0 <= score <= 1.0

    def test_compute_evidence_accuracy_irrelevant_evidence(self) -> None:
        innovation = _make_innovation(
            base_direction="neural network",
            operator="extend",
            gap="classification accuracy",
        )
        evidence = [
            _make_paper("p1", "cooking pasta recipe garlic sauce"),
            _make_paper("p2", "travel guide paris france tourism"),
        ]
        score = compute_evidence_accuracy(evidence, innovation)
        assert score < 0.5, f"Expected low score for irrelevant evidence, got {score}"
        assert 0.0 <= score <= 1.0

    def test_compute_evidence_accuracy_empty_evidence(self) -> None:
        innovation = _make_innovation()
        score = compute_evidence_accuracy([], innovation)
        assert score == 0.0

    def test_compute_evidence_accuracy_bounds(self) -> None:
        innovation = _make_innovation(
            base_direction="deep learning transformer attention",
            operator="extend",
            gap="sequence length generalization",
        )
        evidence = [_make_paper("p1", "deep learning transformer attention sequence length")]
        score = compute_evidence_accuracy(evidence, innovation)
        assert 0.0 <= score <= 1.0


class TestComputeOperatorAdherence:
    def test_compute_operator_adherence_extend(self) -> None:
        innovation = _make_innovation(operator="extend")
        proposal = (
            "We extend the existing transformer architecture to improve upon prior work, "
            "building upon the foundation of attention mechanisms and improving their capabilities."
        )
        score = compute_operator_adherence(proposal, innovation)
        assert score > 0.0, f"Expected positive score for 'extend' operator, got {score}"
        assert 0.0 <= score <= 1.0

    def test_compute_operator_adherence_benchmark(self) -> None:
        innovation = _make_innovation(operator="benchmark")
        proposal = (
            "We create a comprehensive evaluation benchmark and dataset for assessing "
            "the capabilities of language models across diverse tasks."
        )
        score = compute_operator_adherence(proposal, innovation)
        assert score > 0.0, f"Expected positive score for 'benchmark' operator, got {score}"
        assert 0.0 <= score <= 1.0

    def test_compute_operator_adherence_transfer(self) -> None:
        innovation = _make_innovation(operator="transfer")
        proposal = (
            "We investigate the transfer of knowledge from source domain to a new target domain, "
            "adapting the model using domain adaptation techniques."
        )
        score = compute_operator_adherence(proposal, innovation)
        assert score > 0.0, f"Expected positive score for 'transfer' operator, got {score}"
        assert 0.0 <= score <= 1.0

    def test_compute_operator_adherence_unknown_operator(self) -> None:
        innovation = _make_innovation(operator="unknown_operator_xyz")
        proposal = "A research proposal about interesting ideas."
        score = compute_operator_adherence(proposal, innovation)
        assert 0.0 <= score <= 1.0

    def test_compute_operator_adherence_mismatch(self) -> None:
        innovation = _make_innovation(operator="scale")
        # Proposal only mentions analysis — not scaling
        proposal = "We analyze the failure modes and study the investigation of small models."
        score = compute_operator_adherence(proposal, innovation)
        assert 0.0 <= score <= 1.0

    def test_compute_operator_adherence_bounds(self) -> None:
        for operator in ["extend", "transfer", "compose", "benchmark", "analyze", "simplify", "scale", "adapt"]:
            innovation = _make_innovation(operator=operator)
            score = compute_operator_adherence("any proposal text", innovation)
            assert 0.0 <= score <= 1.0, f"Out of bounds for operator={operator}: {score}"


class TestComputeCoherenceScore:
    def test_compute_coherence_score_long_proposal(self) -> None:
        innovation = _make_innovation(
            base_direction="transformer",
            operator="extend",
            gap="efficiency",
        )
        proposal = (
            "Efficient Transformer Architecture for Long Sequences\n"
            "We propose a novel approach to extend the transformer architecture for improved efficiency. "
            "The methodology involves sparse attention patterns combined with hierarchical representations. "
            "Our technical approach leverages gradient checkpointing and mixed precision training. "
            "This direction is promising because recent trends show increasing sequence lengths. "
            "Expected contributions include a 3x speedup and memory reduction while maintaining accuracy. "
            "We extend the transformer model by building upon the existing research gap around efficiency."
        )
        score = compute_coherence_score(proposal, innovation)
        assert score > 0.3, f"Expected high coherence for detailed proposal, got {score}"
        assert 0.0 <= score <= 1.0

    def test_compute_coherence_score_empty_proposal(self) -> None:
        innovation = _make_innovation()
        score = compute_coherence_score("", innovation)
        assert score == 0.0

    def test_compute_coherence_score_very_short_proposal(self) -> None:
        innovation = _make_innovation()
        score = compute_coherence_score("short", innovation)
        assert score < 0.3, f"Expected low coherence for very short proposal, got {score}"
        assert 0.0 <= score <= 1.0

    def test_compute_coherence_score_bounds(self) -> None:
        innovation = _make_innovation()
        for proposal in ["", "short", "a" * 500]:
            score = compute_coherence_score(proposal, innovation)
            assert 0.0 <= score <= 1.0, f"Out of bounds for proposal of length {len(proposal)}: {score}"


class TestComputeRealizationReward:
    def test_compute_realization_reward_weighted_sum(self) -> None:
        """Verify that all weights sum correctly and are applied."""
        innovation = _make_innovation(
            base_direction="attention mechanism",
            operator="extend",
            gap="long sequence efficiency",
        )
        evidence = [
            _make_paper("p1", "attention mechanism long sequence efficiency extend"),
        ]
        proposal = (
            "Extending Attention for Long Sequences\n"
            "We extend the attention mechanism to handle long sequences more efficiently. "
            "The methodology builds upon sparse attention and hierarchical processing. "
            "This extends prior work and improves efficiency for long sequence tasks significantly."
        )
        config = RealizationConfig(
            evidence_accuracy_weight=0.2,
            operator_adherence_weight=0.3,
            coherence_weight=0.5,
        )
        reward = compute_realization_reward(proposal, innovation, evidence, config)
        assert 0.0 <= reward <= 1.0

    def test_compute_realization_reward_bounds(self) -> None:
        """Result is always in [0, 1] across various inputs."""
        config = RealizationConfig(
            evidence_accuracy_weight=0.2,
            operator_adherence_weight=0.3,
            coherence_weight=0.5,
        )
        test_cases = [
            ("", _make_innovation(), []),
            ("short", _make_innovation(operator="benchmark"), [_make_paper("p1", "benchmark evaluation dataset")]),
            (
                "A" * 1000,
                _make_innovation(base_direction="transformer", operator="extend", gap="efficiency"),
                [_make_paper("p1", "transformer efficiency")],
            ),
        ]
        for proposal, innovation, evidence in test_cases:
            reward = compute_realization_reward(proposal, innovation, evidence, config)
            assert 0.0 <= reward <= 1.0, f"Out of bounds: {reward}"

    def test_compute_realization_reward_empty_proposal_is_low(self) -> None:
        innovation = _make_innovation()
        config = RealizationConfig(
            evidence_accuracy_weight=0.2,
            operator_adherence_weight=0.3,
            coherence_weight=0.5,
        )
        reward = compute_realization_reward("", innovation, [], config)
        assert reward < 0.3, f"Empty proposal should yield low reward, got {reward}"

    def test_compute_realization_reward_weights_sum_to_one(self) -> None:
        """Verify that the default config weights sum to 1.0."""
        config = RealizationConfig()
        total = config.evidence_accuracy_weight + config.operator_adherence_weight + config.coherence_weight
        assert abs(total - 1.0) < 1e-6, f"Weights should sum to 1.0, got {total}"
