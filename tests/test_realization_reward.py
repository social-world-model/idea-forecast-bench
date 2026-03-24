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


class TestRLRewardPopularity:
    """Test popularity term in RL reward (forecaster/realization/reward.py)."""

    def _make_paper(self, paper_id: str, title: str = "Test paper") -> PaperRecord:
        return PaperRecord(
            paper_id=paper_id,
            title=title,
            month="2024-02",
            summary=title + " methods and evaluation",
            keywords=["cs.AI"],
            source_path=f"/fake/{paper_id}.md",
            published_date="2024-02-15",
        )

    def _make_prediction(self, title: str) -> "IdeaPrediction":
        from live_idea_bench.models import IdeaPrediction
        return IdeaPrediction(rank=1, title=title, rationale=title, approach=title)

    def test_reward_weights_default_popularity_is_zero(self) -> None:
        """Default RewardWeights has popularity=0.0 (opt-in)."""
        from forecaster.realization.config import RewardWeights
        weights = RewardWeights()
        assert weights.popularity == 0.0

    def test_reward_weights_popularity_configurable(self) -> None:
        """RewardWeights.popularity can be set."""
        from forecaster.realization.config import RewardWeights
        weights = RewardWeights(popularity=0.15)
        assert weights.popularity == 0.15

    def test_evaluate_rl_reward_popular_matched_paper_raises_reward(self) -> None:
        """When matched paper has high popularity_score, reward is higher than with low popularity."""
        from forecaster.realization.reward import evaluate_rl_reward
        from forecaster.realization.config import RewardConfig, RewardWeights

        title = "Neural scaling laws for language models"
        train = [self._make_paper("t1", "old baseline")]
        future_popular = [self._make_paper("f1", title)]
        future_popular[0] = PaperRecord(
            paper_id="f1",
            title=title,
            month="2024-02",
            summary=title,
            keywords=["cs.AI"],
            source_path="/fake/f1.md",
            published_date="2024-02-15",
            popularity_score=1.0,
        )
        future_obscure = [PaperRecord(
            paper_id="f1",
            title=title,
            month="2024-02",
            summary=title,
            keywords=["cs.AI"],
            source_path="/fake/f1.md",
            published_date="2024-02-15",
            popularity_score=0.0,
        )]

        from live_idea_bench.models import IdeaPrediction
        prediction = IdeaPrediction(rank=1, title=title, rationale=title, approach=title)

        config = RewardConfig(weights=RewardWeights(
            future_match=0.7, novelty=0.05, specificity=0.1, lead_time=0.1, popularity=0.05
        ))

        result_popular = evaluate_rl_reward(
            predictions=[prediction],
            train_papers=train,
            future_papers=future_popular,
            reward_config=config,
        )
        result_obscure = evaluate_rl_reward(
            predictions=[prediction],
            train_papers=train,
            future_papers=future_obscure,
            reward_config=config,
        )

        assert result_popular.list_reward >= result_obscure.list_reward

    def test_evaluate_rl_reward_popularity_zero_weight_unchanged(self) -> None:
        """With weights.popularity=0.0, popularity_score on paper has no effect on reward."""
        from forecaster.realization.reward import evaluate_rl_reward
        from forecaster.realization.config import RewardConfig, RewardWeights
        from live_idea_bench.models import IdeaPrediction

        title = "Neural scaling laws for language models"
        train = [self._make_paper("t1", "old baseline")]

        def _make_future(pop_score: float) -> list[PaperRecord]:
            return [PaperRecord(
                paper_id="f1", title=title, month="2024-02", summary=title,
                keywords=["cs.AI"], source_path="/fake/f1.md",
                published_date="2024-02-15", popularity_score=pop_score,
            )]

        prediction = IdeaPrediction(rank=1, title=title, rationale=title, approach=title)
        config = RewardConfig(weights=RewardWeights(popularity=0.0))

        result_high = evaluate_rl_reward(
            predictions=[prediction], train_papers=train,
            future_papers=_make_future(1.0), reward_config=config,
        )
        result_low = evaluate_rl_reward(
            predictions=[prediction], train_papers=train,
            future_papers=_make_future(0.0), reward_config=config,
        )

        assert result_high.list_reward == pytest.approx(result_low.list_reward, abs=1e-4)
