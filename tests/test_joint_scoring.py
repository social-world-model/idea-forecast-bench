"""Tests for forecaster/inference/scoring.py"""

from __future__ import annotations

import math

import pytest

from forecaster.config import InferenceConfig, RealizationConfig
from forecaster.inference.scoring import (
    compute_joint_score,
    compute_prior_score,
    compute_realization_score,
    compute_strict_joint_score,
)
from forecaster.models import Innovation, MemoryEntry, MemoryInventory
from forecaster.prior.memory import MemoryStore
from live_idea_bench.models import PaperRecord


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


def _make_memory_store_with(
    innovation: Innovation, frequency: int = 3, recency_score: float = 0.8
) -> MemoryStore:
    entry = MemoryEntry(
        innovation=innovation,
        source_paper_id="paper_123",
        timestamp_month="2024-01",
        frequency=frequency,
        recency_score=recency_score,
    )
    inventory = MemoryInventory(
        entries=(entry,),
        last_updated_month="2024-01",
    )
    return MemoryStore(inventory)


class TestComputePriorScore:
    def test_compute_prior_score_in_memory(self) -> None:
        """Exact-match heuristic scores are calibrated into log-like space."""
        innovation = _make_innovation()
        store = _make_memory_store_with(innovation, frequency=3, recency_score=0.8)

        score = compute_prior_score(innovation, store)

        # strength = 0.45*0.8 + 0.35*1.0 + 0.2*0.5 = 0.81
        expected = math.log(0.81)
        assert abs(score - expected) < 1e-6
        assert score < 0.0

    def test_compute_prior_score_not_in_memory(self) -> None:
        """Innovation not found in memory returns the calibrated base score."""
        innovation = _make_innovation()
        store = MemoryStore.empty("2024-01")

        score = compute_prior_score(innovation, store)

        assert score == pytest.approx(math.log(1e-6))

    def test_compute_prior_score_different_innovation_semantic_fallback(self) -> None:
        """Innovation with different fields may get a semantic score instead of -2.0.

        With the semantic prior scoring, non-exact matches that share keywords can
        score above the base -2.0. Completely unrelated innovations still score -2.0.
        """
        stored_innovation = _make_innovation(gap="different gap")
        query_innovation = _make_innovation(gap="long sequence efficiency")
        store = _make_memory_store_with(stored_innovation)

        score = compute_prior_score(query_innovation, store)

        # Score is either base (-2.0) or a semantic score; must be >= -2.0
        assert score >= -2.0

    def test_compute_prior_score_unrelated_innovation_returns_base(self) -> None:
        """Completely unrelated innovation returns the calibrated base score."""
        stored_innovation = _make_innovation(
            base_direction="quantum physics",
            operator="apply",
            gap="superconductor materials",
        )
        query_innovation = _make_innovation(
            base_direction="transformer attention",
            operator="extend",
            gap="long sequence efficiency",
        )
        store = _make_memory_store_with(stored_innovation)

        score = compute_prior_score(query_innovation, store)

        assert score == pytest.approx(math.log(1e-6))

    def test_compute_prior_score_in_memory_frequency_one(self) -> None:
        """Frequency/recency/utility blend is normalized before log scaling."""
        innovation = _make_innovation()
        store = _make_memory_store_with(innovation, frequency=1, recency_score=1.0)

        score = compute_prior_score(innovation, store)

        expected = math.log(0.9)
        assert abs(score - expected) < 1e-6


class TestComputeRealizationScore:
    def test_compute_realization_score_good_proposal(self) -> None:
        """Good proposal returns higher score than empty proposal."""
        innovation = _make_innovation(operator="extend")
        evidence = [
            _make_paper(
                "p1",
                "transformer attention extension for long sequences training evaluation",
            )
        ]
        config = RealizationConfig()

        good_text = (
            "Extending Transformer Attention for Long Sequences\n"
            "We extend the transformer model training architecture for evaluation of "
            "sequence efficiency. The approach builds upon attention mechanism with "
            "improvements for long document processing. Baseline experiments demonstrate "
            "improved performance metrics and gradient efficiency."
        )
        good_score = compute_realization_score(good_text, innovation, evidence, config)
        empty_score = compute_realization_score("", innovation, evidence, config)

        assert good_score > empty_score

    def test_compute_realization_score_empty_proposal(self) -> None:
        """Empty proposal returns a low (very negative) score."""
        innovation = _make_innovation()
        evidence: list[PaperRecord] = []
        config = RealizationConfig()

        score = compute_realization_score("", innovation, evidence, config)

        # reward ~0 → log(0 + 1e-6) ≈ -13.8
        assert score < -10.0

    def test_compute_realization_score_returns_float(self) -> None:
        """Always returns a float."""
        innovation = _make_innovation()
        config = RealizationConfig()

        score = compute_realization_score(
            "Some proposal text here.", innovation, [], config
        )

        assert isinstance(score, float)

    def test_compute_realization_score_range(self) -> None:
        """Score is always <= 0 (log of [0, 1] + epsilon)."""
        innovation = _make_innovation(operator="extend")
        evidence = [_make_paper("p1", "attention mechanism training")]
        config = RealizationConfig()

        proposal = "extend transformer model for evaluation benchmark"
        score = compute_realization_score(proposal, innovation, evidence, config)

        # log(reward + 1e-6) where reward in [0,1] → score in (-inf, ~0]
        assert score <= 0.0


class TestComputeJointScore:
    def test_compute_joint_score_linear_blend(self) -> None:
        """Main path follows Algorithm 1's linear combination exactly."""
        config = InferenceConfig(prior_weight=0.5, realization_weight=0.5)
        score = compute_joint_score(-1.2, -0.4, config)
        assert score == pytest.approx((-1.2 * 0.5) + (-0.4 * 0.5))

    def test_compute_joint_score_prior_weight(self) -> None:
        """Higher prior_weight means the joint score tracks prior_score more closely."""
        prior_score = -0.2
        real_score = -3.0

        config_high_prior = InferenceConfig(prior_weight=0.9, realization_weight=0.1)
        config_low_prior = InferenceConfig(prior_weight=0.1, realization_weight=0.9)

        score_high = compute_joint_score(prior_score, real_score, config_high_prior)
        score_low = compute_joint_score(prior_score, real_score, config_low_prior)

        assert score_high > score_low

    def test_compute_joint_score_default_weights(self) -> None:
        """Default config has prior=0.4, realization=0.6 and uses linear blend."""
        config = InferenceConfig()
        assert config.prior_weight == 0.4
        assert config.realization_weight == 0.6

        score = compute_joint_score(-1.0, -0.5, config)
        assert score == pytest.approx((0.4 * -1.0) + (0.6 * -0.5))

    def test_compute_joint_score_realization_dominates(self) -> None:
        """High realization_weight makes realization_score dominate."""
        prior_score = -5.0
        real_score = -0.1

        config = InferenceConfig(prior_weight=0.1, realization_weight=0.9)
        score = compute_joint_score(prior_score, real_score, config)

        assert score == pytest.approx((0.1 * prior_score) + (0.9 * real_score))
        assert score > -1.0

    def test_compute_joint_score_with_popularity_bonus_increases_score(self) -> None:
        """A popularity_bonus > 0 with popularity_weight > 0 increases the score."""
        config = InferenceConfig(
            runtime_mode="demo",
            prior_weight=0.4,
            realization_weight=0.6,
            popularity_weight=0.2,
        )
        base = compute_joint_score(-1.0, -1.0, config, popularity_bonus=0.0)
        boosted = compute_joint_score(-1.0, -1.0, config, popularity_bonus=1.0)
        assert boosted > base

    def test_compute_joint_score_popularity_weight_zero_is_unchanged(self) -> None:
        """When popularity_weight=0.0 (default), bonus has no effect."""
        config_default = InferenceConfig(prior_weight=0.4, realization_weight=0.6)
        config_explicit_zero = InferenceConfig(
            prior_weight=0.4, realization_weight=0.6, popularity_weight=0.0
        )

        score_no_bonus = compute_joint_score(-0.7, -0.4, config_default)
        score_bonus_default = compute_joint_score(
            -0.7, -0.4, config_default, popularity_bonus=1.0
        )
        score_bonus_explicit_zero = compute_joint_score(
            -0.7, -0.4, config_explicit_zero, popularity_bonus=1.0
        )

        assert score_no_bonus == pytest.approx(score_bonus_default)
        assert score_no_bonus == pytest.approx(score_bonus_explicit_zero)

    def test_inference_config_default_popularity_weight_is_zero(self) -> None:
        """Default InferenceConfig has popularity_weight=0.0 (opt-in)."""
        config = InferenceConfig()
        assert config.popularity_weight == 0.0

    def test_compute_strict_joint_score_uses_only_prior_and_realization(self) -> None:
        config = InferenceConfig(
            runtime_mode="strict_eval",
            prior_weight=0.4,
            realization_weight=0.6,
        )
        score = compute_strict_joint_score(-0.7, -0.4, config)

        assert score == pytest.approx((0.4 * -0.7) + (0.6 * -0.4))
