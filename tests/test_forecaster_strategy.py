"""Tests for ForecasterStrategy (Phase 6)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.strategy.forecaster import ForecasterStrategy


def _paper(paper_id: str, month: str, keywords: list[str] | None = None) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        month=month,
        summary="A study about neural networks and deep learning for research.",
        keywords=keywords or ["neural network", "deep learning", "transformer"],
        source_path=f"/fake/{paper_id}.md",
        published_date=f"{month}-01",
    )


def _make_scored_proposal(rank: int = 1, text: str = "Title\nBody text.") -> object:
    """Create a mock ScoredProposal."""
    from forecaster.models import Innovation, ScoredProposal

    return ScoredProposal(
        innovation=Innovation(
            base_direction="deep learning",
            operator="extend",
            gap="improve efficiency",
        ),
        proposal_text=text,
        prior_score=0.7,
        realization_score=0.8,
        joint_score=0.75,
        evidence_paper_ids=("p1",),
        rank=rank,
    )


class TestForecasterStrategyName:
    def test_forecaster_strategy_name(self) -> None:
        """name attribute must be 'forecaster'."""
        strategy = ForecasterStrategy()
        assert strategy.name == "forecaster"


_PATCH_JOINT_INFERENCE = "forecaster.inference.algorithm.run_joint_inference"


class TestForecasterStrategyGenerate:
    def _mock_run_joint_inference(self, proposals: list):
        """Return a patcher for run_joint_inference at the algorithm level."""
        return patch(
            _PATCH_JOINT_INFERENCE,
            return_value=proposals,
        )

    def test_forecaster_strategy_generate_returns_idea_predictions(self) -> None:
        """generate() must return a list of IdeaPrediction objects."""
        train_papers = [_paper("p1", "2024-05"), _paper("p2", "2024-05")]
        proposals = [_make_scored_proposal(rank=1), _make_scored_proposal(rank=2)]

        with self._mock_run_joint_inference(proposals), \
             patch("live_idea_bench.llm.create_client", return_value=(MagicMock(), "gpt-4o")):
            strategy = ForecasterStrategy()
            results = strategy.generate(train_papers, "2024-06", top_k=3)

        assert isinstance(results, list)
        assert all(isinstance(r, IdeaPrediction) for r in results)

    def test_forecaster_strategy_top_k_respected(self) -> None:
        """generate() must not return more items than top_k."""
        train_papers = [_paper(f"p{i}", "2024-05") for i in range(5)]
        proposals = [_make_scored_proposal(rank=i + 1) for i in range(10)]

        with self._mock_run_joint_inference(proposals), \
             patch("live_idea_bench.llm.create_client", return_value=(MagicMock(), "gpt-4o")):
            strategy = ForecasterStrategy()
            results = strategy.generate(train_papers, "2024-06", top_k=3)

        assert len(results) <= 3

    def test_forecaster_strategy_ranks_assigned(self) -> None:
        """Returned IdeaPredictions must have 1-indexed ranks."""
        train_papers = [_paper("p1", "2024-05"), _paper("p2", "2024-05")]
        proposals = [_make_scored_proposal(rank=i + 1) for i in range(3)]

        with self._mock_run_joint_inference(proposals), \
             patch("live_idea_bench.llm.create_client", return_value=(MagicMock(), "gpt-4o")):
            strategy = ForecasterStrategy()
            results = strategy.generate(train_papers, "2024-06", top_k=5)

        for idx, prediction in enumerate(results, start=1):
            assert prediction.rank == idx

    def test_forecaster_strategy_empty_papers(self) -> None:
        """generate() with empty train_papers should return an empty list gracefully."""
        strategy = ForecasterStrategy()
        # No mock needed: empty papers returns early without calling anything
        results = strategy.generate([], "2024-06", top_k=3)
        assert results == []

    def test_forecaster_strategy_heuristic_innovations_from_papers(self) -> None:
        """Without prior_checkpoint, strategy must build innovations from paper keywords."""
        train_papers = [
            _paper("p1", "2024-05", keywords=["attention", "transformers", "nlp"]),
            _paper("p2", "2024-05", keywords=["diffusion", "generation", "images"]),
        ]

        captured_innovations: list = []

        def _fake_joint_inference(innovations, papers, memory_store, llm_client, model, inference_config, realization_config):  # type: ignore[no-untyped-def]
            captured_innovations.extend(innovations)
            return []

        with patch(_PATCH_JOINT_INFERENCE, side_effect=_fake_joint_inference), \
             patch("live_idea_bench.llm.create_client", return_value=(MagicMock(), "gpt-4o")):
            strategy = ForecasterStrategy(prior_checkpoint=None)
            strategy.generate(train_papers, "2024-06", top_k=3)

        assert len(captured_innovations) > 0
        # Innovations must have proper fields
        from forecaster.models import Innovation

        for innovation in captured_innovations:
            assert isinstance(innovation, Innovation)

    def test_forecaster_strategy_innovation_count_limited(self) -> None:
        """Without prior_checkpoint, number of innovations should be <= top_k * 3."""
        train_papers = [_paper(f"p{i}", "2024-05") for i in range(20)]
        top_k = 2

        captured_innovations: list = []

        def _fake_joint_inference(innovations, papers, memory_store, llm_client, model, inference_config, realization_config):  # type: ignore[no-untyped-def]
            captured_innovations.extend(innovations)
            return []

        with patch(_PATCH_JOINT_INFERENCE, side_effect=_fake_joint_inference), \
             patch("live_idea_bench.llm.create_client", return_value=(MagicMock(), "gpt-4o")):
            strategy = ForecasterStrategy(prior_checkpoint=None)
            strategy.generate(train_papers, "2024-06", top_k=top_k)

        assert len(captured_innovations) <= top_k * 3


class TestForecasterStrategyInit:
    def test_init_defaults(self) -> None:
        """ForecasterStrategy should initialise with sensible defaults."""
        strategy = ForecasterStrategy()
        assert strategy.name == "forecaster"
        # These should default to None/empty-string without raising
        assert strategy.prior_checkpoint is None or isinstance(strategy.prior_checkpoint, str)

    def test_init_with_model_name(self) -> None:
        """Should accept model_name kwarg."""
        strategy = ForecasterStrategy(model_name="gpt-4o")
        assert strategy.model_name == "gpt-4o"


class TestForecasterStrategyPriorWiring:
    """generate() must use trained prior when prior_checkpoint is set and path exists."""

    def test_generate_uses_sample_innovations_when_checkpoint_exists(
        self, tmp_path: Path
    ) -> None:
        """When prior_checkpoint is set and path exists, sample_innovations should be called.

        The import happens lazily inside forecaster.py via
        ``from forecaster.prior.sampler import sample_innovations``.
        We patch at the source module so the lazy import picks up the mock.
        """
        train_papers = [_paper("p1", "2024-05")]
        from forecaster.models import Innovation

        fake_innovations = [
            Innovation(base_direction="novel A", operator="extend", gap="gap A"),
            Innovation(base_direction="novel B", operator="combine", gap="gap B"),
        ]
        proposals = [_make_scored_proposal(rank=1), _make_scored_proposal(rank=2)]

        captured_innovations: list = []

        def _fake_joint_inference(innovations, papers, memory_store, llm_client, model, inference_config, realization_config):  # type: ignore[no-untyped-def]
            captured_innovations.extend(innovations)
            return proposals

        # Patch at the source so the ``from forecaster.prior.sampler import`` picks it up
        with patch(_PATCH_JOINT_INFERENCE, side_effect=_fake_joint_inference), \
             patch("live_idea_bench.llm.create_client", return_value=(MagicMock(), "gpt-4o")), \
             patch("forecaster.prior.sampler.sample_innovations", return_value=fake_innovations):
            strategy = ForecasterStrategy(prior_checkpoint=str(tmp_path))
            results = strategy.generate(train_papers, "2024-06", top_k=3)

        assert captured_innovations == fake_innovations

    def test_generate_falls_back_to_heuristic_when_prior_sampling_raises(
        self, tmp_path: Path
    ) -> None:
        """When sample_innovations raises, generate() should fall back to heuristic."""
        train_papers = [_paper("p1", "2024-05")]
        proposals = [_make_scored_proposal(rank=1)]

        captured_innovations: list = []

        def _fake_joint_inference(innovations, papers, memory_store, llm_client, model, inference_config, realization_config):  # type: ignore[no-untyped-def]
            captured_innovations.extend(innovations)
            return proposals

        with patch(_PATCH_JOINT_INFERENCE, side_effect=_fake_joint_inference), \
             patch("live_idea_bench.llm.create_client", return_value=(MagicMock(), "gpt-4o")), \
             patch("forecaster.prior.sampler.sample_innovations", side_effect=RuntimeError("model load failed")):
            strategy = ForecasterStrategy(prior_checkpoint=str(tmp_path))
            results = strategy.generate(train_papers, "2024-06", top_k=3)

        # Should still have produced heuristic innovations (no crash)
        assert len(captured_innovations) >= 1
        from forecaster.models import Innovation
        for inn in captured_innovations:
            assert isinstance(inn, Innovation)

    def test_generate_uses_heuristic_when_no_checkpoint(self) -> None:
        """When prior_checkpoint is None, heuristic innovations should be used."""
        train_papers = [_paper("p1", "2024-05", keywords=["attention", "transformers", "nlp"])]
        proposals = [_make_scored_proposal(rank=1)]

        captured_innovations: list = []

        def _fake_joint_inference(innovations, papers, memory_store, llm_client, model, inference_config, realization_config):  # type: ignore[no-untyped-def]
            captured_innovations.extend(innovations)
            return proposals

        with patch(_PATCH_JOINT_INFERENCE, side_effect=_fake_joint_inference), \
             patch("live_idea_bench.llm.create_client", return_value=(MagicMock(), "gpt-4o")):
            strategy = ForecasterStrategy(prior_checkpoint=None)
            strategy.generate(train_papers, "2024-06", top_k=3)

        assert len(captured_innovations) >= 1
