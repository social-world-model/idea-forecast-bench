"""Tests for ForecasterPipeline orchestrator (Phase 6)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from live_idea_bench.models import PaperRecord
from forecaster.models import HindsightSample, Innovation, ScoredProposal
from forecaster.orchestrator import ForecasterPipeline


def _paper(paper_id: str, month: str) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        month=month,
        summary="A study on machine learning.",
        keywords=["ml", "ai"],
        source_path=f"/fake/{paper_id}.md",
        published_date=f"{month}-01",
    )


def _make_innovation(direction: str = "deep learning") -> Innovation:
    return Innovation(base_direction=direction, operator="extend", gap="improve efficiency")


def _make_hindsight_sample(paper_id: str = "p1", month: str = "2024-01") -> HindsightSample:
    return HindsightSample(
        context_paper_ids=("ctx1",),
        cutoff_month=month,
        future_paper_id=paper_id,
        innovation=_make_innovation(),
    )


def _make_scored_proposal(rank: int = 1) -> ScoredProposal:
    return ScoredProposal(
        innovation=_make_innovation(),
        proposal_text="Title\nBody.",
        prior_score=0.5,
        realization_score=0.7,
        joint_score=0.62,
        evidence_paper_ids=(),
        rank=rank,
    )


class TestForecasterPipelineInit:
    def test_forecaster_pipeline_init(self, tmp_path: Path) -> None:
        """ForecasterPipeline should initialize with default configs."""
        papers = [_paper("p1", "2024-01")]
        pipeline = ForecasterPipeline(papers=papers, output_dir=tmp_path / "out")
        assert pipeline.papers is papers
        assert isinstance(pipeline.output_dir, Path)

    def test_forecaster_pipeline_stores_papers(self, tmp_path: Path) -> None:
        """Papers should be stored on the pipeline."""
        papers = [_paper("p1", "2024-01"), _paper("p2", "2024-02")]
        pipeline = ForecasterPipeline(papers=papers, output_dir=tmp_path / "out")
        assert len(pipeline.papers) == 2

    def test_forecaster_pipeline_default_llm_model(self, tmp_path: Path) -> None:
        """Default LLM model should be gpt-4o."""
        pipeline = ForecasterPipeline(papers=[], output_dir=tmp_path)
        assert pipeline.llm_model == "gpt-4o"

    def test_forecaster_pipeline_custom_llm_model(self, tmp_path: Path) -> None:
        """Custom LLM model should be stored."""
        pipeline = ForecasterPipeline(papers=[], output_dir=tmp_path, llm_model="claude-3-haiku-20240307")
        assert pipeline.llm_model == "claude-3-haiku-20240307"

    def test_forecaster_pipeline_output_dir_as_string(self, tmp_path: Path) -> None:
        """output_dir as a plain string should be wrapped in Path."""
        pipeline = ForecasterPipeline(papers=[], output_dir=str(tmp_path / "out"))
        assert isinstance(pipeline.output_dir, Path)


class TestForecasterPipelineOutputDir:
    def test_forecaster_pipeline_output_dir_created(self, tmp_path: Path) -> None:
        """Calling run_hindsight_extraction should create the output_dir."""
        out_dir = tmp_path / "nested" / "output"
        papers = [_paper("p1", "2024-01"), _paper("p2", "2024-06")]
        pipeline = ForecasterPipeline(papers=papers, output_dir=out_dir)

        mock_client = MagicMock()
        samples = [_make_hindsight_sample()]

        with patch("forecaster.orchestrator.create_client", return_value=(mock_client, "gpt-4o")), \
             patch("forecaster.orchestrator.build_hindsight_dataset", return_value=samples):
            pipeline.run_hindsight_extraction(cutoff_months=["2024-01"], horizon_months=6)

        assert out_dir.exists()


class TestForecasterPipelineRunHindsightExtraction:
    def test_forecaster_pipeline_run_hindsight_extraction_with_mock(
        self, tmp_path: Path
    ) -> None:
        """run_hindsight_extraction should call build_hindsight_dataset and return samples."""
        papers = [_paper("p1", "2024-01"), _paper("p2", "2024-06")]
        pipeline = ForecasterPipeline(papers=papers, output_dir=tmp_path / "out")

        mock_client = MagicMock()
        expected_samples = [_make_hindsight_sample("p2", "2024-01")]

        with patch("forecaster.orchestrator.create_client", return_value=(mock_client, "gpt-4o")), \
             patch("forecaster.orchestrator.build_hindsight_dataset", return_value=expected_samples) as mock_build:
            samples = pipeline.run_hindsight_extraction(
                cutoff_months=["2024-01"],
                horizon_months=6,
            )

        assert samples == expected_samples
        mock_build.assert_called_once()

    def test_hindsight_extraction_passes_papers_to_builder(self, tmp_path: Path) -> None:
        """Papers should be passed to build_hindsight_dataset."""
        papers = [_paper("p1", "2024-01")]
        pipeline = ForecasterPipeline(papers=papers, output_dir=tmp_path)

        captured_args: dict = {}

        def _capture_call(papers, cutoff_months, horizon_months, config, llm_client, model, **kwargs):  # type: ignore[no-untyped-def]
            captured_args["papers"] = papers
            captured_args["cutoff_months"] = cutoff_months
            return []

        mock_client = MagicMock()
        with patch("forecaster.orchestrator.create_client", return_value=(mock_client, "gpt-4o")), \
             patch("forecaster.orchestrator.build_hindsight_dataset", side_effect=_capture_call):
            pipeline.run_hindsight_extraction(cutoff_months=["2024-01"], horizon_months=3)

        assert captured_args["papers"] is papers
        assert captured_args["cutoff_months"] == ["2024-01"]


class TestForecasterPipelineRunJointInference:
    def test_forecaster_pipeline_run_joint_inference_with_mock(
        self, tmp_path: Path
    ) -> None:
        """run_joint_inference should return ScoredProposal list."""
        papers = [_paper("p1", "2024-01"), _paper("p2", "2024-02")]
        pipeline = ForecasterPipeline(papers=papers, output_dir=tmp_path)

        innovations = [_make_innovation()]
        expected_proposals = [_make_scored_proposal(rank=1)]
        mock_client = MagicMock()

        with patch("forecaster.orchestrator.create_client", return_value=(mock_client, "gpt-4o")), \
             patch("forecaster.orchestrator.run_joint_inference_fn", return_value=expected_proposals):
            proposals = pipeline.run_joint_inference(
                cutoff_month="2024-02",
                innovations=innovations,
            )

        assert proposals == expected_proposals

    def test_joint_inference_filters_papers_by_cutoff(self, tmp_path: Path) -> None:
        """Papers after cutoff_month should not be in the training set passed to inference."""
        papers = [
            _paper("p1", "2024-01"),
            _paper("p2", "2024-02"),
            _paper("p3", "2024-06"),  # future paper — should be filtered out
        ]
        pipeline = ForecasterPipeline(papers=papers, output_dir=tmp_path)

        captured_papers: list = []

        def _capture_inference(innovations, papers, memory_store, llm_client, model, inference_config, realization_config):  # type: ignore[no-untyped-def]
            captured_papers.extend(papers)
            return []

        mock_client = MagicMock()
        with patch("forecaster.orchestrator.create_client", return_value=(mock_client, "gpt-4o")), \
             patch("forecaster.orchestrator.run_joint_inference_fn", side_effect=_capture_inference):
            pipeline.run_joint_inference(
                cutoff_month="2024-02",
                innovations=[_make_innovation()],
            )

        paper_ids = [p.paper_id for p in captured_papers]
        assert "p3" not in paper_ids


class TestForecasterPipelineRunPriorTraining:
    def test_run_prior_training_returns_checkpoint_path(self, tmp_path: Path) -> None:
        """run_prior_training should return a string checkpoint path."""
        pipeline = ForecasterPipeline(papers=[], output_dir=tmp_path)
        samples = [_make_hindsight_sample()]

        fake_checkpoint = str(tmp_path / "checkpoint")

        with patch("forecaster.orchestrator.build_sft_samples", return_value=[{"input": "x", "target": "y"}]), \
             patch("forecaster.orchestrator.train_prior", return_value=fake_checkpoint):
            result = pipeline.run_prior_training(hindsight_samples=samples)

        assert result == fake_checkpoint
