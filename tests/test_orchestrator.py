"""Tests for ForecasterPipeline orchestrator (Phase 6)."""
from __future__ import annotations

import json
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
        future_paper_published_date=f"{month}-15",
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

        def _capture_inference(innovations, papers, memory_store, llm_client, model, inference_config, realization_config, **kwargs):  # type: ignore[no-untyped-def]
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


class TestRunFullPipelineMemoryPopulation:
    """run_full_pipeline must materialize the legal eval memory snapshot."""

    def test_memory_populated_after_hindsight_extraction(self, tmp_path: Path) -> None:
        """Eval memory should exclude hindsight labels whose source papers are still in the future."""
        papers = [_paper("p1", "2024-01"), _paper("p2", "2024-02"), _paper("p3", "2024-03")]
        pipeline = ForecasterPipeline(papers=papers, output_dir=tmp_path / "out")

        # Two hindsight samples with distinct innovations
        sample1 = _make_hindsight_sample("p1", "2024-01")
        sample2 = HindsightSample(
            context_paper_ids=("ctx2",),
            cutoff_month="2024-02",
            future_paper_id="p3",
            future_paper_published_date="2024-03-15",
            innovation=Innovation(
                base_direction="reinforcement learning",
                operator="compose",
                gap="sample efficiency",
            ),
        )

        mock_client = MagicMock()
        with patch("forecaster.orchestrator.create_client", return_value=(mock_client, "gpt-4o")), \
             patch("forecaster.orchestrator.build_hindsight_dataset", return_value=[sample1, sample2]), \
             patch("forecaster.orchestrator.train_prior", return_value=""), \
             patch("forecaster.orchestrator.build_sft_samples", return_value=[]), \
             patch("forecaster.orchestrator.run_joint_inference_fn", return_value=[]):
            pipeline.run_full_pipeline(
                cutoff_months=["2024-01", "2024-02"],
                skip_training=True,
                strict_eval=False,
            )

        assert pipeline._memory_store.size == 1
        assert pipeline._memory_store.inventory.entries[0].source_paper_id == "p1"

    def test_memory_persisted_to_disk(self, tmp_path: Path) -> None:
        """run_full_pipeline should persist memory_inventory.json to output_dir."""
        papers = [_paper("p1", "2024-01")]
        pipeline = ForecasterPipeline(papers=papers, output_dir=tmp_path / "out")

        sample1 = _make_hindsight_sample("p1", "2024-01")

        mock_client = MagicMock()
        with patch("forecaster.orchestrator.create_client", return_value=(mock_client, "gpt-4o")), \
             patch("forecaster.orchestrator.build_hindsight_dataset", return_value=[sample1]), \
             patch("forecaster.orchestrator.train_prior", return_value=""), \
             patch("forecaster.orchestrator.build_sft_samples", return_value=[]), \
             patch("forecaster.orchestrator.run_joint_inference_fn", return_value=[]):
            pipeline.run_full_pipeline(
                cutoff_months=["2024-01"],
                skip_training=True,
                strict_eval=False,
            )

        assert (tmp_path / "out" / "memory_inventory.json").exists()
        assert (tmp_path / "out" / "runtime_contract.json").exists()

    def test_strict_eval_does_not_pass_future_memory_to_inference(self, tmp_path: Path) -> None:
        """The eval memory snapshot passed into inference must exclude eval future labels."""
        papers = [_paper("p1", "2024-01"), _paper("p2", "2024-02"), _paper("p3", "2024-03")]
        pipeline = ForecasterPipeline(papers=papers, output_dir=tmp_path / "out")

        legal = HindsightSample(
            context_paper_ids=("ctx1",),
            cutoff_month="2024-01",
            future_paper_id="p2",
            future_paper_published_date="2024-02-01",
            innovation=Innovation(
                base_direction="legal memory",
                operator="extend",
                gap="visible by eval",
            ),
        )
        future_only = HindsightSample(
            context_paper_ids=("ctx2",),
            cutoff_month="2024-02",
            future_paper_id="p3",
            future_paper_published_date="2024-03-20",
            innovation=Innovation(
                base_direction="future memory",
                operator="extend",
                gap="not visible by eval",
            ),
        )

        captured_memory = {}
        fake_checkpoint = str(tmp_path / "ckpt")
        fake_realization = tmp_path / "realization"
        Path(fake_checkpoint).mkdir(parents=True)
        fake_realization.mkdir(parents=True)

        def _capture_inference(innovations, papers, memory_store, llm_client, model, inference_config, realization_config, **kwargs):  # type: ignore[no-untyped-def]
            captured_memory["entries"] = memory_store.inventory.entries
            return []

        mock_client = MagicMock()
        with patch("forecaster.orchestrator.create_client", return_value=(mock_client, "gpt-4o")), \
             patch("forecaster.orchestrator.build_hindsight_dataset", return_value=[legal, future_only]), \
             patch("forecaster.orchestrator.build_sft_samples", return_value=[]), \
             patch("forecaster.orchestrator.train_prior", return_value=fake_checkpoint), \
             patch("forecaster.orchestrator.sample_innovations", return_value=[_make_innovation("strict")]), \
             patch("forecaster.orchestrator.ForecasterPipeline.run_realization_training", return_value="manifest.json"), \
             patch("forecaster.orchestrator._extract_realization_model_path", return_value=str(fake_realization)), \
             patch("forecaster.orchestrator.run_joint_inference_fn", side_effect=_capture_inference):
            pipeline.run_full_pipeline(
                cutoff_months=["2024-01", "2024-02"],
                skip_training=False,
                strict_eval=True,
            )

        source_ids = [entry.source_paper_id for entry in captured_memory["entries"]]
        assert source_ids == ["p2"]


class TestRunFullPipelinePriorWiring:
    """run_full_pipeline must call sample_innovations when a checkpoint is available."""

    def test_filters_eval_future_labels_before_prior_training(self, tmp_path: Path) -> None:
        """Strict eval must not train the prior on labels from the eval future window."""
        papers = [_paper("p1", "2024-01"), _paper("p2", "2024-02"), _paper("p3", "2024-03")]
        pipeline = ForecasterPipeline(papers=papers, output_dir=tmp_path / "out")

        legal_training_sample = HindsightSample(
            context_paper_ids=("ctx1",),
            cutoff_month="2024-01",
            future_paper_id="p2",
            future_paper_published_date="2024-02-01",
            innovation=_make_innovation("legal direction"),
        )
        eval_future_sample = HindsightSample(
            context_paper_ids=("ctx2",),
            cutoff_month="2024-02",
            future_paper_id="p3",
            future_paper_published_date="2024-03-15",
            innovation=_make_innovation("eval future direction"),
        )

        captured_future_ids: list[str] = []

        def _capture_build_sft(samples, **kwargs):  # type: ignore[no-untyped-def]
            if "memory_snapshots_by_cutoff" not in kwargs:
                captured_future_ids.extend(sample.future_paper_id for sample in samples)
            return []

        mock_client = MagicMock()
        fake_checkpoint = str(tmp_path / "ckpt")
        fake_realization = tmp_path / "realization"
        Path(fake_checkpoint).mkdir(parents=True)
        fake_realization.mkdir(parents=True)
        with patch("forecaster.orchestrator.create_client", return_value=(mock_client, "gpt-4o")), \
             patch("forecaster.orchestrator.build_hindsight_dataset", return_value=[legal_training_sample, eval_future_sample]), \
             patch("forecaster.orchestrator.build_sft_samples", side_effect=_capture_build_sft), \
             patch("forecaster.orchestrator.train_prior", return_value=fake_checkpoint), \
             patch("forecaster.orchestrator.sample_innovations", return_value=[_make_innovation("strict")]), \
             patch("forecaster.orchestrator.ForecasterPipeline.run_realization_training", return_value="manifest.json"), \
             patch("forecaster.orchestrator._extract_realization_model_path", return_value=str(fake_realization)), \
             patch("forecaster.orchestrator.run_joint_inference_fn", return_value=[]):
            pipeline.run_full_pipeline(
                cutoff_months=["2024-01", "2024-02"],
                skip_training=False,
                strict_eval=True,
            )

        assert captured_future_ids == ["p2"]

    def test_calls_sample_innovations_when_checkpoint_exists(self, tmp_path: Path) -> None:
        """When prior_checkpoint exists, run_full_pipeline should call sample_innovations."""
        papers = [_paper("p1", "2024-01"), _paper("p2", "2024-02")]
        pipeline = ForecasterPipeline(papers=papers, output_dir=tmp_path / "out")

        fake_checkpoint = str(tmp_path / "ckpt")
        Path(fake_checkpoint).mkdir(parents=True)

        fake_innovations = [_make_innovation("novel direction A"), _make_innovation("novel direction B")]
        legal_training_sample = HindsightSample(
            context_paper_ids=("ctx1",),
            cutoff_month="2024-01",
            future_paper_id="p2",
            future_paper_published_date="2024-02-01",
            innovation=_make_innovation("legal direction"),
        )

        mock_client = MagicMock()
        captured_innovations: list = []

        def _capture_inference(innovations, papers, memory_store, llm_client, model, inference_config, realization_config, **kwargs):  # type: ignore[no-untyped-def]
            captured_innovations.extend(innovations)
            return []

        with patch("forecaster.orchestrator.create_client", return_value=(mock_client, "gpt-4o")), \
             patch("forecaster.orchestrator.build_hindsight_dataset", return_value=[legal_training_sample]), \
             patch("forecaster.orchestrator.build_sft_samples", return_value=[]), \
             patch("forecaster.orchestrator.train_prior", return_value=fake_checkpoint), \
             patch("forecaster.orchestrator.sample_innovations", return_value=fake_innovations) as mock_sample, \
             patch("forecaster.orchestrator.ForecasterPipeline.run_realization_training", return_value=None), \
             patch("forecaster.orchestrator.run_joint_inference_fn", side_effect=_capture_inference):
            pipeline.run_full_pipeline(
                cutoff_months=["2024-01", "2024-02"],
                skip_training=False,
                strict_eval=False,
            )

        mock_sample.assert_called_once()
        assert captured_innovations == fake_innovations

    def test_falls_back_to_heuristic_when_no_checkpoint(self, tmp_path: Path) -> None:
        """When skip_training=True and no prior_checkpoint, heuristic innovations are used."""
        papers = [_paper("p1", "2024-01")]
        pipeline = ForecasterPipeline(papers=papers, output_dir=tmp_path / "out")

        mock_client = MagicMock()
        captured_innovations: list = []

        def _capture_inference(innovations, papers, memory_store, llm_client, model, inference_config, realization_config, **kwargs):  # type: ignore[no-untyped-def]
            captured_innovations.extend(innovations)
            return []

        with patch("forecaster.orchestrator.create_client", return_value=(mock_client, "gpt-4o")), \
             patch("forecaster.orchestrator.build_hindsight_dataset", return_value=[]), \
             patch("forecaster.orchestrator.sample_innovations") as mock_sample, \
             patch("forecaster.orchestrator.run_joint_inference_fn", side_effect=_capture_inference):
            pipeline.run_full_pipeline(
                cutoff_months=["2024-01"],
                skip_training=True,
                strict_eval=False,
            )

        mock_sample.assert_not_called()
        # Heuristic from a single paper that falls in the training window
        assert len(captured_innovations) >= 1

    def test_falls_back_to_heuristic_when_sample_innovations_raises(self, tmp_path: Path) -> None:
        """When sample_innovations raises, heuristic innovations are used as fallback."""
        papers = [_paper("p1", "2024-01"), _paper("p2", "2024-02")]
        pipeline = ForecasterPipeline(papers=papers, output_dir=tmp_path / "out")

        fake_checkpoint = str(tmp_path / "ckpt")
        Path(fake_checkpoint).mkdir(parents=True)
        legal_training_sample = HindsightSample(
            context_paper_ids=("ctx1",),
            cutoff_month="2024-01",
            future_paper_id="p2",
            future_paper_published_date="2024-02-01",
            innovation=_make_innovation("legal direction"),
        )

        mock_client = MagicMock()
        captured_innovations: list = []

        def _capture_inference(innovations, papers, memory_store, llm_client, model, inference_config, realization_config, **kwargs):  # type: ignore[no-untyped-def]
            captured_innovations.extend(innovations)
            return []

        with patch("forecaster.orchestrator.create_client", return_value=(mock_client, "gpt-4o")), \
             patch("forecaster.orchestrator.build_hindsight_dataset", return_value=[legal_training_sample]), \
             patch("forecaster.orchestrator.build_sft_samples", return_value=[]), \
             patch("forecaster.orchestrator.train_prior", return_value=fake_checkpoint), \
             patch("forecaster.orchestrator.sample_innovations", side_effect=RuntimeError("GPU OOM")), \
             patch("forecaster.orchestrator.ForecasterPipeline.run_realization_training", return_value=None), \
             patch("forecaster.orchestrator.run_joint_inference_fn", side_effect=_capture_inference):
            pipeline.run_full_pipeline(
                cutoff_months=["2024-01", "2024-02"],
                skip_training=False,
                strict_eval=False,
            )

        # Should have fallen back to heuristic (paper p1 has month 2024-01 <= cutoff)
        assert len(captured_innovations) >= 1


class TestForecasterPipelineStrictMode:
    def test_strict_mode_raises_when_prior_checkpoint_missing(self, tmp_path: Path) -> None:
        """Strict mode should fail closed instead of constructing heuristic innovations."""
        pipeline = ForecasterPipeline(papers=[_paper("p1", "2024-01")], output_dir=tmp_path / "out")
        mock_client = MagicMock()

        with patch("forecaster.orchestrator.create_client", return_value=(mock_client, "gpt-4o")), \
             patch("forecaster.orchestrator.build_hindsight_dataset", return_value=[]):
            with pytest.raises(RuntimeError, match="prior checkpoint"):
                pipeline.run_full_pipeline(
                    cutoff_months=["2024-01"],
                    skip_training=True,
                    strict_eval=True,
                )

    def test_run_realization_training_enables_alignment_gate_by_default(self, tmp_path: Path) -> None:
        """Real training should run the alignment gate."""
        pipeline = ForecasterPipeline(papers=[_paper("p1", "2024-01")], output_dir=tmp_path / "out")
        captured_kwargs: dict[str, object] = {}

        def _capture_pipeline(*args, **kwargs):  # type: ignore[no-untyped-def]
            captured_kwargs.update(kwargs)
            return {"trainer_output_dir": str(tmp_path / "trainer")}

        with patch("forecaster.realization.pipeline.run_policy_rl_pipeline", side_effect=_capture_pipeline), \
             patch("forecaster.realization.config.load_episode_build_config"), \
             patch("forecaster.realization.config.load_candidate_generation_config"), \
             patch("forecaster.realization.config.load_grpo_train_config"), \
             patch("forecaster.realization.config.load_reward_config"), \
             patch("forecaster.realization.config.load_selection_config"):
            result = pipeline.run_realization_training(cutoff_months=["2024-01"])

        assert captured_kwargs["skip_alignment_check"] is False
        assert result is not None

    def test_run_realization_training_dry_run_skips_alignment_gate(self, tmp_path: Path) -> None:
        """Dry-run training opts out of the alignment gate."""
        pipeline = ForecasterPipeline(papers=[_paper("p1", "2024-01")], output_dir=tmp_path / "out")
        captured_kwargs: dict[str, object] = {}

        def _capture_pipeline(*args, **kwargs):  # type: ignore[no-untyped-def]
            captured_kwargs.update(kwargs)
            return {"trainer_output_dir": str(tmp_path / "trainer")}

        with patch("forecaster.realization.pipeline.run_policy_rl_pipeline", side_effect=_capture_pipeline), \
             patch("forecaster.realization.config.load_episode_build_config"), \
             patch("forecaster.realization.config.load_candidate_generation_config"), \
             patch("forecaster.realization.config.load_grpo_train_config"), \
             patch("forecaster.realization.config.load_reward_config"), \
             patch("forecaster.realization.config.load_selection_config"):
            result = pipeline.run_realization_training(
                cutoff_months=["2024-01"],
                dry_run=True,
            )

        assert captured_kwargs["skip_alignment_check"] is True
        assert result is not None

    def test_strict_mode_runs_prior_refresh_and_persists_refresh_artifacts(self, tmp_path: Path) -> None:
        pipeline = ForecasterPipeline(
            papers=[_paper("p1", "2024-01"), _paper("p2", "2024-02")],
            output_dir=tmp_path / "out",
        )
        mock_client = MagicMock()
        bootstrap_checkpoint = str(tmp_path / "bootstrap")
        refresh_checkpoint = str(tmp_path / "refresh")
        fake_realization = tmp_path / "realization"
        Path(bootstrap_checkpoint).mkdir(parents=True)
        Path(refresh_checkpoint).mkdir(parents=True)
        fake_realization.mkdir(parents=True)

        with patch("forecaster.orchestrator.create_client", return_value=(mock_client, "gpt-4o")), \
             patch("forecaster.orchestrator.build_hindsight_dataset", return_value=[_make_hindsight_sample("p2", "2024-01")]), \
             patch("forecaster.orchestrator.build_sft_samples", return_value=[]), \
             patch("forecaster.orchestrator.train_prior", side_effect=[bootstrap_checkpoint, refresh_checkpoint]), \
             patch("forecaster.orchestrator.sample_innovations", return_value=[_make_innovation("strict")]), \
             patch("forecaster.orchestrator.ForecasterPipeline.run_realization_training", return_value="manifest.json"), \
             patch("forecaster.orchestrator._extract_realization_model_path", return_value=str(fake_realization)), \
             patch("forecaster.orchestrator.run_joint_inference_fn", return_value=[_make_scored_proposal()]), \
             patch("forecaster.orchestrator._score_proposals_for_delayed_feedback", return_value=[]):
            result = pipeline.run_full_pipeline(
                cutoff_months=["2024-01", "2024-02"],
                skip_training=False,
                strict_eval=True,
            )

        runtime_contract = json.loads((tmp_path / "out" / "runtime_contract.json").read_text(encoding="utf-8"))
        assert result["bootstrap_prior_checkpoint"] == bootstrap_checkpoint
        assert result["refresh_prior_checkpoint"] == refresh_checkpoint
        assert result["prior_checkpoint"] == refresh_checkpoint
        assert (tmp_path / "out" / "prior_refresh" / "refresh_manifest.json").exists()
        assert runtime_contract["artifacts"]["bootstrap_prior_checkpoint"] == bootstrap_checkpoint
        assert runtime_contract["artifacts"]["refresh_prior_checkpoint"] == refresh_checkpoint
        assert runtime_contract["artifacts"]["final_prior_checkpoint"] == refresh_checkpoint
        assert runtime_contract["score_contract"]["prior_score_method"] == "conditional_logprob"
        assert runtime_contract["score_contract"]["realization_score_method"] == "conditional_logprob"
        assert runtime_contract["score_contract"]["joint_score_mode"] == "linear_blend"
        assert runtime_contract["score_contract"]["joint_score_formula"] == "linear_blend(prior_score, realization_score)"
        assert runtime_contract["score_contract"]["joint_score_components"] == ["prior_score", "realization_score"]
        assert runtime_contract["score_contract"]["popularity_weight"] == 0.0
