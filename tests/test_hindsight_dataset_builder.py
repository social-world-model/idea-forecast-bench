"""Tests for hindsight dataset builder."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from live_idea_bench.models import PaperRecord
from forecaster.models import HindsightSample, Innovation
from forecaster.config import HindsightConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_paper(
    paper_id: str,
    title: str,
    month: str,
    published_date: str | None = None,
) -> PaperRecord:
    if published_date is None:
        published_date = f"{month}-15"
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        month=month,
        summary=f"Abstract for {title}.",
        keywords=["ml"],
        source_path="",
        published_date=published_date,
    )


def _make_innovation(idx: int = 0) -> Innovation:
    return Innovation(
        base_direction=f"direction_{idx}",
        operator="extend",
        gap=f"gap description {idx}",
    )


# Build a small paper corpus spanning 3 months
TRAIN_PAPERS = [
    _make_paper(f"train-{i:03d}", f"Train Paper {i}", month="2024-01")
    for i in range(5)
]
FUTURE_PAPERS_FEB = [
    _make_paper(f"fut-feb-{i:03d}", f"Future Feb Paper {i}", month="2024-02")
    for i in range(4)
]
FUTURE_PAPERS_MAR = [
    _make_paper(f"fut-mar-{i:03d}", f"Future Mar Paper {i}", month="2024-03")
    for i in range(3)
]
ALL_PAPERS = TRAIN_PAPERS + FUTURE_PAPERS_FEB + FUTURE_PAPERS_MAR


# ---------------------------------------------------------------------------
# Dataset builder tests
# ---------------------------------------------------------------------------


class TestBuildHindsightDataset:
    """Tests for forecaster.hindsight.dataset_builder.build_hindsight_dataset."""

    def test_build_hindsight_dataset_returns_samples(self):
        """With mocked extractor, returns a list of HindsightSample."""
        from forecaster.hindsight.dataset_builder import build_hindsight_dataset

        config = HindsightConfig(max_retries=1)
        client = MagicMock()

        innovation = _make_innovation(0)

        with patch(
            "forecaster.hindsight.dataset_builder.extract_innovation",
            return_value=innovation,
        ):
            samples = build_hindsight_dataset(
                papers=ALL_PAPERS,
                cutoff_months=["2024-01"],
                horizon_months=1,
                config=config,
                llm_client=client,
                model="gpt-4o",
                max_future_papers_per_cutoff=10,
            )

        assert isinstance(samples, list)
        assert len(samples) > 0
        for sample in samples:
            assert isinstance(sample, HindsightSample)
            assert sample.cutoff_month == "2024-01"
            assert isinstance(sample.innovation, Innovation)
            # context_paper_ids must be a tuple (immutable)
            assert isinstance(sample.context_paper_ids, tuple)

    def test_build_hindsight_dataset_temporal_ordering(self):
        """Samples are in chronological order by cutoff_month."""
        from forecaster.hindsight.dataset_builder import build_hindsight_dataset

        config = HindsightConfig(max_retries=1)
        client = MagicMock()

        with patch(
            "forecaster.hindsight.dataset_builder.extract_innovation",
            return_value=_make_innovation(0),
        ):
            samples = build_hindsight_dataset(
                papers=ALL_PAPERS,
                cutoff_months=["2024-02", "2024-01"],  # reversed on purpose
                horizon_months=1,
                config=config,
                llm_client=client,
                model="gpt-4o",
                max_future_papers_per_cutoff=10,
            )

        cutoff_months = [s.cutoff_month for s in samples]
        assert cutoff_months == sorted(cutoff_months)

    def test_build_hindsight_dataset_skips_failed_extractions(self, caplog):
        """If extract_innovation raises ValueError, that paper is skipped."""
        from forecaster.hindsight.dataset_builder import build_hindsight_dataset

        config = HindsightConfig(max_retries=1)
        client = MagicMock()

        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Extraction failed for paper")
            return _make_innovation(call_count)

        with patch(
            "forecaster.hindsight.dataset_builder.extract_innovation",
            side_effect=_side_effect,
        ):
            with caplog.at_level(logging.WARNING, logger="forecaster.hindsight.dataset_builder"):
                samples = build_hindsight_dataset(
                    papers=ALL_PAPERS,
                    cutoff_months=["2024-01"],
                    horizon_months=1,
                    config=config,
                    llm_client=client,
                    model="gpt-4o",
                    max_future_papers_per_cutoff=10,
                )

        # At least one warning should have been logged about failure
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("skip" in msg.lower() or "fail" in msg.lower() or "error" in msg.lower() for msg in warning_messages)

        # Remaining papers should still produce samples
        assert isinstance(samples, list)

    def test_build_hindsight_dataset_max_future_papers(self):
        """max_future_papers_per_cutoff is respected."""
        from forecaster.hindsight.dataset_builder import build_hindsight_dataset

        config = HindsightConfig(max_retries=1)
        client = MagicMock()
        max_future = 2

        extract_call_args: list = []

        def _record_call(future_paper, context_papers, llm_client, model, config):
            extract_call_args.append(future_paper.paper_id)
            return _make_innovation(len(extract_call_args))

        with patch(
            "forecaster.hindsight.dataset_builder.extract_innovation",
            side_effect=_record_call,
        ):
            samples = build_hindsight_dataset(
                papers=ALL_PAPERS,
                cutoff_months=["2024-01"],
                horizon_months=1,
                config=config,
                llm_client=client,
                model="gpt-4o",
                max_future_papers_per_cutoff=max_future,
            )

        # Only max_future future papers should have been processed per cutoff
        assert len(samples) <= max_future
        assert len(extract_call_args) <= max_future

    def test_build_hindsight_dataset_empty_papers(self):
        """Empty paper list returns empty sample list."""
        from forecaster.hindsight.dataset_builder import build_hindsight_dataset

        config = HindsightConfig(max_retries=1)
        client = MagicMock()

        samples = build_hindsight_dataset(
            papers=[],
            cutoff_months=["2024-01"],
            horizon_months=1,
            config=config,
            llm_client=client,
            model="gpt-4o",
        )

        assert samples == []
