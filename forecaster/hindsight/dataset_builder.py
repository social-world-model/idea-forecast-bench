"""Build the hindsight dataset D_z = {(X_<=t, z_tilde_{t+1})}."""
from __future__ import annotations

import logging
from typing import Any

from live_idea_bench.backtest import split_train_future_by_cutoff
from live_idea_bench.models import PaperRecord

from forecaster.config import HindsightConfig
from forecaster.hindsight.extractor import extract_innovation
from forecaster.models import HindsightSample

logger = logging.getLogger(__name__)


def build_hindsight_dataset(
    papers: list[PaperRecord],
    cutoff_months: list[str],
    horizon_months: int,
    config: HindsightConfig,
    llm_client: Any,
    model: str,
    *,
    max_future_papers_per_cutoff: int = 10,
) -> list[HindsightSample]:
    """Build hindsight dataset by extracting innovations from historical episodes.

    For each cutoff month: splits papers into train/future, then for each
    future paper calls extract_innovation and returns a HindsightSample.

    Args:
        papers: All papers (train + future).
        cutoff_months: List of cutoff month strings ("YYYY-MM") for episodes.
        horizon_months: Number of months in the future window.
        config: HindsightConfig with LLM settings.
        llm_client: Initialized LLM client.
        model: LLM model name.
        max_future_papers_per_cutoff: Max future papers to extract per episode.

    Returns:
        Temporally ordered list of HindsightSample objects.
    """
    # Process cutoffs in chronological order
    sorted_cutoffs = sorted(cutoff_months)
    samples: list[HindsightSample] = []

    for cutoff_month in sorted_cutoffs:
        train_papers, future_papers, _future_end_month, _future_end_date = (
            split_train_future_by_cutoff(
                papers=papers,
                cutoff_month=cutoff_month,
                horizon_months=horizon_months,
            )
        )

        if not future_papers:
            logger.info(
                "No future papers for cutoff %r — skipping episode.", cutoff_month
            )
            continue

        context_paper_ids = tuple(p.paper_id for p in train_papers)
        limited_future = future_papers[:max_future_papers_per_cutoff]

        for future_paper in limited_future:
            try:
                innovation = extract_innovation(
                    future_paper=future_paper,
                    context_papers=train_papers,
                    llm_client=llm_client,
                    model=model,
                    config=config,
                )
            except ValueError as exc:
                logger.warning(
                    "Skipping paper %r at cutoff %r — extraction error: %s",
                    future_paper.paper_id,
                    cutoff_month,
                    exc,
                )
                continue

            samples.append(
                HindsightSample(
                    context_paper_ids=context_paper_ids,
                    cutoff_month=cutoff_month,
                    future_paper_id=future_paper.paper_id,
                    innovation=innovation,
                )
            )

    return samples
