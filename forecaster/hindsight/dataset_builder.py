from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from forecaster.config import HindsightConfig
from forecaster.hindsight.extractor import extract_innovation
from forecaster.models import HindsightSample, Innovation
from idea_forecast_bench.backtest import split_train_future_by_cutoff
from idea_forecast_bench.models import PaperRecord

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
                    future_paper_published_date=future_paper.published_date,
                    innovation=innovation,
                )
            )

    return samples


def load_hindsight_samples_jsonl(path: str | Path) -> list[HindsightSample]:
    """Load HindsightSample objects from a batch-extraction JSONL file.

    The JSONL schema (from batch hindsight extraction) uses ``cutoff_date``
    ("YYYY-MM-DD") rather than ``cutoff_month`` ("YYYY-MM") and stores
    ``context_paper_count`` instead of ``context_paper_ids``.  This loader
    bridges that gap so the samples can be fed directly into the SFT pipeline.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of HindsightSample objects sorted by (cutoff_month, published_date).
    """
    from forecaster.foresight.cutoffs import TRAIN_CUTOFF_MAX

    path = Path(path)
    samples: list[HindsightSample] = []
    dropped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        # A hindsight file built before the training window was tightened can
        # carry episodes whose future window overlaps evaluation. Training on
        # them leaks the test period, so they are dropped here rather than
        # trusted because the file exists.
        if str(row["cutoff_date"])[:10] > TRAIN_CUTOFF_MAX:
            dropped += 1
            continue
        innovation = Innovation(
            base_direction=str(row["innovation"]["base_direction"]),
            operator=str(row["innovation"]["operator"]),
            gap=str(row["innovation"]["gap"]),
        )
        cutoff_month = str(row["cutoff_date"])[:7]
        samples.append(
            HindsightSample(
                context_paper_ids=(),
                cutoff_month=cutoff_month,
                future_paper_id=str(row["future_paper_id"]),
                future_paper_published_date=str(row["future_paper_published_date"]),
                innovation=innovation,
            )
        )
    samples.sort(
        key=lambda s: (
            s.cutoff_month,
            s.future_paper_published_date,
            s.future_paper_id,
        ),
    )
    if dropped:
        logger.warning(
            "dropped %d hindsight rows with cutoff_date > %s (training must not "
            "see the evaluation window)",
            dropped,
            TRAIN_CUTOFF_MAX,
        )
    return samples
