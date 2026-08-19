from __future__ import annotations

import dataclasses
import logging
import re

from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.predictor import _base_score, _dedup_predictions, _jaccard, _prediction_text, _top_terms
from forecaster.realization.config import SelectionConfig

logger = logging.getLogger(__name__)


def _title_key(prediction: IdeaPrediction) -> str:
    return re.sub(r"\s+", " ", prediction.title.lower()).strip()


def _signal_terms(train_papers: list[PaperRecord]) -> list[str]:
    recent = train_papers[-20:]
    return _top_terms(
        list(paper.summary for paper in recent) + [keyword for paper in recent for keyword in paper.keywords],
        limit=20,
    )


def select_top_k_predictions(
    candidates: list[IdeaPrediction],
    train_papers: list[PaperRecord],
    selection_config: SelectionConfig,
    *,
    top_k: int | None = None,
) -> list[IdeaPrediction]:
    if not candidates:
        return []

    requested_k = top_k or selection_config.output_top_k
    target_k = min(requested_k, selection_config.output_top_k)
    if top_k is not None and top_k > selection_config.output_top_k:
        logger.warning(
            "Requested top_k=%d exceeds selection_config.output_top_k=%d; capping to %d.",
            top_k,
            selection_config.output_top_k,
            target_k,
        )
    title_frequency: dict[str, int] = {}
    for candidate in candidates:
        key = _title_key(candidate)
        title_frequency[key] = title_frequency.get(key, 0) + 1

    unique_candidate_titles = len(title_frequency)
    deduped = _dedup_predictions(candidates, threshold=selection_config.dedup_similarity_threshold)
    dedup_retention_ratio = round(len(deduped) / max(1, len(candidates)), 4)
    logger.info(
        "Selector candidate pool: total=%d unique_titles=%d deduped=%d retention=%.4f",
        len(candidates),
        unique_candidate_titles,
        len(deduped),
        dedup_retention_ratio,
    )
    signal_terms = _signal_terms(train_papers)
    total_candidates = max(1, len(candidates))

    scored_pool: list[tuple[IdeaPrediction, float]] = []
    for candidate in deduped:
        frequency = title_frequency.get(_title_key(candidate), 1) / total_candidates
        confidence = candidate.confidence
        if confidence is None:
            confidence = candidate.score
        confidence = max(0.0, min(1.0, float(confidence or 0.0)))
        heuristic = _base_score(candidate, signal_terms)
        relevance = (
            (selection_config.relevance_frequency_weight * frequency)
            + (selection_config.relevance_confidence_weight * confidence)
            + (selection_config.relevance_heuristic_weight * heuristic)
        )
        metadata = {
            **candidate.metadata,
            "sample_frequency": round(frequency, 4),
            "mean_model_confidence": round(confidence, 4),
            "heuristic_base_score": round(heuristic, 4),
            "selector_relevance": round(relevance, 4),
            "unique_candidate_titles": unique_candidate_titles,
            "dedup_retention_ratio": dedup_retention_ratio,
        }
        scored_pool.append((dataclasses.replace(candidate, metadata=metadata), relevance))

    scored_pool.sort(key=lambda item: (-item[1], item[0].title.lower()))
    selected: list[IdeaPrediction] = []

    while scored_pool and len(selected) < target_k:
        if not selected:
            prediction, relevance = scored_pool.pop(0)
            selected.append(
                dataclasses.replace(
                    prediction,
                    rank=len(selected) + 1,
                    score=round(relevance, 4),
                    confidence=round(relevance, 4),
                )
            )
            continue

        best_idx = 0
        best_score = float("-inf")
        for idx, (candidate, relevance) in enumerate(scored_pool):
            similarity = max(_jaccard(_prediction_text(candidate), _prediction_text(chosen)) for chosen in selected)
            novelty_to_selected = 1.0 - similarity
            mmr_score = (
                (selection_config.mmr_relevance_weight * relevance)
                + (selection_config.mmr_diversity_weight * novelty_to_selected)
            )
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        prediction, relevance = scored_pool.pop(best_idx)
        selected.append(
            dataclasses.replace(
                prediction,
                rank=len(selected) + 1,
                score=round(best_score, 4),
                confidence=round(relevance, 4),
                metadata={**prediction.metadata, "selector_mmr_score": round(best_score, 4)},
            )
        )

    return selected
