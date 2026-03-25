from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from live_idea_bench.models import EvaluationResult, IdeaPrediction, PaperRecord, PredictionMatchDetail
from forecaster.config import RealizationConfig
from forecaster.models import Innovation, innovation_from_dict
from forecaster.realization.config import RewardConfig
from forecaster.realization.local_generation import _completion_to_text, parse_single_completion_prediction
from forecaster.realization.proposal_generator import proposal_to_idea_prediction
from forecaster.realization.realization_reward import evaluate_realization_reward
from live_idea_bench.similarity import idea_text, paper_text, score_prediction_list

logger = logging.getLogger(__name__)


@dataclass
class PerIdeaReward:
    rank: int
    title: str
    matched_paper_id: str | None
    evidence_quality: float
    operator_adherence: float
    coherence: float
    benchmark_match: float
    duplicate_penalty: float
    total: float


@dataclass
class RLRewardEvaluation:
    benchmark_evaluation: EvaluationResult
    benchmark_score: float
    list_reward: float
    invalid_completion: bool = False
    per_idea_rewards: list[PerIdeaReward] = field(default_factory=list)
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    match_details: list[PredictionMatchDetail] = field(default_factory=list)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _text_similarity(a: str, b: str) -> float:
    sa = _tokenize(a)
    sb = _tokenize(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


_SPECIFICITY_MIN_CHARS = 80


def _specificity_score(prediction: IdeaPrediction, config: RewardConfig) -> float:
    title_score = 1.0 if prediction.title.strip() else 0.0
    rationale_score = min(1.0, len(prediction.rationale.strip()) / _SPECIFICITY_MIN_CHARS)
    approach_score = min(1.0, len(prediction.approach.strip()) / _SPECIFICITY_MIN_CHARS)
    total_weight = (
        config.specificity_title_weight
        + config.specificity_rationale_weight
        + config.specificity_approach_weight
    )
    if total_weight <= 0:
        return 0.0
    weighted = (
        (config.specificity_title_weight * title_score)
        + (config.specificity_rationale_weight * rationale_score)
        + (config.specificity_approach_weight * approach_score)
    ) / total_weight
    return round(max(0.0, min(1.0, weighted)), 4)


def _novelty_score(prediction: IdeaPrediction, train_papers: Iterable[PaperRecord]) -> float:
    refs = [paper_text(paper) for paper in train_papers]
    if not refs:
        return 1.0
    pred_text = idea_text(prediction)
    max_sim = max(_text_similarity(pred_text, ref) for ref in refs)
    return round(max(0.0, min(1.0, 1.0 - max_sim)), 4)


def _duplicate_penalty(
    prediction: IdeaPrediction,
    detail: PredictionMatchDetail,
    prior_predictions: Sequence[IdeaPrediction],
    config: RewardConfig,
) -> float:
    pred_text = idea_text(prediction)
    similarity_penalty = 0.0
    for prior in prior_predictions:
        sim = _text_similarity(pred_text, idea_text(prior))
        if sim >= config.duplicate_similarity_threshold:
            similarity_penalty = max(
                similarity_penalty,
                (sim - config.duplicate_similarity_threshold)
                / max(1e-6, 1.0 - config.duplicate_similarity_threshold),
            )
    match_penalty = 1.0 if detail.duplicate_candidate_paper_ids else 0.0
    return round(max(match_penalty, similarity_penalty), 4)


def _value_for_index(rows: Sequence[Any] | None, index: int, total: int) -> Any:
    if not rows:
        return None
    if len(rows) == total:
        return rows[index]
    if len(rows) == 1:
        return rows[0]
    if total > 0 and total % len(rows) == 0:
        return rows[index // (total // len(rows))]
    logger.warning(
        "Batch size mismatch in reward function: rows=%d, total=%d at index=%d — "
        "reward will be zeroed for this completion.",
        len(rows),
        total,
        index,
    )
    return None


def benchmark_score(evaluation: EvaluationResult) -> float:
    return round((0.7 * evaluation.hit_at_k) + (0.3 * evaluation.mrr), 4)


def _empty_evaluation() -> EvaluationResult:
    return EvaluationResult(
        hit_at_k=0.0,
        recall_at_k=0.0,
        precision_at_k=0.0,
        mrr=0.0,
        novelty=0.0,
        diversity=0.0,
        matched_prediction_ranks=[],
        matched_paper_ids=[],
        lead_time=0.0,
        duplicate_rate=0.0,
    )


def build_invalid_reward_evaluation(
    reward_config: RewardConfig,
    *,
    parse_failure: bool = True,
) -> RLRewardEvaluation:
    invalid_reward = round(reward_config.invalid_completion_reward, 4)
    return RLRewardEvaluation(
        benchmark_evaluation=_empty_evaluation(),
        benchmark_score=0.0,
        list_reward=invalid_reward,
        invalid_completion=True,
        reward_breakdown={
            "dense_reward": invalid_reward,
            "evidence_quality": 0.0,
            "operator_adherence": 0.0,
            "coherence": 0.0,
            "benchmark_match": 0.0,
            "benchmark_score": 0.0,
            "lead_time": 0.0,
            "duplicate_rate": 0.0,
            "parse_failure": 1.0 if parse_failure else 0.0,
            "invalid_completion": 1.0,
            "invalid_completion_reward": invalid_reward,
        },
        match_details=[],
    )


def _innovation_from_prediction(prediction: IdeaPrediction) -> Innovation:
    approach = str(prediction.approach or "").strip()
    if ":" in approach:
        operator, _, base_direction = approach.partition(":")
        operator = operator.strip().lower() or "extend"
        base_direction = base_direction.strip() or prediction.title.strip() or "emerging direction"
    else:
        operator = "extend"
        base_direction = prediction.title.strip() or approach or "emerging direction"
    gap = prediction.rationale.strip() or prediction.title.strip() or "research gap"
    return Innovation(
        base_direction=base_direction,
        operator=operator,
        gap=gap,
    )


def _proposal_text_from_prediction(
    prediction: IdeaPrediction,
    *,
    proposal_text: str | None = None,
) -> str:
    explicit = str(proposal_text or "").strip()
    if explicit:
        return explicit
    metadata_text = str(prediction.metadata.get("proposal_text", "") or "").strip()
    if metadata_text:
        return metadata_text
    body_parts = [prediction.rationale.strip(), prediction.approach.strip()]
    body = "\n".join(part for part in body_parts if part)
    if body:
        return f"{prediction.title.strip()}\n{body}".strip()
    return prediction.title.strip()


def coerce_reward_prediction(
    raw_completion: Any,
    *,
    prompt_mode: str = "",
    innovation: Innovation | None = None,
) -> tuple[IdeaPrediction | None, str]:
    """Parse a reward-model completion into the shared single-idea contract."""
    raw_text = _completion_to_text(raw_completion).strip()
    if not raw_text:
        return None, ""

    normalized_mode = str(prompt_mode or "").strip().lower()
    if normalized_mode == "z_conditioned_realization" and innovation is not None:
        prediction = proposal_to_idea_prediction(raw_text, innovation, rank=1)
        prediction.metadata["proposal_text"] = raw_text
        prediction.metadata["prompt_mode"] = normalized_mode
        return prediction, raw_text

    prediction = parse_single_completion_prediction(raw_completion)
    if prediction is None:
        return None, raw_text
    prediction.metadata.setdefault("proposal_text", raw_text)
    return prediction, raw_text


def evaluate_rl_reward(
    predictions: list[IdeaPrediction],
    train_papers: list[PaperRecord],
    future_papers: list[PaperRecord],
    reward_config: RewardConfig,
    *,
    innovation: Innovation | None = None,
    evidence_papers: list[PaperRecord] | None = None,
    proposal_text: str | None = None,
    realization_config: RealizationConfig | None = None,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
    model_name: str | None = None,
    cutoff_date: str | None = None,
    future_end_date: str | None = None,
) -> RLRewardEvaluation:
    scored = score_prediction_list(
        predictions=predictions,
        train_papers=train_papers,
        future_papers=future_papers,
        k=reward_config.top_k,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
        model_name=model_name,
        cutoff_date=cutoff_date,
        future_end_date=future_end_date,
        candidate_limit=reward_config.candidate_limit,
    )
    evaluation = scored.evaluation
    reward_items: list[PerIdeaReward] = []
    top_predictions = predictions[:1]
    prediction = top_predictions[0] if top_predictions else IdeaPrediction(rank=1, title="", rationale="", approach="")
    detail = scored.matches[0] if scored.matches else PredictionMatchDetail(
        prediction_rank=prediction.rank,
        prediction_title=prediction.title,
    )
    resolved_innovation = innovation or _innovation_from_prediction(prediction)
    resolved_proposal_text = _proposal_text_from_prediction(
        prediction,
        proposal_text=proposal_text,
    )
    resolved_realization_config = realization_config or RealizationConfig()
    paper_reward = evaluate_realization_reward(
        resolved_proposal_text,
        resolved_innovation,
        list(evidence_papers or []),
        resolved_realization_config,
    )
    benchmark_match_value = detail.score if detail.is_match else 0.0
    lead_time_value = round(max(0.0, min(1.0, detail.lead_time if detail.is_match else 0.0)), 4)
    total = paper_reward.total_reward
    reward_items.append(
        PerIdeaReward(
            rank=prediction.rank,
            title=prediction.title,
            matched_paper_id=detail.paper_id,
            evidence_quality=paper_reward.evidence_quality,
            operator_adherence=paper_reward.operator_adherence,
            coherence=paper_reward.coherence,
            benchmark_match=round(benchmark_match_value, 4),
            duplicate_penalty=0.0,
            total=round(total, 4),
        )
    )

    dense_reward = round(total, 4)
    bench_score = benchmark_score(evaluation)
    final_reward = dense_reward

    return RLRewardEvaluation(
        benchmark_evaluation=evaluation,
        benchmark_score=bench_score,
        list_reward=round(final_reward, 4),
        invalid_completion=False,
        per_idea_rewards=reward_items,
        reward_breakdown={
            "dense_reward": round(dense_reward, 4),
            "evidence_quality": paper_reward.evidence_quality,
            "operator_adherence": paper_reward.operator_adherence,
            "coherence": paper_reward.coherence,
            "benchmark_match": round(benchmark_match_value, 4),
            "benchmark_score": bench_score,
            "lead_time": evaluation.lead_time,
            "duplicate_rate": evaluation.duplicate_rate,
            "invalid_completion": 0.0,
            "parse_failure": 0.0,
            "invalid_completion_reward": round(reward_config.invalid_completion_reward, 4),
        },
        match_details=scored.matches,
    )


def serialize_reward_evaluation(result: RLRewardEvaluation) -> dict[str, Any]:
    return asdict(result)


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return 0.0

    def _ranks(values: Sequence[float]) -> list[float]:
        ordered = sorted((value, idx) for idx, value in enumerate(values))
        ranks = [0.0] * len(values)
        cursor = 0
        while cursor < len(ordered):
            end = cursor
            while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[cursor][0]:
                end += 1
            avg_rank = (cursor + end + 2) / 2.0
            for _, original_idx in ordered[cursor : end + 1]:
                ranks[original_idx] = avg_rank
            cursor = end + 1
        return ranks

    x_ranks = _ranks(xs)
    y_ranks = _ranks(ys)
    x_mean = sum(x_ranks) / len(x_ranks)
    y_mean = sum(y_ranks) / len(y_ranks)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_ranks, y_ranks))
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in x_ranks))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in y_ranks))
    if x_denom == 0.0 or y_denom == 0.0:
        return 0.0
    return round(numerator / (x_denom * y_denom), 4)


def build_online_rl_reward_function(
    reward_config: RewardConfig,
    *,
    realization_config: RealizationConfig | None = None,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
    model_name: str | None = None,
) -> Callable[..., list[float]]:
    def reward_func(
        completions: Sequence[Any],
        train_papers: Sequence[list[dict[str, Any]]] | None = None,
        future_papers: Sequence[list[dict[str, Any]]] | None = None,
        prompt_mode: Sequence[str] | None = None,
        innovation: Sequence[dict[str, Any]] | None = None,
        evidence_papers: Sequence[list[dict[str, Any]]] | None = None,
        realization_config_payload: Sequence[dict[str, Any]] | None = None,
        cutoff_date: Sequence[str] | None = None,
        future_end_date: Sequence[str] | None = None,
        **_: Any,
    ) -> list[float]:
        rewards: list[float] = []
        total = len(completions)
        for idx, completion in enumerate(completions):
            train_payload = _value_for_index(train_papers, idx, total) or []
            future_payload = _value_for_index(future_papers, idx, total) or []
            prompt_mode_value = str(_value_for_index(prompt_mode, idx, total) or "")
            cutoff_value = _value_for_index(cutoff_date, idx, total)
            future_end_value = _value_for_index(future_end_date, idx, total)
            try:
                reconstructed_train = [PaperRecord(**paper) for paper in train_payload]
                reconstructed_future = [PaperRecord(**paper) for paper in future_payload]
            except (TypeError, KeyError) as exc:
                logger.warning("Failed to reconstruct PaperRecord at index %d: %s", idx, exc)
                rewards.append(round(reward_config.invalid_completion_reward, 4))
                continue
            innovation_payload = _value_for_index(innovation, idx, total)
            evidence_payload = _value_for_index(evidence_papers, idx, total) or []
            config_payload = _value_for_index(realization_config_payload, idx, total)
            try:
                innovation_value = (
                    innovation_from_dict(innovation_payload)
                    if isinstance(innovation_payload, dict) and innovation_payload
                    else None
                )
                evidence_value = [PaperRecord(**paper) for paper in evidence_payload]
                resolved_realization_config = (
                    RealizationConfig(**config_payload)
                    if isinstance(config_payload, dict) and config_payload
                    else (realization_config or RealizationConfig())
                )
            except (TypeError, KeyError, ValueError) as exc:
                logger.warning("Failed to reconstruct realization reward context at index %d: %s", idx, exc)
                innovation_value = None
                evidence_value = []
                resolved_realization_config = realization_config or RealizationConfig()
            prediction, proposal_text_value = coerce_reward_prediction(
                completion,
                prompt_mode=prompt_mode_value,
                innovation=innovation_value,
            )
            if prediction is None:
                rewards.append(round(reward_config.invalid_completion_reward, 4))
                continue
            try:
                evaluation = evaluate_rl_reward(
                    predictions=[prediction],
                    train_papers=reconstructed_train,
                    future_papers=reconstructed_future,
                    reward_config=reward_config,
                    innovation=innovation_value,
                    evidence_papers=evidence_value,
                    proposal_text=proposal_text_value,
                    realization_config=resolved_realization_config,
                    similarity_config_path=similarity_config_path,
                    runtime_config_path=runtime_config_path,
                    model_name=model_name,
                    cutoff_date=str(cutoff_value) if cutoff_value else None,
                    future_end_date=str(future_end_value) if future_end_value else None,
                )
            except Exception as exc:
                logger.warning("evaluate_rl_reward failed at index %d: %s", idx, exc, exc_info=True)
                rewards.append(round(reward_config.invalid_completion_reward, 4))
                continue
            rewards.append(evaluation.list_reward)
        return rewards

    return reward_func
