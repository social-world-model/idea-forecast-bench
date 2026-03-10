from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from live_idea_bench.models import EvaluationResult, IdeaPrediction, PaperRecord, PredictionMatchDetail

logger = logging.getLogger(__name__)
from live_idea_bench.rl.config import RewardConfig
from live_idea_bench.rl.local_generation import parse_completion_predictions
from live_idea_bench.similarity import idea_text, paper_text, score_prediction_list


@dataclass
class PerIdeaReward:
    rank: int
    title: str
    matched_paper_id: str | None
    future_match: float
    novelty: float
    specificity: float
    lead_time: float
    duplicate_penalty: float
    total: float


@dataclass
class RLRewardEvaluation:
    benchmark_evaluation: EvaluationResult
    benchmark_score: float
    list_reward: float
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


def _specificity_score(prediction: IdeaPrediction, config: RewardConfig) -> float:
    title_score = 1.0 if prediction.title.strip() else 0.0
    rationale_score = min(1.0, len(prediction.rationale.strip()) / 80.0)
    approach_score = min(1.0, len(prediction.approach.strip()) / 80.0)
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
    if total > 0 and len(rows) > 0 and total % len(rows) == 0:
        return rows[index // (total // len(rows))]
    return None


def benchmark_score(evaluation: EvaluationResult) -> float:
    return round((0.7 * evaluation.hit_at_k) + (0.3 * evaluation.mrr), 4)


def evaluate_rl_reward(
    predictions: list[IdeaPrediction],
    train_papers: list[PaperRecord],
    future_papers: list[PaperRecord],
    reward_config: RewardConfig,
    *,
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
    weighted_total = 0.0
    total_rank_weight = 0.0
    prior_predictions: list[IdeaPrediction] = []

    top_predictions = predictions[: reward_config.top_k]
    for idx, prediction in enumerate(top_predictions):
        detail = scored.matches[idx] if idx < len(scored.matches) else PredictionMatchDetail(
            prediction_rank=prediction.rank,
            prediction_title=prediction.title,
        )
        future_match_value = detail.score if detail.is_match else 0.0
        novelty_value = _novelty_score(prediction, train_papers)
        specificity_value = _specificity_score(prediction, reward_config)
        lead_time_value = round(max(0.0, min(1.0, detail.lead_time if detail.is_match else 0.0)), 4)
        duplicate_penalty = _duplicate_penalty(prediction, detail, prior_predictions, reward_config)
        total = (
            (reward_config.weights.future_match * future_match_value)
            + (reward_config.weights.novelty * novelty_value)
            + (reward_config.weights.specificity * specificity_value)
            + (reward_config.weights.lead_time * lead_time_value)
            - (reward_config.weights.duplicate_penalty * duplicate_penalty)
        )
        rank_weight = 1.0 / (1.0 + (reward_config.rank_decay * max(0, prediction.rank - 1)))
        weighted_total += rank_weight * total
        total_rank_weight += rank_weight
        reward_items.append(
            PerIdeaReward(
                rank=prediction.rank,
                title=prediction.title,
                matched_paper_id=detail.paper_id,
                future_match=round(future_match_value, 4),
                novelty=novelty_value,
                specificity=specificity_value,
                lead_time=lead_time_value,
                duplicate_penalty=duplicate_penalty,
                total=round(total, 4),
            )
        )
        prior_predictions.append(prediction)

    dense_reward = weighted_total / total_rank_weight if total_rank_weight else 0.0
    bench_score = benchmark_score(evaluation)
    final_reward = (
        ((1.0 - reward_config.benchmark_score_weight) * dense_reward)
        + (reward_config.benchmark_score_weight * bench_score)
    )

    return RLRewardEvaluation(
        benchmark_evaluation=evaluation,
        benchmark_score=bench_score,
        list_reward=round(final_reward, 4),
        per_idea_rewards=reward_items,
        reward_breakdown={
            "dense_reward": round(dense_reward, 4),
            "benchmark_score": bench_score,
            "lead_time": evaluation.lead_time,
            "duplicate_rate": evaluation.duplicate_rate,
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


def build_grpo_reward_function(
    reward_config: RewardConfig,
    *,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
    model_name: str | None = None,
) -> Callable[..., list[float]]:
    def reward_func(
        completions: Sequence[Any],
        train_papers: Sequence[list[dict[str, Any]]] | None = None,
        future_papers: Sequence[list[dict[str, Any]]] | None = None,
        cutoff_date: Sequence[str] | None = None,
        future_end_date: Sequence[str] | None = None,
        **_: Any,
    ) -> list[float]:
        rewards: list[float] = []
        total = len(completions)
        for idx, completion in enumerate(completions):
            predictions = parse_completion_predictions(completion, limit=reward_config.top_k)
            train_payload = _value_for_index(train_papers, idx, total) or []
            future_payload = _value_for_index(future_papers, idx, total) or []
            cutoff_value = _value_for_index(cutoff_date, idx, total)
            future_end_value = _value_for_index(future_end_date, idx, total)
            try:
                reconstructed_train = [PaperRecord(**paper) for paper in train_payload]
                reconstructed_future = [PaperRecord(**paper) for paper in future_payload]
            except (TypeError, KeyError) as exc:
                logger.warning("Failed to reconstruct PaperRecord at index %d: %s", idx, exc)
                rewards.append(0.0)
                continue
            if not predictions:
                rewards.append(0.0)
                continue
            evaluation = evaluate_rl_reward(
                predictions=predictions,
                train_papers=reconstructed_train,
                future_papers=reconstructed_future,
                reward_config=reward_config,
                similarity_config_path=similarity_config_path,
                runtime_config_path=runtime_config_path,
                model_name=model_name,
                cutoff_date=str(cutoff_value) if cutoff_value else None,
                future_end_date=str(future_end_value) if future_end_value else None,
            )
            rewards.append(evaluation.list_reward)
        return rewards

    return reward_func
