from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    month: str
    summary: str
    keywords: list[str]
    source_path: str
    published_date: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    references: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    popularity_score: float = 0.0


@dataclass
class IdeaPrediction:
    rank: int
    title: str
    rationale: str
    approach: str = ""
    score: float = 0.0
    confidence: float | None = None
    key_terms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    hit_at_k: float
    precision_at_k: float
    mrr: float
    novelty: float
    diversity: float
    matched_prediction_ranks: list[int]
    matched_paper_ids: list[str]
    # coverage_at_k = matched / len(future_papers): fraction of ALL future papers
    # hit by the top-k predictions. Upper-bounded by min(k, |future|)/|future|, so
    # it cannot reach 1.0 when |future| > k -- read it as coverage, not recall.
    coverage_at_k: float = 0.0
    lead_time: float = 0.0
    duplicate_rate: float = 0.0


@dataclass
class BacktestWindowResult:
    cutoff_month: str
    cutoff_date: str
    future_end_month: str
    future_end_date: str
    train_papers: int
    future_papers: int
    predictions: list[IdeaPrediction]
    evaluation: EvaluationResult
    matches: list[PredictionMatchDetail] = field(default_factory=list)
    # arXiv IDs of the training-window papers (date <= cutoff). Stored so the
    # citation/co-author validity analyses can target the train community
    # instead of a global candidate union. train_papers (the int count) is kept.
    train_paper_ids: list[str] = field(default_factory=list)


@dataclass
class MatchResult:
    score: float
    reasoning: str | None = None
    engine_name: str = "hybrid"
    paper_id: str | None = None
    # Hybrid-engine component scores, populated only by the hybrid branch of
    # compute_similarity. is_match (hybrid) reads these so the match decision
    # uses the exact same numbers as the sort score (no recompute / no drift).
    semantic: float | None = None
    keyword: float | None = None


@dataclass
class PredictionMatchDetail:
    prediction_rank: int
    prediction_title: str
    paper_id: str | None = None
    score: float = 0.0
    is_match: bool = False
    lead_time: float = 0.0
    matched_reasoning: str | None = None
    duplicate_candidate_paper_ids: list[str] = field(default_factory=list)
    matched_paper_popularity: float = 0.0


@dataclass
class ScoredPredictionList:
    evaluation: EvaluationResult
    matches: list[PredictionMatchDetail] = field(default_factory=list)
    unmatched_future_paper_ids: list[str] = field(default_factory=list)


@dataclass
class SimilarityPrompt:
    system_prompt: str
    user_prompt_template: str
