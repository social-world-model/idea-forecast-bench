from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    month: str
    summary: str
    keywords: List[str]
    source_path: str
    published_date: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    references: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    popularity_score: float = 0.0


@dataclass
class IdeaPrediction:
    rank: int
    title: str
    rationale: str
    approach: str = ""
    score: float = 0.0
    confidence: float | None = None
    key_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    hit_at_k: float
    recall_at_k: float
    precision_at_k: float
    mrr: float
    novelty: float
    diversity: float
    matched_prediction_ranks: List[int]
    matched_paper_ids: List[str]
    # coverage_at_k = matched / len(future_papers): fraction of ALL future papers
    # hit by the top-k predictions. Upper-bounded by min(k, |future|)/|future|, so
    # it is NOT a true recall when |future| > k (it cannot reach 1.0). Kept for
    # backward comparability with earlier runs that stored this under recall_at_k.
    coverage_at_k: float = 0.0
    lead_time: float = 0.0
    duplicate_rate: float = 0.0
    # Popularity-weighted metrics (opt-in; 0.0 when no popularity_weights provided)
    weighted_hit_at_k: float = 0.0
    weighted_precision_at_k: float = 0.0
    weighted_mrr: float = 0.0
    popularity_recall_at_k: float = 0.0


@dataclass
class BacktestWindowResult:
    cutoff_month: str
    cutoff_date: str
    future_end_month: str
    future_end_date: str
    train_papers: int
    future_papers: int
    predictions: List[IdeaPrediction]
    evaluation: EvaluationResult
    matches: List["PredictionMatchDetail"] = field(default_factory=list)


@dataclass
class MatchResult:
    score: float
    reasoning: Optional[str] = None
    engine_name: str = "hybrid"
    paper_id: Optional[str] = None
    # Hybrid-engine component scores, populated only by the hybrid branch of
    # compute_similarity. is_match (hybrid) reads these so the match decision
    # uses the exact same numbers as the sort score (no recompute / no drift).
    semantic: Optional[float] = None
    keyword: Optional[float] = None


@dataclass
class PredictionMatchDetail:
    prediction_rank: int
    prediction_title: str
    paper_id: Optional[str] = None
    score: float = 0.0
    is_match: bool = False
    lead_time: float = 0.0
    matched_reasoning: Optional[str] = None
    duplicate_candidate_paper_ids: List[str] = field(default_factory=list)
    matched_paper_popularity: float = 0.0


@dataclass
class ScoredPredictionList:
    evaluation: EvaluationResult
    matches: List[PredictionMatchDetail] = field(default_factory=list)
    unmatched_future_paper_ids: List[str] = field(default_factory=list)


@dataclass
class SimilarityPrompt:
    system_prompt: str
    user_prompt_template: str
