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
    metadata: Dict[str, str] = field(default_factory=dict)


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


@dataclass
class MatchResult:
    score: float
    reasoning: Optional[str] = None
    engine_name: str = "hybrid"
    paper_id: Optional[str] = None


@dataclass
class SimilarityPrompt:
    system_prompt: str
    user_prompt_template: str
