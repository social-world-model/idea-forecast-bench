from dataclasses import dataclass, field
from typing import Dict, List


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
    key_terms: List[str]
    confidence: float


@dataclass
class EvaluationResult:
    hit_at_k: float
    recall_at_k: float
    precision_at_k: float
    mrr: float
    novelty: float
    diversity: float
    matched_prediction_ranks: List[int]
    matched_terms: List[str]


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
