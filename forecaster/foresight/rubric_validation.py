"""Phase-2 rubric validation: discriminative AUC + leakage detection.

A rubric R is accepted iff:
  1. ROC AUC on (positive, negative) idea/candidate pairs >= rubric_auc_min
     (default 0.70, configurable per-call).
  2. No "leakage hits": negative pairs scoring at/above the positive median.

Both signals are derived from the same scored-pair table so the runner
can persist a single CSV for inspection.

This module is *backend-agnostic*: pass any callable
`scorer(idea, candidate, rubric) -> JudgeResult` (e.g. RubricJudge.score
or a stubbed function used in tests).
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from forecaster.foresight.judge import JudgeResult, RubricJudge
from forecaster.foresight.rubric import Rubric

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabeledPair:
    idea_text: str
    candidate_text: str
    label: int  # 1 = positive (post-cutoff emerged), 0 = negative (pre-cutoff existing)
    meta: dict = field(default_factory=dict)


@dataclass
class ScoredPair:
    pair: LabeledPair
    score: float


@dataclass
class RubricValidationReport:
    topic_id: str
    cutoff_t: str
    n_positive: int
    n_negative: int
    auc: float
    positive_median: float
    negative_max: float
    leakage_hits: int  # negatives with score >= positive_median
    leakage_examples: list[dict] = field(default_factory=list)
    threshold_used: float = 0.70
    passed: bool = False

    def to_json(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "cutoff_t": self.cutoff_t,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "auc": round(self.auc, 4),
            "positive_median": round(self.positive_median, 4),
            "negative_max": round(self.negative_max, 4),
            "leakage_hits": self.leakage_hits,
            "leakage_examples": list(self.leakage_examples),
            "threshold_used": self.threshold_used,
            "passed": self.passed,
        }


# ---------------------------------------------------------------- AUC


def compute_auc(
    pos_scores: Sequence[float],
    neg_scores: Sequence[float],
) -> float:
    """ROC AUC by the Mann-Whitney U identity.

    Returns 0.5 when either group is empty (uninformative).
    """
    if not pos_scores or not neg_scores:
        return 0.5
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    # Combined ranking with tie averaging.
    combined = sorted(
        [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores],
        key=lambda x: x[0],
    )
    ranks: list[float] = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed average
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(
        r for r, (_, lbl) in zip(ranks, combined, strict=False) if lbl == 1
    )
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


# ---------------------------------------------------------------- driver


ScoreOne = Callable[[str, str, Rubric], JudgeResult]


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def score_pairs(
    pairs: Iterable[LabeledPair],
    rubric: Rubric,
    *,
    judge: RubricJudge | None = None,
    score_fn: ScoreOne | None = None,
) -> list[ScoredPair]:
    """Score every pair under the rubric using the supplied judge or score_fn."""
    if (judge is None) == (score_fn is None):
        raise ValueError("provide exactly one of `judge` or `score_fn`")
    scored: list[ScoredPair] = []
    for p in pairs:
        if judge is not None:
            res = judge.score(p.idea_text, p.candidate_text, rubric)
        else:
            assert score_fn is not None
            res = score_fn(p.idea_text, p.candidate_text, rubric)
        scored.append(ScoredPair(pair=p, score=float(res.score)))
    return scored


def validate_rubric(
    rubric: Rubric,
    pairs: Iterable[LabeledPair],
    *,
    judge: RubricJudge | None = None,
    score_fn: ScoreOne | None = None,
    threshold: float = 0.70,
    leakage_top_n: int = 3,
) -> tuple[RubricValidationReport, list[ScoredPair]]:
    """Score pairs, compute AUC + leakage diagnostics.

    Leakage is defined as a negative pair scoring at or above the median
    of the positive pairs — i.e., the judge can't tell pre-cutoff existing
    work from post-cutoff emergence. That signals the judge has likely
    memorized the topic.
    """
    scored = score_pairs(pairs, rubric, judge=judge, score_fn=score_fn)
    pos = [s for s in scored if s.pair.label == 1]
    neg = [s for s in scored if s.pair.label == 0]
    auc = compute_auc([s.score for s in pos], [s.score for s in neg])
    pos_med = _median([s.score for s in pos])
    neg_max = max((s.score for s in neg), default=float("nan"))
    leakage = [s for s in neg if not math.isnan(pos_med) and s.score >= pos_med]
    leakage.sort(key=lambda s: -s.score)
    leakage_examples = [
        {
            "score": round(s.score, 4),
            "idea_preview": s.pair.idea_text[:200],
            "candidate_preview": s.pair.candidate_text[:200],
            "meta": dict(s.pair.meta),
        }
        for s in leakage[:leakage_top_n]
    ]
    report = RubricValidationReport(
        topic_id=rubric.topic_id,
        cutoff_t=rubric.cutoff_t,
        n_positive=len(pos),
        n_negative=len(neg),
        auc=auc,
        positive_median=pos_med,
        negative_max=neg_max,
        leakage_hits=len(leakage),
        leakage_examples=leakage_examples,
        threshold_used=threshold,
        passed=(auc >= threshold and len(leakage) == 0),
    )
    return report, scored


def write_scored_pairs_csv(scored: Sequence[ScoredPair], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "score", "idea_preview", "candidate_preview", "meta"])
        for s in scored:
            w.writerow(
                [
                    s.pair.label,
                    f"{s.score:.4f}",
                    s.pair.idea_text[:160].replace("\n", " "),
                    s.pair.candidate_text[:160].replace("\n", " "),
                    str(s.pair.meta),
                ]
            )
    return p


__all__ = [
    "LabeledPair",
    "ScoredPair",
    "RubricValidationReport",
    "compute_auc",
    "score_pairs",
    "validate_rubric",
    "write_scored_pairs_csv",
]
