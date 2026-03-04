from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from src.backtest.data import (
    add_months,
    date_to_ordinal,
    get_paper_published_date,
    month_end_date,
    month_start_date,
    month_to_index,
    normalize_date,
    normalize_month,
)
from src.backtest.evaluator import evaluate_predictions
from src.backtest.models import (
    BacktestWindowResult,
    EvaluationResult,
    IdeaPrediction,
    PaperRecord,
)
from src.strategy import IdeaStrategy


@dataclass
class BacktestConfig:
    top_k: int = 5
    horizon_months: int = 3
    min_train_papers: int = 6
    start_month: Optional[str] = None
    end_month: Optional[str] = None


def _filter_by_month(
    papers: List[PaperRecord],
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
) -> List[PaperRecord]:
    start_idx = month_to_index(start_month) if start_month else None
    end_idx = month_to_index(end_month) if end_month else None
    output: List[PaperRecord] = []
    for paper in papers:
        idx = month_to_index(paper.month)
        if start_idx is not None and idx < start_idx:
            continue
        if end_idx is not None and idx > end_idx:
            continue
        output.append(paper)
    return output


def _resolve_cutoff(
    *,
    cutoff_month: Optional[str],
    cutoff_date: Optional[str],
) -> Tuple[str, str]:
    if cutoff_date:
        resolved_cutoff_date = normalize_date(cutoff_date)
        resolved_cutoff_month = normalize_month(resolved_cutoff_date)
        return resolved_cutoff_month, resolved_cutoff_date

    if cutoff_month:
        resolved_cutoff_month = normalize_month(cutoff_month)
        return resolved_cutoff_month, month_start_date(resolved_cutoff_month)

    raise ValueError("cutoff_month or cutoff_date is required")


def generate_at_cutoff(
    papers: List[PaperRecord],
    strategy: IdeaStrategy,
    cutoff_month: Optional[str] = None,
    top_k: int = 5,
    cutoff_date: Optional[str] = None,
) -> List[IdeaPrediction]:
    resolved_month, resolved_date = _resolve_cutoff(
        cutoff_month=cutoff_month,
        cutoff_date=cutoff_date,
    )
    cutoff_ord = date_to_ordinal(resolved_date)
    train = [
        p
        for p in papers
        if date_to_ordinal(get_paper_published_date(p)) <= cutoff_ord
    ]
    return strategy.generate(train_papers=train, cutoff_month=resolved_month, top_k=top_k)


def split_train_future_by_cutoff(
    papers: List[PaperRecord],
    cutoff_month: Optional[str] = None,
    horizon_months: int = 3,
    cutoff_date: Optional[str] = None,
) -> Tuple[List[PaperRecord], List[PaperRecord], str, str]:
    resolved_cutoff_month, resolved_cutoff_date = _resolve_cutoff(
        cutoff_month=cutoff_month,
        cutoff_date=cutoff_date,
    )
    future_end_month = add_months(resolved_cutoff_month, horizon_months)
    future_end_date = month_end_date(future_end_month)
    cutoff_ord = date_to_ordinal(resolved_cutoff_date)
    future_end_ord = date_to_ordinal(future_end_date)

    train = [
        p
        for p in papers
        if date_to_ordinal(get_paper_published_date(p)) <= cutoff_ord
    ]
    future = [
        p
        for p in papers
        if cutoff_ord < date_to_ordinal(get_paper_published_date(p)) <= future_end_ord
    ]
    return train, future, future_end_month, future_end_date


def evaluate_at_cutoff(
    papers: List[PaperRecord],
    strategy: IdeaStrategy,
    cutoff_month: Optional[str] = None,
    top_k: int = 5,
    horizon_months: int = 3,
    cutoff_date: Optional[str] = None,
) -> EvaluationResult:
    resolved_cutoff_month, _ = _resolve_cutoff(
        cutoff_month=cutoff_month,
        cutoff_date=cutoff_date,
    )
    train, future, _, _ = split_train_future_by_cutoff(
        papers=papers,
        cutoff_month=resolved_cutoff_month,
        horizon_months=horizon_months,
        cutoff_date=cutoff_date,
    )
    preds = strategy.generate(
        train_papers=train,
        cutoff_month=resolved_cutoff_month,
        top_k=top_k,
    )
    return evaluate_predictions(
        predictions=preds,
        train_papers=train,
        future_papers=future,
        k=top_k,
    )


def run_backtest(
    papers: List[PaperRecord],
    strategy: IdeaStrategy,
    config: BacktestConfig,
) -> Dict[str, object]:
    scoped_papers = _filter_by_month(
        papers,
        start_month=config.start_month,
        end_month=config.end_month,
    )
    if not scoped_papers:
        return {
            "summary": {
                "windows": 0,
                "avg_hit_at_k": 0.0,
                "avg_recall_at_k": 0.0,
                "avg_precision_at_k": 0.0,
                "avg_mrr": 0.0,
                "avg_novelty": 0.0,
                "avg_diversity": 0.0,
            },
            "windows": [],
        }

    month_values = sorted(set(p.month for p in scoped_papers), key=month_to_index)
    last_allowed_cutoff = add_months(month_values[-1], -config.horizon_months)
    max_cutoff_idx = month_to_index(last_allowed_cutoff)

    window_results: List[BacktestWindowResult] = []
    for cutoff in month_values:
        cutoff_idx = month_to_index(cutoff)
        if cutoff_idx > max_cutoff_idx:
            continue

        cutoff_date = month_start_date(cutoff)
        train, future, future_end, future_end_date = split_train_future_by_cutoff(
            papers=scoped_papers,
            cutoff_month=cutoff,
            horizon_months=config.horizon_months,
            cutoff_date=cutoff_date,
        )
        if len(train) < config.min_train_papers or not future:
            continue

        preds = strategy.generate(train_papers=train, cutoff_month=cutoff, top_k=config.top_k)
        evaluation = evaluate_at_cutoff(
            papers=scoped_papers,
            strategy=strategy,
            cutoff_month=cutoff,
            cutoff_date=cutoff_date,
            top_k=config.top_k,
            horizon_months=config.horizon_months,
        )

        window_results.append(
            BacktestWindowResult(
                cutoff_month=cutoff,
                cutoff_date=cutoff_date,
                future_end_month=future_end,
                future_end_date=future_end_date,
                train_papers=len(train),
                future_papers=len(future),
                predictions=preds,
                evaluation=evaluation,
            )
        )

    summary = _summarize_windows(window_results)
    return {
        "summary": summary,
        "windows": [asdict(w) for w in window_results],
    }


def _summarize_windows(windows: List[BacktestWindowResult]) -> Dict[str, float]:
    if not windows:
        return {
            "windows": 0,
            "avg_hit_at_k": 0.0,
            "avg_recall_at_k": 0.0,
            "avg_precision_at_k": 0.0,
            "avg_mrr": 0.0,
            "avg_novelty": 0.0,
            "avg_diversity": 0.0,
        }

    count = float(len(windows))
    return {
        "windows": int(count),
        "avg_hit_at_k": round(sum(w.evaluation.hit_at_k for w in windows) / count, 4),
        "avg_recall_at_k": round(sum(w.evaluation.recall_at_k for w in windows) / count, 4),
        "avg_precision_at_k": round(sum(w.evaluation.precision_at_k for w in windows) / count, 4),
        "avg_mrr": round(sum(w.evaluation.mrr for w in windows) / count, 4),
        "avg_novelty": round(sum(w.evaluation.novelty for w in windows) / count, 4),
        "avg_diversity": round(sum(w.evaluation.diversity for w in windows) / count, 4),
    }


# Symmetric aliases for external API usage.
generate = generate_at_cutoff
evaluate = evaluate_at_cutoff
backtest = run_backtest
