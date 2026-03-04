from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from backend import strategy_store
from backend.services.arxiv_ingest import ingest_latest_arxiv_papers
from src.backtest import load_papers_from_markdown
from src.backtest.data import month_to_index
from src.backtest.evaluator import evaluate_predictions
from src.backtest.models import IdeaPrediction


class PipelineAlreadyRunningError(RuntimeError):
    pass


class _FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: Optional[int] = None

    def __enter__(self) -> "_FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise PipelineAlreadyRunningError(f"Lock exists: {self.path}") from exc
        payload = f"pid={os.getpid()} created_at={datetime.now(timezone.utc).isoformat()}\n"
        os.write(self.fd, payload.encode("utf-8"))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _coerce_prediction(raw: Dict[str, Any], rank_fallback: int) -> IdeaPrediction:
    rank_raw = raw.get("rank", rank_fallback)
    confidence_raw = raw.get("confidence", 0.0)
    key_terms_raw = raw.get("key_terms") or []
    if not isinstance(key_terms_raw, list):
        key_terms_raw = []

    try:
        rank = int(rank_raw)
    except (TypeError, ValueError):
        rank = rank_fallback

    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0

    key_terms = [str(term).strip() for term in key_terms_raw if str(term).strip()]
    return IdeaPrediction(
        rank=rank,
        title=str(raw.get("title", "")),
        rationale=str(raw.get("rationale", "")),
        key_terms=key_terms,
        confidence=confidence,
    )


def _latest_month_from_data_dir(data_dir: Path) -> Optional[str]:
    papers = load_papers_from_markdown(data_dir)
    if not papers:
        return None
    return papers[-1].month


def _fallback_backtest_score(strategy: Dict[str, Any]) -> Optional[float]:
    summary = (strategy.get("backtest_result") or {}).get("summary") or {}
    score = summary.get("avg_hit_at_k")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _compute_leaderboard_score(daily_eval: Dict[str, Any]) -> float:
    hit = float(daily_eval.get("hit_at_k", 0.0))
    mrr = float(daily_eval.get("mrr", 0.0))
    return round((0.7 * hit) + (0.3 * mrr), 4)


def _evaluate_previous_generation(
    strategy: Dict[str, Any],
    *,
    new_paper_ids: Set[str],
    evaluated_at: datetime,
) -> Optional[Dict[str, Any]]:
    generation = strategy.get("generation") or {}
    cutoff_month = str(generation.get("cutoff_month") or "").strip()
    predictions_raw = generation.get("predictions")
    if not cutoff_month or not isinstance(predictions_raw, list) or not predictions_raw:
        return None

    papers = strategy_store.load_papers_for_strategy(strategy)
    cutoff_idx = month_to_index(cutoff_month)
    train = [p for p in papers if month_to_index(p.month) <= cutoff_idx]
    future = [
        p
        for p in papers
        if p.paper_id in new_paper_ids and month_to_index(p.month) > cutoff_idx
    ]

    predictions = [
        _coerce_prediction(raw, idx + 1)
        for idx, raw in enumerate(predictions_raw)
        if isinstance(raw, dict)
    ]
    if not predictions:
        return None

    top_k_raw = (strategy.get("config") or {}).get("top_k", len(predictions))
    try:
        top_k = max(1, int(top_k_raw))
    except (TypeError, ValueError):
        top_k = max(1, len(predictions))

    evaluation = evaluate_predictions(
        predictions=predictions,
        train_papers=train,
        future_papers=future,
        k=top_k,
    )
    return {
        "evaluated_at": _iso(evaluated_at),
        "prediction_cutoff_month": cutoff_month,
        "new_papers_count": len(future),
        "prediction_count": len(predictions),
        **asdict(evaluation),
    }


def _ensure_strategy_end_month(strategy: Dict[str, Any], latest_month: Optional[str]) -> Dict[str, Any]:
    if not latest_month:
        return strategy

    current = dict(strategy.get("config") or {})
    current_end = str(current.get("end_month") or "").strip()
    should_update = False
    if not current_end:
        should_update = True
    else:
        try:
            should_update = month_to_index(current_end) < month_to_index(latest_month)
        except Exception:
            should_update = True

    if not should_update:
        return strategy

    current["end_month"] = latest_month
    updated = strategy_store.update_strategy(strategy["id"], {"config": current})
    return updated or strategy


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _extract_new_paper_ids(ingest_result: Dict[str, Any]) -> Set[str]:
    raw = ingest_result.get("new_papers") or []
    ids: Set[str] = set()
    if not isinstance(raw, list):
        return ids
    for item in raw:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or "").strip()
        if paper_id:
            ids.add(paper_id)
    return ids


def run_daily_pipeline(
    *,
    now: Optional[datetime] = None,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    utc_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    project_root = strategy_store.PROJECT_ROOT
    run_dir = project_root / "data" / "daily_runs"
    lock_path = run_dir / "pipeline.lock"

    with _FileLock(lock_path):
        ingest_result = ingest_latest_arxiv_papers(data_dir=data_dir, now=utc_now)
        raw_data_dir = str(ingest_result.get("data_dir") or "").strip()
        resolved_data_dir = (
            Path(raw_data_dir).expanduser()
            if raw_data_dir
            else strategy_store.DEFAULT_DATA_DIR
        )
        latest_month = _latest_month_from_data_dir(resolved_data_dir)
        new_paper_ids = _extract_new_paper_ids(ingest_result)

        strategy_results: List[Dict[str, Any]] = []
        for strategy in strategy_store.list_strategies():
            strategy_id = strategy.get("id")
            if not strategy_id:
                continue

            current = strategy_store.get_strategy(strategy_id) or strategy
            current = _ensure_strategy_end_month(current, latest_month)
            daily_eval = _evaluate_previous_generation(
                current,
                new_paper_ids=new_paper_ids,
                evaluated_at=utc_now,
            )

            updates: Dict[str, Any] = {"last_daily_run_at": _iso(utc_now)}
            if latest_month:
                updates["last_generation_cutoff_month"] = latest_month
            if daily_eval is not None:
                updates["daily_evaluation"] = daily_eval
                updates["leaderboard_score"] = _compute_leaderboard_score(daily_eval)
            elif current.get("leaderboard_score") is None:
                updates["leaderboard_score"] = _fallback_backtest_score(current)
            strategy_store.update_strategy(strategy_id, updates)

            generation_status = "skipped_no_data"
            generation_error = None
            if latest_month:
                strategy_store.run_generation_sync(strategy_id, cutoff_month=latest_month)
                after_generation = strategy_store.get_strategy(strategy_id) or {}
                generation_status = str(after_generation.get("generation_status") or "unknown")
                generation_error = after_generation.get("generation_error")

            final = strategy_store.get_strategy(strategy_id) or {}
            strategy_results.append(
                {
                    "id": strategy_id,
                    "name": final.get("name"),
                    "leaderboard_score": final.get("leaderboard_score"),
                    "generation_status": generation_status,
                    "generation_error": generation_error,
                    "daily_evaluation": final.get("daily_evaluation"),
                }
            )

        report = {
            "ran_at": _iso(utc_now),
            "latest_month": latest_month,
            "ingest": ingest_result,
            "strategies_processed": len(strategy_results),
            "strategies": strategy_results,
        }
        report_path = run_dir / f"{utc_now.date().isoformat()}.json"
        latest_path = run_dir / "latest.json"
        _write_json(report_path, report)
        _write_json(latest_path, report)
        return report
