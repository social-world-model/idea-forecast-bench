"""
Strategy store: CRUD for Strategy objects, backed by per-file JSON in backend/strategies/.

A Strategy bundles:
  - strategy_name : which IdeaStrategy implementation ("keyword_trend", …)
  - params        : strategy hyper-params (recent_months, min_keyword_freq)
  - config        : BacktestConfig fields + data_dir
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from live_idea_bench.config import TopicDefinition, load_topics
from live_idea_bench.models import PaperRecord
from live_idea_bench.papers import load_papers_from_markdown
from live_idea_bench.topics import classify_papers_by_topic

if TYPE_CHECKING:
    from typing import SupportsFloat, SupportsInt

    from live_idea_bench.strategy.base import IdeaStrategy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "arxiv_csml" / "raw_markdown"
DEFAULT_MODEL_NAME = "gpt-4o"

STRATEGIES_DIR = Path(__file__).parent / "strategies"
STRATEGIES_DIR.mkdir(exist_ok=True)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _path(strategy_id: str) -> Path:
    return STRATEGIES_DIR / f"{strategy_id}.json"


def _coerce_float(value: object) -> float | None:
    """Best-effort float, mirroring _coerce_int's tolerance of raw JSON.

    Returns None rather than raising: a stored strategy carrying a junk
    temperature used to make _read() swallow the exception and drop the
    record entirely, so one bad field silently deleted a whole strategy.
    """
    if value is None:
        return None
    try:
        return float(cast("SupportsFloat | str", value))
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object, default: int) -> int:
    try:
        # Deliberate duck-typing of an untyped JSON value; the cast keeps the
        # runtime call unchanged and the failure modes are caught below.
        return int(cast("SupportsInt | str", value))
    except (TypeError, ValueError):
        return default


def _runtime_default_data_dir() -> Path:
    override = os.environ.get("LIVE_IDEA_BENCH_DATA_DIR", "").strip()
    if not override:
        return DEFAULT_DATA_DIR
    return Path(override).expanduser()


def _normalize_params(
    strategy_name: str,
    params: object,
    *,
    strict_keyword_coercion: bool = False,
) -> dict[str, Any]:
    raw_params = params if isinstance(params, dict) else {}
    normalized_params = dict(raw_params)

    if strategy_name == "keyword_trend":
        if strict_keyword_coercion:
            normalized_params["recent_months"] = int(raw_params.get("recent_months", 3))
            normalized_params["min_keyword_freq"] = int(
                raw_params.get("min_keyword_freq", 2)
            )
        else:
            normalized_params["recent_months"] = _coerce_int(
                raw_params.get("recent_months", 3), 3
            )
            normalized_params["min_keyword_freq"] = _coerce_int(
                raw_params.get("min_keyword_freq", 2), 2
            )
    else:
        if raw_params.get("model_name") in {None, ""} and raw_params.get(
            "model_id"
        ) not in {None, ""}:
            normalized_params["model_name"] = raw_params.get("model_id")
        normalized_params.setdefault("model_name", DEFAULT_MODEL_NAME)
        normalized_params.setdefault("predictor_config", "predictor.yaml")
        normalized_params.setdefault("similarity_config", "similarity.yaml")
        temperature = _coerce_float(raw_params.get("temperature"))
        if temperature is not None:
            normalized_params["temperature"] = temperature

    normalized_params.pop("model_id", None)
    normalized_params.pop("prompt_id", None)
    normalized_params.pop("prompt_version", None)

    return normalized_params


def _normalize_prediction(raw: object, rank_fallback: int) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    key_terms_raw = raw.get("key_terms") or raw.get("keywords") or []
    if not isinstance(key_terms_raw, list):
        key_terms_raw = []

    def _coerce_score(value: object, default: float = 0.0) -> float:
        try:
            # Same duck-typing as _coerce_int: JSON scores may arrive as str.
            resolved = float(cast("SupportsFloat | str", value))
        except (TypeError, ValueError):
            resolved = default
        if resolved > 1.0:
            resolved = resolved / 10.0
        return round(min(1.0, max(0.0, resolved)), 4)

    confidence = _coerce_score(
        raw.get(
            "confidence", raw.get("Confidence", raw.get("score", raw.get("Score", 0.0)))
        ),
        default=0.0,
    )
    score = _coerce_score(
        raw.get("score", raw.get("Score", confidence)), default=confidence
    )

    try:
        rank = int(raw.get("rank", rank_fallback))
    except (TypeError, ValueError):
        rank = rank_fallback

    return {
        "rank": rank,
        "title": str(raw.get("title") or raw.get("Title") or ""),
        "rationale": str(raw.get("rationale") or raw.get("Rationale") or ""),
        "approach": str(raw.get("approach") or raw.get("Approach") or ""),
        "score": score,
        "confidence": confidence,
        "key_terms": [str(term).strip() for term in key_terms_raw if str(term).strip()],
        "metadata": raw.get("metadata")
        if isinstance(raw.get("metadata"), dict)
        else {},
    }


def _normalize_generation(generation: object) -> object:
    if not isinstance(generation, dict):
        return generation

    predictions_raw = generation.get("predictions")
    predictions = []
    if isinstance(predictions_raw, list):
        for idx, raw in enumerate(predictions_raw, start=1):
            normalized = _normalize_prediction(raw, idx)
            if normalized is not None:
                predictions.append(normalized)

    normalized = dict(generation)
    if predictions:
        normalized["predictions"] = predictions
    cutoff_month = str(normalized.get("cutoff_month") or "").strip()
    cutoff_date = str(normalized.get("cutoff_date") or "").strip()
    if not cutoff_date and cutoff_month:
        normalized["cutoff_date"] = f"{cutoff_month}-01"
    return normalized


def _normalize_evaluation(evaluation: object) -> object:
    if not isinstance(evaluation, dict):
        return evaluation
    normalized = dict(evaluation)
    normalized.setdefault("matched_paper_ids", [])
    normalized.setdefault("matched_prediction_ranks", [])
    normalized.setdefault("lead_time", 0.0)
    normalized.setdefault("duplicate_rate", 0.0)
    normalized.pop("matched_terms", None)
    return normalized


def _normalize_backtest_result(backtest_result: object) -> object:
    if not isinstance(backtest_result, dict):
        return backtest_result
    normalized = dict(backtest_result)
    windows = normalized.get("windows")
    if isinstance(windows, list):
        normalized_windows = []
        for window in windows:
            if not isinstance(window, dict):
                continue
            updated = dict(window)
            predictions_raw = updated.get("predictions")
            if isinstance(predictions_raw, list):
                predictions = []
                for idx, raw in enumerate(predictions_raw, start=1):
                    pred = _normalize_prediction(raw, idx)
                    if pred is not None:
                        predictions.append(pred)
                updated["predictions"] = predictions
            updated["evaluation"] = _normalize_evaluation(updated.get("evaluation"))
            normalized_windows.append(updated)
        normalized["windows"] = normalized_windows
    return normalized


def _normalize_status(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"pending", "running", "done", "failed"}:
        return raw
    return "pending"


def _normalize_topic_run(topic_run: object) -> dict[str, Any] | None:
    if not isinstance(topic_run, dict):
        return None

    normalized = dict(topic_run)
    normalized["topic_id"] = str(normalized.get("topic_id") or "").strip()
    normalized["topic_name"] = str(normalized.get("topic_name") or "").strip()
    normalized["matched_paper_count"] = _coerce_int(
        normalized.get("matched_paper_count", 0), 0
    )
    normalized["generation_status"] = _normalize_status(
        normalized.get("generation_status")
    )
    normalized["backtest_status"] = _normalize_status(normalized.get("backtest_status"))
    normalized["generation"] = _normalize_generation(normalized.get("generation"))
    normalized["backtest_result"] = _normalize_backtest_result(
        normalized.get("backtest_result")
    )
    normalized["generation_error"] = (
        str(normalized.get("generation_error")).strip()
        if normalized.get("generation_error") not in {None, ""}
        else None
    )
    normalized["backtest_error"] = (
        str(normalized.get("backtest_error")).strip()
        if normalized.get("backtest_error") not in {None, ""}
        else None
    )
    if not normalized["topic_id"] or not normalized["topic_name"]:
        return None
    return normalized


def _normalize_topic_runs(topic_runs: object) -> list[dict[str, Any]]:
    if not isinstance(topic_runs, list):
        return []

    normalized: list[dict[str, Any]] = []
    for raw in topic_runs:
        topic_run = _normalize_topic_run(raw)
        if topic_run is not None:
            normalized.append(topic_run)
    return normalized


def _aggregate_topic_backtest_summary(
    topic_runs: list[dict[str, Any]],
) -> dict[str, float] | None:
    summaries: list[tuple[dict[str, Any], int]] = []
    for topic_run in topic_runs:
        summary = (topic_run.get("backtest_result") or {}).get("summary") or {}
        try:
            windows = max(0, int(summary.get("windows", 0)))
        except (TypeError, ValueError):
            windows = 0
        if windows <= 0:
            continue
        summaries.append((summary, windows))

    if not summaries:
        return None

    total_windows = sum(windows for _, windows in summaries)

    def _weighted_avg(name: str) -> float:
        total = 0.0
        for summary, windows in summaries:
            try:
                value = float(summary.get(name, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            total += value * windows
        return round(total / total_windows, 4) if total_windows else 0.0

    return {
        "windows": total_windows,
        "avg_hit_at_k": _weighted_avg("avg_hit_at_k"),
        "avg_recall_at_k": _weighted_avg("avg_recall_at_k"),
        "avg_precision_at_k": _weighted_avg("avg_precision_at_k"),
        "avg_mrr": _weighted_avg("avg_mrr"),
        "avg_novelty": _weighted_avg("avg_novelty"),
        "avg_diversity": _weighted_avg("avg_diversity"),
    }


def _configured_topics() -> list[TopicDefinition]:
    return load_topics()


def _seed_topic_runs(
    topics: list[TopicDefinition], existing_runs: object
) -> list[dict[str, Any]]:
    existing_by_id = {
        topic_run["topic_id"]: topic_run
        for topic_run in _normalize_topic_runs(existing_runs)
    }
    seeded: list[dict[str, Any]] = []
    for topic in topics:
        current = dict(existing_by_id.get(topic.id) or {})
        seeded.append(
            {
                "topic_id": topic.id,
                "topic_name": topic.name,
                "matched_paper_count": _coerce_int(
                    current.get("matched_paper_count", 0), 0
                ),
                "generation_status": _normalize_status(
                    current.get("generation_status")
                ),
                "backtest_status": _normalize_status(current.get("backtest_status")),
                "generation": _normalize_generation(current.get("generation")),
                "backtest_result": _normalize_backtest_result(
                    current.get("backtest_result")
                ),
                "generation_error": current.get("generation_error"),
                "backtest_error": current.get("backtest_error"),
            }
        )
    return seeded


def _normalize_strategy(strategy: dict[str, Any]) -> dict[str, Any]:
    strategy_name = str(strategy.get("strategy_name") or "keyword_trend")
    if strategy_name == "prompt_llm":
        strategy_name = "predictor_llm"
    normalized = dict(strategy)
    normalized["strategy_name"] = strategy_name
    normalized["params"] = _normalize_params(strategy_name, strategy.get("params"))
    normalized["generation"] = _normalize_generation(strategy.get("generation"))
    normalized["backtest_result"] = _normalize_backtest_result(
        strategy.get("backtest_result")
    )
    normalized["topic_runs"] = _normalize_topic_runs(strategy.get("topic_runs"))
    normalized["daily_evaluation"] = _normalize_evaluation(
        strategy.get("daily_evaluation")
    )
    normalized.setdefault("leaderboard_score", None)
    normalized.setdefault("last_daily_run_at", None)
    normalized.setdefault("last_generation_cutoff_month", None)
    return normalized


def _read(strategy_id: str) -> dict[str, Any] | None:
    p = _path(strategy_id)
    if not p.exists():
        return None
    try:
        loaded = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return None
        return _normalize_strategy(loaded)
    except Exception:
        return None


def _write(strategy: dict[str, Any]) -> None:
    _path(strategy["id"]).write_text(
        json.dumps(strategy, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _sort_key(s: dict[str, Any]) -> tuple[float, str]:
    score = s.get("leaderboard_score")
    if score is not None:
        try:
            return (float(score), s.get("created_at", ""))
        except (TypeError, ValueError):
            pass
    summary = _aggregate_topic_backtest_summary(s.get("topic_runs") or [])
    if summary is None:
        summary = (s.get("backtest_result") or {}).get("summary") or {}
    # Read straight from unvalidated JSON. Without coercion a single record
    # storing avg_hit_at_k as a string makes strategies.sort() raise
    # TypeError comparing str with float, taking down GET /api/strategies
    # for every strategy, not just the malformed one.
    hit = _coerce_float(summary.get("avg_hit_at_k"))
    return (hit if hit is not None else -1.0, s.get("created_at", ""))


# ── Public API ────────────────────────────────────────────────────────────────


def list_strategies() -> list[dict[str, Any]]:
    strategies: list[dict[str, Any]] = []
    for p in sorted(STRATEGIES_DIR.glob("*.json")):
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                strategies.append(_normalize_strategy(loaded))
        except Exception:
            pass
    strategies.sort(key=_sort_key, reverse=True)
    return strategies


def get_strategy(strategy_id: str) -> dict[str, Any] | None:
    return _read(strategy_id)


def create_strategy(data: dict[str, Any]) -> dict[str, Any]:
    strategy_id = uuid.uuid4().hex[:8]
    strategy_name = str(data.get("strategy_name", "keyword_trend"))
    if strategy_name == "prompt_llm":
        strategy_name = "predictor_llm"
    config = data.get("config") or {}
    raw_data_dir = config.get("data_dir", "")
    data_dir = str(raw_data_dir).strip() if raw_data_dir is not None else ""

    strategy: dict[str, Any] = {
        "id": strategy_id,
        "name": data.get("name") or f"{strategy_name} [{strategy_id}]",
        # Which IdeaStrategy implementation to use (matches IdeaStrategy.name)
        "strategy_name": strategy_name,
        # Strategy hyper-parameters passed to the strategy constructor
        "params": _normalize_params(
            strategy_name,
            data.get("params") or {},
            strict_keyword_coercion=True,
        ),
        # BacktestConfig fields + data path
        "config": {
            "top_k": int(config.get("top_k", 5)),
            "horizon_months": int(config.get("horizon_months", 3)),
            "min_train_papers": int(config.get("min_train_papers", 6)),
            "start_month": config.get("start_month", "2024-01"),
            "end_month": config.get("end_month", "2024-12"),
            "data_dir": data_dir,
        },
        "created_at": datetime.now().isoformat(),
        "backtest_status": "pending",
        "generation_status": "pending",
        "backtest_result": None,  # { summary: {...}, windows: [...] }
        "generation": None,  # { cutoff_date: str, cutoff_month: str, predictions: [...] }
        "topic_runs": [],
        "leaderboard_score": None,
        "daily_evaluation": None,
        "last_daily_run_at": None,
        "last_generation_cutoff_month": None,
    }
    _write(strategy)
    return strategy


def update_strategy(strategy_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    strategy = _read(strategy_id)
    if strategy is None:
        return None
    strategy.update(updates)
    _write(strategy)
    return strategy


def delete_strategy(strategy_id: str) -> bool:
    p = _path(strategy_id)
    if p.exists():
        p.unlink()
        return True
    return False


# ── Engine helpers ─────────────────────────────────────────────────────────────


def _resolve_data_dir(s: dict[str, Any]) -> Path:
    raw_data_dir = (s.get("config") or {}).get("data_dir", "")
    data_dir = str(raw_data_dir).strip() if raw_data_dir is not None else ""

    if not data_dir:
        return _runtime_default_data_dir()

    configured = Path(data_dir).expanduser()
    if configured.is_absolute() and not configured.exists():
        return _runtime_default_data_dir()

    if configured.is_absolute():
        return configured

    return (PROJECT_ROOT / configured).resolve()


def _load_papers(s: dict[str, Any]) -> list[PaperRecord]:
    """Load PaperRecord list for this strategy's data_dir and time window."""
    return load_papers_from_markdown(
        _resolve_data_dir(s),
        start_month=s["config"].get("start_month"),
        end_month=s["config"].get("end_month"),
    )


def resolve_data_dir_for_strategy(strategy: dict[str, Any]) -> Path:
    return _resolve_data_dir(strategy)


def load_papers_for_strategy(strategy: dict[str, Any]) -> list[PaperRecord]:
    return _load_papers(strategy)


def _make_strategy_obj(s: dict[str, Any]) -> "IdeaStrategy":
    """Instantiate the IdeaStrategy from the live_idea_bench registry."""
    from live_idea_bench.strategy.execution import build_strategy

    return build_strategy(s)


def _classify_strategy_papers(
    papers: list[PaperRecord],
    topics: list[TopicDefinition],
) -> dict[str, list[PaperRecord]]:
    return dict(classify_papers_by_topic(papers, topics))


def _aggregate_mode_status(topic_runs: list[dict[str, Any]], mode: str) -> str:
    statuses = [
        _normalize_status(topic_run.get(f"{mode}_status")) for topic_run in topic_runs
    ]
    if not statuses:
        return "pending"
    if any(status == "running" for status in statuses):
        return "running"
    if any(status == "failed" for status in statuses):
        return "failed"
    if all(status == "done" for status in statuses):
        return "done"
    return "pending"


def _collect_mode_errors(topic_runs: list[dict[str, Any]], mode: str) -> str | None:
    errors: list[str] = []
    for topic_run in topic_runs:
        error = topic_run.get(f"{mode}_error")
        if error in {None, ""}:
            continue
        errors.append(f"{topic_run['topic_id']}: {error}")
    return "; ".join(errors) if errors else None


# ── Synchronous execution (used for seeding / testing) ───────────────────────


def run_backtest_sync(strategy_id: str) -> None:
    """Run backtest synchronously and persist the result."""
    from live_idea_bench.strategy.execution import run_strategy_backtest

    s = _read(strategy_id)
    if s is None:
        return

    update_strategy(strategy_id, {"backtest_status": "running"})
    try:
        topics = _configured_topics()
        if not topics:
            raise ValueError("No topics configured")

        papers = _load_papers(s)
        topic_papers = _classify_strategy_papers(papers, topics)
        topic_runs = _seed_topic_runs(topics, s.get("topic_runs"))

        for topic_run in topic_runs:
            topic_run["matched_paper_count"] = len(
                topic_papers.get(topic_run["topic_id"], [])
            )
            topic_run["backtest_status"] = "running"
            topic_run["backtest_result"] = None
            topic_run["backtest_error"] = None

        update_strategy(
            strategy_id,
            {
                "backtest_status": "running",
                "backtest_result": None,
                "topic_runs": topic_runs,
                "backtest_error": None,
            },
        )

        for topic_run in topic_runs:
            scoped_papers = topic_papers.get(topic_run["topic_id"], [])
            if not scoped_papers:
                topic_run["backtest_status"] = "done"
                topic_run["backtest_result"] = None
                continue
            try:
                topic_run["backtest_result"] = run_strategy_backtest(s, scoped_papers)
                topic_run["backtest_status"] = "done"
            except Exception as exc:
                topic_run["backtest_status"] = "failed"
                topic_run["backtest_result"] = None
                topic_run["backtest_error"] = str(exc)

        update_strategy(
            strategy_id,
            {
                "backtest_status": _aggregate_mode_status(topic_runs, "backtest"),
                "backtest_result": None,
                "topic_runs": topic_runs,
                "backtest_error": _collect_mode_errors(topic_runs, "backtest"),
            },
        )
    except Exception as e:
        update_strategy(
            strategy_id,
            {
                "backtest_status": "failed",
                "backtest_result": None,
                "backtest_error": str(e),
            },
        )


def run_generation_sync(strategy_id: str, cutoff_date: str | None = None) -> None:
    """Run idea generation synchronously and persist the result."""
    s = _read(strategy_id)
    if s is None:
        return

    update_strategy(strategy_id, {"generation_status": "running"})
    try:
        topics = _configured_topics()
        if not topics:
            raise ValueError("No topics configured")

        # Resolve cutoff_date: use supplied value or derive from config end_month.
        if not cutoff_date:
            config_end_month = (s.get("config") or {}).get("end_month") or ""
            cutoff_date = f"{config_end_month}-01" if config_end_month else ""

        if not cutoff_date:
            raise ValueError("cutoff_date is required for generation")

        from live_idea_bench.strategy.execution import run_strategy_generation

        papers = _load_papers(s)
        topic_papers = _classify_strategy_papers(papers, topics)
        topic_runs = _seed_topic_runs(topics, s.get("topic_runs"))

        for topic_run in topic_runs:
            topic_run["matched_paper_count"] = len(
                topic_papers.get(topic_run["topic_id"], [])
            )
            topic_run["generation_status"] = "running"
            topic_run["generation"] = None
            topic_run["generation_error"] = None

        update_strategy(
            strategy_id,
            {
                "generation_status": "running",
                "generation": None,
                "topic_runs": topic_runs,
                "generation_error": None,
            },
        )

        for topic_run in topic_runs:
            scoped_papers = topic_papers.get(topic_run["topic_id"], [])
            if not scoped_papers:
                topic_run["generation_status"] = "done"
                topic_run["generation"] = None
                continue
            try:
                topic_run["generation"] = run_strategy_generation(
                    s,
                    scoped_papers,
                    cutoff_date=cutoff_date,
                )
                topic_run["generation_status"] = "done"
            except Exception as exc:
                topic_run["generation_status"] = "failed"
                topic_run["generation"] = None
                topic_run["generation_error"] = str(exc)

        update_strategy(
            strategy_id,
            {
                "generation_status": _aggregate_mode_status(topic_runs, "generation"),
                "generation": None,
                "topic_runs": topic_runs,
                "generation_error": _collect_mode_errors(topic_runs, "generation"),
            },
        )
    except Exception as e:
        update_strategy(
            strategy_id,
            {
                "generation_status": "failed",
                "generation": None,
                "generation_error": str(e),
            },
        )


# ── Demo seeding ───────────────────────────────────────────────────────────────


def seed_demo_strategies() -> None:
    """Seed demo strategies and run their backtests if no strategies exist yet."""
    if list_strategies():
        return

    demos = [
        {
            "name": "Keyword Trend · 2024-H1 → 2024-H2",
            "strategy_name": "keyword_trend",
            "params": {"recent_months": 3, "min_keyword_freq": 1},
            "config": {
                "top_k": 5,
                "horizon_months": 3,
                "min_train_papers": 2,
                "start_month": "2024-01",
                "end_month": "2024-09",
            },
        },
        {
            "name": "Keyword Trend · 2024 Full Year",
            "strategy_name": "keyword_trend",
            "params": {"recent_months": 6, "min_keyword_freq": 1},
            "config": {
                "top_k": 8,
                "horizon_months": 3,
                "min_train_papers": 2,
                "start_month": "2024-01",
                "end_month": "2024-12",
            },
        },
        {
            "name": "Keyword Trend · 2025 Preview",
            "strategy_name": "keyword_trend",
            "params": {"recent_months": 2, "min_keyword_freq": 1},
            "config": {
                "top_k": 5,
                "horizon_months": 2,
                "min_train_papers": 2,
                "start_month": "2025-01",
                "end_month": "2025-06",
            },
        },
    ]

    for demo in demos:
        s = create_strategy(demo)
        print(f"  Seeding: {s['name']}")
        run_backtest_sync(s["id"])
        result = _read(s["id"]) or {}
        summary = _aggregate_topic_backtest_summary(result.get("topic_runs") or [])
        if summary is None:
            summary = (result.get("backtest_result") or {}).get("summary", {})
        print(
            f"    → status={result.get('backtest_status')}  "
            f"windows={summary.get('windows', 0)}  "
            f"avg_hit@k={summary.get('avg_hit_at_k', 0)}"
        )


def has_leaderboard_baseline() -> bool:
    for strategy in list_strategies():
        if _aggregate_topic_backtest_summary(strategy.get("topic_runs") or []):
            return True
        backtest_result = strategy.get("backtest_result")
        if isinstance(backtest_result, dict) and backtest_result:
            return True
    return False


def bootstrap_backtest_if_missing() -> dict[str, Any]:
    strategies = list_strategies()
    if not strategies:
        return {
            "triggered": False,
            "reason": "no_strategies",
            "count": 0,
            "strategy_ids": [],
        }

    if has_leaderboard_baseline():
        return {
            "triggered": False,
            "reason": "baseline_exists",
            "count": 0,
            "strategy_ids": [],
        }

    executed_ids = []
    for strategy in strategies:
        strategy_id = strategy.get("id")
        if not strategy_id:
            continue
        run_backtest_sync(strategy_id)
        executed_ids.append(strategy_id)

    return {
        "triggered": True,
        "reason": "missing_baseline",
        "count": len(executed_ids),
        "strategy_ids": executed_ids,
    }
