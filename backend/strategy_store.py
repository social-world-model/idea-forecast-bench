"""
Strategy store: CRUD for Strategy objects, backed by per-file JSON in backend/strategies/.

A Strategy bundles:
  - strategy_name : which IdeaStrategy implementation ("keyword_trend", …)
  - params        : strategy hyper-params (recent_months, min_keyword_freq)
  - config        : BacktestConfig fields + data_dir
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data" / "arxiv_csml" / "raw_markdown")

STRATEGIES_DIR = Path(__file__).parent / "strategies"
STRATEGIES_DIR.mkdir(exist_ok=True)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _path(strategy_id: str) -> Path:
    return STRATEGIES_DIR / f"{strategy_id}.json"


def _read(strategy_id: str) -> Optional[dict]:
    p = _path(strategy_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write(strategy: dict) -> None:
    _path(strategy["id"]).write_text(
        json.dumps(strategy, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _sort_key(s: dict) -> tuple:
    summary = (s.get("backtest_result") or {}).get("summary") or {}
    hit = summary.get("avg_hit_at_k", -1)
    return (hit, s.get("created_at", ""))


# ── Public API ────────────────────────────────────────────────────────────────

def list_strategies() -> List[dict]:
    strategies = []
    for p in sorted(STRATEGIES_DIR.glob("*.json")):
        try:
            strategies.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    strategies.sort(key=_sort_key, reverse=True)
    return strategies


def get_strategy(strategy_id: str) -> Optional[dict]:
    return _read(strategy_id)


def create_strategy(data: dict) -> dict:
    strategy_id = uuid.uuid4().hex[:8]
    strategy = {
        "id": strategy_id,
        "name": data.get("name") or "{} [{}]".format(
            data.get("strategy_name", "keyword_trend"), strategy_id
        ),
        # Which IdeaStrategy implementation to use (matches IdeaStrategy.name)
        "strategy_name": data.get("strategy_name", "keyword_trend"),
        # Strategy hyper-parameters passed to the strategy constructor
        "params": {
            "recent_months": int((data.get("params") or {}).get("recent_months", 3)),
            "min_keyword_freq": int((data.get("params") or {}).get("min_keyword_freq", 2)),
        },
        # BacktestConfig fields + data path
        "config": {
            "top_k": int((data.get("config") or {}).get("top_k", 5)),
            "horizon_months": int((data.get("config") or {}).get("horizon_months", 3)),
            "min_train_papers": int((data.get("config") or {}).get("min_train_papers", 6)),
            "start_month": (data.get("config") or {}).get("start_month", "2024-01"),
            "end_month": (data.get("config") or {}).get("end_month", "2024-12"),
            # Leave blank to use DEFAULT_DATA_DIR
            "data_dir": (data.get("config") or {}).get("data_dir", "") or DEFAULT_DATA_DIR,
        },
        "created_at": datetime.now().isoformat(),
        "backtest_status": "pending",
        "generation_status": "pending",
        "backtest_result": None,    # { summary: {...}, windows: [...] }
        "generation": None,         # { cutoff_month: str, predictions: [...] }
    }
    _write(strategy)
    return strategy


def update_strategy(strategy_id: str, updates: dict) -> Optional[dict]:
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

def _resolve_data_dir(s: dict) -> Path:
    d = (s.get("config") or {}).get("data_dir", "") or DEFAULT_DATA_DIR
    return Path(d)


def _load_papers(s: dict):
    """Load PaperRecord list for this strategy's data_dir and time window."""
    from src.backtest import load_papers_from_markdown
    return load_papers_from_markdown(
        _resolve_data_dir(s),
        start_month=s["config"].get("start_month"),
        end_month=s["config"].get("end_month"),
    )


def _make_strategy_obj(s: dict):
    """Instantiate the IdeaStrategy from src.strategy registry."""
    from src import create_strategy as src_create
    return src_create(
        s["strategy_name"],
        recent_months=s["params"]["recent_months"],
        min_keyword_freq=s["params"]["min_keyword_freq"],
    )


# ── Synchronous execution (used for seeding / testing) ───────────────────────

def run_backtest_sync(strategy_id: str) -> None:
    """Run backtest synchronously and persist the result."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from src import BacktestConfig, backtest

    s = _read(strategy_id)
    if s is None:
        return

    update_strategy(strategy_id, {"backtest_status": "running"})
    try:
        papers = _load_papers(s)
        strategy_obj = _make_strategy_obj(s)
        cfg = BacktestConfig(
            top_k=s["config"]["top_k"],
            horizon_months=s["config"]["horizon_months"],
            min_train_papers=s["config"]["min_train_papers"],
            start_month=s["config"].get("start_month"),
            end_month=s["config"].get("end_month"),
        )
        result = backtest(papers, strategy_obj, cfg)
        update_strategy(strategy_id, {
            "backtest_status": "done",
            "backtest_result": result,
        })
    except Exception as e:
        update_strategy(strategy_id, {
            "backtest_status": "failed",
            "backtest_error": str(e),
        })


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
        result = _read(s["id"])
        summary = (result.get("backtest_result") or {}).get("summary", {})
        print(f"    → status={result.get('backtest_status')}  "
              f"windows={summary.get('windows', 0)}  "
              f"avg_hit@k={summary.get('avg_hit_at_k', 0)}")
