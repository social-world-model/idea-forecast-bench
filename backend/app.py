from __future__ import annotations

import json
import os
import secrets
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

from flask import Flask, jsonify, request
from flask_cors import CORS

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from backend import config
from backend.services.run_service import RunService
from backend.strategy_store import (
    bootstrap_backtest_if_missing,
    create_strategy,
    delete_strategy,
    get_strategy,
    list_strategies,
    run_backtest_sync,
    run_generation_sync,
    update_strategy,
)

app = Flask(__name__)

GENERATED_IDEAS_FILE = os.path.join(project_root, "backend", "generated_ideas.json")
VIEWS_FILE = os.path.join(project_root, "data", "views.json")

run_service = RunService(project_root=project_root)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return value not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _cors_origins() -> List[str]:
    raw = os.environ.get("LIVE_IDEA_CORS_ORIGINS", "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def _admin_token() -> str:
    return os.environ.get("LIVE_IDEA_ADMIN_TOKEN", "").strip()


def _request_admin_token() -> str:
    bearer = request.headers.get("Authorization", "").strip()
    if bearer.lower().startswith("bearer "):
        return bearer[7:].strip()
    return request.headers.get("X-Live-Idea-Admin-Token", "").strip()


def _auth_bypass_enabled() -> bool:
    return app.testing or bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _is_protected_write_request() -> bool:
    if request.method not in {"POST", "DELETE", "PUT", "PATCH"}:
        return False
    protected_paths = (
        "/api/generate-ideas",
        "/api/runs/start",
        "/api/strategies",
    )
    return any(
        request.path == path or request.path.startswith(f"{path}/")
        for path in protected_paths
    )


CORS(
    app,
    resources={
        r"/api/*": {"origins": _cors_origins()},
        r"/healthz": {"origins": _cors_origins()},
        r"/metrics": {"origins": _cors_origins()},
    },
)


def _bootstrap_leaderboard_async() -> None:
    if not _env_flag("LIVE_IDEA_BOOTSTRAP_BACKTEST", True):
        return

    def _worker() -> None:
        try:
            bootstrap_backtest_if_missing()
        except Exception as exc:  # pragma: no cover - defensive startup guard
            print(f"[bootstrap] backtest bootstrap failed: {exc}")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


_bootstrap_leaderboard_async()


class APIError(Exception):
    def __init__(self, message: str, status_code: int = 400, code: str = "bad_request", details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details


@app.before_request
def guard_protected_write_endpoints():
    if not _is_protected_write_request() or _auth_bypass_enabled():
        return None

    expected = _admin_token()
    if not expected:
        raise APIError(
            "Protected write endpoints are disabled until LIVE_IDEA_ADMIN_TOKEN is configured",
            status_code=503,
            code="admin_token_not_configured",
        )

    provided = _request_admin_token()
    if not provided or not secrets.compare_digest(provided, expected):
        raise APIError("Admin token required", status_code=403, code="forbidden")
    return None


@app.errorhandler(APIError)
def handle_api_error(error: APIError):
    payload = {
        "error": {
            "code": error.code,
            "message": error.message,
            "details": error.details if app.debug or app.testing else None,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(payload), error.status_code


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    app.logger.exception("Unhandled server error")
    payload = {
        "error": {
            "code": "internal_error",
            "message": "Unexpected server error",
            "details": None,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(payload), 500


def _load_generated_ideas() -> List[Dict[str, Any]]:
    if not os.path.exists(GENERATED_IDEAS_FILE):
        return []
    with open(GENERATED_IDEAS_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_generated_ideas(ideas: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(GENERATED_IDEAS_FILE), exist_ok=True)
    with open(GENERATED_IDEAS_FILE, "w", encoding="utf-8") as fh:
        json.dump(ideas, fh, indent=2)


def _transform_ideas_for_dashboard(ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    transformed_ideas = []
    for idea in ideas:
        transformed = {
            "id": idea.get("id", f"gen_{os.urandom(4).hex()}"),
            "title": idea.get("Title", "Untitled Idea"),
            "description": f"**Problem:** {idea.get('Problem', '')}\n\n**Approach:** {idea.get('Approach', '')}",
            "author": "AI Researcher",
            "institution": "Live Idea Bench",
            "tags": ["AI Generated", "ICLR 2025"],
            "upvotes": int(float(idea.get("Score", 0)) * 10 + float(idea.get("Interestingness", 0))),
            "impact_score": float(idea.get("Score", 0)),
            "created_at": now,
            "updated_at": now,
            "url": idea.get("source_url"),
            "citations": 0,
        }
        if float(idea.get("Novelty", 0)) > 8:
            transformed["tags"].append("High Novelty")
        if float(idea.get("Feasibility", 0)) > 8:
            transformed["tags"].append("High Feasibility")
        transformed_ideas.append(transformed)
    return transformed_ideas


def _read_views() -> int:
    if not os.path.exists(VIEWS_FILE):
        return 0
    try:
        with open(VIEWS_FILE, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return int(payload.get("views", 0))
    except Exception:
        return 0


def _write_views(views: int) -> None:
    os.makedirs(os.path.dirname(VIEWS_FILE), exist_ok=True)
    with open(VIEWS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"views": views}, fh)


@app.route("/healthz", methods=["GET"])
@app.route("/api/health", methods=["GET", "POST"])
def healthz():
    if request.method == "POST":
        return jsonify({"error": "Method not allowed"}), 405
    return jsonify(
        {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "live-idea-bench-backend",
        }
    )


@app.route("/metrics", methods=["GET"])
def metrics():
    report = run_service.build_global_report()
    summary = report.get("summary", {})
    return jsonify(
        {
            "runs_total": summary.get("total_runs", 0),
            "runs_running": summary.get("running_runs", 0),
            "runs_successful": summary.get("successful_runs", 0),
            "runs_failed": summary.get("failed_runs", 0),
            "success_rate": summary.get("success_rate", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.route("/api/generate-ideas", methods=["GET", "POST"])
def api_generate_ideas():
    if request.method == "GET":
        return jsonify(_load_generated_ideas())

    payload = request.get_json(silent=True) or {}
    keywords = payload.get("keywords") if request.method == "POST" else config.KEYWORDS
    n = payload.get("n", config.NUM_PAPERS_TO_FETCH)

    if not isinstance(keywords, list) or not keywords:
        raise APIError("keywords must be a non-empty array", status_code=400)
    if not isinstance(n, int) or n <= 0:
        raise APIError("n must be a positive integer", status_code=400)

    from backend.idea_generator import generate_ideas

    ideas = generate_ideas(keywords, n)
    _save_generated_ideas(ideas)
    return jsonify(ideas)


@app.route("/api/research-ideas", methods=["GET"])
def api_research_ideas():
    ideas = _load_generated_ideas()
    return jsonify(_transform_ideas_for_dashboard(ideas))


@app.route("/api/views", methods=["GET", "POST"])
def api_views():
    views = _read_views()
    if request.method == "POST":
        views += 1
        _write_views(views)
    return jsonify({"views": views})


@app.route("/api/strategies", methods=["GET", "POST"])
def api_strategies():
    if request.method == "GET":
        return jsonify(list_strategies())

    payload = request.get_json(silent=True) or {}
    strategy = create_strategy(payload)
    return jsonify(strategy), 201


@app.route("/api/strategies/<strategy_id>", methods=["GET", "DELETE"])
def api_strategy(strategy_id: str):
    if request.method == "GET":
        strategy = get_strategy(strategy_id)
        if strategy is None:
            raise APIError("strategy not found", status_code=404, code="not_found")
        return jsonify(strategy)

    deleted = delete_strategy(strategy_id)
    if not deleted:
        raise APIError("strategy not found", status_code=404, code="not_found")
    return "", 204


@app.route("/api/strategies/<strategy_id>/status", methods=["GET"])
def api_strategy_status(strategy_id: str):
    strategy = get_strategy(strategy_id)
    if strategy is None:
        raise APIError("strategy not found", status_code=404, code="not_found")

    return jsonify(
        {
            "id": strategy["id"],
            "backtest_status": strategy.get("backtest_status", "pending"),
            "generation_status": strategy.get("generation_status", "pending"),
        }
    )


@app.route("/api/strategies/<strategy_id>/backtest", methods=["POST"])
def api_strategy_backtest(strategy_id: str):
    strategy = get_strategy(strategy_id)
    if strategy is None:
        raise APIError("strategy not found", status_code=404, code="not_found")

    # Persist running status synchronously before spawning thread (prevents status race)
    update_strategy(strategy_id, {"backtest_status": "running"})

    def _worker() -> None:
        run_backtest_sync(strategy_id)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return jsonify({"id": strategy_id, "backtest_status": "running"}), 202


@app.route("/api/strategies/<strategy_id>/generate", methods=["POST"])
def api_strategy_generate(strategy_id: str):
    strategy = get_strategy(strategy_id)
    if strategy is None:
        raise APIError("strategy not found", status_code=404, code="not_found")

    payload = request.get_json(silent=True) or {}
    cutoff_date_raw = payload.get("cutoff_date")
    if not isinstance(cutoff_date_raw, str) or not cutoff_date_raw.strip():
        raise APIError("cutoff_date is required (YYYY-MM-DD)", status_code=400)
    from src.backtest.data import normalize_date

    try:
        cutoff_date = normalize_date(cutoff_date_raw.strip())
    except ValueError as exc:
        raise APIError(str(exc), status_code=400) from exc

    # Persist running status synchronously before spawning thread (prevents status race)
    update_strategy(strategy_id, {"generation_status": "running"})

    def _worker() -> None:
        run_generation_sync(strategy_id, cutoff_date=cutoff_date)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return jsonify({"id": strategy_id, "generation_status": "running"}), 202

@app.route("/api/runs/start", methods=["POST"])
def api_runs_start():
    payload = request.get_json(silent=True) or {}
    keywords = payload.get("keywords", config.KEYWORDS)
    n = payload.get("n", config.NUM_PAPERS_TO_FETCH)

    try:
        run = run_service.start_run(keywords=keywords, n=n)
    except ValueError as exc:
        raise APIError(str(exc), status_code=400) from exc

    return jsonify({"run": run.__dict__}), 202


@app.route("/api/runs", methods=["GET"])
@app.route("/api/runs/list", methods=["GET"])
def api_runs_list():
    limit_raw = request.args.get("limit", "50")
    try:
        limit = int(limit_raw)
    except ValueError as exc:
        raise APIError("limit must be an integer", status_code=400) from exc
    if limit <= 0 or limit > 500:
        raise APIError("limit must be between 1 and 500", status_code=400)
    return jsonify({"runs": run_service.list_runs(limit=limit)})


@app.route("/api/runs/<run_id>", methods=["GET"])
@app.route("/api/runs/detail/<run_id>", methods=["GET"])
def api_runs_detail(run_id: str):
    include_ideas = request.args.get("includeIdeas", "false").lower() == "true"
    run = run_service.get_run(run_id, include_ideas=include_ideas)
    if run is None:
        raise APIError("run not found", status_code=404, code="not_found")
    return jsonify({"run": run})


@app.route("/api/runs/report", methods=["GET"])
def api_runs_report():
    return jsonify(run_service.build_global_report())


if __name__ == "__main__":
    app.run(
        debug=_env_flag("FLASK_DEBUG", False),
        port=_env_int("PORT", 5000),
    )
