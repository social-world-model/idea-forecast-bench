from __future__ import annotations

import json
import threading
import os
import sys
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
CORS(app)

GENERATED_IDEAS_FILE = os.path.join(project_root, "backend", "generated_ideas.json")
VIEWS_FILE = os.path.join(project_root, "data", "views.json")

run_service = RunService(project_root=project_root)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return value not in {"0", "false", "no", "off"}


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


@app.errorhandler(APIError)
def handle_api_error(error: APIError):
    payload = {
        "error": {
            "code": error.code,
            "message": error.message,
            "details": error.details,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(payload), error.status_code


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    payload = {
        "error": {
            "code": "internal_error",
            "message": "Unexpected server error",
            "details": str(error),
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
    cutoff_month: str | None = payload.get("cutoff_month") or None

    # Persist running status synchronously before spawning thread (prevents status race)
    update_strategy(strategy_id, {"generation_status": "running"})

    def _worker() -> None:
        run_generation_sync(strategy_id, cutoff_month=cutoff_month)

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
    app.run(debug=True, port=5000)
