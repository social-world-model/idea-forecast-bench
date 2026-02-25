"""
Live Idea Bench — Flask backend.
All backtest/generation logic delegates to the src.* engine.
"""
import sys
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

# Put project root on sys.path so that `import src.*` works.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.strategy_store import (
    DEFAULT_DATA_DIR,
    _load_papers,
    _make_strategy_obj,
    create_strategy,
    delete_strategy,
    get_strategy,
    list_strategies,
    run_backtest_sync,
    seed_demo_strategies,
    update_strategy,
)

app = Flask(__name__)
CORS(app)


# ── Strategy CRUD ──────────────────────────────────────────────────────────────

@app.route("/api/strategies", methods=["GET"])
def api_list_strategies():
    """Return all strategies, sorted by avg_hit_at_k desc."""
    return jsonify(list_strategies())


@app.route("/api/strategies/<strategy_id>", methods=["GET"])
def api_get_strategy(strategy_id):
    s = get_strategy(strategy_id)
    if s is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(s)


@app.route("/api/strategies", methods=["POST"])
def api_create_strategy():
    data = request.get_json(force=True) or {}
    s = create_strategy(data)
    return jsonify(s), 201


@app.route("/api/strategies/<strategy_id>", methods=["DELETE"])
def api_delete_strategy(strategy_id):
    if delete_strategy(strategy_id):
        return jsonify({"ok": True})
    return jsonify({"error": "Not found"}), 404


# ── Status polling ─────────────────────────────────────────────────────────────

@app.route("/api/strategies/<strategy_id>/status", methods=["GET"])
def api_strategy_status(strategy_id):
    s = get_strategy(strategy_id)
    if s is None:
        return jsonify({"error": "Not found"}), 404
    summary = (s.get("backtest_result") or {}).get("summary")
    gen = s.get("generation") or {}
    return jsonify({
        "backtest_status": s.get("backtest_status", "pending"),
        "generation_status": s.get("generation_status", "pending"),
        "backtest_summary": summary,
        "generation_predictions": len(gen.get("predictions", [])),
    })


# ── Backtest ───────────────────────────────────────────────────────────────────

def _backtest_thread(strategy_id: str) -> None:
    """Background thread: runs the src.backtest engine and saves the result."""
    try:
        from src import BacktestConfig, backtest

        s = get_strategy(strategy_id)
        if s is None:
            return

        update_strategy(strategy_id, {"backtest_status": "running"})
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
    except Exception as exc:
        update_strategy(strategy_id, {
            "backtest_status": "failed",
            "backtest_error": str(exc),
        })


@app.route("/api/strategies/<strategy_id>/backtest", methods=["POST"])
def api_run_backtest(strategy_id):
    s = get_strategy(strategy_id)
    if s is None:
        return jsonify({"error": "Not found"}), 404
    if s.get("backtest_status") == "running":
        return jsonify({"message": "Already running"}), 202

    threading.Thread(target=_backtest_thread, args=(strategy_id,), daemon=True).start()
    return jsonify({"message": "Backtest started", "status": "running"}), 202


# ── Generation ─────────────────────────────────────────────────────────────────

def _generation_thread(strategy_id: str, cutoff_month: str) -> None:
    """Background thread: generates ideas at a given cutoff and saves them."""
    try:
        from dataclasses import asdict
        from src import generate

        s = get_strategy(strategy_id)
        if s is None:
            return

        update_strategy(strategy_id, {"generation_status": "running"})
        papers = _load_papers(s)
        strategy_obj = _make_strategy_obj(s)
        predictions = generate(
            papers=papers,
            strategy=strategy_obj,
            cutoff_month=cutoff_month,
            top_k=s["config"]["top_k"],
        )
        update_strategy(strategy_id, {
            "generation_status": "done",
            "generation": {
                "cutoff_month": cutoff_month,
                "predictions": [asdict(p) for p in predictions],
            },
        })
    except Exception as exc:
        update_strategy(strategy_id, {
            "generation_status": "failed",
            "generation_error": str(exc),
        })


@app.route("/api/strategies/<strategy_id>/generate", methods=["POST"])
def api_run_generation(strategy_id):
    s = get_strategy(strategy_id)
    if s is None:
        return jsonify({"error": "Not found"}), 404
    if s.get("generation_status") == "running":
        return jsonify({"message": "Already running"}), 202

    body = request.get_json(force=True) or {}
    # Default cutoff = strategy's end_month
    cutoff = body.get("cutoff_month") or s["config"].get("end_month", "2024-12")

    threading.Thread(
        target=_generation_thread, args=(strategy_id, cutoff), daemon=True
    ).start()
    return jsonify({"message": "Generation started", "cutoff_month": cutoff}), 202


# ── Health ─────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "ok",
        "time": datetime.now().isoformat(),
        "data_dir": DEFAULT_DATA_DIR,
        "strategies": len(list_strategies()),
    })


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 58)
    print("Live Idea Bench Backend")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data dir     : {DEFAULT_DATA_DIR}")
    print("Seeding demo strategies …")
    seed_demo_strategies()
    strats = list_strategies()
    print(f"Strategies   : {len(strats)}")
    for s in strats:
        summary = (s.get("backtest_result") or {}).get("summary") or {}
        print(
            f"  [{s['backtest_status']:7}] {s['name'][:45]:<45}  "
            f"hit@k={summary.get('avg_hit_at_k', 'N/A')}  "
            f"windows={summary.get('windows', 0)}"
        )
    print("Listening on http://localhost:5000")
    print("=" * 58)
    app.run(debug=True, port=5000)
