#!/usr/bin/env python3
"""Outcome-blind specificity / breadth rating of saved predictions.

Reads backtest or judged artifacts, rates every prediction from its text
alone (no papers, no dates), caches by prediction hash, and prints one row
per (strategy, model)."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from idea_forecast_bench.atomic import atomic_write_text
from idea_forecast_bench.combinatorial.config import (
    load_combinatorial_config,
    load_prompt_pair,
)
from idea_forecast_bench.combinatorial.llm_caller import (
    caller_for_model,
    callers_for_base_urls,
)
from idea_forecast_bench.combinatorial.specificity import rate_specificity
from idea_forecast_bench.judge.identity import pred_hash, pred_text
from idea_forecast_bench.judge.windows import load_predictions
from idea_forecast_bench.models import IdeaPrediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json", action="append", required=True, help="Path or glob; repeatable"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="gpt-4o-qwen35")
    parser.add_argument("--base-urls", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Max predictions")
    return parser.parse_args()


def _iter_predictions(
    artifact: dict[str, Any],
) -> list[tuple[str, str, IdeaPrediction]]:
    strategy = str(artifact.get("strategy") or "?")
    model = str(artifact.get("model_name") or "?")
    out: list[tuple[str, str, IdeaPrediction]] = []
    for topic in artifact.get("topic_results", {}).values():
        backtest = topic.get("backtest") or {}
        for window in backtest.get("windows", []):
            for pred in load_predictions(window.get("predictions", [])):
                out.append((strategy, model, pred))
    return out


def _mean_se(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return mean, se


def main() -> int:
    args = parse_args()
    paths: list[str] = []
    for pattern in args.input_json:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        print("no input artifacts matched", file=sys.stderr)
        return 1

    cfg = load_combinatorial_config(args.config)
    prompt = load_prompt_pair(cfg.specificity.prompt)
    caller = (
        callers_for_base_urls(args.model_name, args.base_urls.split(","))
        if args.base_urls
        else caller_for_model(args.model_name)
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = out_path.with_suffix(".state.json")
    ratings: dict[str, dict[str, Any]] = {}
    if state_path.exists():
        ratings = json.loads(state_path.read_text(encoding="utf-8"))
    lock = threading.Lock()

    items: list[tuple[str, str, IdeaPrediction]] = []
    for path in paths:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        items.extend(_iter_predictions(artifact))
    if args.limit is not None:
        items = items[: args.limit]
    todo = {pred_hash(pred_text(p)): p for _s, _m, p in items}
    todo = {h: p for h, p in todo.items() if h not in ratings}
    print(f"{len(items)} predictions, {len(todo)} to rate", flush=True)

    def _rate(item: tuple[str, IdeaPrediction]) -> None:
        h, pred = item
        result = rate_specificity(pred, caller, prompt, cfg.specificity.temperature)
        with lock:
            ratings[h] = result
            if len(ratings) % 50 == 0:
                atomic_write_text(state_path, json.dumps(ratings))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_rate, item) for item in todo.items()]
        for future in tqdm(as_completed(futures), total=len(futures), unit="pred"):
            future.result()
    atomic_write_text(state_path, json.dumps(ratings))

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for strategy, model, pred in items:
        rating = ratings.get(pred_hash(pred_text(pred)))
        if rating is not None:
            groups.setdefault((strategy, model), []).append(rating)

    rows: list[dict[str, Any]] = []
    print(
        f"\n{'strategy':<27} {'model':<18} {'n':>5} {'spec':>11} {'breadth':>11} {'testable':>9} {'parse_fail':>10}"
    )
    for (strategy, model), rs in sorted(groups.items()):
        spec = [float(r["specificity"]) for r in rs if r.get("specificity") is not None]
        breadth = [float(r["breadth"]) for r in rs if r.get("breadth") is not None]
        testable = [1.0 for r in rs if r.get("testable") is True]
        parse_fail = sum(1 for r in rs if r.get("parse_failed"))
        s_mean, s_se = _mean_se(spec)
        b_mean, b_se = _mean_se(breadth)
        row = {
            "strategy": strategy,
            "model": model,
            "n": len(rs),
            "specificity_mean": s_mean,
            "specificity_se": s_se,
            "breadth_mean": b_mean,
            "breadth_se": b_se,
            "testable_rate": len(testable) / len(rs) if rs else float("nan"),
            "parse_failures": parse_fail,
        }
        rows.append(row)
        print(
            f"{strategy:<27} {model:<18} {len(rs):>5} "
            f"{s_mean:>6.2f}±{s_se:<4.2f} {b_mean:>6.2f}±{b_se:<4.2f} "
            f"{row['testable_rate']:>9.2f} {parse_fail:>10}"
        )
    atomic_write_text(
        out_path,
        json.dumps(
            {"rater": args.model_name, "rows": rows, "sources": paths}, indent=2
        ),
    )
    print(f"\nSaved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
