#!/usr/bin/env python3
"""Citation analysis: do hit papers cite papers from their training window?

For each matched (hit) prediction, fetch the hit paper's references from the
Semantic Scholar API and check whether any reference is a paper in the SAME
window's training set (papers published <= cutoff for that topic). A higher
citation rate for hit papers than for a non-hit control set suggests predictions
capture genuine community continuity rather than arbitrary topical overlap.

This targets the per-window ``train_paper_ids`` (topic-scoped, date<=cutoff) that
llm_judge_eval.py now serializes — NOT a global union of future candidates,
which any real arXiv paper would almost certainly cite (no discriminative power).

Usage:
    python examples/benchmark/analysis_citation.py \\
        --input memory_prompting_llmjudge.json \\
        --output citation_report.json \\
        [--s2-key YOUR_SEMANTIC_SCHOLAR_API_KEY]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from idea_forecast_bench.semantic_scholar import fetch_paper

DEFAULT_DELAY = 1.1  # seconds between requests when no API key


def _require_llmjudge_schema(data: dict, source: str) -> None:
    """A6 schema guard: fail loud unless this is an llm_judge_eval output.

    Citation/coauthor analyses only understand the llmjudge schema
    (topic_results[*].backtest.windows[*] with per_prediction + train_paper_ids).
    The canonical backtest.py output stores `matches` instead, so silently
    reading it would yield empty results that look like a successful run.
    """
    for tr in data.get("topic_results", {}).values():
        bt = tr.get("backtest")
        if not bt or not bt.get("windows"):
            continue
        w = bt["windows"][0]
        if "per_prediction" not in w:
            raise SystemExit(
                f"{source}: expected llm_judge_eval output (window has no "
                "'per_prediction'). This analysis does not support canonical "
                "backtest JSON."
            )
        if "train_paper_ids" not in w:
            raise SystemExit(
                f"{source}: llm_judge_eval output predates train_paper_ids. "
                "Re-run llm_judge_eval.py to populate per-window train ids."
            )
        return


def _extract_arxiv_id(ext_ids: dict | None) -> str | None:
    if not ext_ids:
        return None
    return ext_ids.get("ArXiv") or ext_ids.get("arxiv")


def _analyze(data: dict, api_key: str | None, delay: float) -> dict:
    # Per window: hit papers, a non-hit control set, and the window's TRAIN ids.
    hit_samples: list[dict] = []
    for topic_id, tr in data.get("topic_results", {}).items():
        bt = tr.get("backtest")
        if not bt:
            continue
        for w in bt.get("windows", []):
            cutoff = w.get("cutoff_month", "")
            train_ids = set(w.get("train_paper_ids", []))
            hit_ids: set[str] = set()
            future_ids: set[str] = set()
            for pred in w.get("per_prediction", []):
                if pred.get("is_match") and pred.get("matched_paper_id"):
                    hit_ids.add(pred["matched_paper_id"])
                for cand in pred.get("top_candidates", []):
                    future_ids.add(cand["paper_id"])
            if not hit_ids or not train_ids:
                continue
            hit_samples.append(
                {
                    "topic_id": topic_id,
                    "cutoff": cutoff,
                    "train_ids": train_ids,
                    "hit_ids": sorted(hit_ids),
                    "future_ids": sorted(future_ids - hit_ids),  # non-hit control
                }
            )

    total_hit = total_hit_with_cite = 0
    total_ctrl = total_ctrl_with_cite = 0
    checked: dict[tuple[str, frozenset], bool] = {}

    def _cites_train(arxiv_id: str, train_ids: set[str]) -> bool:
        result = fetch_paper(arxiv_id, "references.externalIds", api_key)
        if not result:
            return False
        for ref in result.get("references", []):
            ref_arxiv = _extract_arxiv_id(ref.get("externalIds"))
            if ref_arxiv and ref_arxiv in train_ids:
                return True
        return False

    def _get_or_check(pid: str, train_ids: set[str]) -> bool:
        key = (pid, frozenset(train_ids))
        if key in checked:
            return checked[key]
        time.sleep(delay)
        out = _cites_train(pid, train_ids)
        checked[key] = out
        return out

    for sample in hit_samples:
        train_ids = sample["train_ids"]
        print(
            f"  Checking {len(sample['hit_ids'])} hit + "
            f"{min(len(sample['future_ids']), 3)} ctrl papers vs "
            f"{len(train_ids)} train ids "
            f"[{sample['topic_id']} cutoff={sample['cutoff']}]",
            flush=True,
        )
        for pid in sample["hit_ids"]:
            total_hit += 1
            if _get_or_check(pid, train_ids):
                total_hit_with_cite += 1
        for pid in sample["future_ids"][:3]:
            total_ctrl += 1
            if _get_or_check(pid, train_ids):
                total_ctrl_with_cite += 1

    return {
        "target": "train_window",
        "hit_papers_checked": total_hit,
        "hit_papers_with_citation": total_hit_with_cite,
        "hit_citation_rate": round(total_hit_with_cite / total_hit, 4)
        if total_hit
        else None,
        "control_papers_checked": total_ctrl,
        "control_papers_with_citation": total_ctrl_with_cite,
        "control_citation_rate": round(total_ctrl_with_cite / total_ctrl, 4)
        if total_ctrl
        else None,
        "interpretation": (
            "Hit papers cite their training-window papers more than control "
            "→ predictions capture genuine research continuity"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="llmjudge output JSON")
    parser.add_argument("--output", default="citation_report.json")
    parser.add_argument(
        "--s2-key",
        default=None,
        help="Semantic Scholar API key (optional; higher rate limit)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Seconds between API requests (default 1.1)",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    label = Path(args.input).stem

    _require_llmjudge_schema(data, args.input)
    print(f"Analyzing citations for: {label}", flush=True)
    result = _analyze(data, api_key=args.s2_key, delay=args.delay)
    result["source"] = args.input

    print(f"\n=== Citation Analysis: {label} ===")
    print(
        f"  Hit papers:     {result['hit_papers_checked']} checked, "
        f"{result['hit_papers_with_citation']} cite train "
        f"(rate={result['hit_citation_rate']})"
    )
    print(
        f"  Control papers: {result['control_papers_checked']} checked, "
        f"{result['control_papers_with_citation']} cite train "
        f"(rate={result['control_citation_rate']})"
    )

    Path(args.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
