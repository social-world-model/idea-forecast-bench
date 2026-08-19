#!/usr/bin/env python3
"""Co-author analysis: do hit papers share authors with the training community?

The research "community" is defined by the authors of the TRAINING-window papers
(published <= cutoff). For each hit paper and a non-hit control paper we measure
author overlap with that community, EXCLUDING the paper's own authors so a paper
cannot trivially overlap with itself. Higher hit-paper overlap than control
suggests predictions track established research groups rather than arbitrary work.

The previous version built the community from the hit papers' own candidate pool
(circular: a hit paper's authors are in the pool by construction) and did not
exclude self-authors — both forced a non-trivial overlap regardless of leakage.

Usage:
    python examples/benchmark/analysis_coauthor.py \\
        --input memory_prompting_llmjudge.json \\
        --output coauthor_report.json \\
        [--s2-key YOUR_SEMANTIC_SCHOLAR_API_KEY]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
DEFAULT_DELAY = 1.1


def _require_llmjudge_schema(data: dict, source: str) -> None:
    """A6 schema guard: fail loud unless this is an llm_judge_eval output."""
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


def _s2_fetch(arxiv_id: str, fields: str, api_key: str | None) -> dict | None:
    url = f"{S2_BASE}/arXiv:{arxiv_id}?fields={fields}"
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("x-api-key", api_key)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception as e:  # noqa: BLE001 — network best-effort, surfaced below
        print(f"  [s2 error] {arxiv_id}: {e}")
        return None


def _get_authors(arxiv_id: str, api_key: str | None, delay: float) -> set[str]:
    time.sleep(delay)
    result = _s2_fetch(arxiv_id, "authors.name", api_key)
    if not result:
        return set()
    return {a.get("name", "") for a in result.get("authors", []) if a.get("name")}


def _analyze(data: dict, api_key: str | None, delay: float) -> dict:
    # Step 1: the community is the union of TRAIN-window papers (date <= cutoff),
    # NOT the candidate/future pool. hits/control are drawn from future papers.
    community_paper_ids: set[str] = set()
    hit_ids: set[str] = set()
    future_non_hit: set[str] = set()

    for tr in data.get("topic_results", {}).values():
        bt = tr.get("backtest")
        if not bt:
            continue
        for w in bt.get("windows", []):
            community_paper_ids.update(w.get("train_paper_ids", []))
            for pred in w.get("per_prediction", []):
                if pred.get("is_match") and pred.get("matched_paper_id"):
                    hit_ids.add(pred["matched_paper_id"])
                for cand in pred.get("top_candidates", []):
                    future_non_hit.add(cand["paper_id"])

    future_non_hit -= hit_ids
    control_ids = set(list(future_non_hit)[: len(hit_ids)])

    print(
        f"Hit papers: {len(hit_ids)} | Control: {len(control_ids)} | "
        f"Community (train) pool: {len(community_paper_ids)}",
        flush=True,
    )

    # Step 2: build the train-community author set (sampled to stay tractable).
    community_sample = list(community_paper_ids)[:200]
    print(f"Fetching authors for {len(community_sample)} train papers ...", flush=True)
    community_authors: set[str] = set()
    for pid in community_sample:
        community_authors.update(_get_authors(pid, api_key, delay))
    print(f"Community author pool size: {len(community_authors)}", flush=True)

    # Step 3: overlap, EXCLUDING each paper's own authors (anti self-confirmation).
    author_cache: dict[str, set[str]] = {}

    def _authors(pid: str) -> set[str]:
        if pid not in author_cache:
            author_cache[pid] = _get_authors(pid, api_key, delay)
        return author_cache[pid]

    def _overlap(pid: str) -> float:
        own = _authors(pid)
        if not own:
            return 0.0
        pool = community_authors - own  # exclude self so overlap isn't trivial
        if not pool:
            return 0.0
        return len(own & pool) / len(own)

    hit_overlaps = [_overlap(pid) for pid in hit_ids]
    ctrl_overlaps = [_overlap(pid) for pid in control_ids]

    def _mean(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 4) if vals else None

    result: dict = {
        "hit_papers": len(hit_ids),
        "control_papers": len(control_ids),
        "community_pool": "train_window",
        "community_author_pool_size": len(community_authors),
        "hit_author_overlap_mean": _mean(hit_overlaps),
        "control_author_overlap_mean": _mean(ctrl_overlaps),
    }

    if hit_overlaps and ctrl_overlaps:
        try:
            from scipy.stats import mannwhitneyu

            stat, pval = mannwhitneyu(
                hit_overlaps, ctrl_overlaps, alternative="greater"
            )
            result["overlap_test"] = {
                "test": "Mann-Whitney U (hit > control)",
                "statistic": round(float(stat), 4),
                "p_value": round(float(pval), 4),
                "significant_at_05": float(pval) < 0.05,
            }
        except ImportError:
            result["overlap_test"] = {
                "test": "mean comparison (scipy not available)",
                "delta": round(
                    (_mean(hit_overlaps) or 0) - (_mean(ctrl_overlaps) or 0), 4
                ),
            }

    result["interpretation"] = (
        "Higher hit-paper author overlap with the TRAIN community (excluding "
        "self-authors) → predictions track established research groups"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="llmjudge output JSON")
    parser.add_argument("--output", default="coauthor_report.json")
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
    print(f"Co-author analysis for: {label}", flush=True)
    result = _analyze(data, api_key=args.s2_key, delay=args.delay)
    result["source"] = args.input

    print(f"\n=== Co-author Analysis: {label} ===")
    print(f"  Hit  author overlap: {result['hit_author_overlap_mean']}")
    print(f"  Ctrl author overlap: {result['control_author_overlap_mean']}")
    if "overlap_test" in result:
        print(f"  Test: {result['overlap_test']}")

    Path(args.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
