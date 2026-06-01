#!/usr/bin/env python3
"""Co-author analysis: do hit papers share authors with training-window papers?

Fetches author lists for hit papers and a random control set of non-hit future
papers from Semantic Scholar. Compares author overlap with the set of authors
appearing in the candidate pool (proxy for the research community).

Usage:
    python examples/analysis_coauthor.py \\
        --input memory_prompting_llmjudge.json \\
        --output coauthor_report.json \\
        [--s2-key YOUR_SEMANTIC_SCHOLAR_API_KEY]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
DEFAULT_DELAY = 1.1


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
    except Exception as e:
        print(f"  [s2 error] {arxiv_id}: {e}")
        return None


def _get_authors(arxiv_id: str, api_key: str | None, delay: float) -> set[str]:
    time.sleep(delay)
    result = _s2_fetch(arxiv_id, "authors.name", api_key)
    if not result:
        return set()
    return {a.get("name", "") for a in result.get("authors", []) if a.get("name")}


def _analyze(data: dict, api_key: str | None, delay: float) -> dict:
    # Step 1: collect all candidate paper IDs as a proxy for the research community pool
    # These are future papers (after cutoff), so the "community" is defined by
    # authors of all papers in the dataset that appeared as embedding candidates.
    community_paper_ids: set[str] = set()
    hit_ids: set[str] = set()
    control_ids: set[str] = set()

    for tr in data.get("topic_results", {}).values():
        bt = tr.get("backtest")
        if not bt:
            continue
        for w in bt.get("windows", []):
            for pred in w.get("per_prediction", []):
                for cand in pred.get("top_candidates", []):
                    community_paper_ids.add(cand["paper_id"])
                if pred.get("is_match") and pred.get("matched_paper_id"):
                    hit_ids.add(pred["matched_paper_id"])

    # Control: non-hit papers from the candidate pool (sample up to same size as hits)
    all_non_hit = community_paper_ids - hit_ids
    control_ids = set(list(all_non_hit)[: len(hit_ids)])

    print(
        f"Hit papers: {len(hit_ids)} | Control: {len(control_ids)} | "
        f"Community pool: {len(community_paper_ids)}",
        flush=True,
    )

    # Step 2: fetch authors for community papers (sampled subset to keep it tractable)
    # Use up to 200 community papers to build the author set
    community_sample = list(community_paper_ids)[:200]
    print(f"Fetching authors for {len(community_sample)} community papers ...", flush=True)
    community_authors: set[str] = set()
    for pid in community_sample:
        authors = _get_authors(pid, api_key, delay)
        community_authors.update(authors)
    print(f"Community author pool size: {len(community_authors)}", flush=True)

    # Step 3: check hit papers
    author_cache: dict[str, set[str]] = {}

    def _overlap(pid: str) -> float:
        if pid not in author_cache:
            author_cache[pid] = _get_authors(pid, api_key, delay)
        authors = author_cache[pid]
        if not authors or not community_authors:
            return 0.0
        return len(authors & community_authors) / len(authors)

    hit_overlaps = []
    for i, pid in enumerate(hit_ids):
        o = _overlap(pid)
        hit_overlaps.append(o)
        if (i + 1) % 10 == 0:
            print(f"  Hit {i+1}/{len(hit_ids)} done", flush=True)

    ctrl_overlaps = []
    for i, pid in enumerate(control_ids):
        o = _overlap(pid)
        ctrl_overlaps.append(o)
        if (i + 1) % 10 == 0:
            print(f"  Ctrl {i+1}/{len(control_ids)} done", flush=True)

    def _mean(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 4) if vals else None

    result: dict = {
        "hit_papers": len(hit_ids),
        "control_papers": len(control_ids),
        "community_author_pool_size": len(community_authors),
        "hit_author_overlap_mean": _mean(hit_overlaps),
        "control_author_overlap_mean": _mean(ctrl_overlaps),
    }

    if hit_overlaps and ctrl_overlaps:
        try:
            from scipy.stats import mannwhitneyu
            stat, pval = mannwhitneyu(hit_overlaps, ctrl_overlaps, alternative="greater")
            result["overlap_test"] = {
                "test": "Mann-Whitney U (hit > control)",
                "statistic": round(float(stat), 4),
                "p_value": round(float(pval), 4),
                "significant_at_05": float(pval) < 0.05,
            }
        except ImportError:
            result["overlap_test"] = {
                "test": "mean comparison (scipy not available)",
                "delta": round((_mean(hit_overlaps) or 0) - (_mean(ctrl_overlaps) or 0), 4),
            }

    result["interpretation"] = (
        "Higher hit-paper author overlap with community → "
        "predictions capture work from established research groups, validating evaluation"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="llmjudge output JSON")
    parser.add_argument("--output", default="coauthor_report.json")
    parser.add_argument("--s2-key", default=None,
                        help="Semantic Scholar API key (optional; higher rate limit)")
    parser.add_argument("--delay",  type=float, default=DEFAULT_DELAY,
                        help="Seconds between API requests (default 1.1)")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    label = Path(args.input).stem

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
