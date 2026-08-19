"""Tests for the evaluation-validity analysis scripts (A5-A8 fixes).

The scripts live under examples/benchmark/ (not an importable package), so
they are loaded by path. These tests cover the schema guard, the leakage
bucketing fix (by per-match lead_time, not the constant horizon), and the
co-author self-author exclusion — without hitting the Semantic Scholar API.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "examples" / "benchmark"


def _load(name: str):
    path = _ANALYSIS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _canonical_window(matches):
    return {"topic_results": {"t1": {"backtest": {"windows": [{"matches": matches}]}}}}


def test_leakage_buckets_by_lead_time_not_horizon() -> None:
    leakage = _load("analysis_leakage")
    # Early matches (small lead_time) all hit; late matches (large lead_time) miss.
    matches = [{"is_match": True, "lead_time": 0.1} for _ in range(5)] + [
        {"is_match": False, "lead_time": 0.9} for _ in range(5)
    ]
    r = leakage._analyze(_canonical_window(matches), "demo")
    assert r["buckets"]["early"]["match_rate"] == 1.0
    assert r["buckets"]["late"]["match_rate"] == 0.0
    # With two populated buckets the test actually fires (not the old no-op).
    assert r["leakage_test"]["test"] != "skipped"


def test_leakage_single_bucket_is_flagged_not_silent() -> None:
    leakage = _load("analysis_leakage")
    # All matches land in one lead_time bucket -> comparison impossible.
    matches = [{"is_match": True, "lead_time": 0.1} for _ in range(6)]
    r = leakage._analyze(_canonical_window(matches), "demo")
    assert r["leakage_test"]["test"] == "skipped"
    assert "bucket" in r["leakage_test"]["reason"]


def test_leakage_rejects_non_canonical_schema() -> None:
    leakage = _load("analysis_leakage")
    # llm_judge schema (per_prediction, no 'matches') -> loud failure, not empty.
    bad = {"topic_results": {"t1": {"backtest": {"windows": [{"per_prediction": []}]}}}}
    with pytest.raises(SystemExit):
        leakage._analyze(bad, "demo")


def test_citation_schema_guard_rejects_canonical_and_missing_train_ids() -> None:
    citation = _load("analysis_citation")
    # canonical output (has 'matches', no 'per_prediction') -> reject
    canonical = {"topic_results": {"t1": {"backtest": {"windows": [{"matches": []}]}}}}
    with pytest.raises(SystemExit, match="per_prediction"):
        citation._require_llmjudge_schema(canonical, "x.json")
    # llm_judge but pre-train_paper_ids -> reject
    old = {"topic_results": {"t1": {"backtest": {"windows": [{"per_prediction": []}]}}}}
    with pytest.raises(SystemExit, match="train_paper_ids"):
        citation._require_llmjudge_schema(old, "x.json")
    # valid llm_judge schema -> passes
    good = {
        "topic_results": {
            "t1": {
                "backtest": {
                    "windows": [
                        {"per_prediction": [], "train_paper_ids": ["2401.0001"]}
                    ]
                }
            }
        }
    }
    citation._require_llmjudge_schema(good, "x.json")  # no raise


def test_coauthor_overlap_excludes_self_authors(monkeypatch) -> None:
    coauthor = _load("analysis_coauthor")
    # Build community from a train paper whose authors are {A, B}; the hit paper
    # is authored by {A, C}. Overlap must exclude the paper's own authors, so the
    # community pool seen by the hit paper is {B} (A removed) -> overlap 0/2 = 0.
    authors = {
        "train-1": {"A", "B"},
        "hit-1": {"A", "C"},
    }
    monkeypatch.setattr(
        coauthor, "_get_authors", lambda pid, *_a, **_k: authors.get(pid, set())
    )

    data = {
        "topic_results": {
            "t1": {
                "backtest": {
                    "windows": [
                        {
                            "train_paper_ids": ["train-1"],
                            "per_prediction": [
                                {
                                    "is_match": True,
                                    "matched_paper_id": "hit-1",
                                    "top_candidates": [],
                                },
                            ],
                        }
                    ]
                }
            }
        }
    }
    r = coauthor._analyze(data, api_key=None, delay=0.0)
    # hit paper's own author A is excluded from the pool -> no trivial overlap.
    assert r["hit_author_overlap_mean"] == 0.0
    assert r["community_pool"] == "train_window"
