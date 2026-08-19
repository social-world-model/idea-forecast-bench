"""Tests for the GRPO grouping invariant + in-group dedup penalty."""

from __future__ import annotations

import json

import pytest

from forecaster.foresight.grouping import (
    GroupingInvariantError,
    assert_group_invariant,
    compute_dedup_penalties,
    grouping_report,
)


def _extra(cutoff: str, base: str, op: str, gap: str) -> str:
    return json.dumps(
        {
            "cutoff_date": cutoff,
            "innovation": {"base_direction": base, "operator": op, "gap": gap},
        }
    )


# --------------------------------------------------------------------------- invariant


def test_assert_invariant_pass_for_two_groups():
    extras = [_extra("2024-06-30", "rag", "extend", "x")] * 4 + [
        _extra("2024-05-31", "agents", "compose", "y")
    ] * 4
    assert_group_invariant(extras, num_generations=4)


def test_assert_invariant_rejects_mixed_group():
    extras = [
        _extra("2024-06-30", "rag", "extend", "x"),
        _extra("2024-06-30", "rag", "extend", "x"),
        _extra("2024-06-30", "rag", "compose", "x"),  # different operator → drift
        _extra("2024-06-30", "rag", "extend", "x"),
    ]
    with pytest.raises(GroupingInvariantError, match="distinct"):
        assert_group_invariant(extras, num_generations=4)


def test_assert_invariant_rejects_non_divisible_batch():
    extras = [_extra("2024-06-30", "rag", "extend", "x")] * 3
    with pytest.raises(GroupingInvariantError, match="not a multiple"):
        assert_group_invariant(extras, num_generations=4)


def test_assert_invariant_rejects_g_below_two():
    extras = [_extra("2024-06-30", "rag", "extend", "x")]
    with pytest.raises(GroupingInvariantError, match="degenerate"):
        assert_group_invariant(extras, num_generations=1)


def test_assert_invariant_rejects_empty_z_key():
    extras = [json.dumps({"cutoff_date": "2024-06-30", "innovation": {}})] * 4
    with pytest.raises(GroupingInvariantError, match="empty"):
        assert_group_invariant(extras, num_generations=4)


def test_grouping_report_summarizes_topology():
    extras = [_extra("2024-06-30", "rag", "extend", "x")] * 4 + [
        _extra("2024-05-31", "agents", "compose", "y")
    ] * 4
    rep = grouping_report(extras, num_generations=4)
    assert rep["batch_size"] == 8
    assert rep["num_groups"] == 2
    assert rep["violations"] == []
    assert len(rep["key_histogram_sample"]) == 2


def test_grouping_report_flags_violations_without_raising():
    extras = [_extra("2024-06-30", "rag", "extend", "x")] * 3
    rep = grouping_report(extras, num_generations=4)
    assert rep["violations"]


# --------------------------------------------------------------------------- dedup penalty


def test_dedup_penalty_zero_when_unique():
    completions = [
        "alpha beta gamma",
        "delta epsilon zeta",
        "eta theta iota",
        "kappa lambda mu",
    ]
    p = compute_dedup_penalties(completions, num_generations=4, penalty=0.1)
    assert p == [0.0, 0.0, 0.0, 0.0]


def test_dedup_penalty_fires_within_group():
    completions = [
        "alpha beta gamma delta",
        "alpha beta gamma delta",
        "alpha beta gamma delta",
        "totally different content here",
    ]
    p = compute_dedup_penalties(
        completions, num_generations=4, penalty=0.2, threshold=0.5
    )
    # First three are duplicates of each other → each has 2 dupes → penalty=0.4.
    assert p[0] == pytest.approx(0.4)
    assert p[1] == pytest.approx(0.4)
    assert p[2] == pytest.approx(0.4)
    assert p[3] == 0.0


def test_dedup_penalty_respects_group_boundaries():
    completions = [
        "alpha beta gamma",  # group 1
        "alpha beta gamma",
        "alpha beta gamma",  # group 2 (same text but a different group)
        "alpha beta gamma",
    ]
    p = compute_dedup_penalties(
        completions, num_generations=2, penalty=0.5, threshold=0.5
    )
    # Group 1: each has 1 dupe → penalty=0.5.
    # Group 2: each has 1 dupe → penalty=0.5.
    # Dedup is intra-group; the second group's penalty must not look across the boundary.
    assert p == [0.5, 0.5, 0.5, 0.5]


def test_dedup_penalty_disabled_when_zero():
    completions = ["a a a", "a a a", "a a a", "a a a"]
    p = compute_dedup_penalties(completions, num_generations=4, penalty=0.0)
    assert p == [0.0, 0.0, 0.0, 0.0]
