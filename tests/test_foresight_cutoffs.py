"""Tests for the train/test window leakage invariants."""

from __future__ import annotations

import pytest

from forecaster.foresight.cutoffs import (
    assert_no_test_window_leakage,
    assert_train_test_disjoint,
)


def test_disjoint_windows_pass():
    assert_train_test_disjoint(
        train_cutoffs=["2023-04-30", "2024-06-30"],
        test_cutoffs=["2024-10-01", "2024-12-31"],
    )


def test_train_after_hard_limit_fails():
    with pytest.raises(AssertionError):
        assert_train_test_disjoint(
            train_cutoffs=["2024-10-31"],
            test_cutoffs=["2024-12-01"],
        )


def test_test_before_hard_limit_fails():
    with pytest.raises(AssertionError):
        assert_train_test_disjoint(
            train_cutoffs=["2024-06-30"],
            test_cutoffs=["2024-09-30"],
        )


def test_overlap_fails():
    with pytest.raises(AssertionError):
        assert_train_test_disjoint(
            train_cutoffs=["2024-10-15"],
            test_cutoffs=["2024-10-01"],
        )


def test_empty_window_fails():
    with pytest.raises(AssertionError):
        assert_train_test_disjoint(train_cutoffs=[], test_cutoffs=["2024-10-01"])
    with pytest.raises(AssertionError):
        assert_train_test_disjoint(train_cutoffs=["2024-01-01"], test_cutoffs=[])


def test_yyyy_mm_parses_as_first_of_month():
    # 2024-09 must compare as 2024-09-01, which is < FUTURE_WINDOW_HARD_LIMIT.
    assert_train_test_disjoint(
        train_cutoffs=["2024-06"],
        test_cutoffs=["2024-10"],
    )


def test_no_test_window_leakage_pass():
    # All dates strictly before the hard limit.
    assert_no_test_window_leakage(
        [
            "2023-04-15",
            "2024-09-30",
            "2024-08-01",
        ]
    )


def test_no_test_window_leakage_catches_oct_first():
    with pytest.raises(AssertionError):
        assert_no_test_window_leakage(
            ["2024-09-30", "2024-10-01", "2024-11-05"],
            context="future_index@2024-07-01",
        )
