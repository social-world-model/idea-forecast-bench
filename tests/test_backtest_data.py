from __future__ import annotations

import pytest

from src.backtest.data import normalize_month


def test_normalize_month_accepts_yymm() -> None:
    assert normalize_month("2401") == "2024-01"


def test_normalize_month_rejects_invalid_four_digit_value() -> None:
    with pytest.raises(ValueError):
        normalize_month("2024")
