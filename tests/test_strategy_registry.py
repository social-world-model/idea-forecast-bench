from __future__ import annotations

import pytest

from src.strategy.keyword_trend import KeywordTrendStrategy
from src.strategy.prompt_llm import PromptLLMStrategy
from src.strategy.registry import create_strategy


def test_create_strategy_prompt_llm() -> None:
    strategy = create_strategy(
        "prompt_llm",
        model_id="gpt-4o-mini",
        prompt_id="llm_baseline",
        prompt_version="v1",
    )
    assert isinstance(strategy, PromptLLMStrategy)
    assert strategy.name == "prompt_llm"
    assert strategy.model_id == "gpt-4o-mini"
    assert strategy.prompt_id == "llm_baseline"
    assert strategy.prompt_version == "v1"


def test_create_strategy_keyword_trend_unchanged() -> None:
    strategy = create_strategy(
        "keyword_trend",
        recent_months=5,
        min_keyword_freq=4,
    )
    assert isinstance(strategy, KeywordTrendStrategy)
    assert strategy.recent_months == 5
    assert strategy.min_keyword_freq == 4


def test_create_strategy_unsupported_raises() -> None:
    with pytest.raises(ValueError) as exc_info:
        create_strategy("not-a-real-strategy")
    assert "Unsupported strategy" in str(exc_info.value)
