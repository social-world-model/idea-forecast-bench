from __future__ import annotations

import pytest

from live_idea_bench.strategy.keyword_trend import KeywordTrendStrategy
from live_idea_bench.strategy.predictor_llm import PredictorLLMStrategy
from live_idea_bench.strategy.registry import create_strategy


def test_create_strategy_predictor_llm() -> None:
    strategy = create_strategy(
        "predictor_llm",
        model_name="gpt-4o-mini",
        predictor_config="predictor.yaml",
        similarity_config="similarity.yaml",
    )
    assert isinstance(strategy, PredictorLLMStrategy)
    assert strategy.name == "predictor_llm"
    assert strategy.model_name == "gpt-4o-mini"
    assert strategy.predictor_config == "predictor.yaml"
    assert strategy.similarity_config == "similarity.yaml"


def test_create_strategy_prompt_llm_alias_migrates_to_predictor_llm() -> None:
    strategy = create_strategy("prompt_llm", model_id="gpt-4o-mini")
    assert isinstance(strategy, PredictorLLMStrategy)
    assert strategy.name == "predictor_llm"
    assert strategy.model_name == "gpt-4o-mini"


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
