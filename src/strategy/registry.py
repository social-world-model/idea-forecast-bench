from src.strategy.base import IdeaStrategy
from src.strategy.keyword_trend import KeywordTrendStrategy
from src.strategy.prompt_llm import PromptLLMStrategy


def create_strategy(
    strategy_name: str,
    recent_months: int = 3,
    min_keyword_freq: int = 2,
    model_id: str = "gpt-4o-mini",
    prompt_id: str = "llm_baseline",
    prompt_version: str = "v1",
    temperature: float | None = None,
) -> IdeaStrategy:
    normalized = strategy_name.strip().lower()
    if normalized == KeywordTrendStrategy.name:
        return KeywordTrendStrategy(
            recent_months=recent_months,
            min_keyword_freq=min_keyword_freq,
        )
    if normalized == PromptLLMStrategy.name:
        return PromptLLMStrategy(
            model_id=model_id,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            temperature=temperature,
        )
    raise ValueError("Unsupported strategy: {}".format(strategy_name))
