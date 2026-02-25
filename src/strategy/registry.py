from src.strategy.base import IdeaStrategy
from src.strategy.keyword_trend import KeywordTrendStrategy


def create_strategy(
    strategy_name: str,
    recent_months: int = 3,
    min_keyword_freq: int = 2,
) -> IdeaStrategy:
    normalized = strategy_name.strip().lower()
    if normalized == KeywordTrendStrategy.name:
        return KeywordTrendStrategy(
            recent_months=recent_months,
            min_keyword_freq=min_keyword_freq,
        )
    raise ValueError("Unsupported strategy: {}".format(strategy_name))

