"""
Backward-compatible re-export.
Prefer importing strategy APIs from src.strategy.
"""

from src.strategy import IdeaStrategy, KeywordTrendStrategy, create_strategy

__all__ = [
    "IdeaStrategy",
    "KeywordTrendStrategy",
    "create_strategy",
]

