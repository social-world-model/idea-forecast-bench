from src.configs import Config, EmbeddingConfig, PromptTemplate
from src.backtest import BacktestConfig, backtest, evaluate, generate
from src.matching import (
    LLMSimilarityEngine,
    ResearchMatcher,
    SentenceTransformerSimilarityEngine,
    SimilarityEngine,
    create_similarity_engine,
)
from src.prompting import extract_abstract, extract_keywords, predict_future_ideas
from src.types import MatchResult, SimilarityPrompt, clean_paper_content
from src.strategy import IdeaStrategy, KeywordTrendStrategy, create_strategy
from src.utils import (
    filter_by_arxiv_date,
    find_markdown_files,
    group_by_keywords,
    load_json,
    read_file_content,
    read_text,
    save_json,
    truncate,
)

__all__ = [
    "Config",
    "EmbeddingConfig",
    "PromptTemplate",
    "BacktestConfig",
    "generate",
    "evaluate",
    "backtest",
    "IdeaStrategy",
    "KeywordTrendStrategy",
    "create_strategy",
    "SimilarityEngine",
    "LLMSimilarityEngine",
    "SentenceTransformerSimilarityEngine",
    "ResearchMatcher",
    "create_similarity_engine",
    "SimilarityPrompt",
    "MatchResult",
    "clean_paper_content",
    "extract_abstract",
    "extract_keywords",
    "predict_future_ideas",
    "find_markdown_files",
    "read_text",
    "read_file_content",
    "truncate",
    "load_json",
    "save_json",
    "filter_by_arxiv_date",
    "group_by_keywords",
]
