"""LiveIdeaBench: temporal backtesting of scientific idea generation.

The names re-exported here are the package's intended public surface: config,
data models, strategies, and the entry points for running and scoring a
backtest. Internal helpers -- date arithmetic, file IO, markdown parsing,
prompt lookup -- stay importable from their own modules; they used to sit in
``__all__`` only because it had accumulated every symbol in the package, which
turned each of them into an implicit compatibility promise.
"""

from live_idea_bench.backtest import (
    BacktestConfig,
    BacktestRunner,
    TimeWindow,
    backtest,
    evaluate,
    generate,
    generate_windows,
    load_papers_from_markdown,
    run_backtest,
)
from live_idea_bench.config import (
    Config,
    EmbeddingConfig,
    PredictorConfig,
    SimilarityConfig,
    TopicDefinition,
    load_predictor_config,
    load_runtime_config,
    load_similarity_config,
    load_topics,
)
from live_idea_bench.daily import (
    coerce_prediction,
    compute_leaderboard_score,
    daily_cutoff_date,
    evaluate_previous_generation,
)
from live_idea_bench.ingest import ingest_latest_arxiv_papers
from live_idea_bench.models import (
    BacktestWindowResult,
    EvaluationResult,
    IdeaPrediction,
    MatchResult,
    PaperRecord,
    PredictionMatchDetail,
    ScoredPredictionList,
    SimilarityPrompt,
)
from live_idea_bench.predictor import generate_predictions
from live_idea_bench.similarity import evaluate_predictions, score_prediction_list
from live_idea_bench.strategy import (
    IdeaStrategy,
    PolicyRLStrategy,
    PredictorLLMStrategy,
    create_strategy,
)
from live_idea_bench.strategy.execution import (
    build_strategy,
    run_strategy_backtest,
    run_strategy_generation,
)
from live_idea_bench.topics import classify_paper_topics, classify_papers_by_topic

__all__ = [
    # configuration
    "Config",
    "EmbeddingConfig",
    "PredictorConfig",
    "SimilarityConfig",
    "TopicDefinition",
    "load_predictor_config",
    "load_runtime_config",
    "load_similarity_config",
    "load_topics",
    # data models
    "BacktestWindowResult",
    "EvaluationResult",
    "IdeaPrediction",
    "MatchResult",
    "PaperRecord",
    "PredictionMatchDetail",
    "ScoredPredictionList",
    "SimilarityPrompt",
    "TimeWindow",
    # strategies
    "IdeaStrategy",
    "PolicyRLStrategy",
    "PredictorLLMStrategy",
    "build_strategy",
    "create_strategy",
    # running a backtest
    "BacktestConfig",
    "BacktestRunner",
    "backtest",
    "evaluate",
    "generate",
    "generate_predictions",
    "generate_windows",
    "load_papers_from_markdown",
    "run_backtest",
    "run_strategy_backtest",
    "run_strategy_generation",
    # scoring and analysis
    "classify_paper_topics",
    "classify_papers_by_topic",
    "coerce_prediction",
    "compute_leaderboard_score",
    "daily_cutoff_date",
    "evaluate_predictions",
    "evaluate_previous_generation",
    "ingest_latest_arxiv_papers",
    "score_prediction_list",
]
