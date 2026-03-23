from live_idea_bench.strategy.base import IdeaStrategy
from live_idea_bench.strategy.keyword_trend import KeywordTrendStrategy
from live_idea_bench.strategy.policy_rl import PolicyRLStrategy
from live_idea_bench.strategy.predictor_llm import PredictorLLMStrategy


def create_strategy(
    strategy_name: str,
    recent_months: int = 3,
    min_keyword_freq: int = 2,
    model_name: str | None = None,
    predictor_config: str = "predictor.yaml",
    selection_config: str = "selection.yaml",
    similarity_config: str = "similarity.yaml",
    temperature: float | None = None,
    **legacy_params,
) -> IdeaStrategy:
    normalized = strategy_name.strip().lower()
    if normalized == KeywordTrendStrategy.name:
        return KeywordTrendStrategy(
            recent_months=recent_months,
            min_keyword_freq=min_keyword_freq,
        )
    if normalized in {PredictorLLMStrategy.name, "prompt_llm"}:
        resolved_model = model_name or legacy_params.get("model_id")
        return PredictorLLMStrategy(
            model_name=str(resolved_model) if resolved_model else None,
            predictor_config=str(legacy_params.get("predictor_config", predictor_config)),
            similarity_config=str(legacy_params.get("similarity_config", similarity_config)),
            temperature=temperature,
        )
    if normalized == PolicyRLStrategy.name:
        resolved_model = model_name or legacy_params.get("model_id")
        return PolicyRLStrategy(
            model_name=str(resolved_model) if resolved_model else None,
            predictor_config=str(legacy_params.get("predictor_config", predictor_config)),
            selection_config=str(legacy_params.get("selection_config", selection_config)),
            similarity_config=str(legacy_params.get("similarity_config", similarity_config)),
            temperature=temperature,
            policy_manifest_path=(
                str(legacy_params.get("policy_manifest_path"))
                if legacy_params.get("policy_manifest_path") not in {None, ""}
                else None
            ),
        )
    if normalized == "forecaster":
        from live_idea_bench.strategy.forecaster import ForecasterStrategy

        return ForecasterStrategy(
            model_name=model_name,
            memory_path=str(legacy_params.get("memory_path") or ""),
            prior_checkpoint=str(legacy_params.get("prior_checkpoint") or ""),
        )
    raise ValueError(f"Unsupported strategy: {strategy_name}")
