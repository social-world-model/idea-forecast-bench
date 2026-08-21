from typing import Any

from live_idea_bench.strategy.base import IdeaStrategy
from live_idea_bench.strategy.memory_prompting import MemoryPromptingStrategy
from live_idea_bench.strategy.policy_rl import PolicyRLStrategy
from live_idea_bench.strategy.predictor_llm import PredictorLLMStrategy
from live_idea_bench.strategy.retrieval_prompting import RetrievalPromptingStrategy
from live_idea_bench.strategy.summary_prompting import SummaryPromptingStrategy
from live_idea_bench.strategy.topic_trend import TopicTrendStrategy


def create_strategy(
    strategy_name: str,
    recent_months: int = 3,
    min_keyword_freq: int = 2,
    model_name: str | None = None,
    predictor_config: str = "predictor.yaml",
    selection_config: str = "selection.yaml",
    similarity_config: str = "similarity.yaml",
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    **legacy_params: Any,
) -> IdeaStrategy:
    normalized = strategy_name.strip().lower()
    if normalized in {PredictorLLMStrategy.name, "prompt_llm"}:
        resolved_model = model_name or legacy_params.get("model_id")
        return PredictorLLMStrategy(
            model_name=str(resolved_model) if resolved_model else None,
            predictor_config=str(
                legacy_params.get("predictor_config", predictor_config)
            ),
            similarity_config=str(
                legacy_params.get("similarity_config", similarity_config)
            ),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
    if normalized == PolicyRLStrategy.name:
        resolved_model = model_name or legacy_params.get("model_id")
        return PolicyRLStrategy(
            model_name=str(resolved_model) if resolved_model else None,
            predictor_config=str(
                legacy_params.get("predictor_config", predictor_config)
            ),
            selection_config=str(
                legacy_params.get("selection_config", selection_config)
            ),
            similarity_config=str(
                legacy_params.get("similarity_config", similarity_config)
            ),
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
            memory_path=str(legacy_params["memory_path"])
            if legacy_params.get("memory_path")
            else None,
            prior_checkpoint=str(legacy_params["prior_checkpoint"])
            if legacy_params.get("prior_checkpoint")
            else None,
            realization_checkpoint=str(legacy_params["realization_checkpoint"])
            if legacy_params.get("realization_checkpoint")
            else None,
            inference_config_path=str(
                legacy_params.get("inference_config_path") or "inference.yaml"
            ),
            realization_config_path=str(
                legacy_params.get("realization_config_path") or "realization.yaml"
            ),
        )
    if normalized == TopicTrendStrategy.name:
        resolved_model = model_name or legacy_params.get("model_id")
        return TopicTrendStrategy(
            model_name=str(resolved_model) if resolved_model else None,
            recent_months=recent_months,
            min_keyword_freq=min_keyword_freq,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
    if normalized == MemoryPromptingStrategy.name:
        resolved_model = model_name or legacy_params.get("model_id")
        return MemoryPromptingStrategy(
            model_name=str(resolved_model) if resolved_model else None,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
    if normalized == SummaryPromptingStrategy.name:
        resolved_model = model_name or legacy_params.get("model_id")
        return SummaryPromptingStrategy(
            model_name=str(resolved_model) if resolved_model else None,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
    if normalized == RetrievalPromptingStrategy.name:
        resolved_model = model_name or legacy_params.get("model_id")
        retrieval_top_n = legacy_params.get("retrieval_top_n")
        return RetrievalPromptingStrategy(
            model_name=str(resolved_model) if resolved_model else None,
            retrieval_top_n=int(retrieval_top_n) if retrieval_top_n else 20,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
    raise ValueError(f"Unsupported strategy: {strategy_name}")
