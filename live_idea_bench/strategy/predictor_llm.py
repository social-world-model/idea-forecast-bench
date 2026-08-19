from __future__ import annotations

from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.predictor import generate_predictions
from live_idea_bench.strategy.base import IdeaStrategy


class PredictorLLMStrategy(IdeaStrategy):
    name = "predictor_llm"

    def __init__(
        self,
        model_name: str | None = None,
        predictor_config: str = "predictor.yaml",
        similarity_config: str = "similarity.yaml",
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.predictor_config = predictor_config
        self.similarity_config = similarity_config
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def generate(
        self,
        train_papers: list[PaperRecord],
        cutoff_month: str,
        top_k: int,
    ) -> list[IdeaPrediction]:
        # fail-loud: on LLM failure return an empty prediction set rather than
        # fabricating lexical-template ideas (consistent with the no-fallback
        # embedding policy). The benchmark records the empty/partial set.
        return generate_predictions(
            train_papers=train_papers,
            cutoff_month=cutoff_month,
            top_k=top_k,
            model_name=self.model_name,
            predictor_config_path=self.predictor_config,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            fallback_to_heuristic=False,
        )
