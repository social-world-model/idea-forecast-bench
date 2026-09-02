from abc import ABC, abstractmethod

from idea_forecast_bench.models import IdeaPrediction, PaperRecord


class IdeaStrategy(ABC):
    name = "base"

    @abstractmethod
    def generate(
        self,
        train_papers: list[PaperRecord],
        cutoff_month: str,
        top_k: int,
    ) -> list[IdeaPrediction]:
        raise NotImplementedError
