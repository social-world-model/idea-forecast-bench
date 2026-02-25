from abc import ABC, abstractmethod
from typing import List

from src.backtest.models import IdeaPrediction, PaperRecord


class IdeaStrategy(ABC):
    name = "base"

    @abstractmethod
    def generate(
        self,
        train_papers: List[PaperRecord],
        cutoff_month: str,
        top_k: int,
    ) -> List[IdeaPrediction]:
        raise NotImplementedError

