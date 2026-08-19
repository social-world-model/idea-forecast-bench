from collections import Counter

from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.papers import add_months, month_to_index
from live_idea_bench.strategy.base import IdeaStrategy


def _unique_ordered(items: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


class KeywordTrendStrategy(IdeaStrategy):
    name = "keyword_trend"

    def __init__(self, recent_months: int = 3, min_keyword_freq: int = 2):
        self.recent_months = max(1, recent_months)
        self.min_keyword_freq = max(1, min_keyword_freq)
        self.stop_terms = {
            "cs.ml",
            "cs",
            "ml",
            "arxiv",
            "machine learning",
            "deep learning",
        }

    def generate(
        self,
        train_papers: list[PaperRecord],
        cutoff_month: str,
        top_k: int,
    ) -> list[IdeaPrediction]:
        if not train_papers:
            return []

        cutoff_idx = month_to_index(cutoff_month)
        recent_start_idx = month_to_index(
            add_months(cutoff_month, -(self.recent_months - 1))
        )
        overall = Counter()
        recent = Counter()

        for paper in train_papers:
            unique_keys = _unique_ordered(
                [keyword.lower() for keyword in paper.keywords if keyword.strip()]
            )
            for key in unique_keys:
                if key in self.stop_terms:
                    continue
                overall[key] += 1
                month_idx = month_to_index(paper.month)
                if recent_start_idx <= month_idx <= cutoff_idx:
                    recent[key] += 1

        if not overall:
            return []

        scored: list[dict[str, object]] = []
        for keyword, total_count in overall.items():
            if total_count < self.min_keyword_freq:
                continue
            recent_count = recent.get(keyword, 0)
            trend_gain = recent_count - max(1, total_count // self.recent_months)
            score = float(recent_count * 2 + trend_gain)
            scored.append(
                {
                    "keyword": keyword,
                    "score": score,
                    "recent_count": float(recent_count),
                    "total_count": float(total_count),
                }
            )

        if not scored:
            for keyword, total_count in overall.most_common(top_k):
                scored.append(
                    {
                        "keyword": keyword,
                        "score": float(total_count),
                        "recent_count": float(recent.get(keyword, 0)),
                        "total_count": float(total_count),
                    }
                )

        scored.sort(
            key=lambda item: (-item["score"], -item["recent_count"], item["keyword"])
        )
        results: list[IdeaPrediction] = []
        for rank, item in enumerate(scored[:top_k], start=1):
            keyword = item["keyword"]
            confidence = min(
                0.99,
                0.35 + (0.08 * item["recent_count"]) + (0.02 * item["total_count"]),
            )
            results.append(
                IdeaPrediction(
                    rank=rank,
                    title=f"Next-stage direction: {keyword.title()}",
                    rationale=(
                        f"Keyword '{keyword}' appears {int(item['recent_count'])} times in the recent "
                        f"{self.recent_months} month(s), {int(item['total_count'])} times overall before {cutoff_month}."
                    ),
                    approach=(
                        f"Track papers centered on {keyword}, cluster the recent methodology shifts, "
                        "and benchmark whether the trend remains predictive over the next few months."
                    ),
                    score=round(min(1.0, max(0.0, float(item["score"]) / 10.0)), 4),
                    confidence=round(float(confidence), 4),
                    key_terms=[keyword],
                )
            )
        return results
