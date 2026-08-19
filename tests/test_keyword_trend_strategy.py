from __future__ import annotations

from live_idea_bench.models import PaperRecord
from live_idea_bench.strategy.keyword_trend import KeywordTrendStrategy


def _paper(paper_id: str, month: str, keywords: list[str]) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        month=month,
        summary="summary",
        keywords=keywords,
        source_path=f"/fake/{paper_id}.md",
    )


def test_trend_gain_respects_recent_months_param() -> None:
    """trend_gain should divide by self.recent_months, not a hardcoded 3.

    Setup: keyword "transformer" appears 6 times total, 3 of which are
    inside the recent window when recent_months=6.

    Correct   (// self.recent_months == //6):
        trend_gain = 3 - max(1, 6//6) = 3 - 1 = 2
        raw_score  = 3*2 + 2 = 8  →  IdeaPrediction.score = 0.8

    Buggy     (// 3 hardcoded):
        trend_gain = 3 - max(1, 6//3) = 3 - 2 = 1
        raw_score  = 3*2 + 1 = 7  →  IdeaPrediction.score = 0.7
    """
    # 3 papers outside the recent window (before 2025-01)
    old_papers = [
        _paper(f"old-{i}", f"2024-{7 + i:02d}", ["transformer"]) for i in range(3)
    ]
    # 3 papers inside the recent window (2025-01 to 2025-03)
    # recent_months=6, cutoff=2025-06  →  recent window starts at 2025-01
    recent_papers = [
        _paper(f"new-{i}", f"2025-{1 + i:02d}", ["transformer"]) for i in range(3)
    ]

    strategy = KeywordTrendStrategy(recent_months=6, min_keyword_freq=2)
    predictions = strategy.generate(
        train_papers=old_papers + recent_papers,
        cutoff_month="2025-06",
        top_k=1,
    )

    assert len(predictions) == 1
    assert predictions[0].key_terms == ["transformer"]
    assert predictions[0].score == 0.8, (
        f"Expected 0.8 (correct //6), got {predictions[0].score} — "
        "this would be 0.7 if the hardcoded //3 bug were still present"
    )
