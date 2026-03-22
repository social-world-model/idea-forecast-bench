from __future__ import annotations

from live_idea_bench.config import TopicDefinition
from live_idea_bench.models import PaperRecord
from live_idea_bench.topics import classify_paper_topics, classify_papers_by_topic


def _paper(*, paper_id: str, title: str, summary: str, keywords: list[str]) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        month="2024-06",
        summary=summary,
        keywords=keywords,
        source_path=f"/tmp/{paper_id}.md",
        published_date="2024-06-01",
    )


def test_classify_paper_topics_matches_keywords_and_summary() -> None:
    topics = [
        TopicDefinition(id="optimizer", name="Optimizer", keywords=["adamw"]),
        TopicDefinition(id="agents", name="GUI / Computer Use / Web Agent", aliases=["web agent"]),
    ]
    paper = _paper(
        paper_id="p1",
        title="Adaptive schedulers for browser agents",
        summary="We improve web agent planning with stronger verification.",
        keywords=["adamw"],
    )

    matched = classify_paper_topics(paper, topics)

    assert [topic.id for topic in matched] == ["optimizer", "agents"]


def test_classify_papers_by_topic_supports_multi_topic_membership() -> None:
    topics = [
        TopicDefinition(id="diffusion", name="Diffusion Language Model", aliases=["diffusion lm"]),
        TopicDefinition(id="forecasting", name="Time-series Forecasting", keywords=["time series"]),
    ]
    papers = [
        _paper(
            paper_id="p1",
            title="Diffusion LM for demand prediction",
            summary="A diffusion lm adapted to time series forecasting.",
            keywords=["time series"],
        ),
        _paper(
            paper_id="p2",
            title="Unrelated topic",
            summary="No configured topic terms.",
            keywords=["retrieval"],
        ),
    ]

    grouped = classify_papers_by_topic(papers, topics)

    assert [paper.paper_id for paper in grouped["diffusion"]] == ["p1"]
    assert [paper.paper_id for paper in grouped["forecasting"]] == ["p1"]
    assert all("p2" not in [paper.paper_id for paper in matched] for matched in grouped.values())
