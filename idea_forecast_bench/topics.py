from __future__ import annotations

import re
from collections import OrderedDict

from idea_forecast_bench.config import TopicDefinition
from idea_forecast_bench.models import PaperRecord


def normalize_topic_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _paper_topic_corpus(paper: PaperRecord) -> str:
    return normalize_topic_text(
        " ".join([paper.title, paper.summary, " ".join(paper.keywords)])
    )


def _topic_terms(topic: TopicDefinition) -> list[str]:
    terms = [topic.name, *topic.aliases, *topic.keywords]
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = normalize_topic_text(term)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def classify_paper_topics(
    paper: PaperRecord,
    topics: list[TopicDefinition],
) -> list[TopicDefinition]:
    corpus = f" {_paper_topic_corpus(paper)} "
    matches: list[TopicDefinition] = []
    for topic in topics:
        if any(f" {term} " in corpus for term in _topic_terms(topic)):
            matches.append(topic)
    return matches


def classify_papers_by_topic(
    papers: list[PaperRecord],
    topics: list[TopicDefinition],
) -> OrderedDict[str, list[PaperRecord]]:
    grouped: OrderedDict[str, list[PaperRecord]] = OrderedDict(
        (topic.id, []) for topic in topics
    )
    for paper in papers:
        for topic in classify_paper_topics(paper, topics):
            grouped[topic.id].append(paper)
    return grouped
