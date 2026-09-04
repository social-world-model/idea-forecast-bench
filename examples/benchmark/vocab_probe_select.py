#!/usr/bin/env python3
"""Choose a FIXED probe set of 10 papers per topic, selected once from the v1
element cache so that every vocabulary version is eyeballed on the same
papers. Deterministic given ``--seed``; makes no LLM calls."""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from idea_forecast_bench.atomic import atomic_write_text
from idea_forecast_bench.combinatorial.cache import ElementCache
from idea_forecast_bench.combinatorial.canonicalize import element_key
from idea_forecast_bench.combinatorial.types import ExtractionRecord
from idea_forecast_bench.models import PaperRecord
from idea_forecast_bench.paper_cache import load_papers_and_topics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OLD_CACHE = "output/pilot/elements/a176b6f85981"
DEFAULT_OUTPUT = "config/vocab_probe_set.yaml"
FALLBACK_OUTPUT = "output/vocab/vocab_probe_set.yaml"

_SURVEY_RE = re.compile(r"survey|benchmark|review|empirical study", re.IGNORECASE)
_LABEL_TYPES = ("theme", "domain", "method")

_N_MAINSTREAM = 3
_N_NICHE = 3
_N_NEW_CONCEPT = 2
_N_SURVEY = 2
_MIN_NICHE_LABELS = 3
_MIN_LABEL_DOC_FREQ = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/hf_full/raw_markdown")
    parser.add_argument("--start-month", default="2024-04")
    parser.add_argument("--end-month", default="2025-09")
    parser.add_argument(
        "--topics", required=True, help="Comma-separated topic ids (required)."
    )
    parser.add_argument(
        "--old-cache",
        default=DEFAULT_OLD_CACHE,
        help="v1 element cache directory (extract-elements output).",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@dataclass(frozen=True)
class ProbeEntry:
    paper_id: str
    title: str
    month: str
    reason: str
    score: int = 0


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _paper_labels(record: ExtractionRecord) -> frozenset[str]:
    return frozenset(
        element_key(element_type, text)
        for element_type, text in record.typed_elements()
        if element_type in _LABEL_TYPES and text
    )


def _label_stats(
    eligible: list[PaperRecord], labels_of: dict[str, frozenset[str]]
) -> tuple[dict[str, int], dict[str, str]]:
    """Document frequency and first-appearance month per label, within the
    given (topic-scoped) eligible papers only."""
    df: Counter[str] = Counter()
    first_seen: dict[str, str] = {}
    for paper in eligible:
        for label in labels_of[paper.paper_id]:
            df[label] += 1
            if label not in first_seen or paper.month < first_seen[label]:
                first_seen[label] = paper.month
    return dict(df), first_seen


def _new_concept_score(
    paper: PaperRecord,
    labels_of: dict[str, frozenset[str]],
    df: dict[str, int],
    first_seen: dict[str, str],
) -> int:
    return sum(
        1
        for label in labels_of[paper.paper_id]
        if first_seen.get(label) == paper.month
        and df.get(label, 0) >= _MIN_LABEL_DOC_FREQ
    )


def _rank_mainstream(
    eligible: list[PaperRecord], score: dict[str, int]
) -> list[PaperRecord]:
    return sorted(eligible, key=lambda p: (-score[p.paper_id], p.paper_id))


def _rank_niche(
    eligible: list[PaperRecord],
    labels_of: dict[str, frozenset[str]],
    score: dict[str, int],
) -> list[PaperRecord]:
    candidates = [
        p for p in eligible if len(labels_of[p.paper_id]) >= _MIN_NICHE_LABELS
    ]
    return sorted(candidates, key=lambda p: (score[p.paper_id], p.paper_id))


def _rank_new_concept(
    eligible: list[PaperRecord],
    labels_of: dict[str, frozenset[str]],
    df: dict[str, int],
    first_seen: dict[str, str],
) -> list[PaperRecord]:
    scored = [(p, _new_concept_score(p, labels_of, df, first_seen)) for p in eligible]
    ranked = sorted(scored, key=lambda ps: (-ps[1], ps[0].paper_id))
    return [p for p, s in ranked if s > 0]


def _rank_survey(papers: list[PaperRecord]) -> list[PaperRecord]:
    matches = [p for p in papers if _SURVEY_RE.search(p.title or "")]
    return sorted(matches, key=lambda p: (p.month, p.paper_id))


def _entry(paper: PaperRecord, reason: str, score: dict[str, int]) -> ProbeEntry:
    return ProbeEntry(
        paper_id=paper.paper_id,
        title=paper.title,
        month=paper.month,
        reason=reason,
        score=score.get(paper.paper_id, 0),
    )


def _take(
    pool: list[PaperRecord],
    used: set[str],
    n: int,
    reason: str,
    score: dict[str, int],
) -> list[ProbeEntry]:
    picked: list[ProbeEntry] = []
    for paper in pool:
        if len(picked) >= n:
            break
        if paper.paper_id in used:
            continue
        used.add(paper.paper_id)
        picked.append(_entry(paper, reason, score))
    return picked


def _take_random(
    papers: list[PaperRecord],
    used: set[str],
    rng: random.Random,
    n: int,
    reason: str,
    score: dict[str, int],
) -> list[ProbeEntry]:
    """Deterministic fallback when a category ran out of qualifying
    candidates: shuffle the remaining unused papers with the topic-seeded
    RNG and take the first ``n``."""
    if n <= 0:
        return []
    remaining = [p for p in papers if p.paper_id not in used]
    rng.shuffle(remaining)
    return _take(remaining, used, n, reason, score)


def select_probe_papers(
    topic_id: str,
    papers: list[PaperRecord],
    records: dict[str, ExtractionRecord],
    seed: int,
) -> list[ProbeEntry]:
    """10 papers per topic: 3 mainstream, 3 niche, 2 new-concept, 2
    survey/benchmark, no paper picked twice. Deterministic given ``seed``."""
    eligible = [p for p in papers if records.get(p.paper_id) and records[p.paper_id].ok]
    labels_of = {p.paper_id: _paper_labels(records[p.paper_id]) for p in eligible}
    df, first_seen = _label_stats(eligible, labels_of)
    score = {
        p.paper_id: sum(df[label] for label in labels_of[p.paper_id]) for p in eligible
    }

    used: set[str] = set()
    rng = random.Random(f"{seed}:{topic_id}")
    entries: list[ProbeEntry] = []

    entries += _take(
        _rank_mainstream(eligible, score), used, _N_MAINSTREAM, "mainstream", score
    )
    entries += _take(
        _rank_niche(eligible, labels_of, score), used, _N_NICHE, "niche", score
    )

    new_concept = _take(
        _rank_new_concept(eligible, labels_of, df, first_seen),
        used,
        _N_NEW_CONCEPT,
        "new-concept",
        score,
    )
    new_concept += _take_random(
        papers, used, rng, _N_NEW_CONCEPT - len(new_concept), "new-concept", score
    )
    entries += new_concept

    survey = _take(_rank_survey(papers), used, _N_SURVEY, "survey/benchmark", score)
    survey += _take_random(
        papers, used, rng, _N_SURVEY - len(survey), "survey/benchmark", score
    )
    entries += survey

    return entries


def _print_topic_table(topic_id: str, entries: list[ProbeEntry]) -> None:
    print(f"\n{topic_id} ({len(entries)} papers)")
    header = f"  {'paper_id':<16}{'month':<9}{'score':>7}  {'reason':<18}title"
    print(header)
    for e in entries:
        title = e.title if len(e.title) <= 60 else e.title[:57] + "..."
        print(f"  {e.paper_id:<16}{e.month:<9}{e.score:>7}  {e.reason:<18}{title}")


def _write_output(
    path_str: str, payload: dict[str, dict[str, list[dict[str, str]]]]
) -> Path:
    target = _resolve(path_str)
    fallback = PROJECT_ROOT / FALLBACK_OUTPUT
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)
    try:
        atomic_write_text(target, text)
        return target
    except PermissionError:
        print(
            f"WARNING: cannot write {target} (permission denied); "
            f"falling back to {fallback}",
            file=sys.stderr,
        )
        atomic_write_text(fallback, text)
        return fallback


def main() -> int:
    args = parse_args()
    input_dir = _resolve(args.input_dir)
    papers, _topics, grouped = load_papers_and_topics(
        input_dir, args.start_month, args.end_month
    )

    topic_ids = [t.strip() for t in args.topics.split(",") if t.strip()]
    if not topic_ids:
        print("--topics must name at least one topic id", file=sys.stderr)
        return 2
    unknown = [t for t in topic_ids if t not in grouped]
    if unknown:
        print(f"Unknown topic id(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    old_cache_dir = _resolve(args.old_cache)
    cache = ElementCache.locate(old_cache_dir)
    records = cache.load()
    print(f"v1 element cache: {cache.directory} ({len(records)} records)", flush=True)

    output: dict[str, list[dict[str, str]]] = {}
    for topic_id in topic_ids:
        entries = select_probe_papers(topic_id, grouped[topic_id], records, args.seed)
        output[topic_id] = [
            {
                "paper_id": e.paper_id,
                "title": e.title,
                "month": e.month,
                "reason": e.reason,
            }
            for e in entries
        ]
        _print_topic_table(topic_id, entries)
        if len(entries) < _N_MAINSTREAM + _N_NICHE + _N_NEW_CONCEPT + _N_SURVEY:
            print(
                f"  WARNING: only {len(entries)}/10 probe papers selected for "
                f"{topic_id} (not enough eligible papers in the v1 cache)",
                file=sys.stderr,
            )

    output_path = _write_output(args.output, {"topics": output})
    total = sum(len(v) for v in output.values())
    print(
        f"\nwrote {total} probe papers across {len(topic_ids)} topic(s) -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
