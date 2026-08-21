"""Cached corpus loading, keyed on everything that changes the result.

Three entry scripts had copy-pasted the same cache block, and all three keyed
it on ``input_dir | start_month | end_month`` with the comment "topics config
is stable". The cache also stores ``grouped`` -- the topic-to-paper assignment
-- which depends entirely on the taxonomy. Editing config/topics_v2.yaml
therefore left the key unchanged and silently reused the old grouping, so a
run could report results computed against a taxonomy that no longer existed.
That is the kind of failure a benchmark cannot afford: it is invisible and it
changes the numbers.

The key here includes a fingerprint of the resolved topic definitions, so any
taxonomy edit misses the cache. Topics are loaded first because that is a
cheap YAML parse; the expensive steps are reading the corpus and classifying
it.

It also includes a fingerprint of the corpus. `fetch` is incremental, so the
documented workflow -- fetch, run, fetch more, run again -- kept every input to
the old key identical while the corpus underneath had grown, and the second run
silently reported the first run's numbers.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from live_idea_bench.config import TopicDefinition, load_topics
from live_idea_bench.models import PaperRecord
from live_idea_bench.papers import corpus_fingerprint, load_papers_from_markdown
from live_idea_bench.topics import classify_papers_by_topic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
#: Not under data/. That directory holds corpora people download and share,
#: and this cache is a pickle -- loading one from an untrusted source executes
#: whatever it contains.
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "papers"


def topics_fingerprint(topics: list[TopicDefinition]) -> str:
    """Stable hash of the resolved taxonomy, independent of which file it came from."""
    payload = json.dumps(
        [
            {
                "id": t.id,
                "name": t.name,
                "aliases": sorted(t.aliases or []),
                "keywords": sorted(t.keywords or []),
            }
            for t in sorted(topics, key=lambda t: t.id)
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_key(
    input_dir: Path | str,
    start_month: str,
    end_month: str,
    topics_fp: str,
    corpus_fp: str,
) -> str:
    raw = (
        f"{Path(input_dir).resolve()}|{start_month}|{end_month}|{topics_fp}|{corpus_fp}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_papers_and_topics(
    input_dir: Path | str,
    start_month: str,
    end_month: str,
    *,
    cache_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[list[PaperRecord], list[TopicDefinition], dict[str, list[PaperRecord]]]:
    """Load the corpus, the taxonomy, and the grouping, caching the slow parts.

    Returns ``(papers, topics, grouped)``. Raises if the corpus is empty --
    an empty corpus silently scores zero on every metric, which is
    indistinguishable from a model that predicted badly.
    """
    topics = load_topics()
    key = cache_key(
        input_dir,
        start_month,
        end_month,
        topics_fingerprint(topics),
        corpus_fingerprint(input_dir),
    )
    resolved_cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_path = resolved_cache_dir / f"papers_{key}.pkl"

    if cache_path.exists():
        if verbose:
            print(f"Loading papers+topics from cache ({cache_path.name}) ...")
        with open(cache_path, "rb") as handle:
            cached: dict[str, Any] = pickle.load(handle)  # noqa: S301 - self-written
        papers = cached["papers"]
        grouped = cached["grouped"]
        if verbose:
            print(
                f"Loaded {len(papers)} papers ({start_month} to {end_month}) [cached]"
            )
        return papers, cached["topics"], grouped

    if verbose:
        print(f"Loading papers from {input_dir} ...")
    papers = load_papers_from_markdown(
        Path(input_dir), start_month=start_month, end_month=end_month
    )
    if not papers:
        raise SystemExit(
            f"No papers found in {input_dir} for {start_month}..{end_month}. "
            "Fetch a corpus first (`live-idea-bench fetch`), or widen the window."
        )
    if verbose:
        print(f"Loaded {len(papers)} papers ({start_month} to {end_month})")
        print(f"Topics configured: {len(topics)}")
        print("Classifying papers by topic (this may take a minute) ...")
    grouped = classify_papers_by_topic(papers, topics)

    try:
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as handle:
            pickle.dump(
                {"papers": papers, "topics": topics, "grouped": grouped}, handle
            )
        if verbose:
            print(f"  [cache] saved to {cache_path.name}")
    except OSError as exc:
        if verbose:
            print(f"  [cache] could not save: {exc}")

    return papers, topics, grouped
