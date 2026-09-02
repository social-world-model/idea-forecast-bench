from __future__ import annotations

from collections.abc import Mapping, Sequence

from idea_forecast_bench.combinatorial.types import Combo, Evidence
from idea_forecast_bench.models import PaperRecord
from idea_forecast_bench.papers import get_paper_published_date
from idea_forecast_bench.similarity import _sanitize, lexical_similarity, paper_text

_LEXICAL_POOL = 300


def _snippet(paper: PaperRecord, chars: int) -> str:
    return _sanitize(paper.summary)[:chars].replace("\n", " ").strip()


def retrieve_evidence(
    combo: Combo,
    papers_by_id: Mapping[str, PaperRecord],
    train_papers: Sequence[PaperRecord],
    *,
    n: int,
    snippet_chars: int,
) -> tuple[Evidence, ...]:
    """Supporting pre-cutoff papers for one combination.

    Papers that contain the most of the combo's elements win, newest first;
    if fewer than two qualify, the most lexically similar recent train
    papers fill the gap. Candidates come only from ``train_papers`` and the
    elements' own ``paper_ids`` (which are a subset of the train set)."""
    hits: dict[str, set[str]] = {}
    for element in combo.elements:
        for pid in element.paper_ids:
            hits.setdefault(pid, set()).add(element.label)
    # Most matched elements first, then newest (ISO dates sort lexically).
    ranked = sorted(
        (pid for pid in hits if pid in papers_by_id),
        key=lambda pid: (len(hits[pid]), get_paper_published_date(papers_by_id[pid])),
        reverse=True,
    )

    out: list[Evidence] = []
    for pid in ranked[:n]:
        paper = papers_by_id[pid]
        out.append(
            Evidence(
                paper_id=pid,
                title=_sanitize(paper.title),
                month=paper.month,
                snippet=_snippet(paper, snippet_chars),
                matched_elements=tuple(sorted(hits[pid])),
            )
        )
    if len(out) >= min(2, n):
        return tuple(out)

    query = " ".join(e.label for e in combo.elements)
    chosen = {e.paper_id for e in out}
    pool = [p for p in train_papers[-_LEXICAL_POOL:] if p.paper_id not in chosen]
    pool.sort(key=lambda p: lexical_similarity(query, paper_text(p)), reverse=True)
    for paper in pool[: n - len(out)]:
        out.append(
            Evidence(
                paper_id=paper.paper_id,
                title=_sanitize(paper.title),
                month=paper.month,
                snippet=_snippet(paper, snippet_chars),
                matched_elements=(),
            )
        )
    return tuple(out)
