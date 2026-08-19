"""build_memory(papers_before_t) -> str.

A compact text inventory of *active research directions* at cutoff t.
Each entry is a short topic summary + recency/momentum count over the
most recent window. The function is intentionally simple and swappable;
downstream code (prior SFT, rubric retrieval, reward grounding) only
relies on the returned string being a stable, deterministic projection
of `papers_before_t`.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import date

from live_idea_bench.models import PaperRecord


def _paper_date(p: PaperRecord) -> date | None:
    s = (p.published_date or p.month or "").strip()
    if not s:
        return None
    if len(s) == 7:
        s = s + "-01"
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _bucket_for_paper(p: PaperRecord) -> str:
    """Topic-ish bucket for a single paper.

    Order of preference:
      1) metadata.topic_id (already topic-labelled in the corpus)
      2) first keyword (lower-cased)
      3) fallback string "uncategorized"
    """
    meta = p.metadata or {}
    topic_id = meta.get("topic_id") or meta.get("topic") or ""
    if isinstance(topic_id, str) and topic_id.strip():
        return topic_id.strip().lower()
    kws = list(p.keywords or [])
    if kws:
        return str(kws[0]).strip().lower() or "uncategorized"
    return "uncategorized"


def _short_summary(p: PaperRecord, max_chars: int = 160) -> str:
    text = " ".join((p.summary or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def build_memory(
    papers_before_t: Sequence[PaperRecord],
    *,
    cutoff_t: str | None = None,
    recency_window_months: int = 6,
    max_entries: int = 25,
    max_chars: int = 4000,
) -> str:
    """Return a compact memory string M_t for the cutoff.

    Args:
        papers_before_t: Papers with published_date <= cutoff t.
        cutoff_t: YYYY-MM or YYYY-MM-DD. Used to compute "recency" relative
            to t. If None, recency is computed relative to the latest paper.
        recency_window_months: How many months back from t count toward
            the per-bucket "momentum" tally.
        max_entries: Cap on number of bucket lines in the returned string.
        max_chars: Hard char cap on the returned string.

    Returns:
        Newline-separated text inventory. Empty corpus -> empty string.
    """
    if not papers_before_t:
        return ""

    # Determine anchor date (t). Prefer explicit cutoff; else use latest paper.
    anchor: date | None = None
    if cutoff_t:
        s = cutoff_t.strip()
        if len(s) == 7:
            s = s + "-01"
        try:
            anchor = date.fromisoformat(s)
        except ValueError:
            anchor = None
    if anchor is None:
        dates = [d for d in (_paper_date(p) for p in papers_before_t) if d is not None]
        if dates:
            anchor = max(dates)

    # Group + summarize per bucket.
    buckets: dict[str, list[PaperRecord]] = defaultdict(list)
    for p in papers_before_t:
        buckets[_bucket_for_paper(p)].append(p)

    # Per-bucket: total count, recent count (within window), representative summary.
    cutoff_ordinal = anchor.toordinal() if anchor else None
    window_days = int(recency_window_months * 30.44)
    bucket_stats: list[tuple[str, int, int, str]] = []
    for name, papers in buckets.items():
        total = len(papers)
        recent = 0
        # pick a "representative" by latest published_date
        latest: PaperRecord | None = None
        latest_ord: int | None = None
        for p in papers:
            d = _paper_date(p)
            if d is None:
                continue
            ord_d = d.toordinal()
            if cutoff_ordinal is not None and (cutoff_ordinal - ord_d) <= window_days:
                recent += 1
            if latest_ord is None or ord_d > latest_ord:
                latest_ord = ord_d
                latest = p
        rep = _short_summary(latest) if latest else ""
        bucket_stats.append((name, total, recent, rep))

    # Sort by (recent desc, total desc) → high-momentum buckets first.
    bucket_stats.sort(key=lambda r: (-r[2], -r[1], r[0]))
    bucket_stats = bucket_stats[:max_entries]

    header = (
        f"Memory snapshot at t={cutoff_t} | papers={len(papers_before_t)} "
        f"| recency_window={recency_window_months}mo"
    ) if cutoff_t else f"Memory snapshot | papers={len(papers_before_t)}"
    lines: list[str] = [header, ""]
    for name, total, recent, rep in bucket_stats:
        line = f"- {name} (total={total}, recent={recent}): {rep}".rstrip()
        lines.append(line)
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text
