from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

DEFAULT_QUERY = "cat:cs.AI OR cat:cs.ML"
DEFAULT_MAX_RESULTS = 100
DEFAULT_LOOKBACK_DAYS = 1


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_utc(raw: str | None) -> datetime | None:
    if not raw:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _sanitize_ws(raw: str | None) -> str:
    if not raw:
        return ""
    return " ".join(raw.split())


def _paper_id_from_url(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    return value.rstrip("/").split("/")[-1]


def _extract_entries(feed_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(feed_text)
    entries: list[dict[str, Any]] = []
    for node in root.findall("atom:entry", ATOM_NS):
        source_url = _sanitize_ws(node.findtext("atom:id", default="", namespaces=ATOM_NS))
        paper_id = _paper_id_from_url(source_url)
        title = _sanitize_ws(node.findtext("atom:title", default="", namespaces=ATOM_NS))
        summary = _sanitize_ws(node.findtext("atom:summary", default="", namespaces=ATOM_NS))
        published = _parse_utc(node.findtext("atom:published", default="", namespaces=ATOM_NS))
        updated = _parse_utc(node.findtext("atom:updated", default="", namespaces=ATOM_NS))
        keywords = []
        for cat in node.findall("atom:category", ATOM_NS):
            term = (cat.attrib.get("term") or "").strip()
            if term:
                keywords.append(term)
        dedup_keywords = sorted(set(keywords))

        if not paper_id or not title or published is None:
            continue

        entries.append(
            {
                "paper_id": paper_id,
                "title": title,
                "summary": summary,
                "published_at": published,
                "updated_at": updated,
                "keywords": dedup_keywords,
                "source_url": source_url or f"https://arxiv.org/abs/{paper_id}",
            }
        )
    return entries


def _paper_path(base_dir: Path, paper_id: str, month: str) -> Path:
    safe_id = paper_id.replace("/", "_")
    return base_dir / month / f"{safe_id}.md"


def _build_markdown(entry: dict[str, Any]) -> str:
    summary = (entry.get("summary") or "").strip()
    if not summary:
        summary = "No abstract provided by arXiv."

    lines = [
        f"# {entry['title'].strip()}",
        "",
        f"Paper ID: {entry['paper_id']}",
        f"Date: {entry['published_at'].date().isoformat()}",
    ]
    keywords = [str(keyword).strip() for keyword in (entry.get("keywords") or []) if str(keyword).strip()]
    if keywords:
        lines.append(f"Keywords: {', '.join(keywords)}")
    if entry.get("source_url"):
        lines.append(f"Source URL: {entry['source_url']}")

    lines.extend(
        [
            "",
            "# Abstract",
            "",
            summary,
        ]
    )

    references = entry.get("references") or []
    rendered_references: list[str] = []
    for index, reference in enumerate(references, start=1):
        if isinstance(reference, dict):
            text = str(
                reference.get("text")
                or reference.get("title")
                or reference.get("paper_title")
                or ""
            ).strip()
        else:
            text = str(reference).strip()
        if text:
            rendered_references.append(f"[{index}] {text}")

    if rendered_references:
        lines.extend(
            [
                "",
                "# References",
                "",
                *rendered_references,
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def ingest_latest_arxiv_papers(
    *,
    data_dir: Path | None = None,
    query: str | None = None,
    max_results: int | None = None,
    lookback_days: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    utc_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    resolved_query = query or _env_str("LIVE_IDEA_ARXIV_QUERY", DEFAULT_QUERY)
    resolved_max_results = max_results if max_results is not None else _env_int(
        "LIVE_IDEA_ARXIV_MAX_RESULTS", DEFAULT_MAX_RESULTS
    )
    resolved_lookback_days = lookback_days if lookback_days is not None else _env_int(
        "LIVE_IDEA_ARXIV_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS
    )
    base_dir = Path(
        data_dir
        or _env_str(
            "LIVE_IDEA_BENCH_DATA_DIR",
            str(Path(__file__).resolve().parents[1] / "data" / "arxiv_csml" / "raw_markdown"),
        )
    ).expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "search_query": resolved_query,
        "start": 0,
        "max_results": max(1, resolved_max_results),
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending",
    }
    response = requests.get(ARXIV_API_URL, params=params, timeout=30)
    response.raise_for_status()

    entries = _extract_entries(response.text)
    cutoff = utc_now - timedelta(days=max(0, resolved_lookback_days))

    ingested: list[dict[str, Any]] = []
    skipped_existing = 0
    skipped_old = 0

    for entry in entries:
        updated_at: datetime = entry.get("updated_at") or entry["published_at"]
        if updated_at < cutoff:
            skipped_old += 1
            continue

        month = entry["published_at"].strftime("%Y-%m")
        path = _paper_path(base_dir, entry["paper_id"], month)
        if path.exists():
            skipped_existing += 1
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_build_markdown(entry), encoding="utf-8")
        ingested.append(
            {
                "paper_id": entry["paper_id"],
                "month": month,
                "path": str(path),
                "published_at": _to_iso(entry["published_at"]),
                "updated_at": _to_iso(updated_at),
                "source_url": entry["source_url"],
                "keywords": entry["keywords"],
            }
        )

    return {
        "query": resolved_query,
        "max_results": resolved_max_results,
        "lookback_days": resolved_lookback_days,
        "fetched_count": len(entries),
        "ingested_count": len(ingested),
        "skipped_existing_count": skipped_existing,
        "skipped_old_count": skipped_old,
        "new_papers": ingested,
        "data_dir": str(base_dir),
        "fetched_at": _to_iso(utc_now),
    }
