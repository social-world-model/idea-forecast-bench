from __future__ import annotations

from datetime import datetime, timezone

from backend.services import arxiv_ingest
from live_idea_bench.papers import load_papers_from_markdown


SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2603.00001v1</id>
    <updated>2026-03-02T10:00:00Z</updated>
    <published>2026-03-01T10:00:00Z</published>
    <title>  Test Paper One  </title>
    <summary>Abstract one.</summary>
    <category term="cs.AI"/>
    <category term="cs.LG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2501.12345v1</id>
    <updated>2025-01-03T10:00:00Z</updated>
    <published>2025-01-01T10:00:00Z</published>
    <title>Old Paper</title>
    <summary>Should be skipped by lookback.</summary>
    <category term="cs.AI"/>
  </entry>
</feed>
"""


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def test_ingest_latest_arxiv_papers_writes_markdown_and_is_idempotent(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        arxiv_ingest.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(SAMPLE_FEED),
    )
    now = datetime(2026, 3, 3, 0, 0, 0, tzinfo=timezone.utc)

    first = arxiv_ingest.ingest_latest_arxiv_papers(
        data_dir=tmp_path,
        query="cat:cs.AI",
        max_results=10,
        lookback_days=7,
        now=now,
    )

    assert first["fetched_count"] == 2
    assert first["ingested_count"] == 1
    assert first["skipped_old_count"] == 1
    assert first["new_papers"][0]["paper_id"] == "2603.00001v1"

    paper_path = tmp_path / "2026-03" / "2603.00001v1.md"
    assert paper_path.exists()
    text = paper_path.read_text(encoding="utf-8")
    assert not text.startswith("---")
    assert "# Test Paper One" in text
    assert "Paper ID: 2603.00001v1" in text
    assert "Date: 2026-03-01" in text
    assert "Source URL: http://arxiv.org/abs/2603.00001v1" in text
    assert "# Abstract" in text
    assert "Abstract one." in text

    loaded = load_papers_from_markdown(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].paper_id == "2603.00001v1"
    assert loaded[0].month == "2026-03"
    assert loaded[0].published_date == "2026-03-01"
    assert loaded[0].summary == "Abstract one."
    # NOTE: load_papers_from_markdown intentionally strips heavy fields
    # (metadata/references/citations) not used by evidence retrieval or the GRPO
    # reward; source_url is still verified on disk above (the "Source URL:" line).
    assert loaded[0].keywords == ["cs.ai", "cs.lg"]

    second = arxiv_ingest.ingest_latest_arxiv_papers(
        data_dir=tmp_path,
        query="cat:cs.AI",
        max_results=10,
        lookback_days=7,
        now=now,
    )
    assert second["ingested_count"] == 0
    assert second["skipped_existing_count"] == 1
