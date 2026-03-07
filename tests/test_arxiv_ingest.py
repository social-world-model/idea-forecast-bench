from __future__ import annotations

from datetime import datetime, timezone

from live_idea_bench import ingest


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
        ingest.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(SAMPLE_FEED),
    )
    now = datetime(2026, 3, 3, 0, 0, 0, tzinfo=timezone.utc)

    first = ingest.ingest_latest_arxiv_papers(
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
    assert "paper_id:" in text
    assert "source_url:" in text
    assert "# Abstract" in text

    second = ingest.ingest_latest_arxiv_papers(
        data_dir=tmp_path,
        query="cat:cs.AI",
        max_results=10,
        lookback_days=7,
        now=now,
    )
    assert second["ingested_count"] == 0
    assert second["skipped_existing_count"] == 1
