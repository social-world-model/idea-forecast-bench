from __future__ import annotations

from pathlib import Path

from live_idea_bench.papers import load_papers_from_markdown, parse_markdown_paper


def test_parse_markdown_paper_extracts_sample_style_abstract_and_references(
    tmp_path: Path,
) -> None:
    path = tmp_path / "2403.12345.md"
    path.write_text(
        (
            "# Retrieval-Augmented Forecasting\n\n"
            "Ada Lovelace, Alan Turing\n\n"
            "Abstract— We ground forecasting models with bibliography evidence.\n\n"
            "This second paragraph still belongs to the abstract.\n\n"
            "# INTRODUCTION\n\n"
            "The full paper starts here.\n\n"
            "# REFERENCES\n\n"
            "[1] Foundational Work on Retrieval.\n"
            "continued context about the same reference.\n\n"
            "[2] Evidence Selection for Forecasting.\n"
        ),
        encoding="utf-8",
    )

    paper = parse_markdown_paper(path)

    assert paper is not None
    assert paper.title == "Retrieval-Augmented Forecasting"
    assert paper.paper_id == "2403.12345"
    assert paper.month == "2024-03"
    assert paper.published_date == "2024-03-31"
    assert "Ada Lovelace" not in paper.summary
    assert paper.summary.startswith(
        "We ground forecasting models with bibliography evidence."
    )
    assert "This second paragraph still belongs to the abstract." in paper.summary
    assert paper.references == [
        {
            "text": "Foundational Work on Retrieval. continued context about the same reference."
        },
        {"text": "Evidence Selection for Forecasting."},
    ]
    assert paper.citations == []
    assert paper.keywords[:3] == ["retrieval", "augmented", "forecasting"]


def test_parse_markdown_paper_reads_preamble_metadata_and_parent_month(
    tmp_path: Path,
) -> None:
    path = tmp_path / "2024-11" / "agent-forecasting.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "# Agent Forecasting\n\n"
            "Paper ID: paper-agent-001\n"
            "Keywords: agents, retrieval\n"
            "Source URL: https://example.com/paper-agent-001\n\n"
            "# Abstract\n\n"
            "Agent forecasting abstract.\n"
        ),
        encoding="utf-8",
    )

    paper = parse_markdown_paper(path)

    assert paper is not None
    assert paper.paper_id == "paper-agent-001"
    assert paper.month == "2024-11"
    assert paper.published_date == "2024-11-30"
    assert paper.summary == "Agent forecasting abstract."
    assert paper.keywords == ["agents", "retrieval"]
    assert paper.metadata["source_url"] == "https://example.com/paper-agent-001"


def test_load_papers_from_markdown_skips_body_file_without_any_date_signal(
    tmp_path: Path,
) -> None:
    dated_path = tmp_path / "2024-12" / "good-paper.md"
    dated_path.parent.mkdir(parents=True, exist_ok=True)
    dated_path.write_text(
        (
            "# Good Paper\n\n"
            "# Summary\n\n"
            "This paper has a month via its parent directory.\n"
        ),
        encoding="utf-8",
    )

    undated_path = tmp_path / "bad-paper.md"
    undated_path.write_text(
        (
            "# Bad Paper\n\n"
            "# Summary\n\n"
            "This file has no date metadata, no arXiv-style filename, and no month directory.\n"
        ),
        encoding="utf-8",
    )

    papers = load_papers_from_markdown(tmp_path)

    assert [paper.paper_id for paper in papers] == ["good-paper"]
    assert papers[0].month == "2024-12"
