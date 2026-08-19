from __future__ import annotations

import json
from pathlib import Path

from forecaster.hindsight.topic_sampling import (
    FIXED_TOPIC_HINDSIGHT_EPISODES,
    build_topic_hindsight_manifest,
    load_topic_hindsight_context,
    select_preview_targets,
)
from forecaster.models import Innovation, innovation_from_dict
from live_idea_bench.config import load_topics
from live_idea_bench.papers import parse_markdown_paper


def _write_markdown_paper(
    root: Path,
    *,
    paper_id: str,
    published_date: str,
    title: str,
    summary: str,
    keywords: str = "time series, tabular data",
) -> Path:
    month = published_date[:7]
    path = root / month / f"{paper_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"Paper ID: {paper_id}",
                f"Date: {published_date}",
                f"Keywords: {keywords}",
                "",
                f"Abstract— {summary}",
                "",
                "# References",
                "[1] Reference entry one.",
                "continued detail for the same reference.",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_legacy_arxiv_markdown(root: Path, *, paper_id: str) -> Path:
    path = root / paper_id / "auto" / f"{paper_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"# Legacy Paper {paper_id}",
                "",
                f"Paper ID: {paper_id}",
                "",
                "Abstract— Legacy paper that should be ignored by the hindsight range filter.",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _build_sample_corpus(root: Path) -> Path:
    data_dir = root / "papers"
    common_title = "Time-series Forecasting for Tabular Data Systems"
    common_summary = (
        "This work studies time-series forecasting for tabular data with stable baselines."
    )
    _write_legacy_arxiv_markdown(data_dir, paper_id="0704.0047")

    _write_markdown_paper(
        data_dir,
        paper_id="ctx-2023-01",
        published_date="2023-01-15",
        title=common_title + " Context",
        summary=common_summary,
    )
    _write_markdown_paper(
        data_dir,
        paper_id="bench-2024-10",
        published_date="2024-10-05",
        title=common_title + " Benchmark Only",
        summary=common_summary,
    )

    for episode_index, episode in enumerate(FIXED_TOPIC_HINDSIGHT_EPISODES, start=1):
        future_month = episode.future_start_date[:7]
        for slot, day in enumerate((5, 15, 25), start=1):
            _write_markdown_paper(
                data_dir,
                paper_id=f"e{episode_index}-p{slot}",
                published_date=f"{future_month}-{day:02d}",
                title=f"{common_title} Episode {episode_index} Paper {slot}",
                summary=common_summary,
            )

    return data_dir


def test_topics_v2_default_loader_has_52_topics() -> None:
    topics = load_topics("config/topics_v2.yaml")
    assert len(topics) == 52


def test_prepare_topic_hindsight_manifest_enforces_fixed_windows_and_sampling(
    tmp_path: Path,
) -> None:
    from examples.data.prepare_topic_hindsight_manifest import (
        prepare_topic_hindsight_manifest,
    )

    data_dir = _build_sample_corpus(tmp_path)
    output_dir = tmp_path / "manifest_out"
    context = load_topic_hindsight_context(data_dir)

    manifest, summary = prepare_topic_hindsight_manifest(
        input_dir=data_dir,
        output_dir=output_dir,
    )
    manifest_again, _summary_again = prepare_topic_hindsight_manifest(
        input_dir=data_dir,
        output_dir=tmp_path / "manifest_out_2",
    )

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "summary.json").exists()
    assert len(manifest["episodes"]) == 6
    assert summary["episode_count"] == 6
    assert summary["topic_count"] == 52
    assert len(manifest["topic_episode_rows"]) == 52 * 6
    assert "bench-2024-10" not in [paper.paper_id for paper in context.papers]
    assert "0704.0047" not in [paper.paper_id for paper in context.papers]

    non_empty_rows = [
        row
        for row in manifest["topic_episode_rows"]
        if row["topic_id"] in {"time_series", "tabular_ml"}
    ]
    assert len(non_empty_rows) == 12
    assert all(row["future_start_date"][:7] >= "2023-04" for row in non_empty_rows)
    assert all(row["future_end_date"][:7] <= "2024-09" for row in non_empty_rows)
    assert all(len(row["selected_future_paper_ids"]) <= 2 for row in manifest["topic_episode_rows"])
    assert all(
        paper_id != "bench-2024-10"
        for row in manifest["topic_episode_rows"]
        for paper_id in row["selected_future_paper_ids"]
    )

    time_series_e1 = next(
        row
        for row in manifest["topic_episode_rows"]
        if row["topic_id"] == "time_series" and row["episode_id"] == "E1"
    )
    assert time_series_e1["future_paper_count"] == 3
    assert time_series_e1["selected_future_paper_ids"] == ["e1-p1", "e1-p3"]

    empty_row = next(
        row
        for row in manifest["topic_episode_rows"]
        if row["topic_id"] == "llm_pretraining" and row["episode_id"] == "E1"
    )
    assert empty_row["future_paper_count"] == 0
    assert empty_row["selected_future_paper_ids"] == []

    assert manifest["topic_episode_rows"] == manifest_again["topic_episode_rows"]


def test_load_topic_hindsight_context_supports_deterministic_smoke_sample(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = _build_sample_corpus(tmp_path)
    monkeypatch.setenv("TOPIC_HINDSIGHT_MAX_FILES", "5")
    monkeypatch.setenv("TOPIC_HINDSIGHT_LOAD_WORKERS", "1")

    context_one = load_topic_hindsight_context(data_dir)
    context_two = load_topic_hindsight_context(data_dir)

    assert 1 <= len(context_one.papers) <= 5
    assert "0704.0047" not in [paper.paper_id for paper in context_one.papers]
    assert [paper.paper_id for paper in context_one.papers] == [
        paper.paper_id for paper in context_two.papers
    ]


def test_run_topic_hindsight_preview_generates_10_rows_with_valid_innovations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from examples.forecaster import run_topic_hindsight as preview_module

    data_dir = _build_sample_corpus(tmp_path)
    output_dir = tmp_path / "preview_out"
    context = load_topic_hindsight_context(data_dir)
    manifest = build_topic_hindsight_manifest(context)
    expected_targets = select_preview_targets(manifest, preview_count=10)

    monkeypatch.setattr(preview_module, "create_client", lambda model: (object(), model))

    def _fake_extract_innovation(*, future_paper, context_papers, llm_client, model, config):
        assert context_papers
        return Innovation(
            base_direction=f"based on {future_paper.paper_id}",
            operator="extend",
            gap=f"gap for {future_paper.paper_id}",
        )

    monkeypatch.setattr(preview_module, "extract_innovation", _fake_extract_innovation)

    rows, summary = preview_module.run_topic_hindsight(
        input_dir=data_dir,
        output_dir=output_dir,
        mode="preview",
    )

    assert summary["manifest_loaded_from_disk"] is False
    assert summary["requested_preview_count"] == 10
    assert summary["extracted_count"] == 10
    assert len(rows) == 10
    assert (output_dir / "preview_hindsight_samples.jsonl").exists()
    assert (output_dir / "preview_summary.json").exists()

    expected_triplets = [
        (target["topic_id"], target["episode_id"], target["future_paper_id"])
        for target in expected_targets
    ]
    actual_triplets = [
        (row["topic_id"], row["episode_id"], row["future_paper_id"])
        for row in rows
    ]
    assert actual_triplets == expected_triplets

    for row in rows:
        innovation = innovation_from_dict(dict(row["innovation"]))
        assert innovation.operator == "extend"
        assert row["context_paper_count"] >= 1
        assert row["future_paper_published_date"][:7] <= "2024-09"

    jsonl_rows = [
        json.loads(line)
        for line in (output_dir / "preview_hindsight_samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(jsonl_rows) == 10


def test_preview_uses_existing_manifest_when_present(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from examples.data.prepare_topic_hindsight_manifest import (
        prepare_topic_hindsight_manifest,
    )
    from examples.forecaster import run_topic_hindsight as preview_module

    data_dir = _build_sample_corpus(tmp_path)
    output_dir = tmp_path / "preview_existing_manifest"
    prepare_topic_hindsight_manifest(input_dir=data_dir, output_dir=output_dir)

    monkeypatch.setattr(preview_module, "create_client", lambda model: (object(), model))
    monkeypatch.setattr(
        preview_module,
        "extract_innovation",
        lambda **kwargs: Innovation(
            base_direction="body-style hindsight",
            operator="extend",
            gap="validate preview schema",
        ),
    )

    _rows, summary = preview_module.run_topic_hindsight(
        input_dir=data_dir,
        output_dir=output_dir,
        mode="preview",
        preview_count=4,
    )

    assert summary["manifest_loaded_from_disk"] is True
    assert summary["extracted_count"] == 4


def test_sample_markdown_parser_round_trip_for_preview_fixture(tmp_path: Path) -> None:
    data_dir = _build_sample_corpus(tmp_path)
    paper = parse_markdown_paper(data_dir / "2023-04" / "e1-p1.md")

    assert paper is not None
    assert paper.paper_id == "e1-p1"
    assert paper.references
    assert paper.month == "2023-04"
