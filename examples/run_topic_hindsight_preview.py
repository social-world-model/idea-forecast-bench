from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from forecaster.config import load_hindsight_config  # noqa: E402
from forecaster.hindsight.extractor import extract_innovation  # noqa: E402
from forecaster.hindsight.topic_sampling import (  # noqa: E402
    DEFAULT_TOPICS_CONFIG_PATH,
    TOPIC_HINDSIGHT_HORIZON_MONTHS,
    build_topic_hindsight_manifest,
    load_topic_hindsight_context,
    select_preview_targets,
    summarize_topic_hindsight_manifest,
    write_json,
)
from forecaster.models import innovation_to_dict  # noqa: E402
from live_idea_bench.backtest import split_train_future_by_cutoff  # noqa: E402
from live_idea_bench.llm import create_client  # noqa: E402


def _load_or_build_manifest(
    *,
    input_dir: str | Path,
    manifest_path: str | Path,
    topics_config: str,
) -> tuple[dict[str, Any], bool]:
    manifest_file = Path(manifest_path).resolve()
    if manifest_file.exists():
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Manifest at {manifest_file} must be a JSON object.")
        return payload, True

    context = load_topic_hindsight_context(
        input_dir=input_dir,
        topics_config_path=topics_config,
    )
    return build_topic_hindsight_manifest(context), False


def run_topic_hindsight_preview(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    topics_config: str = DEFAULT_TOPICS_CONFIG_PATH,
    model: str = "gpt-5.4",
    preview_count: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved_output_dir = Path(output_dir).resolve()
    resolved_manifest_path = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else resolved_output_dir / "manifest.json"
    )
    manifest, manifest_loaded_from_disk = _load_or_build_manifest(
        input_dir=input_dir,
        manifest_path=resolved_manifest_path,
        topics_config=topics_config,
    )
    context = load_topic_hindsight_context(
        input_dir=input_dir,
        topics_config_path=topics_config,
    )
    preview_targets = select_preview_targets(manifest, preview_count=preview_count)

    preview_rows: list[dict[str, Any]] = []
    if preview_targets:
        config = load_hindsight_config()
        llm_client, resolved_model = create_client(model)
        config.llm_model = resolved_model
        for target in preview_targets:
            topic_papers = list(context.grouped_papers.get(target["topic_id"], ()))
            train_papers, future_papers, _future_end_month, _future_end_date = split_train_future_by_cutoff(
                papers=topic_papers,
                cutoff_date=target["cutoff_date"],
                horizon_months=TOPIC_HINDSIGHT_HORIZON_MONTHS,
            )
            future_lookup = {paper.paper_id: paper for paper in future_papers}
            future_paper = future_lookup.get(str(target["future_paper_id"]))
            if future_paper is None:
                raise ValueError(
                    f"Future paper {target['future_paper_id']!r} not found for "
                    f"topic={target['topic_id']!r}, episode={target['episode_id']!r}."
                )
            innovation = extract_innovation(
                future_paper=future_paper,
                context_papers=train_papers,
                llm_client=llm_client,
                model=resolved_model,
                config=config,
            )
            preview_rows.append(
                {
                    "topic_id": target["topic_id"],
                    "episode_id": target["episode_id"],
                    "cutoff_date": target["cutoff_date"],
                    "future_paper_id": future_paper.paper_id,
                    "future_paper_title": future_paper.title,
                    "future_paper_published_date": future_paper.published_date,
                    "innovation": innovation_to_dict(innovation),
                    "context_paper_count": len(train_papers),
                }
            )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = resolved_output_dir / "preview_hindsight_samples.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in preview_rows),
        encoding="utf-8",
    )
    preview_summary = {
        "input_dir": str(Path(input_dir).resolve()),
        "topics_config_path": str(context.topics_config_path),
        "manifest_path": str(resolved_manifest_path),
        "manifest_loaded_from_disk": manifest_loaded_from_disk,
        "manifest_summary": summarize_topic_hindsight_manifest(manifest),
        "requested_preview_count": preview_count,
        "generated_preview_count": len(preview_rows),
        "model": model,
    }
    write_json(resolved_output_dir / "preview_summary.json", preview_summary)
    return preview_rows, preview_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--topics-config", default=DEFAULT_TOPICS_CONFIG_PATH)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--preview-count", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _rows, summary = run_topic_hindsight_preview(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        topics_config=args.topics_config,
        model=args.model,
        preview_count=args.preview_count,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

