"""Unified hindsight innovation extraction script.

Supports both a fast *preview* mode (a small subset of targets) and a *full*
mode that processes every selected paper in the manifest.  Extraction can be
done via sequential synchronous LLM calls (the default, works with any
provider) or via the OpenAI Batch API (``--batch`` flag, 50 % cost savings,
resume-capable).

Examples
--------
Preview mode, synchronous (same as run_topic_hindsight_preview.py)::

    python examples/run_topic_hindsight.py \\
        --input-dir data/papers --output-dir output/ \\
        --mode preview --preview-count 10 --model gpt-4o

Full mode, OpenAI Batch API::

    python examples/run_topic_hindsight.py \\
        --input-dir data/papers --output-dir output/ \\
        --mode full --batch --model gpt-4o

Full mode, synchronous (slow but works with Anthropic / Gemini too)::

    python examples/run_topic_hindsight.py \\
        --input-dir data/papers --output-dir output/ \\
        --mode full --model claude-3-5-sonnet-20241022
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from forecaster.config import load_hindsight_config  # noqa: E402
from forecaster.hindsight.batch import (  # noqa: E402
    encode_custom_id,
    run_batch_extraction,
)
from forecaster.hindsight.extractor import extract_innovation  # noqa: E402
from forecaster.hindsight.topic_sampling import (  # noqa: E402
    DEFAULT_TOPICS_CONFIG_PATH,
    TOPIC_HINDSIGHT_HORIZON_MONTHS,
    build_topic_hindsight_manifest,
    expand_all_targets,
    load_topic_hindsight_context,
    select_preview_targets,
    summarize_topic_hindsight_manifest,
    write_json,
)
from forecaster.models import innovation_to_dict  # noqa: E402
from live_idea_bench.backtest import split_train_future_by_cutoff  # noqa: E402
from live_idea_bench.llm import _is_openai_model, create_client  # noqa: E402


# ---------------------------------------------------------------------------
# Manifest helpers (shared with run_topic_hindsight_preview.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def _load_already_extracted(jsonl_path: Path) -> set[str]:
    """Return the set of custom_ids already written to *jsonl_path*."""
    if not jsonl_path.exists():
        return set()
    custom_ids: set[str] = set()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            cid = encode_custom_id(
                row["topic_id"],
                row["episode_id"],
                str(row["future_paper_id"]),
            )
            custom_ids.add(cid)
        except (json.JSONDecodeError, KeyError):
            continue
    return custom_ids


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------


def run_topic_hindsight(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    topics_config: str = DEFAULT_TOPICS_CONFIG_PATH,
    model: str = "gpt-5.4",
    mode: str = "preview",
    preview_count: int = 10,
    use_batch: bool = False,
    max_retry_rounds: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run hindsight innovation extraction.

    Args:
        input_dir: Directory containing paper markdown files.
        output_dir: Directory for all output files.
        manifest_path: Optional path to a pre-built ``manifest.json``.  If
            ``None`` the script looks for ``{output_dir}/manifest.json`` and
            builds one on-the-fly if it does not exist.
        topics_config: Path to the topics YAML config.
        model: LLM model identifier.
        mode: ``"preview"`` (default) or ``"full"``.
        preview_count: Number of targets in preview mode (ignored for full).
        use_batch: If ``True``, use the OpenAI Batch API.  Requires an OpenAI
            model.
        max_retry_rounds: Max additional batch retry rounds for parse failures
            (batch mode only).

    Returns:
        ``(rows, summary)`` — list of result dicts and a summary dict.
    """
    if use_batch and not _is_openai_model(model):
        raise ValueError(
            f"--batch requires an OpenAI model (gpt-4o* or gpt-5*), "
            f"but got model={model!r}."
        )

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

    if mode == "full":
        targets = expand_all_targets(manifest)
        output_jsonl_name = "hindsight_samples.jsonl"
        output_summary_name = "hindsight_summary.json"
    else:
        targets = select_preview_targets(manifest, preview_count=preview_count)
        output_jsonl_name = "preview_hindsight_samples.jsonl"
        output_summary_name = "preview_summary.json"

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = resolved_output_dir / output_jsonl_name

    already_extracted = _load_already_extracted(jsonl_path)
    if already_extracted:
        import logging
        logging.getLogger(__name__).info(
            "Resuming: %d targets already extracted, skipping them.",
            len(already_extracted),
        )

    rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Load already-written rows so they appear in the final output too
    # ------------------------------------------------------------------
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if use_batch:
        # -----------------------------------------------------------------
        # OpenAI Batch API path
        # -----------------------------------------------------------------
        config = load_hindsight_config()
        llm_client, resolved_model = create_client(model)
        config.llm_model = resolved_model

        successes, failures = run_batch_extraction(
            client=llm_client,
            targets=targets,
            grouped_papers=context.grouped_papers,
            model=resolved_model,
            config=config,
            output_dir=resolved_output_dir,
            max_retry_rounds=max_retry_rounds,
            already_extracted=already_extracted,
        )

        # Build a target lookup to enrich output rows
        target_index = {
            encode_custom_id(t["topic_id"], t["episode_id"], str(t["future_paper_id"])): t
            for t in targets
        }
        # Also need paper metadata for title / published_date
        paper_lookup: dict[str, Any] = {}
        for papers in context.grouped_papers.values():
            for paper in papers:
                paper_lookup[paper.paper_id] = paper

        for custom_id, innovation in successes.items():
            target = target_index.get(custom_id)
            if target is None:
                continue
            paper = paper_lookup.get(str(target["future_paper_id"]))
            topic_papers = list(context.grouped_papers.get(target["topic_id"], ()))
            train_papers, _, _, _ = split_train_future_by_cutoff(
                papers=topic_papers,
                cutoff_date=target["cutoff_date"],
                horizon_months=TOPIC_HINDSIGHT_HORIZON_MONTHS,
            )
            rows.append(
                {
                    "topic_id": target["topic_id"],
                    "episode_id": target["episode_id"],
                    "cutoff_date": target["cutoff_date"],
                    "future_paper_id": str(target["future_paper_id"]),
                    "future_paper_title": paper.title if paper else "",
                    "future_paper_published_date": paper.published_date if paper else "",
                    "innovation": innovation_to_dict(innovation),
                    "context_paper_count": len(train_papers),
                }
            )

        # Overwrite the JSONL with all rows (existing + new successes)
        jsonl_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

        summary: dict[str, Any] = {
            "input_dir": str(Path(input_dir).resolve()),
            "topics_config_path": str(context.topics_config_path),
            "manifest_path": str(resolved_manifest_path),
            "manifest_loaded_from_disk": manifest_loaded_from_disk,
            "manifest_summary": summarize_topic_hindsight_manifest(manifest),
            "mode": mode,
            "use_batch": True,
            "model": model,
            "total_targets": len(targets),
            "extracted_count": len(rows),
            "batch_failures": {cid: msg for cid, msg in failures.items()},
        }

    else:
        # -----------------------------------------------------------------
        # Synchronous path (sequential LLM calls, works for any provider)
        # -----------------------------------------------------------------
        new_targets = [
            t for t in targets
            if encode_custom_id(t["topic_id"], t["episode_id"], str(t["future_paper_id"]))
            not in already_extracted
        ]

        if new_targets:
            config = load_hindsight_config()
            llm_client, resolved_model = create_client(model)
            config.llm_model = resolved_model

            # Open in append mode for incremental writes (supports resume on crash)
            with jsonl_path.open("a", encoding="utf-8") as fh:
                for target in new_targets:
                    topic_papers = list(context.grouped_papers.get(target["topic_id"], ()))
                    train_papers, future_papers, _, _ = split_train_future_by_cutoff(
                        papers=topic_papers,
                        cutoff_date=target["cutoff_date"],
                        horizon_months=TOPIC_HINDSIGHT_HORIZON_MONTHS,
                    )
                    future_lookup = {p.paper_id: p for p in future_papers}
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
                    row = {
                        "topic_id": target["topic_id"],
                        "episode_id": target["episode_id"],
                        "cutoff_date": target["cutoff_date"],
                        "future_paper_id": future_paper.paper_id,
                        "future_paper_title": future_paper.title,
                        "future_paper_published_date": future_paper.published_date,
                        "innovation": innovation_to_dict(innovation),
                        "context_paper_count": len(train_papers),
                    }
                    rows.append(row)
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = {
            "input_dir": str(Path(input_dir).resolve()),
            "topics_config_path": str(context.topics_config_path),
            "manifest_path": str(resolved_manifest_path),
            "manifest_loaded_from_disk": manifest_loaded_from_disk,
            "manifest_summary": summarize_topic_hindsight_manifest(manifest),
            "mode": mode,
            "use_batch": False,
            "model": model,
            "total_targets": len(targets),
            "extracted_count": len(rows),
        }
        if mode == "preview":
            summary["requested_preview_count"] = preview_count

    write_json(resolved_output_dir / output_summary_name, summary)
    return rows, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hindsight innovation extraction (preview or full, sync or batch)."
    )
    parser.add_argument("--input-dir", required=True, help="Directory of paper markdown files.")
    parser.add_argument("--output-dir", required=True, help="Directory for output files.")
    parser.add_argument("--manifest-path", default=None, help="Optional pre-built manifest.json.")
    parser.add_argument("--topics-config", default=DEFAULT_TOPICS_CONFIG_PATH)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument(
        "--mode",
        choices=["preview", "full"],
        default="preview",
        help="'preview' processes --preview-count targets; 'full' processes all manifest targets.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=10,
        help="Number of targets in preview mode (ignored for full mode).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use the OpenAI Batch API instead of sequential synchronous calls. "
             "Requires an OpenAI model (gpt-4o* or gpt-5*).",
    )
    parser.add_argument(
        "--max-retry-rounds",
        type=int,
        default=1,
        help="Max additional batch retry rounds for parse failures (batch mode only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _rows, summary = run_topic_hindsight(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        topics_config=args.topics_config,
        model=args.model,
        mode=args.mode,
        preview_count=args.preview_count,
        use_batch=args.batch,
        max_retry_rounds=args.max_retry_rounds,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
