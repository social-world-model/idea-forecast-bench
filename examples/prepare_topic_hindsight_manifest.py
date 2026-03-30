from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from forecaster.hindsight.topic_sampling import (  # noqa: E402
    DEFAULT_TOPICS_CONFIG_PATH,
    build_topic_hindsight_manifest,
    load_topic_hindsight_context,
    summarize_topic_hindsight_manifest,
    write_json,
)


def prepare_topic_hindsight_manifest(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    topics_config: str = DEFAULT_TOPICS_CONFIG_PATH,
    per_topic_per_episode: int = 2,
) -> tuple[dict[str, object], dict[str, object]]:
    context = load_topic_hindsight_context(
        input_dir=input_dir,
        topics_config_path=topics_config,
    )
    manifest = build_topic_hindsight_manifest(
        context,
        per_topic_per_episode=per_topic_per_episode,
    )
    summary = summarize_topic_hindsight_manifest(manifest)

    resolved_output_dir = Path(output_dir).resolve()
    write_json(resolved_output_dir / "manifest.json", manifest)
    write_json(resolved_output_dir / "summary.json", summary)
    return manifest, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topics-config", default=DEFAULT_TOPICS_CONFIG_PATH)
    parser.add_argument("--per-topic-per-episode", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _manifest, summary = prepare_topic_hindsight_manifest(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        topics_config=args.topics_config,
        per_topic_per_episode=args.per_topic_per_episode,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

