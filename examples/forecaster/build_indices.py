from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

from forecaster.foresight.dz import augment_hindsight_rows
from forecaster.foresight.indices import (
    SentenceTransformerEmbedder,
    build_cutoff_indices,
)
from forecaster.foresight.operators import load_operator_inventory
from idea_forecast_bench.papers import load_papers_from_markdown

HORIZON_MONTHS = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build foresight D_z and cutoff indices.")
    p.add_argument(
        "--papers-dir",
        default=os.environ.get(
            "IDEA_FORECAST_BENCH_PAPERS_DIR", "data/csml/raw_markdown"
        ),
    )
    p.add_argument(
        "--hindsight",
        default="data/topic_hindsight/hindsight_samples.jsonl",
        help="Raw hindsight labels from `idea-forecast-bench hindsight`. Used only "
        "when --dz does not exist yet.",
    )
    p.add_argument("--dz", default="data/topic_hindsight/dz.jsonl")
    p.add_argument("--art", default="output/foresight_artifacts")
    p.add_argument("--start-month", default="2022-06")
    p.add_argument("--end-month", default="2024-12")
    p.add_argument("--embedder-model", default="sentence-transformers/allenai-specter")
    return p.parse_args()


def ensure_dz(dz_path: Path, hindsight_path: Path) -> None:
    """Derive D_z from the hindsight JSONL unless it already exists."""
    if dz_path.is_file():
        print(f"[dz] using existing {dz_path}", flush=True)
        return
    if not hindsight_path.is_file():
        raise SystemExit(
            f"[dz] neither {dz_path} nor {hindsight_path} exists; run "
            "`idea-forecast-bench hindsight` first"
        )
    dz_path.parent.mkdir(parents=True, exist_ok=True)
    summary = augment_hindsight_rows(
        hindsight_path,
        dz_path,
        inventory=load_operator_inventory(),
        # Corpus-free pass: memory_text is built at training time from the
        # indices; D_z only needs the operator mapping and the cutoff keying.
        papers_by_id=None,
        summary_path=dz_path.with_name("dz_summary.json"),
    )
    print(
        f"[dz] wrote {dz_path}: kept={summary.train_window_rows} "
        f"dropped(test window)={summary.dropped_test_window} "
        f"other-operator ratio={summary.other_ratio:.1%}",
        flush=True,
    )


def dataset_cutoffs(dz_path: Path) -> list[str]:
    """The GRPO episode dataset keys cutoffs as the FIRST day of the next
    period (dz "2023-03-31" -> dataset "2023-04-01"). The reward looks indices
    up by the dataset's cutoff_date, so build them keyed by dz+1day."""
    with open(dz_path, encoding="utf-8") as fh:
        raw = sorted({json.loads(line)["cutoff_t"] for line in fh if line.strip()})
    shifted = [
        (datetime.date.fromisoformat(c) + datetime.timedelta(days=1)).isoformat()
        for c in raw
    ]
    print(f"[idx] dz cutoffs={raw}", flush=True)
    print(f"[idx] dataset-aligned cutoffs (dz+1day)={shifted}", flush=True)
    return shifted


def main() -> int:
    args = parse_args()
    papers_dir = Path(args.papers_dir)
    if not papers_dir.is_dir():
        raise SystemExit(f"[idx] papers dir not found: {papers_dir}")

    ensure_dz(Path(args.dz), Path(args.hindsight))
    cutoffs = dataset_cutoffs(Path(args.dz))

    papers = load_papers_from_markdown(
        papers_dir, start_month=args.start_month, end_month=args.end_month
    )
    print(f"[idx] loaded {len(papers)} papers", flush=True)

    art = Path(args.art)
    (art / "indices").mkdir(parents=True, exist_ok=True)
    bundles = build_cutoff_indices(
        papers,
        cutoffs,
        horizon_months=HORIZON_MONTHS,
        embedder=SentenceTransformerEmbedder(model_name=args.embedder_model),
        save_dir=str(art / "indices"),
    )
    print(f"[idx] built {len(bundles)} cutoff bundles -> {art / 'indices'}", flush=True)
    for cutoff, bundle in bundles.items():
        print(
            f"[idx]   {cutoff}: future={bundle.future.size} history={bundle.history.size}",
            flush=True,
        )
    print("[idx] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
