#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def month_to_index(month: str) -> int:
    year, mm = month.split("-")
    return int(year) * 12 + int(mm) - 1


def index_to_month(index: int) -> str:
    year = index // 12
    month = index % 12 + 1
    return f"{year:04d}-{month:02d}"


@dataclass
class Paper:
    paper_id: str
    date: str

    @property
    def month_idx(self) -> int:
        return month_to_index(self.date)


def load_papers(input_jsonl: Path) -> list[Paper]:
    papers: list[Paper] = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            papers.append(Paper(paper_id=row["paper_id"], date=row["date"]))
    return papers


def ids_in_range(papers: list[Paper], start_idx: int, end_idx: int) -> list[str]:
    return [p.paper_id for p in papers if start_idx <= p.month_idx <= end_idx]


def month_bounds(papers: list[Paper]) -> tuple[int, int]:
    indices = [p.month_idx for p in papers]
    return min(indices), max(indices)


def build_splits(
    papers: list[Paper],
    mode: str,
    train_window_months: int,
    min_train_months: int,
    horizon_months: int,
    step_months: int,
    gap_months: int,
    start_month: str | None,
    end_month: str | None,
    min_test_papers: int,
) -> list[dict]:
    if not papers:
        return []

    global_min, global_max = month_bounds(papers)
    cutoff_min = month_to_index(start_month) if start_month else global_min
    cutoff_max = month_to_index(end_month) if end_month else global_max

    # Ensure the horizon stays inside observed timeline.
    last_valid_cutoff = global_max - gap_months - horizon_months
    cutoff_max = min(cutoff_max, last_valid_cutoff)

    splits: list[dict] = []
    split_idx = 0
    for cutoff in range(cutoff_min, cutoff_max + 1, step_months):
        if mode == "rolling":
            train_start = cutoff - train_window_months + 1
            train_end = cutoff
            if train_start < global_min:
                continue
        else:
            train_start = global_min
            train_end = cutoff
            if (train_end - train_start + 1) < min_train_months:
                continue

        test_start = cutoff + gap_months + 1
        test_end = test_start + horizon_months - 1
        if test_end > global_max:
            continue

        train_ids = ids_in_range(papers, train_start, train_end)
        test_ids = ids_in_range(papers, test_start, test_end)

        if len(test_ids) < min_test_papers:
            continue

        split_idx += 1
        splits.append(
            {
                "split_id": f"{mode}_{split_idx:04d}",
                "mode": mode,
                "cutoff": {
                    "month": index_to_month(cutoff),
                    "month_index": cutoff,
                },
                "train": {
                    "start_month": index_to_month(train_start),
                    "end_month": index_to_month(train_end),
                    "paper_ids": train_ids,
                    "paper_count": len(train_ids),
                },
                "test": {
                    "start_month": index_to_month(test_start),
                    "end_month": index_to_month(test_end),
                    "paper_ids": test_ids,
                    "paper_count": len(test_ids),
                },
                "horizon_config": {
                    "horizon_months": horizon_months,
                    "gap_months": gap_months,
                    "step_months": step_months,
                },
                "leakage_safe": True,
            }
        )
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe rolling/expanding time splits from papers JSONL."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("data/csml/papers.jsonl"),
        help="Path to canonical papers JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/csml/time_slices"),
        help="Output directory for split JSON files",
    )
    parser.add_argument(
        "--mode",
        choices=["rolling", "expanding", "both"],
        default="both",
        help="Split type to generate",
    )
    parser.add_argument("--train-window-months", type=int, default=6)
    parser.add_argument("--min-train-months", type=int, default=6)
    parser.add_argument("--horizon-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--gap-months", type=int, default=0)
    parser.add_argument("--start-month", type=str, default=None)
    parser.add_argument("--end-month", type=str, default=None)
    parser.add_argument("--min-test-papers", type=int, default=1)
    args = parser.parse_args()

    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")

    papers = load_papers(args.input_jsonl)
    if not papers:
        raise ValueError("No papers found in input JSONL.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = ["rolling", "expanding"] if args.mode == "both" else [args.mode]

    all_splits: list[dict] = []
    for mode in modes:
        splits = build_splits(
            papers=papers,
            mode=mode,
            train_window_months=args.train_window_months,
            min_train_months=args.min_train_months,
            horizon_months=args.horizon_months,
            step_months=args.step_months,
            gap_months=args.gap_months,
            start_month=args.start_month,
            end_month=args.end_month,
            min_test_papers=args.min_test_papers,
        )
        for split in splits:
            split_path = args.output_dir / f"{split['split_id']}.json"
            split_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
        all_splits.extend(splits)

    bounds = month_bounds(papers)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_jsonl": str(args.input_jsonl),
        "dataset_bounds": {
            "min_month": index_to_month(bounds[0]),
            "max_month": index_to_month(bounds[1]),
            "paper_count": len(papers),
        },
        "config": {
            "mode": args.mode,
            "train_window_months": args.train_window_months,
            "min_train_months": args.min_train_months,
            "horizon_months": args.horizon_months,
            "step_months": args.step_months,
            "gap_months": args.gap_months,
            "start_month": args.start_month,
            "end_month": args.end_month,
            "min_test_papers": args.min_test_papers,
        },
        "split_count": len(all_splits),
        "split_ids": [s["split_id"] for s in all_splits],
    }

    manifest_path = args.output_dir / "time_slices_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(all_splits)} split files to {args.output_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
