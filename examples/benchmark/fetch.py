from __future__ import annotations

import argparse
from pathlib import Path

from idea_forecast_bench.ingest import ingest_latest_arxiv_papers

DEFAULT_OUT_DIR = "data/csml/raw_markdown"
DEFAULT_HF_REPO = "4R5T/idea-forecast-bench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download an arXiv corpus for the benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Where to write the corpus; pass this same path to `benchmark --input-dir`.",
    )
    parser.add_argument(
        "--from-hf",
        nargs="?",
        const=DEFAULT_HF_REPO,
        default=None,
        metavar="REPO_ID",
        help="Download the frozen paper corpus from this Hugging Face dataset instead "
        f"of querying arXiv (default repo when given without a value: {DEFAULT_HF_REPO}).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="cat:cs.LG",
        help="arXiv search query, e.g. 'cat:cs.CL' or 'cat:cs.LG OR cat:cs.AI'.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=2000,
        help="Upper bound on papers to fetch.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=540,
        help="How far back to reach. A backtest needs several months of history "
        "before its first cutoff, so keep this comfortably larger than the "
        "window you intend to score.",
    )
    return parser.parse_args()


def download_from_hf(repo_id: str, out_dir: Path) -> int:
    """Download the per-month Parquet files and write them out as Markdown."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".hf_parquet"
    print(f"Downloading {repo_id} ...")
    snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=cache_dir,
        allow_patterns=["*.parquet", "manifest.json"],
    )
    files = sorted(cache_dir.glob("*.parquet"))
    if not files:
        print(f"No Parquet files found in {repo_id}.")
        return 1
    written = 0
    for file in files:
        month = file.stem
        month_dir = out_dir / month
        if month_dir.is_dir():
            continue
        month_dir.mkdir()
        table = pq.read_table(file, columns=["arxiv_id", "text"])
        ids = table.column("arxiv_id").to_pylist()
        texts = table.column("text").to_pylist()
        for arxiv_id, text in zip(ids, texts, strict=True):
            (month_dir / f"{arxiv_id}.md").write_bytes(text.encode("utf-8"))
            written += 1
        file.unlink()
    print(f"  wrote {written} papers from {len(files)} monthly files into {out_dir}")
    return 0


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)

    if args.from_hf:
        status = download_from_hf(args.from_hf, out_dir)
        if status:
            return status
    else:
        print(f"Fetching up to {args.max_results} papers for {args.query!r} ...")
        fetch_from_arxiv(args, out_dir)

    paths = sorted(out_dir.rglob("*.md"))
    months = sorted({p.parent.name for p in paths})
    print(f"  corpus now: {len(paths)} papers across {len(months)} months")
    if months:
        print(f"  range: {months[0]} .. {months[-1]}")
        print(f"\nNext: idea-forecast-bench benchmark --input-dir {out_dir}")
    else:
        print("\nNo papers written. Widen --lookback-days or loosen --query.")
        return 1
    return 0


def fetch_from_arxiv(args: argparse.Namespace, out_dir: Path) -> None:
    result = ingest_latest_arxiv_papers(
        data_dir=out_dir,
        query=args.query,
        max_results=args.max_results,
        lookback_days=args.lookback_days,
    )
    print(
        f"  fetched {result['fetched_count']}, "
        f"new {result['ingested_count']}, "
        f"already present {result['skipped_existing_count']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
