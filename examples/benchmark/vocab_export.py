#!/usr/bin/env python3
"""Export the locked concept vocabulary to plain files: one JSON per topic
(every concept with its slot, parent, counts, variants and tags) and one
CSV across topics, so the vocabulary can be saved, diffed and shared
without rebuilding it from the extraction cache."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from idea_forecast_bench.backtest import split_train_future_by_cutoff
from idea_forecast_bench.combinatorial.embeddings import VectorStore
from idea_forecast_bench.paper_cache import load_papers_and_topics
from idea_forecast_bench.papers import month_start_date
from idea_forecast_bench.vocab.build import build_vocabulary
from idea_forecast_bench.vocab.config import load_vocab_config
from idea_forecast_bench.vocab.store import ConceptStore
from idea_forecast_bench.vocab.types import Vocabulary

FIELDS = [
    "topic", "cutoff", "id", "slot", "label", "parent", "count", "doc_frac",
    "first_seen", "recent_count", "background", "emerging", "variants",
]  # fmt: skip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--store", default="b493410c0021")
    parser.add_argument("--cache-dir", default="output/vocab/cache")
    parser.add_argument("--config", default="config/vocab.yaml")
    parser.add_argument("--input-dir", default="data/hf_full/raw_markdown")
    parser.add_argument("--start-month", default="2024-10")
    parser.add_argument("--end-month", default="2025-09")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _rows(vocab: Vocabulary) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for c in sorted(vocab.concepts.values(), key=lambda c: (-c.count, c.id)):
        rows.append(
            {
                "topic": vocab.topic_id,
                "cutoff": vocab.cutoff_month,
                "id": c.id,
                "slot": c.slot,
                "label": c.label,
                "parent": c.parent,
                "count": c.count,
                "doc_frac": round(c.doc_frac, 4),
                "first_seen": c.first_seen,
                "recent_count": c.recent_count,
                "background": c.background,
                "emerging": c.emerging,
                "variants": " | ".join(v for v in c.variants if v != c.label),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_vocab_config(args.config)
    store = ConceptStore(Path(args.cache_dir), args.store)
    records = store.load()
    vectors = VectorStore(store.vectors_dir / f"{cfg.cluster.embed_model}.json").view()
    _papers, _topics, grouped = load_papers_and_topics(
        args.input_dir, args.start_month, args.end_month, verbose=False
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for topic in [t.strip() for t in args.topics.split(",") if t.strip()]:
        train, _future, _m, _d = split_train_future_by_cutoff(
            grouped[topic],
            cutoff_month=args.cutoff,
            horizon_months=cfg.checks.horizon_months,
        )
        vocab = build_vocabulary(
            topic_id=topic,
            cutoff_month=args.cutoff,
            cutoff_date=month_start_date(args.cutoff),
            train_papers=train,
            records=records,
            vectors=vectors,
            cfg=cfg,
        )
        rows = _rows(vocab)
        all_rows.extend(rows)
        payload = {
            "topic": topic,
            "cutoff": args.cutoff,
            "config_sha": vocab.config_sha,
            "store": args.store,
            "n_train": vocab.n_train,
            "n_with_records": vocab.n_with_records,
            "n_concepts": len(rows),
            "n_background": sum(1 for r in rows if r["background"]),
            "n_emerging": sum(1 for r in rows if r["emerging"]),
            "concepts": rows,
        }
        (out / f"{topic}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(
            f"{topic}: {len(rows)} concepts ({payload['n_background']} background, "
            f"{payload['n_emerging']} emerging)"
        )
    with open(out / "vocabulary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
