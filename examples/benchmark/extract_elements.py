#!/usr/bin/env python3
"""Offline, resumable per-paper element extraction for the combinatorial
forecaster. Each paper is processed once and cached by paper id, so the
result is shared by every topic and every cutoff."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from idea_forecast_bench.combinatorial.cache import ElementCache
from idea_forecast_bench.combinatorial.canonicalize import (
    element_key,
    merge_elements,
    split_key,
)
from idea_forecast_bench.combinatorial.config import (
    load_combinatorial_config,
    load_prompt_pair,
)
from idea_forecast_bench.combinatorial.embeddings import (
    HASH_BACKEND,
    VectorStore,
    make_embedder,
)
from idea_forecast_bench.combinatorial.extraction import (
    FAKE_MODEL,
    extract_paper,
    extraction_fingerprint,
    fake_extraction,
)
from idea_forecast_bench.combinatorial.llm_caller import (
    TextCaller,
    caller_for_model,
    callers_for_base_urls,
)
from idea_forecast_bench.combinatorial.types import ExtractionRecord
from idea_forecast_bench.models import PaperRecord
from idea_forecast_bench.paper_cache import load_papers_and_topics
from idea_forecast_bench.papers import corpus_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/csml/raw_markdown")
    parser.add_argument("--start-month", default="2024-04")
    parser.add_argument("--end-month", default="2025-09")
    parser.add_argument(
        "--topics",
        default=None,
        help="Comma-separated topic ids; default = every topic's papers (union).",
    )
    parser.add_argument("--cache-dir", default=".cache/elements")
    parser.add_argument("--config", default=None, help="combinatorial.yaml override")
    parser.add_argument(
        "--model-name",
        default="gpt-4o-qwen35",
        help="Served alias (must start with gpt-4o/gpt-4.1/gpt-5 when using "
        "OPENAI_BASE_URL / --base-urls).",
    )
    parser.add_argument(
        "--base-urls",
        default=None,
        help="Comma-separated OpenAI-compatible base URLs, used round-robin.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Max papers (testing)")
    parser.add_argument("--dry-run", action="store_true", help="Fake extractor, no LLM")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--embed", action="store_true", help="Embed element labels")
    parser.add_argument("--embed-backend", default=None, help="voyage | hash")
    parser.add_argument(
        "--dump-clusters",
        type=int,
        default=0,
        help="Print the N largest merge clusters",
    )
    parser.add_argument("--allow-fingerprint-mismatch", action="store_true")
    parser.add_argument("--selfcheck", action="store_true", help="Closed-form checks")
    return parser.parse_args()


def _union_papers(
    grouped: dict[str, list[PaperRecord]], topic_ids: list[str] | None
) -> list[PaperRecord]:
    seen: dict[str, PaperRecord] = {}
    for topic_id, papers in grouped.items():
        if topic_ids is not None and topic_id not in topic_ids:
            continue
        for paper in papers:
            seen.setdefault(paper.paper_id, paper)
    return sorted(seen.values(), key=lambda p: (p.month, p.paper_id))


def _run_extraction(
    papers: list[PaperRecord],
    cache: ElementCache,
    existing: dict[str, ExtractionRecord],
    args: argparse.Namespace,
    caller: TextCaller | None,
    fingerprint: str,
) -> None:
    cfg = load_combinatorial_config(args.config)
    prompt = load_prompt_pair(cfg.extraction.prompt)
    aliases = cfg.canonicalize.aliases
    todo = [
        p
        for p in papers
        if p.paper_id not in existing
        or (args.retry_failed and not existing[p.paper_id].ok)
    ]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(
        f"{len(papers)} papers in scope, {len(existing)} cached, {len(todo)} to extract",
        flush=True,
    )
    if not todo:
        return

    def _one(paper: PaperRecord) -> ExtractionRecord:
        if caller is None:
            return fake_extraction(paper, aliases, fingerprint)
        return extract_paper(
            paper, caller, prompt, cfg.extraction, aliases, fingerprint
        )

    writer = cache.writer()
    failures = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_one, p): p for p in todo}
            for future in tqdm(as_completed(futures), total=len(futures), unit="paper"):
                record = future.result()
                writer.append(record)
                if not record.ok:
                    failures += 1
    finally:
        writer.close()
    print(f"extracted {len(todo)} papers, {failures} failed", flush=True)


def _element_texts(records: dict[str, ExtractionRecord]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for record in records.values():
        if not record.ok:
            continue
        for element_type, text in record.typed_elements():
            if text:
                texts[element_key(element_type, text)] = text
    return texts


def _run_embed(
    cache: ElementCache, records: dict[str, ExtractionRecord], args: argparse.Namespace
) -> None:
    cfg = load_combinatorial_config(args.config)
    backend = args.embed_backend or cfg.canonicalize.embed_backend
    embedder = make_embedder(backend, cfg.canonicalize.embed_model)
    store = VectorStore(cache.vectors_path(cfg.canonicalize.embed_model))
    texts = _element_texts(records)
    print(
        f"{len(texts)} unique element labels, {len(store)} already embedded", flush=True
    )
    added = store.ensure(texts.keys(), texts, embedder)
    print(f"embedded {added} new labels with {embedder.model_name}", flush=True)


def _dump_clusters(
    cache: ElementCache, records: dict[str, ExtractionRecord], args: argparse.Namespace
) -> None:
    cfg = load_combinatorial_config(args.config)
    store = VectorStore(cache.vectors_path(cfg.canonicalize.embed_model))
    counts: Counter[str] = Counter()
    for record in records.values():
        if record.ok:
            counts.update(
                element_key(t, text) for t, text in record.typed_elements() if text
            )
    leader_of = merge_elements(counts, store.view(), cfg.canonicalize.merge_threshold)
    clusters: dict[str, list[str]] = {}
    for member, leader in leader_of.items():
        clusters.setdefault(leader, []).append(member)
    ranked = sorted(clusters.items(), key=lambda kv: -sum(counts[m] for m in kv[1]))
    print(
        f"{len(counts)} raw keys -> {len(clusters)} elements at threshold {cfg.canonicalize.merge_threshold}"
    )
    for leader, members in ranked[: args.dump_clusters]:
        total = sum(counts[m] for m in members)
        variants = sorted(members, key=lambda m: -counts[m])
        shown = ", ".join(f"{split_key(m)[1]}({counts[m]})" for m in variants[:8])
        print(f"  {leader} [{total} papers, {len(members)} variants]: {shown}")


def _selfcheck() -> int:
    """Closed-form checks of the state formulas on a toy corpus."""
    from idea_forecast_bench.combinatorial.config import (
        CanonicalizeConfig,
        SamplerConfig,
        StateConfig,
    )
    from idea_forecast_bench.combinatorial.state import (
        build_state,
        rising,
        unpaired_bonus,
    )

    def paper(pid: str, date: str) -> PaperRecord:
        return PaperRecord(
            paper_id=pid,
            title=pid,
            month=date[:7],
            summary="",
            keywords=[],
            source_path="",
            published_date=date,
        )

    def record(
        pid: str, date: str, themes: tuple[str, ...], methods: tuple[str, ...]
    ) -> ExtractionRecord:
        return ExtractionRecord(
            paper_id=pid,
            published_date=date,
            status="ok",
            themes=themes,
            methods=methods,
            move="extend",
        )

    state_cfg = StateConfig(
        half_life_months=6.0, recent_months=3.0, smoothing_alpha=0.5, hot_quantile=0.0
    )
    canon_cfg = CanonicalizeConfig(
        embed_backend="hash",
        embed_model="hash",
        merge_threshold=0.99,
        min_count=1,
        aliases={},
    )
    sampler_cfg = SamplerConfig(
        top_m_per_type=10,
        top_m_triple=5,
        type_patterns=(("theme", "method"),),
        score_gamma=1.0,
        lambda_rising=1.0,
        lambda_unpaired=1.0,
        rising_log_clip=2.0,
    )
    # Paper exactly one half-life (6 * 30.44 days) before the cutoff.
    cutoff = "2025-01-01"
    papers = [
        paper("old", "2024-07-02"),
        paper("new", "2024-12-20"),
        paper("new2", "2024-12-25"),
    ]
    records = {
        "old": record("old", "2024-07-02", ("a",), ("x",)),
        "new": record("new", "2024-12-20", ("a",), ("y",)),
        "new2": record("new2", "2024-12-25", ("b",), ("y",)),
    }
    state = build_state(papers, records, cutoff, state_cfg, canon_cfg, {})
    checks: list[tuple[str, bool]] = []
    heat_x = state.elements["method:x"].heat
    checks.append(
        ("heat of a paper one half-life old == 0.5", abs(heat_x - 0.5) < 0.01)
    )
    checks.append(
        (
            "unpaired bonus > 0 only for never-paired hot elements",
            unpaired_bonus(state, "theme:a", "method:y") == 0.0
            and unpaired_bonus(state, "theme:b", "method:x") > 0.0,
        )
    )
    r = rising(state, "theme:a", "method:y", state_cfg, 2.0)
    checks.append(("rising is positive for a pair seen only recently", r > 0))
    checks.append(("rising is clipped", abs(r) <= 2.0))
    checks.append(("freshness config consumed", sampler_cfg.lambda_rising == 1.0))
    try:
        build_state(
            [paper("future", "2025-02-01")], records, cutoff, state_cfg, canon_cfg, {}
        )
        checks.append(("post-cutoff paper rejected", False))
    except ValueError:
        checks.append(("post-cutoff paper rejected", True))
    ok = True
    for name, passed in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    return 0 if ok else 1


def main() -> int:
    args = parse_args()
    if args.selfcheck:
        return _selfcheck()

    cfg = load_combinatorial_config(args.config)
    prompt = load_prompt_pair(cfg.extraction.prompt)
    model = FAKE_MODEL if args.dry_run else args.model_name
    fingerprint = extraction_fingerprint(
        prompt, model, cfg.schema_version, cfg.extraction.temperature
    )

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    papers, _topics, grouped = load_papers_and_topics(
        input_dir, args.start_month, args.end_month
    )
    topic_ids = (
        [t.strip() for t in args.topics.split(",") if t.strip()]
        if args.topics
        else None
    )
    if topic_ids is not None:
        unknown = [t for t in topic_ids if t not in grouped]
        if unknown:
            print(f"Unknown topic id(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
    scoped = _union_papers(grouped, topic_ids)

    cache = ElementCache.open(
        Path(args.cache_dir),
        fingerprint,
        manifest_extra={
            "model": model,
            "prompt": cfg.extraction.prompt,
            "config": cfg.source_path,
            "input_dir": str(input_dir),
            "corpus_fingerprint": corpus_fingerprint(input_dir),
            "start_month": args.start_month,
            "end_month": args.end_month,
        },
        allow_mismatch=args.allow_fingerprint_mismatch,
    )
    stored_corpus = cache.manifest.get("corpus_fingerprint")
    current_corpus = corpus_fingerprint(input_dir)
    if stored_corpus and stored_corpus != current_corpus:
        print(
            f"WARNING: cache was built from corpus {stored_corpus}, current corpus is "
            f"{current_corpus}. Records are per paper id, so this is safe only if the "
            "ids mean the same papers.",
            file=sys.stderr,
        )
    print(f"element cache: {cache.directory}", flush=True)

    existing = cache.load()
    caller: TextCaller | None = None
    if not args.dry_run:
        if args.base_urls:
            caller = callers_for_base_urls(args.model_name, args.base_urls.split(","))
        else:
            caller = caller_for_model(args.model_name)
    _run_extraction(scoped, cache, existing, args, caller, fingerprint)

    records = cache.load()
    summary = cache.summary(records)
    print("summary:", summary, flush=True)

    if args.embed:
        if args.dry_run and not args.embed_backend:
            args.embed_backend = HASH_BACKEND
        _run_embed(cache, records, args)
    if args.dump_clusters > 0:
        _dump_clusters(cache, records, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
