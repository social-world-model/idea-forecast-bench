#!/usr/bin/env python3
"""Run one vocabulary version end to end: extract concepts, embed them,
build + check a vocabulary per (topic, cutoff), and write reports plus a
ledger row. Companion to ``vocab_probe_select.py``, which picks the fixed
probe papers this script's reports are eyeballed on."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

from idea_forecast_bench.atomic import atomic_write_text
from idea_forecast_bench.backtest import split_train_future_by_cutoff
from idea_forecast_bench.combinatorial.cache import ElementCache
from idea_forecast_bench.combinatorial.config import PromptPair
from idea_forecast_bench.combinatorial.embeddings import (
    HASH_BACKEND,
    VectorStore,
    make_embedder,
)
from idea_forecast_bench.combinatorial.llm_caller import (
    TextCaller,
    caller_for_model,
    callers_for_base_urls,
)
from idea_forecast_bench.combinatorial.types import ExtractionRecord
from idea_forecast_bench.models import PaperRecord
from idea_forecast_bench.paper_cache import load_papers_and_topics
from idea_forecast_bench.papers import (
    add_months,
    corpus_fingerprint,
    month_start_date,
    month_to_index,
)
from idea_forecast_bench.vocab.build import all_texts, build_vocabulary
from idea_forecast_bench.vocab.checks import CheckResult, run_checks, stability
from idea_forecast_bench.vocab.config import (
    ExtractionConfig,
    VocabConfig,
    load_prompt,
    load_vocab_config,
)
from idea_forecast_bench.vocab.extract import (
    FAKE_MODEL,
    extract_paper,
    extraction_fingerprint,
    fake_extraction,
)
from idea_forecast_bench.vocab.report import (
    ProbeRow,
    append_ledger,
    render_ledger_row,
    render_vocab_report,
)
from idea_forecast_bench.vocab.store import ConceptStore
from idea_forecast_bench.vocab.types import ConceptRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = "output/vocab/cache"
DEFAULT_PROBE_SET = "config/vocab_probe_set.yaml"
DEFAULT_OLD_CACHE = "output/pilot/elements/a176b6f85981"
DEFAULT_MIN_CUTOFF_MONTH = "2024-07"
_EMPTY_CHECKS = CheckResult(values={}, details={})
_APPEND_BATCH = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/hf_full/raw_markdown")
    parser.add_argument("--start-month", default="2024-04")
    parser.add_argument("--end-month", default="2025-09")
    parser.add_argument("--topics", required=True, help="Comma-separated topic ids.")
    parser.add_argument(
        "--cutoffs",
        default=None,
        help="Comma-separated YYYY-MM cutoffs; default = every month from "
        "--min-cutoff-month to end-month minus the checks horizon.",
    )
    parser.add_argument("--min-cutoff-month", default=DEFAULT_MIN_CUTOFF_MONTH)
    parser.add_argument("--config", default=None, help="vocab.yaml override")
    parser.add_argument("--prompt", default=None, help="Override cfg.extraction.prompt")
    parser.add_argument("--model-name", default="deepseek-v4-flash")
    parser.add_argument("--base-urls", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rpm", type=int, default=0)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--reuse-store",
        default=None,
        help="Fingerprint of an already-populated concept store to use as-is; "
        "skips extraction entirely (e.g. a v1 import).",
    )
    parser.add_argument(
        "--embed-backend", default=None, help="Override cfg.cluster.embed_backend"
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding calls; use whatever vectors are already on disk "
        "and warn about how many needed texts have none.",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Default: output/vocab/<tag>"
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--probe-set", default=DEFAULT_PROBE_SET)
    parser.add_argument("--old-cache", default=DEFAULT_OLD_CACHE)
    parser.add_argument(
        "--only-probe",
        action="store_true",
        help="Extract + probe report only; skip the offline lock-in checks.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Fake extractor, no LLM")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


class _RateLimiter:
    """Blocks so that no more than `rpm` acquisitions happen in any 60s
    window. A hosted quota is enforced per minute, so pacing beats absorbing
    429s. Copied from ``extract_elements.py``."""

    def __init__(self, rpm: int) -> None:
        self.rpm = max(0, rpm)
        self._lock = threading.Lock()
        self._stamps: deque[float] = deque()

    def acquire(self) -> None:
        if not self.rpm:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._stamps and now - self._stamps[0] >= 60.0:
                    self._stamps.popleft()
                if len(self._stamps) < self.rpm:
                    self._stamps.append(now)
                    return
                wait = 60.0 - (now - self._stamps[0])
            time.sleep(max(wait, 0.05))


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _union_papers(
    grouped: dict[str, list[PaperRecord]], topic_ids: list[str]
) -> list[PaperRecord]:
    seen: dict[str, PaperRecord] = {}
    for topic_id in topic_ids:
        for paper in grouped.get(topic_id, ()):
            seen.setdefault(paper.paper_id, paper)
    return sorted(seen.values(), key=lambda p: (p.month, p.paper_id))


def _default_cutoffs(min_month: str, end_month: str, horizon_months: int) -> list[str]:
    end_cutoff = add_months(end_month, -horizon_months)
    if month_to_index(end_cutoff) < month_to_index(min_month):
        return []
    cutoffs = []
    month = min_month
    while month_to_index(month) <= month_to_index(end_cutoff):
        cutoffs.append(month)
        month = add_months(month, 1)
    return cutoffs


def _load_probe_set(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        print(
            f"probe set not found at {path}; probe rows will be empty "
            "(run vocab_probe_select.py first)",
            file=sys.stderr,
        )
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    topics = payload.get("topics") if isinstance(payload, dict) else None
    return dict(topics) if isinstance(topics, dict) else {}


def _load_old_records(path_str: str) -> dict[str, ExtractionRecord]:
    path = _resolve(path_str)
    try:
        cache = ElementCache.locate(path)
    except FileNotFoundError:
        print(
            f"v1 element cache not found at {path}; probe 'old' column will be empty",
            file=sys.stderr,
        )
        return {}
    return cache.load()


def _open_store(cache_dir: Path, fingerprint: str, input_dir: Path) -> ConceptStore:
    """Open the concept store and warn (not fail) if it was built from a
    different corpus snapshot -- records are keyed by paper id, so that is
    only unsafe if the ids stopped meaning the same papers."""
    store = ConceptStore(cache_dir, fingerprint)
    manifest_path = store.dir / "manifest.json"
    if manifest_path.exists():
        stored_corpus = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "corpus_fingerprint"
        )
        current_corpus = corpus_fingerprint(input_dir)
        if stored_corpus and stored_corpus != current_corpus:
            print(
                f"WARNING: store built from corpus {stored_corpus}, now {current_corpus}",
                file=sys.stderr,
            )
    return store


def _summary(records: Mapping[str, ConceptRecord]) -> dict[str, float]:
    ok = sum(1 for r in records.values() if r.ok)
    failed = len(records) - ok
    return {
        "records": len(records),
        "ok": ok,
        "failed": failed,
        "failure_rate": round(failed / len(records), 4) if records else 0.0,
    }


def _run_extraction(
    papers: list[PaperRecord],
    store: ConceptStore,
    existing: dict[str, ConceptRecord],
    args: argparse.Namespace,
    caller: TextCaller | None,
    prompt: PromptPair,
    ext_cfg: ExtractionConfig,
    fingerprint: str,
) -> None:
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

    throttle = _RateLimiter(args.rpm)

    def _one(paper: PaperRecord) -> ConceptRecord | None:
        if caller is None:
            return fake_extraction(paper, ext_cfg.aliases, fingerprint)
        throttle.acquire()
        return extract_paper(paper, caller, prompt, ext_cfg, fingerprint)

    failures = deferred = 0
    batch: list[ConceptRecord] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one, p): p for p in todo}
        for future in tqdm(as_completed(futures), total=len(futures), unit="paper"):
            record = future.result()
            if record is None:
                # Service was busy; leave the paper uncached so a re-run picks
                # it up instead of burning it on an empty reply.
                deferred += 1
                continue
            batch.append(record)
            if not record.ok:
                failures += 1
            if len(batch) >= _APPEND_BATCH:
                store.append(batch)
                batch = []
    if batch:
        store.append(batch)
    print(
        f"extracted {len(todo) - deferred} papers, {failures} unparseable, "
        f"{deferred} deferred (rate limit / timeout -- re-run to pick them up)",
        flush=True,
    )


def _run_embed(
    store: ConceptStore,
    records: Mapping[str, ConceptRecord],
    cfg: VocabConfig,
    embed_backend_override: str | None,
    dry_run: bool,
    skip_embed: bool,
) -> Mapping[str, Sequence[float]]:
    safe_name = "".join(
        ch if ch.isalnum() or ch in "-._" else "_" for ch in cfg.cluster.embed_model
    )
    vector_store = VectorStore(store.vectors_dir / f"{safe_name}.json")
    texts = all_texts(records.values())
    if skip_embed:
        missing = vector_store.missing(texts)
        print(
            f"{len(texts)} unique concept texts, {len(vector_store)} vectors on "
            f"disk, {len(missing)} missing (--skip-embed: no embedding call; "
            "missing texts become their own leaders in build.py)",
            flush=True,
        )
        return vector_store.view()
    backend = embed_backend_override or cfg.cluster.embed_backend
    if dry_run and not embed_backend_override:
        backend = HASH_BACKEND
    embedder = make_embedder(backend, cfg.cluster.embed_model)
    text_map = {t: t for t in texts}
    print(
        f"{len(texts)} unique concept texts, {len(vector_store)} already embedded",
        flush=True,
    )
    added = vector_store.ensure(texts, text_map, embedder)
    print(f"embedded {added} new texts with {embedder.model_name}", flush=True)
    return vector_store.view()


def _old_labels_for(
    paper_id: str, old_records: Mapping[str, ExtractionRecord]
) -> Mapping[str, Sequence[str]] | None:
    record = old_records.get(paper_id)
    if record is None or not record.ok:
        return None
    return {
        "themes": list(record.themes),
        "domains": list(record.domains),
        "methods": list(record.methods),
    }


def _build_probe_rows(
    topic_id: str,
    probe_set: Mapping[str, list[dict[str, str]]],
    papers_by_id: Mapping[str, PaperRecord],
    records: Mapping[str, ConceptRecord],
    old_records: Mapping[str, ExtractionRecord],
) -> list[ProbeRow]:
    rows: list[ProbeRow] = []
    for entry in probe_set.get(topic_id, []):
        paper_id = str(entry.get("paper_id", ""))
        paper = papers_by_id.get(paper_id)
        if paper is None:
            print(
                f"  WARNING: probe paper {paper_id} not in the loaded corpus "
                f"for {topic_id}; skipping its probe row",
                file=sys.stderr,
            )
            continue
        rows.append(
            ProbeRow(
                paper=paper,
                new=records.get(paper_id),
                old=_old_labels_for(paper_id, old_records),
                reason=str(entry.get("reason", "")),
            )
        )
    return rows


def _nanmean(values: Sequence[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else math.nan


def _summarize_windows(window_values: list[Mapping[str, float]]) -> dict[str, float]:
    keys: set[str] = set()
    for values in window_values:
        keys.update(values)
    return {
        key: _nanmean([v.get(key, math.nan) for v in window_values]) for key in keys
    }


def _print_summary_table(summary: Mapping[str, float]) -> None:
    print("\nsummary (nan-aware mean over topics x cutoffs):")
    for key in sorted(summary):
        value = summary[key]
        shown = "n/a" if math.isnan(value) else f"{value:.4f}"
        print(f"  {key:<30}{shown:>10}")


def _run_topic_cutoffs(
    *,
    topic_id: str,
    cutoffs: list[str],
    grouped: Mapping[str, list[PaperRecord]],
    records: Mapping[str, ConceptRecord],
    vectors: Mapping[str, Sequence[float]],
    cfg: VocabConfig,
    fingerprint: str,
    probe_rows: list[ProbeRow],
    output_dir: Path,
    only_probe: bool,
) -> list[Mapping[str, float]]:
    topic_papers = grouped.get(topic_id, [])
    topic_dir = output_dir / topic_id
    prev = None
    window_values: list[Mapping[str, float]] = []
    for cutoff in cutoffs:
        train, future, _end_month, _end_date = split_train_future_by_cutoff(
            topic_papers, cutoff_month=cutoff, horizon_months=cfg.checks.horizon_months
        )
        if not train or not future:
            print(
                f"  WARNING: skipping {topic_id}@{cutoff} "
                f"(train={len(train)}, future={len(future)})",
                file=sys.stderr,
            )
            continue

        cutoff_date = month_start_date(cutoff)
        vocab = build_vocabulary(
            topic_id=topic_id,
            cutoff_month=cutoff,
            cutoff_date=cutoff_date,
            train_papers=train,
            records=records,
            vectors=vectors,
            cfg=cfg,
        )

        if only_probe:
            checks = _EMPTY_CHECKS
            stab = math.nan
            n_future_with_records = 0
        else:
            train_records = [
                records[p.paper_id] for p in train if p.paper_id in records
            ]
            future_records = [
                records[p.paper_id] for p in future if p.paper_id in records
            ]
            checks = run_checks(
                vocab=vocab,
                train_records=train_records,
                future_records=future_records,
                vectors=vectors,
                cfg=cfg,
            )
            stab = stability(prev, vocab)
            n_future_with_records = len(future_records)
        prev = vocab

        report_text = render_vocab_report(vocab=vocab, checks=checks, probe=probe_rows)
        topic_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(topic_dir / f"{cutoff}.md", report_text)

        payload: dict[str, object] = {
            **checks.values,
            "stability": stab,
            "n_train": len(train),
            "n_train_with_records": vocab.n_with_records,
            "n_future": len(future),
            "n_future_with_records": n_future_with_records,
            "config_sha": cfg.sha,
            "prompt_fingerprint": fingerprint,
            "topic_id": topic_id,
            "cutoff_month": cutoff,
        }
        atomic_write_text(
            topic_dir / f"{cutoff}.json", json.dumps(payload, indent=2, sort_keys=True)
        )
        window_values.append({**checks.values, "stability": stab})
    return window_values


def main() -> int:
    args = parse_args()
    input_dir = _resolve(args.input_dir)
    papers, _topics, grouped = load_papers_and_topics(
        input_dir, args.start_month, args.end_month
    )

    topic_ids = [t.strip() for t in args.topics.split(",") if t.strip()]
    if not topic_ids:
        print("--topics must name at least one topic id", file=sys.stderr)
        return 2
    unknown = [t for t in topic_ids if t not in grouped]
    if unknown:
        print(f"Unknown topic id(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    probe_set = _load_probe_set(_resolve(args.probe_set))
    old_records = _load_old_records(args.old_cache)

    scoped = _union_papers(grouped, topic_ids)
    if args.only_probe:
        probe_ids = {
            str(entry.get("paper_id", ""))
            for topic_id in topic_ids
            for entry in probe_set.get(topic_id, [])
        }
        scoped = [p for p in scoped if p.paper_id in probe_ids]
        if not scoped:
            print(
                "--only-probe: no probe papers found for the selected topics "
                "(run vocab_probe_select.py first)",
                file=sys.stderr,
            )
            return 2
    if args.limit is not None:
        scoped = scoped[: args.limit]

    cfg = load_vocab_config(args.config)

    if args.reuse_store:
        fingerprint = args.reuse_store
        store = _open_store(_resolve(args.cache_dir), fingerprint, input_dir)
        print(f"concept store: {store.dir} (reused, extraction skipped)", flush=True)
    else:
        prompt_name = args.prompt or cfg.extraction.prompt
        prompt = load_prompt(prompt_name)
        model = FAKE_MODEL if args.dry_run else args.model_name
        fingerprint = extraction_fingerprint(
            prompt, model, cfg.schema_version, cfg.extraction.temperature
        )

        store = _open_store(_resolve(args.cache_dir), fingerprint, input_dir)
        store.write_manifest(
            {
                "model": model,
                "prompt": prompt_name,
                "config_sha": cfg.sha,
                "corpus_fingerprint": corpus_fingerprint(input_dir),
            }
        )
        print(f"concept store: {store.dir}", flush=True)

        existing = store.load()
        caller: TextCaller | None = None
        if not args.dry_run:
            if args.base_urls:
                caller = callers_for_base_urls(
                    args.model_name, args.base_urls.split(",")
                )
            else:
                caller = caller_for_model(args.model_name)
        _run_extraction(
            scoped, store, existing, args, caller, prompt, cfg.extraction, fingerprint
        )

    records = store.load()
    print("summary:", _summary(records), flush=True)

    vectors = _run_embed(
        store, records, cfg, args.embed_backend, args.dry_run, args.skip_embed
    )

    if args.cutoffs:
        cutoffs = sorted(
            {c.strip() for c in args.cutoffs.split(",") if c.strip()},
            key=month_to_index,
        )
    else:
        cutoffs = _default_cutoffs(
            args.min_cutoff_month, args.end_month, cfg.checks.horizon_months
        )
    if not cutoffs:
        print(
            "no cutoffs to run (check --min-cutoff-month / --end-month)",
            file=sys.stderr,
        )
        return 2

    output_dir = (
        _resolve(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "output" / "vocab" / args.tag
    )
    papers_by_id = {p.paper_id: p for p in papers}

    all_windows: list[Mapping[str, float]] = []
    processed_topics: set[str] = set()
    processed_cutoffs: set[str] = set()
    for topic_id in topic_ids:
        probe_rows = _build_probe_rows(
            topic_id, probe_set, papers_by_id, records, old_records
        )
        windows = _run_topic_cutoffs(
            topic_id=topic_id,
            cutoffs=cutoffs,
            grouped=grouped,
            records=records,
            vectors=vectors,
            cfg=cfg,
            fingerprint=fingerprint,
            probe_rows=probe_rows,
            output_dir=output_dir,
            only_probe=args.only_probe,
        )
        if windows:
            processed_topics.add(topic_id)
            processed_cutoffs.update(cutoffs)
        all_windows.extend(windows)

    summary = _summarize_windows(all_windows)
    _print_summary_table(summary)

    if args.only_probe:
        print("--only-probe: skipping the ledger row (checks were not computed)")
    elif processed_topics:
        row = render_ledger_row(
            tag=args.tag,
            prompt_fingerprint=fingerprint,
            cfg=cfg,
            summary=summary,
            n_topics=len(processed_topics),
            n_cutoffs=len(processed_cutoffs),
        )
        ledger_path = output_dir.parent / "ledger.md"
        append_ledger(ledger_path, row)
        print(f"appended ledger row -> {ledger_path}")
    else:
        print("no window ran to completion; ledger row skipped", file=sys.stderr)

    summary_payload = {
        "tag": args.tag,
        "n_topics": len(processed_topics),
        "n_cutoffs": len(processed_cutoffs),
        "config_sha": cfg.sha,
        "prompt_fingerprint": fingerprint,
        **summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary_payload, indent=2, sort_keys=True),
    )
    print(f"\nwrote reports for {len(processed_topics)} topic(s) -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
