#!/usr/bin/env python3
"""Import the v1 (old-schema) element cache into a v2 concept-vocabulary
store, so a "v1" row can be built in the vocabulary ledger under the exact
same checks as v2: same clustering, same coverage/stability metrics, same
config. Makes no LLM or embedding calls -- v1's own themes/domains/methods
become v2's problems/objects/mechanisms (parent always ""), and v1's own
vectors are reused, stripped of their `type:` key prefix.

Run this once, then feed the result to ``vocab_build.py --reuse-store
v1import --skip-embed``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from idea_forecast_bench.atomic import atomic_write_text
from idea_forecast_bench.combinatorial.cache import ElementCache
from idea_forecast_bench.combinatorial.embeddings import VectorStore
from idea_forecast_bench.combinatorial.types import ExtractionRecord
from idea_forecast_bench.vocab.build import all_texts
from idea_forecast_bench.vocab.config import load_vocab_config
from idea_forecast_bench.vocab.store import ConceptStore
from idea_forecast_bench.vocab.types import RECORD_OK, ConceptRecord, Term

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OLD_CACHE = "output/pilot/elements/a176b6f85981"
DEFAULT_CACHE_DIR = "output/vocab/cache"

#: Constant, not derived from a prompt/model: this store holds exactly one
#: thing -- v1 records pushed through the v2 schema -- so it needs no
#: extraction fingerprint of its own.
FINGERPRINT = "v1import"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-cache", default=DEFAULT_OLD_CACHE)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--config", default=None, help="vocab.yaml override")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _terms(texts: tuple[str, ...]) -> tuple[Term, ...]:
    # v1 had no parent/child distinction, so every term is its own top level.
    return tuple(Term(text=t, parent="") for t in texts if t)


def convert_record(old: ExtractionRecord) -> ConceptRecord:
    """v1's themes/domains/methods -> v2's problems/objects/mechanisms.
    Only called for ``old.ok`` records; frames have no v2 slot and are
    dropped."""
    return ConceptRecord(
        paper_id=old.paper_id,
        published_date=old.published_date,
        status=RECORD_OK,
        objects=_terms(old.domains),
        mechanisms=_terms(old.methods),
        problems=_terms(old.themes),
        model=f"v1:{old.model}",
        fingerprint=FINGERPRINT,
        extracted_at=old.extracted_at,
    )


def _import_records(
    old_records: Mapping[str, ExtractionRecord], store: ConceptStore
) -> dict[str, ConceptRecord]:
    existing = store.load()
    if existing:
        print(
            f"{store.dir / 'records.jsonl'} already has {len(existing)} "
            "records; skipping re-import",
            flush=True,
        )
        return existing
    converted = [convert_record(r) for r in old_records.values() if r.ok]
    n = store.append(converted)
    print(f"imported {n} ok v1 records -> {store.dir}", flush=True)
    return store.load()


def _import_vectors(
    old_cache: ElementCache,
    store: ConceptStore,
    records: Mapping[str, ConceptRecord],
    embed_model: str,
) -> None:
    safe_name = "".join(
        ch if ch.isalnum() or ch in "-._" else "_" for ch in embed_model
    )
    vector_path = store.vectors_dir / f"{safe_name}.json"
    if vector_path.exists() and vector_path.stat().st_size > 0:
        print(f"{vector_path} already exists; skipping vector rebuild", flush=True)
        return

    wanted = all_texts(records.values())
    old_vector_path = old_cache.vectors_path(embed_model)
    old_vectors = VectorStore(old_vector_path)  # loads the ~128 MB file once
    if old_vectors.embedder_name and old_vectors.embedder_name != embed_model:
        raise ValueError(
            f"{old_vector_path} holds vectors from {old_vectors.embedder_name!r}, "
            f"not {embed_model!r}"
        )
    print(
        f"{len(wanted)} texts need a vector; scanning {len(old_vectors)} old "
        f"vectors ({old_vector_path})",
        flush=True,
    )

    new_vectors: dict[str, list[float]] = {}
    for key, vec in old_vectors.view().items():
        _old_type, sep, text = key.partition(":")
        if not sep or text not in wanted or text in new_vectors:
            continue
        new_vectors[text] = list(vec)

    missing = len(wanted) - len(new_vectors)
    payload = json.dumps({"embedder": embed_model, "vectors": new_vectors})
    atomic_write_text(vector_path, payload)
    print(
        f"wrote {len(new_vectors)} vectors -> {vector_path} "
        f"({missing} wanted texts have no v1 vector)",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    old_cache_dir = _resolve(args.old_cache)
    old_cache = ElementCache.locate(old_cache_dir)
    old_records = old_cache.load()
    print(
        f"v1 element cache: {old_cache.directory} ({len(old_records)} records)",
        flush=True,
    )

    cfg = load_vocab_config(args.config)
    store = ConceptStore(_resolve(args.cache_dir), FINGERPRINT)

    records = _import_records(old_records, store)
    store.write_manifest(
        {
            "source": str(old_cache.directory),
            "note": "v1 element cache (themes/domains/methods) imported as v2 "
            "problems/objects/mechanisms, parent='' throughout, so v1 can be "
            "measured under the exact v2 vocabulary/checks pipeline.",
        }
    )
    _import_vectors(old_cache, store, records, cfg.cluster.embed_model)

    ok = sum(1 for r in records.values() if r.ok)
    print(
        f"store ready: {len(records)} records ({ok} ok) under fingerprint "
        f"{FINGERPRINT!r} -> {store.dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
