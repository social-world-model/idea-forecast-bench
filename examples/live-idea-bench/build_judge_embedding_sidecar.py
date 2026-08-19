#!/usr/bin/env python3
"""Convert a flat {paper_id: vector} embedding dump into the embeddings sidecar
that llm_judge_eval.py's RunState reads, so judge-eval reuses pre-computed paper
vectors instead of re-embedding (and re-paying) them.

RunState stores paper vectors under keys ``f"{embed_fp}__{paper_id}"`` where
``embed_fp = sha256(embed_model)[:12]`` and reads them back the same way. The
sidecar file MUST be named ``<state_path_stem>.embeddings.json`` and live next
to the state file; with the default ``--state-file`` that is
``<output>.state.embeddings.json``.

This script only writes the SIDECAR (paper vectors). It deliberately does NOT
create the main state file — let judge-eval create a fresh one so it stamps the
current judge+embed fingerprints itself. As long as ``--embed-model`` matches
the model used here, the fresh state's embed_fp equals this sidecar's key prefix
and the vectors are hit (no re-embedding).

Usage:
    python build_judge_embedding_sidecar.py \
        --embeddings ~/Downloads/baselines/paper_embeddings.json \
        --embed-model voyage-3-large \
        --out data/raw-208/summary_208_judged.state.embeddings.json

    # then run judge-eval with a matching --output / --embed-model so the
    # default state path is <output> -> <output>.state -> sidecar above:
    #   --output data/raw-208/summary_208_judged.json
    #   --embed-model voyage-3-large
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _embed_fingerprint(embed_model: str) -> str:
    """Must match llm_judge_eval._embed_fingerprint exactly."""
    return hashlib.sha256(embed_model.encode()).hexdigest()[:12]


def _extract_flat_map(data) -> dict:
    """Accept either {paper_id: vec} or {"paper_embeddings": {paper_id: vec}}."""
    if isinstance(data, dict) and "paper_embeddings" in data and isinstance(data["paper_embeddings"], dict):
        inner = data["paper_embeddings"]
        # Guard against an already-namespaced dump (keys contain "__").
        if inner and "__" in next(iter(inner)):
            raise SystemExit(
                "Input already looks namespaced (keys contain '__'); expected a "
                "flat {paper_id: vector} map. Refusing to double-prefix."
            )
        return inner
    if isinstance(data, dict):
        return data
    raise SystemExit("Unsupported embeddings JSON shape; expected an object.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="flat {paper_id: vector} JSON")
    ap.add_argument("--embed-model", default="voyage-3-large",
                    help="must equal judge-eval's --embed-model so fingerprints match")
    ap.add_argument("--out", required=True,
                    help="sidecar path; should be <judge-eval --output>.state.embeddings.json")
    args = ap.parse_args()

    src = json.loads(Path(args.embeddings).read_text(encoding="utf-8"))
    flat = _extract_flat_map(src)
    fp = _embed_fingerprint(args.embed_model)

    # Sanity: vectors must be non-empty lists of numbers, all same dimension.
    dims = set()
    namespaced: dict[str, list] = {}
    skipped = 0
    for pid, vec in flat.items():
        if not isinstance(vec, list) or not vec:
            skipped += 1
            continue
        dims.add(len(vec))
        namespaced[f"{fp}__{pid}"] = vec
    if len(dims) > 1:
        raise SystemExit(f"Inconsistent vector dimensions in input: {sorted(dims)}")
    if not namespaced:
        raise SystemExit("No usable vectors found in input.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"paper_embeddings": namespaced}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Wrote {out_path}\n"
        f"  embed_model={args.embed_model}  embed_fp={fp}\n"
        f"  vectors={len(namespaced)}  dim={next(iter(dims))}  skipped={skipped}\n"
        f"  -> run judge-eval with --embed-model {args.embed_model} and an --output "
        f"whose default state path yields this sidecar name."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
