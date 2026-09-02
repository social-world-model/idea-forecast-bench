"""Build foresight future/history indices for all dz cutoffs → artifact dir.
Local embedder (sentence-transformer allenai-specter). Run on one GPU.

Usage:
    python examples/forecaster/build_indices.py \
        --papers-dir data/csml/raw_markdown \
        --dz data/topic_hindsight/dz.jsonl \
        --art output/foresight_artifacts
"""

import argparse
import datetime
import json
import os
from pathlib import Path

from forecaster.foresight.indices import (
    SentenceTransformerEmbedder,
    build_cutoff_indices,
)
from idea_forecast_bench.papers import load_papers_from_markdown

_p = argparse.ArgumentParser(description="Build foresight cutoff indices.")
_p.add_argument(
    "--papers-dir",
    default=os.environ.get("IDEA_FORECAST_BENCH_PAPERS_DIR", "data/csml/raw_markdown"),
)
_p.add_argument("--dz", default="data/topic_hindsight/dz.jsonl")
_p.add_argument("--art", default="output/foresight_artifacts")
_p.add_argument("--start-month", default="2022-06")
_p.add_argument("--end-month", default="2024-12")
_p.add_argument("--embedder-model", default="sentence-transformers/allenai-specter")
_args = _p.parse_args()

PAPERS_DIR = _args.papers_dir
DZ = _args.dz
ART = Path(_args.art)
EMBEDDER_MODEL = _args.embedder_model  # AI2 scientific paper embedder

if not Path(DZ).is_file():
    raise SystemExit(
        f"[idx] dz file not found: {DZ} (generate it first; see forecaster/foresight/README.md)"
    )
if not Path(PAPERS_DIR).is_dir():
    raise SystemExit(
        f"[idx] papers dir not found: {PAPERS_DIR} (provide the corpus at this path, or pass --papers-dir)"
    )

# The GRPO episode dataset keys cutoffs as the FIRST day of the next period
# (e.g. dz "2023-03-31" -> dataset "2023-04-01"). The foresight reward looks up
# indices by the dataset's cutoff_date, so build/key indices with dz+1day.
with open(DZ) as _dz_fh:
    _dz_cutoffs = sorted(
        {json.loads(line)["cutoff_t"] for line in _dz_fh if line.strip()}
    )
cutoffs = [
    (datetime.date.fromisoformat(c) + datetime.timedelta(days=1)).isoformat()
    for c in _dz_cutoffs
]
print(f"[idx] dz_cutoffs={_dz_cutoffs}", flush=True)
print(f"[idx] dataset-aligned cutoffs (dz+1day)={cutoffs}", flush=True)
papers = load_papers_from_markdown(
    Path(PAPERS_DIR), start_month=_args.start_month, end_month=_args.end_month
)
print(f"[idx] loaded {len(papers)} papers", flush=True)
emb = SentenceTransformerEmbedder(model_name=EMBEDDER_MODEL)
(ART / "indices").mkdir(parents=True, exist_ok=True)
bundles = build_cutoff_indices(
    papers, cutoffs, horizon_months=3, embedder=emb, save_dir=str(ART / "indices")
)
print(f"[idx] built {len(bundles)} cutoff bundles -> {ART / 'indices'}", flush=True)
for c, b in bundles.items():
    print(f"[idx]   {c}: future={b.future.size} history={b.history.size}", flush=True)
print("[idx] DONE", flush=True)
