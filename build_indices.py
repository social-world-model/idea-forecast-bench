"""Build foresight future/history indices for all dz cutoffs → artifact dir.
Local embedder (sentence-transformer all-MiniLM-L6-v2). Run on one GPU.
"""
import json, sys
from pathlib import Path
from live_idea_bench.papers import load_papers_from_markdown
from forecaster.foresight.indices import SentenceTransformerEmbedder, build_cutoff_indices

PAPERS_DIR = "/home/max7/live_idea_bench_fenghai/live-idea-bench/data/csml_v2/raw_markdown"
DZ = "data/topic_hindsight/dz.jsonl"
ART = Path("output/foresight_artifacts")
EMBEDDER_MODEL = "sentence-transformers/allenai-specter"  # AI2 scientific paper embedder

import datetime
# The GRPO episode dataset keys cutoffs as the FIRST day of the next period
# (e.g. dz "2023-03-31" -> dataset "2023-04-01"). The foresight reward looks up
# indices by the dataset's cutoff_date, so build/key indices with dz+1day.
_dz_cutoffs = sorted({json.loads(l)["cutoff_t"] for l in open(DZ) if l.strip()})
cutoffs = [(datetime.date.fromisoformat(c) + datetime.timedelta(days=1)).isoformat() for c in _dz_cutoffs]
print(f"[idx] dz_cutoffs={_dz_cutoffs}", flush=True)
print(f"[idx] dataset-aligned cutoffs (dz+1day)={cutoffs}", flush=True)
papers = load_papers_from_markdown(Path(PAPERS_DIR), start_month="2022-06", end_month="2024-12")
print(f"[idx] loaded {len(papers)} papers", flush=True)
emb = SentenceTransformerEmbedder(model_name=EMBEDDER_MODEL)
(ART / "indices").mkdir(parents=True, exist_ok=True)
bundles = build_cutoff_indices(
    papers, cutoffs, horizon_months=3, embedder=emb, save_dir=str(ART / "indices")
)
print(f"[idx] built {len(bundles)} cutoff bundles -> {ART/'indices'}", flush=True)
for c, b in bundles.items():
    print(f"[idx]   {c}: future={b.future.size} history={b.history.size}", flush=True)
print("[idx] DONE", flush=True)
