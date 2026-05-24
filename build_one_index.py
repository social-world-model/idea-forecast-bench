"""Build the single missing foresight index cutoff (2024-07-01)."""
from pathlib import Path
from live_idea_bench.papers import load_papers_from_markdown
from forecaster.foresight.indices import SentenceTransformerEmbedder, build_cutoff_indices

PAPERS_DIR = "/home/max7/live_idea_bench_fenghai/live-idea-bench/data/csml_v2/raw_markdown"
ART = Path("output/foresight_artifacts")
EMBEDDER_MODEL = "sentence-transformers/allenai-specter"
CUTOFFS = ["2024-07-01"]  # the one truncated by the earlier GPU collision

print(f"[idx1] building cutoffs={CUTOFFS}", flush=True)
# Load only through 2024-09: the dz cutoff 2024-06-30 (keyed 2024-07-01) has a
# 3-month future window that would otherwise reach into 2024-10 (the held-out
# test window, FUTURE_WINDOW_HARD_LIMIT). Capping the corpus at 2024-09 keeps
# the future index to Aug/Sep 2024 — exactly what the episode builder sees under
# --end-month 2024-09 — and avoids the test-window leakage assertion.
papers = load_papers_from_markdown(Path(PAPERS_DIR), start_month="2022-06", end_month="2024-09")
print(f"[idx1] loaded {len(papers)} papers", flush=True)
emb = SentenceTransformerEmbedder(model_name=EMBEDDER_MODEL)
(ART / "indices").mkdir(parents=True, exist_ok=True)
bundles = build_cutoff_indices(papers, CUTOFFS, horizon_months=3, embedder=emb, save_dir=str(ART / "indices"))
for c, b in bundles.items():
    print(f"[idx1]   {c}: future={b.future.size} history={b.history.size}", flush=True)
print("[idx1] DONE", flush=True)
