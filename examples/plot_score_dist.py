"""Plot best_score distribution for predictor_llm vs keyword_trend."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── collect all best_scores ────────────────────────────────────────────────────
def collect_scores(reeval_json: str) -> dict[str, list[float]]:
    """Returns {topic_id: [best_score, ...]} and 'all' key for aggregate."""
    data = json.loads(Path(reeval_json).read_text())
    topic_scores: dict[str, list[float]] = {}
    all_scores: list[float] = []
    for tid, tr in data["topic_results"].items():
        bt = tr.get("backtest")
        if not bt or not bt.get("windows"):
            continue
        scores = []
        for w in bt["windows"]:
            for ps in w["evaluation"].get("per_prediction_scores", []):
                scores.append(ps["best_score"])
        topic_scores[tid] = scores
        all_scores.extend(scores)
    topic_scores["__all__"] = all_scores
    return topic_scores

TOPIC_LABELS = {
    "diffusion_language_model":    "Diffusion LM",
    "multimodal_visual_reasoning": "Multimodal",
    "gui_computer_use_web_agent":  "GUI Agent",
    "optimizer":                   "Optimizer",
    "time_series_forecasting":     "Time-series",
    "__all__":                     "All topics",
}

llm_scores = collect_scores("/tmp/reeval_t06_patched.json")
kw_scores  = collect_scores("/tmp/keyword_trend_reeval.json")

TOPICS = ["__all__", "diffusion_language_model", "multimodal_visual_reasoning",
          "gui_computer_use_web_agent", "optimizer", "time_series_forecasting"]

BINS = np.linspace(0, 1, 41)   # 0.025-wide bins

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Best cosine similarity score distribution\n(each bar = one prediction's best match among future papers)",
             fontsize=13, fontweight="bold")

for ax, tid in zip(axes.flat, TOPICS):
    ls = llm_scores.get(tid, [])
    ks = kw_scores.get(tid, [])

    ax.hist(ls, bins=BINS, alpha=0.65, color="#2563eb", label=f"predictor_llm (n={len(ls)})", density=True)
    ax.hist(ks, bins=BINS, alpha=0.65, color="#dc2626", label=f"keyword_trend (n={len(ks)})", density=True)

    # mean lines
    if ls:
        ax.axvline(np.mean(ls), color="#2563eb", linewidth=1.5, linestyle="-", alpha=0.8)
    if ks:
        ax.axvline(np.mean(ks), color="#dc2626", linewidth=1.5, linestyle="-", alpha=0.8)

    ax.set_title(TOPIC_LABELS[tid], fontsize=11)
    ax.set_xlabel("cosine similarity", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_xlim(0, 1)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4)

    # mean annotations — bottom-right to avoid legend
    if ls:
        ax.text(0.98, 0.93, f"μ_llm={np.mean(ls):.3f}", transform=ax.transAxes,
                fontsize=8, color="#2563eb", fontweight="bold", ha="right")
    if ks:
        ax.text(0.98, 0.84, f"μ_kw ={np.mean(ks):.3f}", transform=ax.transAxes,
                fontsize=8, color="#dc2626", fontweight="bold", ha="right")

    # legend — upper left
    ax.legend(fontsize=8, loc="upper left")

# threshold lines + labels after ylim is finalised
for ax, tid in zip(axes.flat, TOPICS):
    ymax = ax.get_ylim()[1]
    for t, style, label in [(0.5, "--", "match threshold 0.5"),
                             (0.6, ":",  "match threshold 0.6")]:
        ax.axvline(t, color="gray", linewidth=1.2, linestyle=style, alpha=0.7, zorder=0)
        ax.text(t + 0.01, ymax * 0.55, label, fontsize=7, color="gray",
                rotation=90, va="bottom")

fig.tight_layout()
out = "/tmp/score_distribution.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
