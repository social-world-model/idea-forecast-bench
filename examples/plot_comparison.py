"""Draw predictor_llm vs keyword_trend comparison charts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from examples.reeval_threshold import recompute

# ── load data ──────────────────────────────────────────────────────────────────
llm_data = json.loads(Path("/tmp/reeval_t06_patched.json").read_text())
kw_data  = json.loads(Path("/tmp/keyword_trend_reeval.json").read_text())

THRESHOLDS = [0.5, 0.6]

llm_results = {t: recompute(llm_data, t) for t in THRESHOLDS}
kw_results  = {t: recompute(kw_data,  t) for t in THRESHOLDS}

TOPIC_ORDER = [
    "diffusion_language_model",
    "multimodal_visual_reasoning",
    "gui_computer_use_web_agent",
    "optimizer",
    "time_series_forecasting",
]
TOPIC_LABELS = {
    "diffusion_language_model":   "Diffusion LM",
    "multimodal_visual_reasoning": "Multimodal",
    "gui_computer_use_web_agent":  "GUI Agent",
    "optimizer":                   "Optimizer",
    "time_series_forecasting":     "Time-series",
}

COLORS = {
    "llm":  {"0.5": "#2563eb", "0.6": "#93c5fd"},
    "kw":   {"0.5": "#dc2626", "0.6": "#fca5a5"},
}

# ── fig layout: 2 rows (hit@k, MRR) × 2 cols (t=0.5, t=0.6) ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("predictor_llm vs keyword_trend  |  Embedding-based Evaluation",
             fontsize=14, fontweight="bold", y=0.98)

metrics = [("avg_hit_at_k", "Hit@k"), ("avg_mrr", "MRR")]

x = np.arange(len(TOPIC_ORDER))
w = 0.35

for row, (metric_key, metric_label) in enumerate(metrics):
    for col, t in enumerate(THRESHOLDS):
        ax = axes[row][col]

        llm_vals = [llm_results[t]["topics"][tid].get(metric_key, 0) for tid in TOPIC_ORDER]
        kw_vals  = [kw_results[t]["topics"][tid].get(metric_key, 0)  for tid in TOPIC_ORDER]

        b1 = ax.bar(x - w/2, llm_vals, w, label="predictor_llm",
                    color=COLORS["llm"][str(t)], zorder=3)
        b2 = ax.bar(x + w/2, kw_vals,  w, label="keyword_trend",
                    color=COLORS["kw"][str(t)],  zorder=3)

        # value labels
        for bar in b1:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)
        for bar in b2:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

        # aggregate line
        llm_agg = llm_results[t]["aggregate"][metric_key]
        kw_agg  = kw_results[t]["aggregate"][metric_key]
        ax.axhline(llm_agg, color=COLORS["llm"][str(t)], linewidth=1.5,
                   linestyle="--", alpha=0.8, zorder=4)
        ax.axhline(kw_agg,  color=COLORS["kw"][str(t)],  linewidth=1.5,
                   linestyle="--", alpha=0.8, zorder=4)

        ax.set_title(f"{metric_label}  (t = {t})", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([TOPIC_LABELS[tid] for tid in TOPIC_ORDER],
                            rotation=18, ha="right", fontsize=9)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

        if row == 0 and col == 1:
            ax.legend(fontsize=9, loc="upper right")

# ── aggregate bar (bottom strip) ──────────────────────────────────────────────
fig.tight_layout(rect=[0, 0.10, 1, 1])

agg_ax = fig.add_axes([0.12, 0.01, 0.76, 0.08])
agg_labels = [f"hit@k\nt={t}" for t in THRESHOLDS] + [f"MRR\nt={t}" for t in THRESHOLDS]
agg_llm = [llm_results[t]["aggregate"]["avg_hit_at_k"] for t in THRESHOLDS] + \
          [llm_results[t]["aggregate"]["avg_mrr"]       for t in THRESHOLDS]
agg_kw  = [kw_results[t]["aggregate"]["avg_hit_at_k"]  for t in THRESHOLDS] + \
          [kw_results[t]["aggregate"]["avg_mrr"]        for t in THRESHOLDS]

xa = np.arange(4)
agg_ax.bar(xa - 0.2, agg_llm, 0.35, color=["#2563eb","#93c5fd","#2563eb","#93c5fd"], zorder=3, label="predictor_llm")
agg_ax.bar(xa + 0.2, agg_kw,  0.35, color=["#dc2626","#fca5a5","#dc2626","#fca5a5"], zorder=3, label="keyword_trend")
for i, (lv, kv) in enumerate(zip(agg_llm, agg_kw)):
    agg_ax.text(xa[i]-0.2, lv+0.02, f"{lv:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    if kv > 0.005:
        agg_ax.text(xa[i]+0.2, kv+0.02, f"{kv:.3f}", ha="center", va="bottom", fontsize=8)
agg_ax.set_xticks(xa)
agg_ax.set_xticklabels(agg_labels, fontsize=9)
agg_ax.set_ylim(0, 1.3)
agg_ax.set_title("Aggregate (weighted avg across 66 windows)", fontsize=9)
agg_ax.yaxis.grid(True, linestyle=":", alpha=0.4)
agg_ax.set_axisbelow(True)
p1 = mpatches.Patch(color="#2563eb", label="predictor_llm  t=0.5")
p2 = mpatches.Patch(color="#93c5fd", label="predictor_llm  t=0.6")
p3 = mpatches.Patch(color="#dc2626", label="keyword_trend  t=0.5")
p4 = mpatches.Patch(color="#fca5a5", label="keyword_trend  t=0.6")
agg_ax.legend(handles=[p1, p2, p3, p4], fontsize=8, loc="upper right", ncol=2)

out = "/tmp/comparison_chart.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
