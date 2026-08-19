#!/usr/bin/env python
"""Phase-1 M1 spot check.

- Streams the existing hindsight JSONL (data/topic_hindsight/hindsight_samples.jsonl)
  through `augment_hindsight_rows`, producing data/topic_hindsight/dz.jsonl.
- Reports:
    * total rows, kept (train-window), dropped (test-window / missing cutoff)
    * closed operator counts + "other" ratio  (Decision 2 health signal)
    * 10-row qualitative spot-check (paper title + b/o/g/closed)
- Writes reports/m1_spot_check.md.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from forecaster.foresight.dz import augment_hindsight_rows, load_dz_rows
from forecaster.foresight.operators import load_operator_inventory

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(REPO_ROOT / "data/topic_hindsight/hindsight_samples.jsonl"))
    ap.add_argument("--output", default=str(REPO_ROOT / "data/topic_hindsight/dz.jsonl"))
    ap.add_argument("--summary", default=str(REPO_ROOT / "data/topic_hindsight/dz_summary.json"))
    ap.add_argument("--report", default=str(REPO_ROOT / "reports/m1_spot_check.md"))
    ap.add_argument("--n-spot", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    inventory = load_operator_inventory()
    summary = augment_hindsight_rows(
        args.input,
        args.output,
        inventory=inventory,
        papers_by_id=None,            # corpus-free pass; M1 only needs operator mapping
        summary_path=args.summary,
    )
    rows = load_dz_rows(args.output)
    rng = random.Random(args.seed)
    spot = rng.sample(rows, min(args.n_spot, len(rows)))

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# M1 spot check — Foresight Phase 1\n")
    lines.append("## Window filtering\n")
    lines.append(f"- input rows: **{summary.total_rows}**")
    lines.append(f"- kept (train window): **{summary.train_window_rows}**")
    lines.append(f"- dropped (test window, cutoff ≥ 2024-10-01): **{summary.dropped_test_window}**")
    lines.append(f"- dropped (missing/malformed cutoff): **{summary.dropped_missing_cutoff}**\n")

    lines.append("## Closed-operator distribution\n")
    lines.append("| closed id | count | share |")
    lines.append("|---|---:|---:|")
    total = sum(summary.operator_closed_counts.values()) or 1
    for op_id, count in sorted(summary.operator_closed_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{op_id}` | {count} | {count / total:.1%} |")
    lines.append("")
    lines.append(f"**`other` ratio: {summary.other_ratio:.1%}**\n")
    if summary.other_ratio < 0.10:
        verdict = "✅ <10% — safe to drop/merge `other` rows for closed-4 training."
    elif summary.other_ratio > 0.20:
        verdict = "🚨 >20% — closed 4-set may not cover the data; escalate before Phase 4."
    else:
        verdict = "⚠️ 10–20% — borderline; carry `other` through Phase 2 rubric AUC before deciding."
    lines.append(verdict + "\n")

    lines.append(f"## 10-row qualitative spot-check (seed={args.seed})\n")
    for i, row in enumerate(spot, 1):
        tz = row.get("target_z", {})
        lines.append(
            f"### {i}. `{row.get('topic_id')}/{row.get('episode_id')}` "
            f"@ {row.get('cutoff_t')} → {row.get('source_future_id')}"
        )
        lines.append(f"- **future title**: {row.get('extra', {}).get('future_paper_title', '')}")
        lines.append(f"- **base_direction**: {tz.get('base_direction', '')}")
        lines.append(f"- **operator (free-text → closed)**: `{tz.get('operator', '')}` → `{row.get('operator_closed', '')}`")
        gap = (tz.get("gap") or "").replace("\n", " ")
        if len(gap) > 280:
            gap = gap[:277] + "…"
        lines.append(f"- **gap**: {gap}\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")
    print(json.dumps(summary.to_json(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
