"""Dataset row builder for the GRPO trainer.

The training pipeline materializes one prompt row per (episode, hindsight innovation)
pair. This module turns those rows into the {prompt, ground_truth, extra_info}
shape that ``trl.GRPOTrainer`` consumes via the ``reward_funcs`` callable, with
``extra_info`` carrying everything ``compute_score`` needs to evaluate the
METHOD §3.3 verifiable rewards (evidence accuracy, operator adherence, coherence).
"""
from __future__ import annotations

import json
from typing import Any


def build_grpo_dataset_rows(
    prompt_rows: list[dict[str, Any]],
    *,
    data_source: str = "live_idea_bench",
) -> list[dict[str, Any]]:
    """Convert pipeline prompt rows to GRPO training rows.

    Each output row matches the shape ``trl.GRPOTrainer`` expects after the
    chat template is applied: ``prompt`` is the raw user message (the runner
    re-renders it with ``apply_chat_template(..., enable_thinking=True)`` for
    Qwen3.5), and ``extra_info`` is a JSON-serialized payload that the reward
    callable decodes back into ``PaperRecord`` lists + ``Innovation`` for
    scoring.
    """
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(prompt_rows):
        extra_info = {
            "row_index": index,
            "cutoff_month": row.get("cutoff_month", ""),
            "cutoff_date": row.get("cutoff_date", ""),
            "future_end_month": row.get("future_end_month", ""),
            "future_end_date": row.get("future_end_date", ""),
            "prompt_mode": row.get("prompt_mode", ""),
            "innovation": row.get("innovation", {}),
            "realization_config": row.get("realization_config", {}),
            "target_future_paper": row.get("target_future_paper", {}),
            "target_future_paper_id": row.get("target_future_paper_id", ""),
            "evidence_papers": row.get("evidence_papers", []),
            "train_papers": row.get("train_papers", []),
            "future_papers": row.get("future_papers", []),
        }
        rows.append(
            {
                "data_source": data_source,
                "prompt": str(row.get("prompt", "")),
                "ground_truth": "",
                "extra_info": json.dumps(extra_info, ensure_ascii=False),
            }
        )
    return rows
