from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from forecaster.realization.io import _write_jsonl


@dataclass
class VerlDatasetArtifacts:
    primary_path: Path
    parquet_path: Path
    preview_path: Path
    row_count: int
    parquet_ready: bool
    parquet_engine: str | None


def build_verl_dataset_rows(prompt_rows: list[dict[str, Any]], *, data_source: str = "live_idea_bench") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(prompt_rows):
        extra_info = {
            "row_index": index,
            "cutoff_month": row.get("cutoff_month", ""),
            "cutoff_date": row.get("cutoff_date", ""),
            "future_end_month": row.get("future_end_month", ""),
            "future_end_date": row.get("future_end_date", ""),
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


def _detect_parquet_engine() -> str | None:
    try:
        import pyarrow  # noqa: F401

        return "pyarrow"
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401

        return "fastparquet"
    except ImportError:
        return None


def write_verl_dataset(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    allow_preview_only: bool = False,
) -> VerlDatasetArtifacts:
    preview_path = path.with_suffix(".preview.jsonl")
    _write_jsonl(preview_path, rows)

    engine = _detect_parquet_engine()
    if engine is None:
        if not allow_preview_only:
            raise RuntimeError(
                "Writing veRL Parquet datasets requires pyarrow or fastparquet. "
                "Install pyarrow in the RL environment before running PPO/GRPO training."
            )
        return VerlDatasetArtifacts(
            primary_path=preview_path,
            parquet_path=path,
            preview_path=preview_path,
            row_count=len(rows),
            parquet_ready=False,
            parquet_engine=None,
        )

    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(rows)
    frame.to_parquet(path, engine=engine, index=False)
    return VerlDatasetArtifacts(
        primary_path=path,
        parquet_path=path,
        preview_path=preview_path,
        row_count=len(rows),
        parquet_ready=True,
        parquet_engine=engine,
    )
