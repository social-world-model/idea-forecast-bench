from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from forecaster.models import strict_runtime_manifest_contract
from forecaster.realization.io import _write_json
from live_idea_bench.models import PaperRecord

if TYPE_CHECKING:  # pragma: no cover - typing-only helper alias
    from _typeshed import DataclassInstance


@dataclass
class PreparedRLContext:
    papers: list[PaperRecord]
    all_episodes: list[Any]
    selected_episodes: list[Any]
    prompt_rows: list[dict[str, Any]]
    shared_dir: Path
    episodes_path: Path
    prompt_rows_path: Path
    paper_lookup: dict[str, PaperRecord]
    config_fingerprint: str
    selected_split: str
    model_name: str
    realization_config: Any
    hindsight_samples: list[Any]
    similarity_config_path: str
    runtime_config_path: str | None
    episode_cache_root: Path
    shared_manifest_path: Path


@dataclass
class TrainerPreparedArtifacts:
    trainer_name: str
    output_dir: Path
    dataset_path: Path
    dataset_rows: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


class RLTrainerRunner(ABC):
    trainer_name: str
    default_config_filename: str
    backend_name: str = "unknown"

    @abstractmethod
    def prepare(
        self, common_context: PreparedRLContext, **kwargs: Any
    ) -> TrainerPreparedArtifacts:
        raise NotImplementedError

    @abstractmethod
    def train(
        self, prepared_artifacts: TrainerPreparedArtifacts, **kwargs: Any
    ) -> dict[str, Any]:
        raise NotImplementedError


def _serialize_for_fingerprint(value: Any) -> Any:
    if is_dataclass(value):
        # `is_dataclass` also accepts dataclass *classes*; only instances ever
        # reach here, and `asdict` would raise on a class anyway.
        return asdict(cast("DataclassInstance", value))
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {
            str(key): _serialize_for_fingerprint(inner)
            for key, inner in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_serialize_for_fingerprint(item) for item in value]
    return value


def build_config_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        _serialize_for_fingerprint(payload), ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_policy_manifest(
    *,
    trainer: str,
    base_model_name: str,
    inference_model_name: str,
    init_policy_path: str | None,
    predictor_config: str,
    trainer_config_path: str,
    output_dir: Path,
    dataset_path: Path,
    dataset_size: int,
    dry_run: bool,
    selection_config_path: str,
    candidate_pool_size: int,
    output_top_k: int,
    training_split_policy: str = "train_only",
    diagnostics: dict[str, Any] | None = None,
    backend: str = "verl",
    launch_config_path: str | None = None,
    launch_command: str | None = None,
    launch_command_path: str | None = None,
    prepared_parquet_path: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = output_dir / "artifacts"
    payload = {
        "policy_manifest_version": 1,
        "policy_type": "policy_rl",
        "backend": backend,
        "trainer": trainer,
        "base_model_name": base_model_name,
        "inference_model_name": inference_model_name,
        "init_policy_path": init_policy_path or "",
        "predictor_config": predictor_config,
        "output_dir": str(output_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_size": dataset_size,
        "trainer_config_path": str(trainer_config_path),
        "reward_mode": "single_idea",
        "selection_config_path": str(selection_config_path),
        "candidate_pool_size": int(candidate_pool_size),
        "output_top_k": int(output_top_k),
        "training_split_policy": training_split_policy,
        "dry_run": dry_run,
        "strict_contract": strict_runtime_manifest_contract(),
    }
    if launch_config_path:
        payload["launch_config_path"] = str(launch_config_path)
    if launch_command:
        payload["launch_command"] = str(launch_command)
    if launch_command_path:
        payload["launch_command_path"] = str(launch_command_path)
    if prepared_parquet_path:
        payload["prepared_parquet_path"] = str(prepared_parquet_path)
    if diagnostics:
        payload["diagnostics"] = diagnostics
        payload["parse_failure_rate"] = float(
            diagnostics.get("parse_failure_rate", 0.0)
        )
        payload["invalid_completion_rate"] = float(
            diagnostics.get("invalid_completion_rate", 0.0)
        )
    return payload


def write_policy_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _write_json(path, manifest)
