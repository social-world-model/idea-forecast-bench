from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

from live_idea_bench.models import PaperRecord
from live_idea_bench.rl.io import _write_json


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

    @abstractmethod
    def prepare(self, common_context: PreparedRLContext, **kwargs: Any) -> TrainerPreparedArtifacts:
        raise NotImplementedError

    @abstractmethod
    def train(self, prepared_artifacts: TrainerPreparedArtifacts, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError


def _serialize_for_fingerprint(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _serialize_for_fingerprint(inner) for key, inner in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_fingerprint(item) for item in value]
    return value


def build_config_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(_serialize_for_fingerprint(payload), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_trl_stack() -> dict[str, Any]:
    try:
        datasets = importlib.import_module("datasets")
        peft = importlib.import_module("peft")
        trl = importlib.import_module("trl")
    except ImportError as exc:
        raise RuntimeError(
            "TRL training dependencies are not installed. Install torch, datasets, transformers, peft, "
            "accelerate, and trl to run non-dry-run RL training."
        ) from exc

    return {
        "Dataset": getattr(datasets, "Dataset"),
        "LoraConfig": getattr(peft, "LoraConfig"),
        "DPOConfig": getattr(trl, "DPOConfig", None),
        "DPOTrainer": getattr(trl, "DPOTrainer", None),
        "GRPOConfig": getattr(trl, "GRPOConfig", None),
        "GRPOTrainer": getattr(trl, "GRPOTrainer", None),
        "RLOOConfig": getattr(trl, "RLOOConfig", None),
        "RLOOTrainer": getattr(trl, "RLOOTrainer", None),
    }


def _peft_config(deps: dict[str, Any], *, r: int, lora_alpha: int, lora_dropout: float) -> Any:
    return deps["LoraConfig"](
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _filter_supported_kwargs(factory: Any, values: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return values

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return values
    accepted = set(signature.parameters)
    return {key: value for key, value in values.items() if key in accepted}


def create_trl_config(factory: Any, values: dict[str, Any]) -> Any:
    if factory is None:
        raise RuntimeError("The selected TRL trainer is not available in the installed trl package.")
    return factory(**_filter_supported_kwargs(factory, values))


def create_trl_trainer(factory: Any, values: dict[str, Any]) -> Any:
    if factory is None:
        raise RuntimeError("The selected TRL trainer is not available in the installed trl package.")
    return factory(**_filter_supported_kwargs(factory, values))


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
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_path = output_dir / "artifacts"
    payload = {
        "policy_manifest_version": 1,
        "policy_type": "policy_rl",
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
        "dry_run": dry_run,
    }
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return payload


def write_policy_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _write_json(path, manifest)
