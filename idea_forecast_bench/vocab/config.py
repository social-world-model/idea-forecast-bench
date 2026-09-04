from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from idea_forecast_bench.combinatorial.config import PromptPair

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "vocab.yaml"


@dataclass(frozen=True)
class ExtractionConfig:
    prompt: str = "vocab_extract_v2.yaml"
    temperature: float = 0.0
    max_summary_chars: int = 1500
    max_terms_per_slot: Mapping[str, int] = field(
        default_factory=lambda: {"object": 2, "mechanism": 3, "problem": 2}
    )
    retry_temperature_bump: float = 0.2
    aliases: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ClusterConfig:
    embed_backend: str = "voyage"
    embed_model: str = "voyage-3-large"
    fine_threshold: float = 0.90
    single_token_threshold: float = 0.95
    shared_prefix_chars: int = 6
    parent_threshold: float = 0.90
    slot_majority_min: float = 0.6


@dataclass(frozen=True)
class TagConfig:
    background_doc_frac: float = 0.20
    min_count: int = 2
    emerging_months: int = 3
    emerging_min_count: int = 1
    #: 0 disables the hybrid level. Above 0, a fine concept with fewer than
    #: this many training papers is folded into a concept named by its
    #: (merged) parent label instead of staying its own node -- see
    #: ``idea_forecast_bench.vocab.build._fold_weak_clusters``.
    promote_min_count: int = 0


@dataclass(frozen=True)
class ChecksConfig:
    horizon_months: int = 3
    assign_threshold: float = 0.90
    mid_layer_min_papers: int = 3


@dataclass(frozen=True)
class VocabConfig:
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    tag: TagConfig = field(default_factory=TagConfig)
    checks: ChecksConfig = field(default_factory=ChecksConfig)
    schema_version: int = 1
    source_path: str = ""
    sha: str = ""


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return payload


def _resolve(name_or_path: str | None) -> Path:
    if not name_or_path:
        return DEFAULT_CONFIG
    candidate = Path(name_or_path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return PROJECT_ROOT / "config" / name_or_path


def _section(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    raw = payload.get(name) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"vocab config section {name!r} must be a mapping")
    return dict(raw)


def _unit(value: Any, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {number}")
    return number


def load_vocab_config(name_or_path: str | None = None) -> VocabConfig:
    path = _resolve(name_or_path)
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    ext = _section(payload, "extraction")
    clu = _section(payload, "cluster")
    tag = _section(payload, "tag")
    chk = _section(payload, "checks")
    caps = ext.get("max_terms_per_slot") or {}
    aliases = ext.get("aliases") or {}
    return VocabConfig(
        extraction=ExtractionConfig(
            prompt=str(ext.get("prompt", "vocab_extract_v2.yaml")),
            temperature=float(ext.get("temperature", 0.0)),
            max_summary_chars=int(ext.get("max_summary_chars", 1500)),
            max_terms_per_slot={str(k): int(v) for k, v in dict(caps).items()},
            retry_temperature_bump=float(ext.get("retry_temperature_bump", 0.2)),
            aliases={str(k).lower(): str(v).lower() for k, v in dict(aliases).items()},
        ),
        cluster=ClusterConfig(
            embed_backend=str(clu.get("embed_backend", "voyage")),
            embed_model=str(clu.get("embed_model", "voyage-3-large")),
            fine_threshold=_unit(clu.get("fine_threshold", 0.90), "fine_threshold"),
            single_token_threshold=_unit(
                clu.get("single_token_threshold", 0.95), "single_token_threshold"
            ),
            shared_prefix_chars=int(clu.get("shared_prefix_chars", 6)),
            parent_threshold=_unit(
                clu.get("parent_threshold", 0.90), "parent_threshold"
            ),
            slot_majority_min=_unit(
                clu.get("slot_majority_min", 0.6), "slot_majority_min"
            ),
        ),
        tag=TagConfig(
            background_doc_frac=_unit(
                tag.get("background_doc_frac", 0.20), "background_doc_frac"
            ),
            min_count=int(tag.get("min_count", 2)),
            emerging_months=int(tag.get("emerging_months", 3)),
            emerging_min_count=int(tag.get("emerging_min_count", 1)),
            promote_min_count=int(tag.get("promote_min_count", 0)),
        ),
        checks=ChecksConfig(
            horizon_months=int(chk.get("horizon_months", 3)),
            assign_threshold=_unit(
                chk.get("assign_threshold", 0.90), "assign_threshold"
            ),
            mid_layer_min_papers=int(chk.get("mid_layer_min_papers", 3)),
        ),
        schema_version=int(payload.get("schema_version", 1)),
        source_path=str(path),
        sha=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    )


def load_prompt(name_or_path: str) -> PromptPair:
    candidate = Path(name_or_path)
    if not (candidate.is_absolute() or candidate.exists()):
        candidate = PROJECT_ROOT / "idea_forecast_bench" / "prompt" / name_or_path
    payload = _read_yaml(candidate)
    system_prompt = str(payload.get("system_prompt", "")).strip()
    user_template = str(payload.get("user_template", "")).strip()
    if not system_prompt or not user_template:
        raise ValueError(
            f"{candidate} requires non-empty system_prompt and user_template"
        )
    return PromptPair(system_prompt=system_prompt, user_template=user_template)
