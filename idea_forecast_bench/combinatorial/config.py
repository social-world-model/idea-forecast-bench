from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from idea_forecast_bench.combinatorial.types import ELEMENT_TYPES, ElementType
from idea_forecast_bench.config import PROJECT_ROOT, _read_yaml, _resolve_prompt_path

DEFAULT_CONFIG_NAME = "combinatorial.yaml"


@dataclass(frozen=True)
class PromptPair:
    system_prompt: str
    user_template: str


@dataclass(frozen=True)
class ExtractionConfig:
    prompt: str
    temperature: float
    max_summary_chars: int
    max_elements_per_type: Mapping[ElementType, int]
    retry_temperature_bump: float


@dataclass(frozen=True)
class CanonicalizeConfig:
    embed_backend: str
    embed_model: str
    merge_threshold: float
    min_count: int
    aliases: Mapping[str, str]


@dataclass(frozen=True)
class StateConfig:
    half_life_months: float
    recent_months: float
    smoothing_alpha: float
    hot_quantile: float


@dataclass(frozen=True)
class SamplerConfig:
    top_m_per_type: int
    top_m_triple: int
    type_patterns: tuple[tuple[ElementType, ...], ...]
    score_gamma: float
    lambda_rising: float
    lambda_unpaired: float
    rising_log_clip: float


@dataclass(frozen=True)
class RealizeConfig:
    prompt: str
    temperature: float
    top_p: float | None
    evidence_per_combo: int
    evidence_snippet_chars: int
    fallback_template: bool
    min_coverage_warn: float


@dataclass(frozen=True)
class SpecificityConfig:
    prompt: str
    temperature: float


@dataclass(frozen=True)
class CombinatorialConfig:
    schema_version: int
    extraction: ExtractionConfig
    canonicalize: CanonicalizeConfig
    state: StateConfig
    sampler: SamplerConfig
    realize: RealizeConfig
    specificity: SpecificityConfig
    source_path: str


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    raw = payload.get(name)
    if not isinstance(raw, Mapping):
        raise ValueError(f"combinatorial config: section '{name}' must be a mapping")
    return raw


def _element_type(raw: object) -> ElementType:
    value = str(raw).strip().lower()
    for known in ELEMENT_TYPES:
        if value == known:
            return known
    raise ValueError(
        f"combinatorial config: unknown element type {value!r}; "
        f"expected one of {', '.join(ELEMENT_TYPES)}"
    )


def _parse_patterns(raw: object) -> tuple[tuple[ElementType, ...], ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("sampler.type_patterns must be a non-empty list")
    patterns: list[tuple[ElementType, ...]] = []
    for item in raw:
        if not isinstance(item, list) or len(item) < 2:
            raise ValueError("each sampler.type_pattern needs at least 2 types")
        types = tuple(_element_type(t) for t in item)
        if len(set(types)) != len(types):
            raise ValueError(f"sampler.type_pattern repeats a type: {types}")
        patterns.append(types)
    return tuple(patterns)


def _parse_limits(raw: object) -> Mapping[ElementType, int]:
    if not isinstance(raw, Mapping):
        raise ValueError("extraction.max_elements_per_type must be a mapping")
    limits: dict[ElementType, int] = {}
    for key, value in raw.items():
        limits[_element_type(key)] = int(value)
    for element_type in ELEMENT_TYPES:
        limits.setdefault(element_type, 3)
    return limits


def _parse_aliases(raw: object) -> Mapping[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("canonicalize.aliases must be a mapping")
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in raw.items()}


def resolve_config_path(name_or_path: str | None) -> Path:
    raw = (name_or_path or DEFAULT_CONFIG_NAME).strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (PROJECT_ROOT / "config" / path.name).resolve()


def load_combinatorial_config(name_or_path: str | None = None) -> CombinatorialConfig:
    path = resolve_config_path(name_or_path)
    payload = _read_yaml(path)

    ext = _section(payload, "extraction")
    canon = _section(payload, "canonicalize")
    state = _section(payload, "state")
    sampler = _section(payload, "sampler")
    realize = _section(payload, "realize")
    spec = _section(payload, "specificity")

    top_p_raw = realize.get("top_p")
    return CombinatorialConfig(
        schema_version=int(payload.get("schema_version", 1)),
        extraction=ExtractionConfig(
            prompt=str(ext.get("prompt", "combinatorial_extract.yaml")),
            temperature=float(ext.get("temperature", 0.0)),
            max_summary_chars=int(ext.get("max_summary_chars", 1500)),
            max_elements_per_type=_parse_limits(ext.get("max_elements_per_type")),
            retry_temperature_bump=float(ext.get("retry_temperature_bump", 0.2)),
        ),
        canonicalize=CanonicalizeConfig(
            embed_backend=str(canon.get("embed_backend", "voyage")).strip().lower(),
            embed_model=str(canon.get("embed_model", "voyage-3-large")),
            merge_threshold=float(canon.get("merge_threshold", 0.86)),
            min_count=int(canon.get("min_count", 2)),
            aliases=_parse_aliases(canon.get("aliases")),
        ),
        state=StateConfig(
            half_life_months=float(state.get("half_life_months", 6.0)),
            recent_months=float(state.get("recent_months", 3.0)),
            smoothing_alpha=float(state.get("smoothing_alpha", 0.5)),
            hot_quantile=float(state.get("hot_quantile", 0.7)),
        ),
        sampler=SamplerConfig(
            top_m_per_type=int(sampler.get("top_m_per_type", 30)),
            top_m_triple=int(sampler.get("top_m_triple", 12)),
            type_patterns=_parse_patterns(sampler.get("type_patterns")),
            score_gamma=float(sampler.get("score_gamma", 1.0)),
            lambda_rising=float(sampler.get("lambda_rising", 1.0)),
            lambda_unpaired=float(sampler.get("lambda_unpaired", 1.0)),
            rising_log_clip=float(sampler.get("rising_log_clip", 2.0)),
        ),
        realize=RealizeConfig(
            prompt=str(realize.get("prompt", "combinatorial_realize.yaml")),
            temperature=float(realize.get("temperature", 0.7)),
            top_p=float(top_p_raw) if top_p_raw is not None else None,
            evidence_per_combo=int(realize.get("evidence_per_combo", 3)),
            evidence_snippet_chars=int(realize.get("evidence_snippet_chars", 220)),
            fallback_template=bool(realize.get("fallback_template", True)),
            min_coverage_warn=float(realize.get("min_coverage_warn", 0.8)),
        ),
        specificity=SpecificityConfig(
            prompt=str(spec.get("prompt", "combinatorial_specificity.yaml")),
            temperature=float(spec.get("temperature", 0.0)),
        ),
        source_path=str(path),
    )


def load_prompt_pair(name_or_path: str) -> PromptPair:
    path = _resolve_prompt_path(name_or_path)
    payload = _read_yaml(path)
    system_prompt = str(payload.get("system_prompt", "")).strip()
    user_template = str(payload.get("user_template", "")).strip()
    if not system_prompt or not user_template:
        raise ValueError(f"{path} requires non-empty system_prompt and user_template")
    return PromptPair(system_prompt=system_prompt, user_template=user_template)
