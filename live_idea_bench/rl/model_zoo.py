from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SmallModelSpec:
    alias: str
    model_id: str
    family: str
    params_billions: float
    variant: str
    license_name: str
    min_transformers_version: str
    gated: bool = False
    notes: str = ""


_SMALL_MODEL_SPECS = (
    SmallModelSpec(
        alias="qwen2.5-3b-base",
        model_id="Qwen/Qwen2.5-3B",
        family="qwen2.5",
        params_billions=3.09,
        variant="base",
        license_name="qwen-research",
        min_transformers_version="4.37.0",
        notes="Best if you want to add your own post-training from a cleaner base model.",
    ),
    SmallModelSpec(
        alias="qwen2.5-3b-instruct",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        family="qwen2.5",
        params_billions=3.09,
        variant="instruct",
        license_name="qwen-research",
        min_transformers_version="4.37.0",
        notes="Safest first DPO baseline for structured JSON idea generation.",
    ),
    SmallModelSpec(
        alias="qwen3-4b-base",
        model_id="Qwen/Qwen3-4B-Base",
        family="qwen3",
        params_billions=4.0,
        variant="base",
        license_name="apache-2.0",
        min_transformers_version="4.51.0",
        notes="Good research baseline if you want a pure Qwen3 4B pretraining checkpoint.",
    ),
    SmallModelSpec(
        alias="qwen3-4b",
        model_id="Qwen/Qwen3-4B",
        family="qwen3",
        params_billions=4.0,
        variant="thinking-switchable",
        license_name="apache-2.0",
        min_transformers_version="4.51.0",
        notes="Supports thinking and non-thinking modes; use non-thinking mode for stable JSON output.",
    ),
    SmallModelSpec(
        alias="qwen3-4b-instruct-2507",
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        family="qwen3",
        params_billions=4.0,
        variant="instruct",
        license_name="apache-2.0",
        min_transformers_version="4.51.0",
        notes="Strongest small Qwen3 non-thinking instruct checkpoint for this repo's JSON-heavy prompt style.",
    ),
    SmallModelSpec(
        alias="llama3.2-3b-instruct",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        family="llama3.2",
        params_billions=3.21,
        variant="instruct",
        license_name="llama3.2",
        min_transformers_version="4.43.0",
        gated=True,
        notes="Useful comparison model, but gated on Hugging Face and license acceptance is required.",
    ),
)


def list_small_model_specs() -> list[SmallModelSpec]:
    return list(_SMALL_MODEL_SPECS)


def list_small_model_payloads() -> list[dict[str, object]]:
    return [asdict(spec) for spec in _SMALL_MODEL_SPECS]


def resolve_small_model(alias_or_model_id: str) -> SmallModelSpec:
    normalized = alias_or_model_id.strip().lower()
    for spec in _SMALL_MODEL_SPECS:
        if spec.alias == normalized or spec.model_id.lower() == normalized:
            return spec
    raise ValueError(f"Unknown RL model preset: {alias_or_model_id}")
