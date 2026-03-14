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
        alias="qwen2.5-7b-base",
        model_id="Qwen/Qwen2.5-7B",
        family="qwen2.5",
        params_billions=7.61,
        variant="base",
        license_name="apache-2.0",
        min_transformers_version="4.37.0",
        notes="Larger Qwen2.5 base checkpoint if you want more headroom for continued pretraining or RL warm starts.",
    ),
    SmallModelSpec(
        alias="qwen2.5-7b-instruct",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        family="qwen2.5",
        params_billions=7.61,
        variant="instruct",
        license_name="apache-2.0",
        min_transformers_version="4.37.0",
        notes="Stronger structured-generation baseline than 3B when you can afford the extra VRAM.",
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
        alias="qwen3-8b-base",
        model_id="Qwen/Qwen3-8B-Base",
        family="qwen3",
        params_billions=8.2,
        variant="base",
        license_name="apache-2.0",
        min_transformers_version="4.51.0",
        notes="Best Qwen3 8B starting point if you want a clean pretraining checkpoint for post-training.",
    ),
    SmallModelSpec(
        alias="qwen3-8b",
        model_id="Qwen/Qwen3-8B",
        family="qwen3",
        params_billions=8.2,
        variant="thinking-switchable",
        license_name="apache-2.0",
        min_transformers_version="4.51.0",
        notes="Qwen3 8B post-trained checkpoint with switchable thinking and strong general instruction following.",
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


_MODEL_ALIAS_REDIRECTS = {
    "qwen3-4b-instruct": "qwen3-4b-instruct-2507",
    "qwen3-8b-instruct": "qwen3-8b",
}


def list_small_model_specs() -> list[SmallModelSpec]:
    return list(_SMALL_MODEL_SPECS)


def list_small_model_payloads() -> list[dict[str, object]]:
    return [asdict(spec) for spec in _SMALL_MODEL_SPECS]


def resolve_small_model(alias_or_model_id: str) -> SmallModelSpec:
    normalized = _MODEL_ALIAS_REDIRECTS.get(alias_or_model_id.strip().lower(), alias_or_model_id.strip().lower())
    for spec in _SMALL_MODEL_SPECS:
        if spec.alias == normalized or spec.model_id.lower() == normalized:
            return spec
    raise ValueError(f"Unknown RL model preset: {alias_or_model_id}")
