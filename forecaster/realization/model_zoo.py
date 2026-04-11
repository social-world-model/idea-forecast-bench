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
    # ----- Qwen 2.5 (legacy, kept for backward compat) -----
    SmallModelSpec(
        alias="qwen2.5-3b-instruct",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        family="qwen2.5",
        params_billions=3.09,
        variant="instruct",
        license_name="qwen-research",
        min_transformers_version="4.37.0",
        notes="Legacy Qwen2.5 baseline; prefer Qwen3.5 models for new work.",
    ),
    # ----- Qwen 3 (text-only, compatible with vLLM) -----
    SmallModelSpec(
        alias="qwen3-0.6b",
        model_id="Qwen/Qwen3-0.6B",
        family="qwen3",
        params_billions=0.6,
        variant="base",
        license_name="apache-2.0",
        min_transformers_version="4.51.0",
        notes="Smallest Qwen3; fast iteration and vLLM-compatible.",
    ),
    SmallModelSpec(
        alias="qwen3-1.7b",
        model_id="Qwen/Qwen3-1.7B",
        family="qwen3",
        params_billions=1.7,
        variant="base",
        license_name="apache-2.0",
        min_transformers_version="4.51.0",
        notes="Good balance of speed and quality; vLLM-compatible.",
    ),
    SmallModelSpec(
        alias="qwen3-4b",
        model_id="Qwen/Qwen3-4B",
        family="qwen3",
        params_billions=4.0,
        variant="base",
        license_name="apache-2.0",
        min_transformers_version="4.51.0",
        notes="Strong text model; vLLM-compatible.",
    ),
    SmallModelSpec(
        alias="qwen3-8b",
        model_id="Qwen/Qwen3-8B",
        family="qwen3",
        params_billions=8.0,
        variant="base",
        license_name="apache-2.0",
        min_transformers_version="4.51.0",
        notes="Largest Qwen3 dense model; vLLM-compatible.",
    ),
    # ----- Qwen 3.5 (VLM, requires transformers >=5.x, no vLLM) -----
    SmallModelSpec(
        alias="qwen3.5-0.8b",
        model_id="Qwen/Qwen3.5-0.8B",
        family="qwen3.5",
        params_billions=0.8,
        variant="thinking-switchable",
        license_name="apache-2.0",
        min_transformers_version="4.52.0",
        notes="Smallest Qwen3.5; good for quick iteration. Thinking disabled by default.",
    ),
    SmallModelSpec(
        alias="qwen3.5-2b",
        model_id="Qwen/Qwen3.5-2B",
        family="qwen3.5",
        params_billions=2.0,
        variant="thinking-switchable",
        license_name="apache-2.0",
        min_transformers_version="4.52.0",
        notes="Best small model for prior SFT on a single GPU. Thinking disabled by default.",
    ),
    SmallModelSpec(
        alias="qwen3.5-4b-base",
        model_id="Qwen/Qwen3.5-4B-Base",
        family="qwen3.5",
        params_billions=4.0,
        variant="base",
        license_name="apache-2.0",
        min_transformers_version="4.52.0",
        notes="Clean 4B pretraining checkpoint for custom post-training or RL warm starts.",
    ),
    SmallModelSpec(
        alias="qwen3.5-4b",
        model_id="Qwen/Qwen3.5-4B",
        family="qwen3.5",
        params_billions=4.0,
        variant="thinking-switchable",
        license_name="apache-2.0",
        min_transformers_version="4.52.0",
        notes="Strong small model with switchable thinking; disable thinking for stable JSON output.",
    ),
    SmallModelSpec(
        alias="qwen3.5-9b-base",
        model_id="Qwen/Qwen3.5-9B-Base",
        family="qwen3.5",
        params_billions=9.0,
        variant="base",
        license_name="apache-2.0",
        min_transformers_version="4.52.0",
        notes="Largest dense base checkpoint in the small-model range for post-training.",
    ),
    SmallModelSpec(
        alias="qwen3.5-9b",
        model_id="Qwen/Qwen3.5-9B",
        family="qwen3.5",
        params_billions=9.0,
        variant="thinking-switchable",
        license_name="apache-2.0",
        min_transformers_version="4.52.0",
        notes="Strongest dense small model; use when you have the VRAM budget.",
    ),
    # ----- Llama 3.2 (comparison baseline) -----
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
    "qwen3-4b-base": "qwen3-4b",
    "qwen3-4b-instruct": "qwen3-4b",
    "qwen3-4b-instruct-2507": "qwen3-4b",
    "qwen3-8b-base": "qwen3.5-9b-base",
    "qwen3-8b-instruct": "qwen3.5-9b",
    # Old Qwen2.5 aliases
    "qwen2.5-3b-base": "qwen2.5-3b-instruct",
    "qwen2.5-7b-base": "qwen2.5-3b-instruct",
    "qwen2.5-7b-instruct": "qwen2.5-3b-instruct",
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
