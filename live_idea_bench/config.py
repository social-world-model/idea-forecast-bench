from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from live_idea_bench.models import SimilarityPrompt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent / "prompt"
DEFAULT_TOPICS: tuple[dict[str, Any], ...] = (
    {
        "id": "diffusion_language_model",
        "name": "Diffusion Language Model",
        "aliases": [
            "diffusion language model",
            "diffusion lm",
            "diffusion llm",
            "text diffusion",
        ],
        "keywords": [
            "discrete diffusion",
            "masked diffusion",
            "diffusion transformer",
            "language diffusion",
        ],
    },
    {
        "id": "multimodal_visual_reasoning",
        "name": "Multimodal Visual Reasoning",
        "aliases": [
            "multimodal visual reasoning",
            "visual reasoning",
            "vision language reasoning",
            "image reasoning",
        ],
        "keywords": [
            "vision-language",
            "vqa",
            "chart reasoning",
            "document reasoning",
            "multimodal reasoning",
        ],
    },
    {
        "id": "gui_computer_use_web_agent",
        "name": "GUI / Computer Use / Web Agent",
        "aliases": [
            "gui agent",
            "computer use",
            "web agent",
            "browser agent",
        ],
        "keywords": [
            "web navigation",
            "browser automation",
            "computer-use",
            "gui grounding",
            "agentic browsing",
        ],
    },
    {
        "id": "optimizer",
        "name": "Optimizer",
        "aliases": [
            "optimizer",
            "optimization",
            "optimization algorithm",
        ],
        "keywords": [
            "gradient descent",
            "adam",
            "adamw",
            "sgd",
            "training dynamics",
        ],
    },
    {
        "id": "time_series_forecasting",
        "name": "Time-series Forecasting",
        "aliases": [
            "time-series forecasting",
            "time series forecasting",
            "time series",
        ],
        "keywords": [
            "temporal forecasting",
            "multivariate forecasting",
            "long-term forecasting",
            "sequence forecasting",
            "timeseries",
        ],
    },
)


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    max_context_chars: int = 50000
    use_section_weights: bool = True
    abstract_weight: float = 0.4
    introduction_weight: float = 0.2
    method_weight: float = 0.2
    conclusion_weight: float = 0.2
    # Voyage embedding endpoint (OpenAI-compatible). Empty → the Voyage default
    # URL is used. There is NO local/lexical fallback: the embedding engine
    # requires VOYAGE_API_KEY and fails loud if the call cannot be made.
    embedding_base_url: str = ""
    api_model: str = "voyage-3-large"


@dataclass
class TopicDefinition:
    id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class PredictorConfig:
    system_prompt: str
    user_template: str
    default_model: str | None = None
    default_temperature: float | None = None
    max_context_papers: int = 20
    parser: str = "json"


@dataclass
class SimilarityConfig:
    engine: str = "hybrid"
    semantic_threshold: float = 0.5
    keyword_threshold: float = 0.3
    embedding_threshold: float = 0.4
    llm_match_threshold: float = 0.7
    system_prompt: str = ""
    user_prompt_template: str = ""


@dataclass
class PromptTemplate:
    similarity_prompt: SimilarityPrompt


@dataclass
class Config:
    model_name: str = "gpt-4o"
    max_context_chars: int = 15000
    temperature: float = 0.0
    openai_api_key: str | None = None
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    topics: list[TopicDefinition] = field(default_factory=list)
    prompt_template: PromptTemplate | None = None

    @classmethod
    def load_config(
        cls,
        config_path: str | None = None,
        prompt_dir: str | None = None,
    ) -> Config:
        runtime = load_runtime_config(config_path)
        similarity = load_similarity_config("similarity.yaml", prompt_dir=prompt_dir)
        runtime.prompt_template = PromptTemplate(
            similarity_prompt=SimilarityPrompt(
                system_prompt=similarity.system_prompt,
                user_prompt_template=similarity.user_prompt_template,
            )
        )
        return runtime

    @staticmethod
    def _load_yaml_file(path: str, model_class: type[Any]) -> Any:
        payload = _read_yaml(Path(path))
        return model_class(**payload)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {path}")

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML at {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"YAML at {path} must decode to a mapping")
    return payload


def _resolve_config_path(config_path: str | None) -> Path:
    if not config_path:
        return DEFAULT_CONFIG_PATH
    path = Path(config_path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _resolve_prompt_path(name_or_path: str, prompt_dir: str | None = None) -> Path:
    raw = str(name_or_path or "").strip()
    if not raw:
        raise ValueError("Prompt config path is required")

    path = Path(raw)
    if path.is_absolute():
        return path

    if path.parts and path.parts[0] == "prompt":
        relative = Path(*path.parts[1:]) if len(path.parts) > 1 else Path()
        base_dir = Path(prompt_dir).resolve() if prompt_dir else DEFAULT_PROMPT_DIR
        return (base_dir / relative).resolve()

    base_dir = Path(prompt_dir).resolve() if prompt_dir else DEFAULT_PROMPT_DIR
    return (base_dir / path.name).resolve()


def _coerce_string_list(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        value = raw.strip()
        return [value] if value else []
    if not isinstance(raw, list):
        raise ValueError("topic aliases/keywords must be a list of strings")

    values: list[str] = []
    for item in raw:
        value = str(item).strip()
        if value:
            values.append(value)
    return values


def _default_topics() -> list[TopicDefinition]:
    return [
        TopicDefinition(
            id=str(item["id"]),
            name=str(item["name"]),
            aliases=list(item.get("aliases", [])),
            keywords=list(item.get("keywords", [])),
        )
        for item in DEFAULT_TOPICS
    ]


def _load_topics(payload: object) -> list[TopicDefinition]:
    if payload is None:
        return _default_topics()
    if not isinstance(payload, list):
        raise ValueError("topics config must be a list")

    topics: list[TopicDefinition] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each topic config must be a mapping")
        topic_id = str(item.get("id") or "").strip()
        name = str(item.get("name") or "").strip()
        if not topic_id or not name:
            raise ValueError("each topic config requires non-empty id and name")
        topics.append(
            TopicDefinition(
                id=topic_id,
                name=name,
                aliases=_coerce_string_list(item.get("aliases")),
                keywords=_coerce_string_list(item.get("keywords")),
            )
        )
    return topics


def load_runtime_config(config_path: str | None = None) -> Config:
    payload = _read_yaml(_resolve_config_path(config_path))
    embedding_payload = payload.pop("embedding", {}) or {}
    topics_payload = payload.pop("topics", None)
    if not isinstance(embedding_payload, dict):
        raise ValueError("embedding config must be a mapping")
    return Config(
        model_name=str(payload.get("model_name", "gpt-4o")),
        max_context_chars=int(payload.get("max_context_chars", 15000)),
        temperature=float(payload.get("temperature", 0.0)),
        openai_api_key=(
            str(payload["openai_api_key"]).strip()
            if payload.get("openai_api_key") not in {None, ""}
            else None
        ),
        embedding=EmbeddingConfig(**embedding_payload),
        topics=_load_topics(topics_payload),
    )


def load_topics(config_path: str | None = None) -> list[TopicDefinition]:
    return load_runtime_config(config_path).topics


def load_predictor_config(
    name_or_path: str = "predictor.yaml",
    *,
    prompt_dir: str | None = None,
) -> PredictorConfig:
    payload = _read_yaml(_resolve_prompt_path(name_or_path, prompt_dir=prompt_dir))
    system_prompt = str(payload.get("system_prompt", "")).strip()
    user_template = str(payload.get("user_template", "")).strip()
    if not system_prompt or not user_template:
        raise ValueError("predictor.yaml requires non-empty system_prompt and user_template")
    return PredictorConfig(
        system_prompt=system_prompt,
        user_template=user_template,
        default_model=(
            str(payload["default_model"]).strip()
            if payload.get("default_model") not in {None, ""}
            else None
        ),
        default_temperature=(
            float(payload["default_temperature"])
            if payload.get("default_temperature") is not None
            else None
        ),
        max_context_papers=int(payload.get("max_context_papers", 20)),
        parser=str(payload.get("parser", "json")).strip() or "json",
    )


def load_similarity_config(
    name_or_path: str = "similarity.yaml",
    *,
    prompt_dir: str | None = None,
) -> SimilarityConfig:
    payload = _read_yaml(_resolve_prompt_path(name_or_path, prompt_dir=prompt_dir))
    return SimilarityConfig(
        engine=str(payload.get("engine", "hybrid")).strip() or "hybrid",
        semantic_threshold=float(payload.get("semantic_threshold", 0.5)),
        keyword_threshold=float(payload.get("keyword_threshold", 0.3)),
        embedding_threshold=float(payload.get("embedding_threshold", 0.4)),
        llm_match_threshold=float(payload.get("llm_match_threshold", 0.7)),
        system_prompt=str(payload.get("system_prompt", "")).strip(),
        user_prompt_template=str(payload.get("user_prompt_template", "")).strip(),
    )


def _resolve_versioned_prompt_path(
    prompt_id: str,
    version: str,
    *,
    prompt_dir: str | None = None,
) -> Path:
    if not prompt_id.strip():
        raise ValueError("prompt_id is required")
    if not version.strip():
        raise ValueError("version is required")
    base_dir = Path(prompt_dir).resolve() if prompt_dir else DEFAULT_PROMPT_DIR
    return (base_dir / prompt_id / f"{version}.yaml").resolve()


def get_prompt_policy(
    prompt_id: str,
    version: str,
    *,
    prompt_dir: str | None = None,
) -> dict[str, Any]:
    payload = _read_yaml(_resolve_versioned_prompt_path(prompt_id, version, prompt_dir=prompt_dir))
    template = str(payload.get("template") or "").strip()
    if not template:
        raise ValueError("versioned prompt assets require a non-empty template field")
    return {
        "prompt_id": str(payload.get("prompt_id") or prompt_id),
        "version": str(payload.get("version") or version),
        "model_id": str(payload.get("model_id") or payload.get("default_model") or "").strip(),
        "temperature": float(payload.get("temperature", payload.get("default_temperature", 0.7))),
        "max_tokens": int(payload.get("max_tokens", 1024)),
        "timeout_seconds": int(payload.get("timeout_seconds", 30)),
        "template": template,
    }


def get_prompt_template(
    prompt_id: str,
    version: str,
    *,
    prompt_dir: str | None = None,
) -> str:
    return str(get_prompt_policy(prompt_id, version, prompt_dir=prompt_dir)["template"])


def list_prompts(*, prompt_dir: str | None = None) -> list[str]:
    base_dir = Path(prompt_dir).resolve() if prompt_dir else DEFAULT_PROMPT_DIR
    if not base_dir.is_dir():
        return []

    prompts: list[str] = []
    for yaml_path in sorted(base_dir.rglob("*.yaml")):
        relative = yaml_path.relative_to(base_dir)
        if len(relative.parts) != 2:
            continue
        prompt_id = relative.parts[0]
        version = yaml_path.stem
        prompts.append(f"{prompt_id}@{version}")
    return prompts
