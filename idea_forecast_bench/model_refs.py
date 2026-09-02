from __future__ import annotations

from pathlib import Path


def resolve_model_reference(raw: str | None) -> str | None:
    """Resolve a local path, HF model id, or model-zoo alias into a loadable reference."""
    if raw is None:
        return None

    text = str(raw).strip()
    if not text:
        return None

    if Path(text).exists():
        return text

    # Hugging Face model ids like `Qwen/Qwen3.5-2B`.
    if "/" in text and not text.startswith((".", "/")):
        return text

    # Short aliases from the local forecaster model zoo like `qwen3.5-2b`.
    try:
        from forecaster.realization.model_zoo import resolve_small_model

        return str(resolve_small_model(text).model_id)
    except Exception:
        return None
