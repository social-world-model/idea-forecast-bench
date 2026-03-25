"""Generate research proposals from innovation triples and evidence."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.llm import get_response_from_llm

from forecaster.models import Innovation
from forecaster.config import RealizationConfig

logger = logging.getLogger(__name__)

PROMPT_FILE = Path(__file__).parent.parent / "prompt" / "realization.yaml"


def _load_prompt() -> dict[str, str]:
    """Load the realization prompt templates from YAML."""
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Realization prompt file not found: {PROMPT_FILE}")
    payload = yaml.safe_load(PROMPT_FILE.read_text(encoding="utf-8")) or {}
    return {
        "system_prompt": str(payload.get("system_prompt", "")).strip(),
        "user_template": str(payload.get("user_template", "")).strip(),
    }


def _format_evidence_summary(evidence: list[PaperRecord]) -> str:
    """Format evidence papers into a readable summary string."""
    if not evidence:
        return "(No supporting evidence retrieved.)"
    lines = []
    for i, paper in enumerate(evidence, start=1):
        lines.append(f"{i}. {paper.title}: {paper.summary[:200]}")
    return "\n".join(lines)


def _build_user_message(prompt_data: dict[str, str], innovation: Innovation, evidence: list[PaperRecord]) -> str:
    """Build the formatted user message for proposal generation."""
    evidence_summary = _format_evidence_summary(evidence)
    return prompt_data["user_template"].format(
        base_direction=innovation.base_direction,
        operator=innovation.operator,
        gap=innovation.gap,
        evidence_summary=evidence_summary,
    )


def _generate_proposal_local(
    user_msg: str,
    system_prompt: str,
    realization_model_path: str,
    config: RealizationConfig,
) -> str:
    """Generate proposal using a local HF model (the GRPO-trained realization policy).

    Args:
        user_msg: Formatted user message containing innovation and evidence.
        system_prompt: System prompt for the model.
        realization_model_path: Path to the trained realization checkpoint (LoRA or full model).
        config: RealizationConfig with max_tokens.

    Returns:
        Generated proposal text.
    """
    from forecaster.realization.local_generation import _load_local_model, _apply_chat_template, _require_local_generation_stack
    from forecaster.prior.sampler import _detect_base_model

    full_prompt = f"{system_prompt}\n\n{user_msg}".strip()

    # Detect if checkpoint is a LoRA adapter
    base_model_name: str | None = None
    adapter_config_path = Path(realization_model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        base_model_name = _detect_base_model(realization_model_path)

    model, tokenizer = _load_local_model(realization_model_path, base_model_name=base_model_name)
    deps = _require_local_generation_stack()
    torch = deps["torch"]

    chat_prompt = _apply_chat_template(tokenizer, full_prompt, None)
    encoded = tokenizer([chat_prompt], return_tensors="pt")
    encoded = {name: value.to(model.device) for name, value in encoded.items()}

    generated = model.generate(
        **encoded,
        max_new_tokens=config.max_tokens,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    output_ids = generated[0][len(encoded["input_ids"][0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def generate_proposal(
    innovation: Innovation,
    evidence: list[PaperRecord],
    llm_client: Any,
    model: str,
    config: RealizationConfig,
    *,
    realization_model_path: str | None = None,
) -> str:
    """Generate a research proposal text given innovation triple and evidence.

    When realization_model_path is provided, uses the GRPO-trained local model
    (p_ψ(y|z,X) from the paper). Otherwise falls back to the generic LLM client.

    Args:
        innovation: The innovation triple (z).
        evidence: Retrieved supporting papers.
        llm_client: Initialized LLM client (used as fallback when no local model).
        model: LLM model name (used as fallback).
        config: RealizationConfig with max_tokens.
        realization_model_path: Optional path to trained realization checkpoint.
            When provided, uses the local HF model for generation instead of the API.

    Returns:
        Proposal text string (title on first line, body following).
    """
    prompt_data = _load_prompt()
    user_msg = _build_user_message(prompt_data, innovation, evidence)

    if realization_model_path and Path(realization_model_path).exists():
        try:
            return _generate_proposal_local(
                user_msg=user_msg,
                system_prompt=prompt_data["system_prompt"],
                realization_model_path=realization_model_path,
                config=config,
            )
        except Exception as exc:
            logger.warning(
                "Local realization model generation failed (%s); falling back to LLM client.", exc
            )

    response_text, _ = get_response_from_llm(
        msg=user_msg,
        client=llm_client,
        model=model,
        system_message=prompt_data["system_prompt"],
    )

    return response_text.strip()


def proposal_to_idea_prediction(
    proposal_text: str,
    innovation: Innovation,
    rank: int = 1,
) -> IdeaPrediction:
    """Convert a proposal text to an IdeaPrediction for benchmark evaluation.

    Extracts title from first line, uses gap as rationale, operator as approach.
    """
    lines = proposal_text.strip().splitlines()
    title = lines[0].strip() if lines else ""
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    return IdeaPrediction(
        rank=rank,
        title=title,
        rationale=f"{innovation.gap}. {body}"[:500] if body else innovation.gap,
        approach=f"{innovation.operator}: {innovation.base_direction}",
        score=0.0,
    )
