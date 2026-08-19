"""Generate research proposals from innovation triples and evidence."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from forecaster.config import RealizationConfig
from forecaster.models import Innovation
from live_idea_bench.llm import get_response_from_llm
from live_idea_bench.model_refs import resolve_model_reference
from live_idea_bench.models import IdeaPrediction, PaperRecord

logger = logging.getLogger(__name__)

PROMPT_FILE = Path(__file__).parent.parent / "prompt" / "realization.yaml"

def strip_think_block(text: str) -> str:
    """Strip Qwen3.5 thinking-mode reasoning from a generated completion.

    Handles four cases:
    - Full block: ``<think>reasoning</think>answer`` → ``answer``
    - Half block: ``reasoning</think>answer`` → ``answer``
      (the ``<think>`` opener was in the chat-template prefill, so the
      generated completion starts inside the think block)
    - Loop: ``<think>reasoning...`` (no closer) → ``""``
      (treated as an invalid completion that should score zero)
    - No tags: ``plain answer`` → ``plain answer`` (unchanged)
    """
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    if "<think>" in text:
        return ""
    return text


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


def _format_context_summary(
    context_papers: list[PaperRecord],
    *,
    top_k: int,
) -> str:
    """Format available historical context as a compact summary."""
    if not context_papers:
        return "(No historical context available.)"
    ranked = sorted(
        context_papers,
        key=lambda paper: (paper.month, paper.paper_id),
        reverse=True,
    )
    lines = []
    for idx, paper in enumerate(ranked[: max(1, top_k)], start=1):
        lines.append(f"{idx}. {paper.title} ({paper.month}): {paper.summary[:200]}")
    return "\n".join(lines)


def _build_user_message(
    prompt_data: dict[str, str],
    innovation: Innovation,
    evidence: list[PaperRecord],
    *,
    context_papers: list[PaperRecord] | None = None,
    config: RealizationConfig | None = None,
) -> str:
    """Build the formatted user message for proposal generation."""
    evidence_summary = _format_evidence_summary(evidence)
    realized_config = config or RealizationConfig()
    context_summary = _format_context_summary(
        context_papers or [],
        top_k=realized_config.context_top_k,
    )
    return prompt_data["user_template"].format(
        base_direction=innovation.base_direction,
        operator=innovation.operator,
        gap=innovation.gap,
        context_summary=context_summary,
        evidence_summary=evidence_summary,
    )


def build_realization_messages(
    innovation: Innovation,
    evidence: list[PaperRecord],
    *,
    context_papers: list[PaperRecord] | None = None,
    config: RealizationConfig | None = None,
) -> tuple[str, str]:
    """Build the shared realization system/user prompts for training and inference."""
    prompt_data = _load_prompt()
    user_msg = _build_user_message(
        prompt_data,
        innovation,
        evidence,
        context_papers=context_papers,
        config=config,
    )
    return prompt_data["system_prompt"], user_msg


def _generate_proposal_local(
    user_msg: str,
    system_prompt: str,
    realization_model_path: str,
    config: RealizationConfig,
    *,
    base_model_name: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
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
    from forecaster.prior.sampler import _detect_base_model
    from forecaster.realization.local_generation import (
        _apply_chat_template,
        _load_local_model,
        _require_local_generation_stack,
    )

    full_prompt = f"{system_prompt}\n\n{user_msg}".strip()

    # Detect if checkpoint is a LoRA adapter
    resolved_base_model_name = base_model_name
    adapter_config_path = Path(realization_model_path) / "adapter_config.json"
    if resolved_base_model_name is None and adapter_config_path.exists():
        resolved_base_model_name = _detect_base_model(realization_model_path)

    model, tokenizer = _load_local_model(
        realization_model_path,
        base_model_name=resolved_base_model_name,
    )
    deps = _require_local_generation_stack()
    torch = deps["torch"]

    if seed is not None:
        torch.manual_seed(seed)

    # Match training: thinking=True so the chat template prefills <think>\n
    # and the GRPO-trained policy sees the same prompt distribution it learned.
    chat_prompt = _apply_chat_template(tokenizer, full_prompt, True)
    encoded = tokenizer([chat_prompt], return_tensors="pt")
    encoded = {name: value.to(model.device) for name, value in encoded.items()}

    # Match training sampling: Qwen3.5 recommended thinking-mode params
    # (temperature=1.0, top_p=0.95, top_k=20, repetition_penalty=1.5).
    # repetition_penalty is critical to prevent thinking-mode loops on the
    # smaller variants. Caller temperature/top_p still override if provided.
    generated = model.generate(
        **encoded,
        max_new_tokens=config.proposal_max_tokens,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=1.0 if temperature is None else temperature,
        top_p=0.95 if top_p is None else top_p,
        top_k=20,
        repetition_penalty=1.5,
    )
    output_ids = generated[0][len(encoded["input_ids"][0]):].tolist()
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    return strip_think_block(decoded).strip()


def _sglang_api_url() -> str | None:
    """Return the SGLang server URL if available, or None.

    Set SGLANG_URL=http://localhost:30000 to enable.
    The server must be launched separately (e.g., from the eval-sglang conda env).
    """
    import os
    url = os.environ.get("SGLANG_URL", "").strip()
    if not url:
        return None
    try:
        import urllib.request
        urllib.request.urlopen(f"{url}/v1/models", timeout=2)
        return url
    except Exception:
        return None


def generate_proposals_batch(
    innovations_and_evidence: list[tuple[Innovation, list[PaperRecord]]],
    realization_model_path: str,
    config: RealizationConfig,
    *,
    context_papers: list[PaperRecord] | None = None,
    base_model_name: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> list[str]:
    """Batch-generate proposals for multiple innovations.

    Uses SGLang offline engine when available (~10x faster than HF generate).
    Falls back to HF model.generate() otherwise.
    """
    if not innovations_and_evidence:
        return []

    from forecaster.prior.sampler import _detect_base_model

    # Build all prompts
    prompt_data = _load_prompt()
    prompts: list[str] = []
    for innovation, evidence in innovations_and_evidence:
        user_msg = _build_user_message(prompt_data, innovation, evidence, context_papers=context_papers, config=config)
        prompts.append(f"{prompt_data['system_prompt']}\n\n{user_msg}".strip())

    # --- SGLang API fast path (~10x faster than HF generate) ---
    # Use chat.completions so the SGLang server applies the Qwen3.5 chat
    # template with enable_thinking=True — matching how training renders
    # prompts and how the HF eval path applies the template. Wrapping the
    # already-concatenated f"{system}\n\n{user}" string as a single user turn
    # keeps all three paths byte-identical.
    sglang_url = _sglang_api_url()
    if sglang_url:
        try:
            import openai
            client = openai.OpenAI(base_url=f"{sglang_url}/v1", api_key="none")
            models = client.models.list()
            model_id = models.data[0].id if models.data else "default"

            results = []
            for prompt in prompts:
                r = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.proposal_max_tokens,
                    # Match training: Qwen3.5 thinking-mode anti-loop sampling.
                    temperature=1.0 if temperature is None else temperature,
                    top_p=0.95 if top_p is None else top_p,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": True},
                        "top_k": 20,
                        "repetition_penalty": 1.5,
                    },
                )
                content = r.choices[0].message.content or ""
                results.append(strip_think_block(content).strip())
            return results
        except Exception as exc:
            logger.warning("SGLang API failed (%s); falling back to HF generate.", exc)

    # --- HF generate fallback ---
    from forecaster.realization.local_generation import (
        _apply_chat_template,
        _load_local_model,
        _require_local_generation_stack,
    )

    resolved_base = base_model_name
    adapter_path = Path(realization_model_path) / "adapter_config.json"
    if resolved_base is None and adapter_path.exists():
        resolved_base = _detect_base_model(realization_model_path)

    model, tokenizer = _load_local_model(realization_model_path, base_model_name=resolved_base)
    deps = _require_local_generation_stack()
    torch = deps["torch"]

    chat_prompts = [_apply_chat_template(tokenizer, p, True) for p in prompts]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    encoded = tokenizer(chat_prompts, return_tensors="pt", padding=True, truncation=True)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=config.proposal_max_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            # Match training: Qwen3.5 thinking-mode anti-loop sampling.
            temperature=1.0 if temperature is None else temperature,
            top_p=0.95 if top_p is None else top_p,
            top_k=20,
            repetition_penalty=1.5,
        )

    results = []
    for seq in generated:
        output_ids = seq[encoded["input_ids"].shape[1]:].tolist()
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
        results.append(strip_think_block(decoded).strip())

    tokenizer.padding_side = "right"
    return results


def _sglang_api_url() -> str | None:
    """Return the SGLang server URL if available, or None.

    Set SGLANG_URL=http://localhost:30000 to enable.
    The server must be launched separately (e.g., from the eval-sglang conda env).
    """
    import os
    url = os.environ.get("SGLANG_URL", "").strip()
    if not url:
        return None
    try:
        import urllib.request
        urllib.request.urlopen(f"{url}/v1/models", timeout=2)
        return url
    except Exception:
        return None


def generate_local_proposal(
    innovation: Innovation,
    evidence: list[PaperRecord],
    *,
    context_papers: list[PaperRecord] | None = None,
    model_name_or_path: str,
    config: RealizationConfig,
    base_model_name: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> str:
    """Generate a proposal with a local model identifier or checkpoint path."""
    system_prompt, user_msg = build_realization_messages(
        innovation,
        evidence,
        context_papers=context_papers,
        config=config,
    )
    return _generate_proposal_local(
        user_msg=user_msg,
        system_prompt=system_prompt,
        realization_model_path=model_name_or_path,
        config=config,
        base_model_name=base_model_name,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )


def score_local_proposal(
    innovation: Innovation,
    evidence: list[PaperRecord],
    proposal_text: str,
    *,
    context_papers: list[PaperRecord] | None = None,
    model_name_or_path: str,
    config: RealizationConfig,
    base_model_name: str | None = None,
    score_normalization: str = "per_token",
    score_temperature: float = 1.0,
) -> float:
    """Score log p_psi(y | z, X) for a concrete proposal under a local policy artifact."""
    from forecaster.prior.sampler import _detect_base_model
    from forecaster.realization.local_generation import (
        _apply_chat_template,
        _load_local_model,
        _require_local_generation_stack,
    )

    if not proposal_text.strip():
        return float("-inf")

    system_prompt, user_msg = build_realization_messages(
        innovation,
        evidence,
        context_papers=context_papers,
        config=config,
    )
    full_prompt = f"{system_prompt}\n\n{user_msg}".strip()
    resolved_base_model_name = base_model_name
    if resolved_base_model_name is None:
        resolved_base_model_name = _detect_base_model(model_name_or_path)

    deps = _require_local_generation_stack()
    torch = deps["torch"]
    import torch.nn.functional as F

    model, tokenizer = _load_local_model(
        model_name_or_path,
        base_model_name=resolved_base_model_name,
    )
    prompt_text = _apply_chat_template(tokenizer, full_prompt, None)
    prompt_encoded = tokenizer([prompt_text], return_tensors="pt")
    prompt_len = prompt_encoded["input_ids"].shape[1]
    encoded = tokenizer([f"{prompt_text}{proposal_text}"], return_tensors="pt")
    encoded = {name: value.to(model.device) for name, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    logits = outputs.logits[:, :-1, :]
    if score_temperature > 0 and score_temperature != 1.0:
        logits = logits / score_temperature
    target_ids = encoded["input_ids"][:, 1:]
    target_log_probs = F.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)
    target_start = max(0, prompt_len - 1)
    conditioned = target_log_probs[:, target_start:]
    if conditioned.numel() == 0:
        return float("-inf")
    if str(score_normalization or "per_token").strip().lower() == "sum":
        return float(conditioned.sum().item())
    return float(conditioned.mean().item())


def generate_proposal(
    innovation: Innovation,
    evidence: list[PaperRecord],
    llm_client: Any,
    model: str,
    config: RealizationConfig,
    *,
    context_papers: list[PaperRecord] | None = None,
    realization_model_path: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    allow_llm_fallback_on_artifact_error: bool | None = None,
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
    system_prompt, user_msg = build_realization_messages(
        innovation,
        evidence,
        context_papers=context_papers,
        config=config,
    )
    allow_fallback = (
        config.allow_artifact_fallback_to_llm
        if allow_llm_fallback_on_artifact_error is None
        else allow_llm_fallback_on_artifact_error
    )

    if realization_model_path:
        resolved_model_ref = resolve_model_reference(realization_model_path)
        if resolved_model_ref is None:
            raise FileNotFoundError(
                f"Realization model reference could not be resolved: {realization_model_path}"
            )
        try:
            return generate_local_proposal(
                innovation,
                evidence,
                context_papers=context_papers,
                model_name_or_path=resolved_model_ref,
                config=config,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    f"Realization artifact generation failed for {realization_model_path}: {exc}"
                ) from exc
            logger.warning(
                "Local realization model generation failed (%s); falling back to LLM client.",
                exc,
            )

    response_text, _ = get_response_from_llm(
        msg=user_msg,
        client=llm_client,
        model=model,
        system_message=system_prompt,
        temperature=0.75 if temperature is None else temperature,
        top_p=top_p,
        seed=seed,
    )

    return response_text.strip()


def proposal_to_idea_prediction(
    proposal_text: str,
    innovation: Innovation,
    rank: int = 1,
) -> IdeaPrediction:
    """Convert a proposal text to an IdeaPrediction for benchmark evaluation.

    Extracts title from first line, uses gap as rationale, operator as approach.
    Strips any leading <think>...</think> block first so reasoning traces from
    Qwen3.5 thinking mode don't leak into the title.
    """
    lines = strip_think_block(proposal_text).strip().splitlines()
    title = lines[0].strip() if lines else ""
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    return IdeaPrediction(
        rank=rank,
        title=title,
        rationale=f"{innovation.gap}. {body}"[:500] if body else innovation.gap,
        approach=f"{innovation.operator}: {innovation.base_direction}",
        score=0.0,
    )
