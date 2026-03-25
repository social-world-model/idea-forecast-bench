"""Strict realization runtime built on the interactive search environment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from live_idea_bench.llm import get_response_from_llm
from live_idea_bench.models import PaperRecord

from forecaster.config import RealizationConfig
from forecaster.models import (
    Innovation,
    RealizationTrajectory,
    STRICT_SEARCH_ENV_DEFAULTS,
    SearchAction,
    SearchState,
    strict_search_contract,
    search_action_from_dict,
)
from forecaster.realization.local_generation import (
    _apply_chat_template,
    _load_local_model,
    _require_local_generation_stack,
)
from forecaster.realization.search_env import (
    apply_search_action,
    initialize_search_state,
    rollout_search_trajectory,
)


def build_default_search_queries(
    innovation: Innovation,
    *,
    max_search_steps: int = STRICT_SEARCH_ENV_DEFAULTS["max_search_steps"],
) -> tuple[str, ...]:
    """Build a deterministic search plan from the innovation triple."""
    raw_queries = (
        f"{innovation.base_direction} {innovation.gap}",
        f"{innovation.base_direction} {innovation.operator}",
        f"{innovation.operator} {innovation.gap}",
    )
    ordered: list[str] = []
    seen: set[str] = set()
    for query in raw_queries:
        normalized = " ".join(query.split()).strip()
        if normalized and normalized not in seen:
            ordered.append(normalized)
            seen.add(normalized)
        if len(ordered) >= max_search_steps:
            break
    return tuple(ordered)


def build_strict_interactive_messages(
    innovation: Innovation,
    *,
    search_env_payload: dict[str, Any] | None = None,
) -> tuple[str, str]:
    contract = strict_search_contract()
    env = dict(contract["search_env_defaults"])
    if search_env_payload:
        env.update(
            {
                "max_search_steps": int(search_env_payload.get("max_search_steps", env["max_search_steps"]) or env["max_search_steps"]),
                "top_k": int(search_env_payload.get("top_k", env["top_k"]) or env["top_k"]),
                "max_selected_evidence": int(
                    search_env_payload.get("max_selected_evidence", env["max_selected_evidence"])
                    or env["max_selected_evidence"]
                ),
            }
        )
    system_prompt = (
        "You are an interactive research policy. "
        "Return ONLY JSON with an `actions` array. "
        "Each action must use one of: search, select, finish."
    )
    user_prompt = (
        "Innovation to realize:\n"
        f"- base_direction: {innovation.base_direction}\n"
        f"- operator: {innovation.operator}\n"
        f"- gap: {innovation.gap}\n\n"
        "Search environment:\n"
        f"- max_search_steps: {env['max_search_steps']}\n"
        f"- top_k per search: {env['top_k']}\n"
        f"- max_selected_evidence: {env['max_selected_evidence']}\n"
        "- observation fields: paper_id, title, month, summary\n"
        "- select can only reference surfaced paper_id values\n"
        "- finish must include proposal_text\n\n"
        'Output JSON like {"actions":[{"action_type":"search","query":"..."},'
        '{"action_type":"finish","proposal_text":"..."}]}'
    )
    return system_prompt, user_prompt


def _ordered_surfaced_ids(trajectory: list[tuple[str, ...]]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for step_ids in trajectory:
        for paper_id in step_ids:
            if paper_id not in seen:
                ordered.append(paper_id)
                seen.add(paper_id)
    return tuple(ordered)


def parse_search_actions_completion(text: str) -> list[SearchAction] | None:
    """Parse a completion payload into strict search actions.

    Accepted forms:
    - {"actions": [...]}
    - [...]
    """
    raw = str(text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    payload: object
    if start != -1 and end > start:
        try:
            payload = json.loads(raw[start:end])
        except json.JSONDecodeError:
            payload = None
    else:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end <= start:
            return None
        try:
            payload = json.loads(raw[start:end])
        except json.JSONDecodeError:
            return None

    rows: object
    if isinstance(payload, dict):
        rows = payload.get("actions", [])
    else:
        rows = payload
    if not isinstance(rows, list):
        return None
    try:
        return [search_action_from_dict(item) for item in rows if isinstance(item, dict)]
    except ValueError:
        return None


def _serialize_actions(actions: list[SearchAction]) -> str:
    rows: list[dict[str, str]] = []
    for action in actions:
        payload = {"action_type": action.action_type}
        if action.query is not None:
            payload["query"] = action.query
        if action.paper_id is not None:
            payload["paper_id"] = action.paper_id
        if action.proposal_text is not None:
            payload["proposal_text"] = action.proposal_text
        rows.append(payload)
    return json.dumps({"actions": rows}, ensure_ascii=False, separators=(",", ":"))


def _heuristic_proposal_text(innovation: Innovation, selected_evidence: list[PaperRecord]) -> str:
    title = f"{innovation.base_direction.title()} via {innovation.operator.title()}"
    evidence_clause = (
        f" It builds on evidence from {', '.join(paper.title for paper in selected_evidence[:2])}."
        if selected_evidence
        else ""
    )
    body = (
        f"We {innovation.operator} {innovation.base_direction} to address {innovation.gap}."
        f"{evidence_clause} The proposal is grounded in historical work surfaced through the strict search loop."
    )
    return f"{title}\n{body}"


def _replay_search_prefix(
    innovation: Innovation,
    papers: list[PaperRecord],
    *,
    max_search_steps: int,
    top_k: int,
    max_selected_evidence: int,
) -> SearchState:
    state = initialize_search_state(innovation)
    for query in build_default_search_queries(innovation, max_search_steps=max_search_steps):
        state, _ = apply_search_action(
            state,
            SearchAction(action_type="search", query=query),
            papers,
            top_k=top_k,
            max_search_steps=max_search_steps,
            max_selected_evidence=max_selected_evidence,
        )
        if state.done:
            break
    return state


def build_heuristic_strict_actions(
    innovation: Innovation,
    papers: list[PaperRecord],
    *,
    max_search_steps: int = STRICT_SEARCH_ENV_DEFAULTS["max_search_steps"],
    top_k: int = STRICT_SEARCH_ENV_DEFAULTS["top_k"],
    max_selected_evidence: int = STRICT_SEARCH_ENV_DEFAULTS["max_selected_evidence"],
) -> list[SearchAction]:
    state = _replay_search_prefix(
        innovation,
        papers,
        max_search_steps=max_search_steps,
        top_k=top_k,
        max_selected_evidence=max_selected_evidence,
    )
    actions = [SearchAction(action_type="search", query=query) for query in state.search_queries]
    selected_ids = tuple(state.surfaced_paper_ids)[:max_selected_evidence]
    for paper_id in selected_ids:
        actions.append(SearchAction(action_type="select", paper_id=paper_id))
    paper_lookup = {paper.paper_id: paper for paper in papers}
    selected_evidence = [paper_lookup[paper_id] for paper_id in selected_ids if paper_id in paper_lookup]
    actions.append(
        SearchAction(
            action_type="finish",
            proposal_text=_heuristic_proposal_text(innovation, selected_evidence),
        )
    )
    return actions


def _generate_local_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    model_name_or_path: str,
    max_new_tokens: int,
    base_model_name: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> str:
    from forecaster.prior.sampler import _detect_base_model

    resolved_base_model_name = base_model_name
    adapter_config_path = Path(model_name_or_path) / "adapter_config.json"
    if resolved_base_model_name is None and adapter_config_path.exists():
        resolved_base_model_name = _detect_base_model(model_name_or_path)

    model, tokenizer = _load_local_model(model_name_or_path, base_model_name=resolved_base_model_name)
    deps = _require_local_generation_stack()
    torch = deps["torch"]

    if seed is not None:
        torch.manual_seed(seed)

    full_prompt = f"{system_prompt}\n\n{user_prompt}".strip()
    chat_prompt = _apply_chat_template(tokenizer, full_prompt, None)
    encoded = tokenizer([chat_prompt], return_tensors="pt")
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature is not None and temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.9 if top_p is None else top_p,
            }
        )
    generated = model.generate(**encoded, **generation_kwargs)
    output_ids = generated[0][len(encoded["input_ids"][0]) :].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def generate_strict_policy_completion(
    innovation: Innovation,
    papers: list[PaperRecord],
    *,
    llm_client: object | None,
    model: str | None,
    realization_config: RealizationConfig,
    realization_model_path: str | None = None,
    search_env_payload: dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    base_model_name: str | None = None,
    backend: str | None = None,
) -> str:
    env = dict(search_env_payload or {})
    max_search_steps = int(env.get("max_search_steps", STRICT_SEARCH_ENV_DEFAULTS["max_search_steps"]) or STRICT_SEARCH_ENV_DEFAULTS["max_search_steps"])
    top_k = int(env.get("top_k", STRICT_SEARCH_ENV_DEFAULTS["top_k"]) or STRICT_SEARCH_ENV_DEFAULTS["top_k"])
    max_selected_evidence = int(
        env.get("max_selected_evidence", STRICT_SEARCH_ENV_DEFAULTS["max_selected_evidence"])
        or STRICT_SEARCH_ENV_DEFAULTS["max_selected_evidence"]
    )
    resolved_backend = str(backend or "").strip().lower()
    if resolved_backend == "heuristic":
        return _serialize_actions(
            build_heuristic_strict_actions(
                innovation,
                papers,
                max_search_steps=max_search_steps,
                top_k=top_k,
                max_selected_evidence=max_selected_evidence,
            )
        )

    system_prompt, user_prompt = build_strict_interactive_messages(
        innovation,
        search_env_payload=env,
    )
    if realization_model_path:
        if not Path(realization_model_path).exists():
            raise FileNotFoundError(f"Realization artifact path does not exist: {realization_model_path}")
        return _generate_local_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name_or_path=realization_model_path,
            max_new_tokens=realization_config.proposal_max_tokens,
            base_model_name=base_model_name,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
    if llm_client is None or not str(model or "").strip():
        raise ValueError("Strict policy completion requires either a realization_model_path or llm_client/model.")
    response_text, _ = get_response_from_llm(
        user_prompt,
        llm_client,
        str(model),
        system_prompt,
        temperature=0.7 if temperature is None else temperature,
        top_p=top_p,
        seed=seed,
    )
    return response_text.strip()


def score_strict_policy_completion(
    innovation: Innovation,
    completion_text: str,
    *,
    model_name_or_path: str,
    search_env_payload: dict[str, Any] | None = None,
    base_model_name: str | None = None,
    score_normalization: str = "per_token",
    score_temperature: float = 1.0,
) -> float:
    from forecaster.prior.sampler import _detect_base_model

    if not completion_text.strip():
        return float("-inf")
    system_prompt, user_prompt = build_strict_interactive_messages(
        innovation,
        search_env_payload=search_env_payload,
    )
    deps = _require_local_generation_stack()
    torch = deps["torch"]
    import torch.nn.functional as F

    resolved_base_model_name = base_model_name or _detect_base_model(model_name_or_path)
    model, tokenizer = _load_local_model(
        model_name_or_path,
        base_model_name=resolved_base_model_name,
    )
    prompt_text = _apply_chat_template(tokenizer, f"{system_prompt}\n\n{user_prompt}".strip(), None)
    prompt_encoded = tokenizer([prompt_text], return_tensors="pt")
    prompt_len = prompt_encoded["input_ids"].shape[1]
    encoded = tokenizer([f"{prompt_text}{completion_text}"], return_tensors="pt")
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


def run_strict_realization_rollout(
    innovation: Innovation,
    papers: list[PaperRecord],
    *,
    llm_client: object | None,
    model: str | None,
    realization_config: RealizationConfig,
    realization_model_path: str | None = None,
    completion_text: str | None = None,
    search_env_payload: dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    base_model_name: str | None = None,
    max_search_steps: int = STRICT_SEARCH_ENV_DEFAULTS["max_search_steps"],
    top_k: int = STRICT_SEARCH_ENV_DEFAULTS["top_k"],
    max_selected_evidence: int = STRICT_SEARCH_ENV_DEFAULTS["max_selected_evidence"],
) -> tuple[RealizationTrajectory, list[PaperRecord]]:
    """Execute the strict inference runtime over the shared search environment."""
    env = dict(search_env_payload or {})
    resolved_max_search_steps = int(env.get("max_search_steps", max_search_steps) or max_search_steps)
    resolved_top_k = int(env.get("top_k", top_k) or top_k)
    resolved_max_selected = int(env.get("max_selected_evidence", max_selected_evidence) or max_selected_evidence)
    resolved_completion = str(completion_text or "").strip()
    if not resolved_completion:
        resolved_completion = generate_strict_policy_completion(
            innovation,
            papers,
            llm_client=llm_client,
            model=model,
            realization_config=realization_config,
            realization_model_path=realization_model_path,
            search_env_payload=env,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            base_model_name=base_model_name,
        )
    actions = parse_search_actions_completion(resolved_completion)
    if not actions:
        return (
            RealizationTrajectory(
                innovation=innovation,
                steps=tuple(),
                result=None,
                invalid_reason="invalid_action_sequence",
            ),
            [],
        )
    trajectory = rollout_search_trajectory(
        innovation,
        actions,
        papers,
        top_k=resolved_top_k,
        max_search_steps=resolved_max_search_steps,
        max_selected_evidence=resolved_max_selected,
    )
    paper_lookup = {paper.paper_id: paper for paper in papers}
    selected_ids = trajectory.result.selected_evidence_ids if trajectory.result is not None else ()
    selected_evidence = [paper_lookup[paper_id] for paper_id in selected_ids if paper_id in paper_lookup]
    return trajectory, selected_evidence
