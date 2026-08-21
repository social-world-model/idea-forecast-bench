"""Strict realization runtime built on the interactive search environment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forecaster.config import RealizationConfig
from forecaster.models import (
    STRICT_SEARCH_ENV_DEFAULTS,
    Innovation,
    RealizationTrajectory,
    RealizationTrajectoryStep,
    SearchAction,
    SearchState,
    realization_trajectory_from_dict,
    realization_trajectory_to_dict,
    search_action_from_dict,
    search_action_to_dict,
    search_observation_to_dict,
    strict_search_contract,
)
from forecaster.realization.local_generation import (
    apply_chat_template,
    load_local_model,
    require_local_generation_stack,
)
from forecaster.realization.search_env import (
    apply_search_action,
    initialize_search_state,
    strict_result_from_state,
)
from live_idea_bench.llm import get_response_from_llm
from live_idea_bench.models import PaperRecord


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


def _resolve_search_env(
    search_env_payload: dict[str, Any] | None = None,
    *,
    max_search_steps: int = STRICT_SEARCH_ENV_DEFAULTS["max_search_steps"],
    top_k: int = STRICT_SEARCH_ENV_DEFAULTS["top_k"],
    max_selected_evidence: int = STRICT_SEARCH_ENV_DEFAULTS["max_selected_evidence"],
) -> dict[str, int]:
    payload = dict(search_env_payload or {})
    return {
        "max_search_steps": int(
            payload.get("max_search_steps", max_search_steps) or max_search_steps
        ),
        "top_k": int(payload.get("top_k", top_k) or top_k),
        "max_selected_evidence": int(
            payload.get("max_selected_evidence", max_selected_evidence)
            or max_selected_evidence
        ),
    }


def build_strict_interactive_messages(
    innovation: Innovation,
    *,
    state: SearchState | None = None,
    search_env_payload: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Build the single-step strict policy prompt from the current search state."""
    resolved_state = state or initialize_search_state(innovation)
    contract = strict_search_contract()
    env = _resolve_search_env(search_env_payload)
    action_history = [
        search_action_to_dict(action) for action in resolved_state.action_history
    ]
    previous_observation = [
        search_observation_to_dict(obs) for obs in resolved_state.last_observation
    ]
    system_prompt = (
        "You are an interactive research policy inside a strict search environment. "
        "Return ONLY one JSON object for the NEXT action. "
        "Valid outputs are exactly one of "
        '{"action_type":"search","query":"..."}, '
        '{"action_type":"select","paper_id":"..."}, or '
        '{"action_type":"finish","proposal_text":"..."}. '
        "Do not return action lists, wrappers, plans, or explanations."
    )
    user_prompt = (
        "Innovation z:\n"
        f"- base_direction: {innovation.base_direction}\n"
        f"- operator: {innovation.operator}\n"
        f"- gap: {innovation.gap}\n\n"
        "Current strict rollout state:\n"
        f"- step_index: {len(resolved_state.action_history)}\n"
        f"- search_steps_used: {resolved_state.step_index}\n"
        f"- action_history: {json.dumps(action_history, ensure_ascii=False)}\n"
        f"- previous_observation: {json.dumps(previous_observation, ensure_ascii=False)}\n"
        f"- surfaced_paper_ids: {json.dumps(list(resolved_state.surfaced_paper_ids), ensure_ascii=False)}\n"
        f"- selected_evidence_ids: {json.dumps(list(resolved_state.selected_evidence_ids), ensure_ascii=False)}\n\n"
        "Environment constraints:\n"
        f"- max_search_steps: {env['max_search_steps']}\n"
        f"- top_k: {env['top_k']}\n"
        f"- max_selected_evidence: {env['max_selected_evidence']}\n"
        f"- observation_fields: {json.dumps(list(contract['observation_fields']), ensure_ascii=False)}\n"
        "- select can only reference surfaced_paper_ids\n"
        "- full action lists are invalid in strict mode\n"
        "- finish must include proposal_text\n\n"
        'Return one JSON object only, for example {"action_type":"search","query":"..."}'
    )
    return system_prompt, user_prompt


def extract_json_payload(text: str) -> object | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    decoder = json.JSONDecoder()
    payload: object
    for index, char in enumerate(raw):
        if char not in "{[":
            continue
        try:
            payload, _ = decoder.raw_decode(raw[index:])
            return payload
        except json.JSONDecodeError:
            continue
    return None


def parse_search_actions_completion(text: str) -> list[SearchAction] | None:
    """Parse a legacy completion payload into a list of search actions."""
    payload = extract_json_payload(text)
    rows: object
    rows = payload.get("actions", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return None
    try:
        return [
            search_action_from_dict(item) for item in rows if isinstance(item, dict)
        ]
    except ValueError:
        return None


def parse_single_search_action_completion(text: str) -> SearchAction | None:
    """Parse a strict single-action completion payload."""
    payload = extract_json_payload(text)
    if not isinstance(payload, dict):
        return None
    if "actions" in payload or "steps" in payload or "strict_rollout" in payload:
        return None
    try:
        return search_action_from_dict(payload)
    except ValueError:
        return None


def _strict_completion_invalid_reason(text: str) -> str:
    payload = extract_json_payload(text)
    if isinstance(payload, list):
        return "full_action_list_invalid_in_strict_mode"
    if isinstance(payload, dict) and "actions" in payload:
        return "full_action_list_invalid_in_strict_mode"
    return "invalid_single_action_completion"


def serialize_search_action_completion(action: SearchAction) -> str:
    """Serialize a strict single-step action completion."""
    return json.dumps(
        search_action_to_dict(action), ensure_ascii=False, separators=(",", ":")
    )


def serialize_strict_rollout_completion(trajectory: RealizationTrajectory) -> str:
    """Serialize a stepwise strict rollout artifact for reward/training adapters."""
    return json.dumps(
        {"strict_rollout": realization_trajectory_to_dict(trajectory)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_strict_rollout_completion(text: str) -> RealizationTrajectory | None:
    """Parse a serialized strict rollout artifact."""
    payload = extract_json_payload(text)
    if not isinstance(payload, dict):
        return None
    trajectory_payload = payload.get("strict_rollout", payload)
    if not isinstance(trajectory_payload, dict):
        return None
    if "steps" not in trajectory_payload or "innovation" not in trajectory_payload:
        return None
    try:
        return realization_trajectory_from_dict(trajectory_payload)
    except ValueError:
        return None


def _heuristic_proposal_text(
    innovation: Innovation, selected_evidence: list[PaperRecord]
) -> str:
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


def _heuristic_next_action(
    state: SearchState,
    papers: list[PaperRecord],
    *,
    max_search_steps: int,
    max_selected_evidence: int,
) -> SearchAction:
    planned_queries = build_default_search_queries(
        state.innovation, max_search_steps=max_search_steps
    )
    for query in planned_queries:
        if query not in state.search_queries:
            return SearchAction(action_type="search", query=query)

    for paper_id in state.surfaced_paper_ids:
        if (
            paper_id not in state.selected_evidence_ids
            and len(state.selected_evidence_ids) < max_selected_evidence
        ):
            return SearchAction(action_type="select", paper_id=paper_id)

    paper_lookup = {paper.paper_id: paper for paper in papers}
    selected_evidence = [
        paper_lookup[paper_id]
        for paper_id in state.selected_evidence_ids
        if paper_id in paper_lookup
    ]
    return SearchAction(
        action_type="finish",
        proposal_text=_heuristic_proposal_text(state.innovation, selected_evidence),
    )


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

    model, tokenizer = load_local_model(
        model_name_or_path, base_model_name=resolved_base_model_name
    )
    deps = require_local_generation_stack()
    torch = deps["torch"]

    if seed is not None:
        torch.manual_seed(seed)

    full_prompt = f"{system_prompt}\n\n{user_prompt}".strip()
    chat_prompt = apply_chat_template(tokenizer, full_prompt, None)
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
    decoded: str = tokenizer.decode(output_ids, skip_special_tokens=True)
    return decoded.strip()


def generate_strict_policy_completion(
    innovation: Innovation,
    papers: list[PaperRecord],
    *,
    state: SearchState | None = None,
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
    """Generate the next strict single-action completion for the current rollout state."""
    resolved_state = state or initialize_search_state(innovation)
    env = _resolve_search_env(search_env_payload)
    resolved_backend = str(backend or "").strip().lower()
    if resolved_backend == "heuristic":
        action = _heuristic_next_action(
            resolved_state,
            papers,
            max_search_steps=env["max_search_steps"],
            max_selected_evidence=env["max_selected_evidence"],
        )
        return serialize_search_action_completion(action)

    system_prompt, user_prompt = build_strict_interactive_messages(
        resolved_state.innovation,
        state=resolved_state,
        search_env_payload=env,
    )
    if realization_model_path:
        if not Path(realization_model_path).exists():
            raise FileNotFoundError(
                f"Realization artifact path does not exist: {realization_model_path}"
            )
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
        raise ValueError(
            "Strict policy completion requires either a realization_model_path or llm_client/model."
        )
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


def _score_conditioned_completion_with_model(
    *,
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    completion_text: str,
    score_normalization: str,
    score_temperature: float,
) -> float:
    deps = require_local_generation_stack()
    torch = deps["torch"]
    import torch.nn.functional as F

    prompt_text = apply_chat_template(
        tokenizer, f"{system_prompt}\n\n{user_prompt}".strip(), None
    )
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
    target_log_probs = (
        F.log_softmax(logits, dim=-1)
        .gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    target_start = max(0, prompt_len - 1)
    conditioned = target_log_probs[:, target_start:]
    if conditioned.numel() == 0:
        return float("-inf")
    if str(score_normalization or "per_token").strip().lower() == "sum":
        return float(conditioned.sum().item())
    return float(conditioned.mean().item())


def _advance_state_from_step_trace(
    state: SearchState, step: RealizationTrajectoryStep
) -> SearchState:
    surfaced_ids = step.surfaced_paper_ids or (
        state.surfaced_paper_ids
        + tuple(
            obs.paper_id
            for obs in step.observation
            if obs.paper_id not in state.surfaced_paper_ids
        )
    )
    if step.action.action_type == "search":
        observation_history = state.observation_history + (step.observation,)
        step_index = state.step_index + 1
        proposal_text = state.proposal_text
        done = False
    elif step.action.action_type == "finish":
        observation_history = state.observation_history
        step_index = state.step_index
        proposal_text = step.action.proposal_text
        done = True
    else:
        observation_history = state.observation_history
        step_index = state.step_index
        proposal_text = state.proposal_text
        done = False
    return SearchState(
        innovation=state.innovation,
        step_index=step_index,
        action_history=state.action_history + (step.action,),
        last_observation=step.observation,
        observation_history=observation_history,
        surfaced_paper_ids=surfaced_ids,
        selected_evidence_ids=step.selected_evidence_ids,
        search_queries=(
            state.search_queries + (step.action.query.strip(),)
            if step.action.action_type == "search"
            else state.search_queries
        ),
        proposal_text=proposal_text,
        done=done,
        invalid_reason=None,
    )


def score_strict_realization_trajectory(
    trajectory: RealizationTrajectory,
    *,
    model_name_or_path: str,
    search_env_payload: dict[str, Any] | None = None,
    base_model_name: str | None = None,
    score_normalization: str = "per_token",
    score_temperature: float = 1.0,
) -> float:
    """Score a strict rollout by aggregating per-step conditional action log-probs."""
    from forecaster.prior.sampler import _detect_base_model

    if not trajectory.steps:
        return float("-inf")

    resolved_base_model_name = base_model_name
    if resolved_base_model_name is None:
        resolved_base_model_name = _detect_base_model(model_name_or_path)

    model, tokenizer = load_local_model(
        model_name_or_path,
        base_model_name=resolved_base_model_name,
    )
    state = initialize_search_state(trajectory.innovation)
    step_scores: list[float] = []
    normalized = str(score_normalization or "per_token").strip().lower()
    for step in trajectory.steps:
        system_prompt = step.prompt_system
        user_prompt = step.prompt_user
        if not system_prompt or not user_prompt:
            system_prompt, user_prompt = build_strict_interactive_messages(
                trajectory.innovation,
                state=state,
                search_env_payload=search_env_payload,
            )
        action_completion = serialize_search_action_completion(step.action)
        step_score = _score_conditioned_completion_with_model(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            completion_text=action_completion,
            score_normalization=normalized,
            score_temperature=score_temperature,
        )
        if step_score == float("-inf"):
            return step_score
        step_scores.append(step_score)
        state = _advance_state_from_step_trace(state, step)
    if normalized == "sum":
        return float(sum(step_scores))
    return float(sum(step_scores) / len(step_scores))


def score_strict_policy_completion(
    trajectory: RealizationTrajectory,
    *,
    model_name_or_path: str,
    search_env_payload: dict[str, Any] | None = None,
    base_model_name: str | None = None,
    score_normalization: str = "per_token",
    score_temperature: float = 1.0,
) -> float:
    """Backward-compatible wrapper for strict per-step trajectory scoring."""
    return score_strict_realization_trajectory(
        trajectory,
        model_name_or_path=model_name_or_path,
        search_env_payload=search_env_payload,
        base_model_name=base_model_name,
        score_normalization=score_normalization,
        score_temperature=score_temperature,
    )


def run_strict_realization_rollout(
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
    max_search_steps: int = STRICT_SEARCH_ENV_DEFAULTS["max_search_steps"],
    top_k: int = STRICT_SEARCH_ENV_DEFAULTS["top_k"],
    max_selected_evidence: int = STRICT_SEARCH_ENV_DEFAULTS["max_selected_evidence"],
) -> tuple[RealizationTrajectory, list[PaperRecord]]:
    """Execute the strict step-interactive realization runtime."""
    env = _resolve_search_env(
        search_env_payload,
        max_search_steps=max_search_steps,
        top_k=top_k,
        max_selected_evidence=max_selected_evidence,
    )
    state = initialize_search_state(innovation)
    steps: list[RealizationTrajectoryStep] = []
    max_policy_turns = max(
        1, env["max_search_steps"] + env["max_selected_evidence"] + 1
    )

    for _ in range(max_policy_turns):
        system_prompt, user_prompt = build_strict_interactive_messages(
            innovation,
            state=state,
            search_env_payload=env,
        )
        completion_text = generate_strict_policy_completion(
            innovation,
            papers,
            state=state,
            llm_client=llm_client,
            model=model,
            realization_config=realization_config,
            realization_model_path=realization_model_path,
            search_env_payload=env,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            base_model_name=base_model_name,
            backend=backend,
        )
        action = parse_single_search_action_completion(completion_text)
        if action is None:
            return (
                RealizationTrajectory(
                    innovation=innovation,
                    steps=tuple(steps),
                    result=None,
                    invalid_reason=_strict_completion_invalid_reason(completion_text),
                ),
                [],
            )
        prior_observation = state.last_observation
        next_state, observation = apply_search_action(
            state,
            action,
            papers,
            top_k=env["top_k"],
            max_search_steps=env["max_search_steps"],
            max_selected_evidence=env["max_selected_evidence"],
        )
        steps.append(
            RealizationTrajectoryStep(
                action=action,
                step_index=len(steps),
                prompt_system=system_prompt,
                prompt_user=user_prompt,
                prior_observation=prior_observation,
                observation=observation,
                surfaced_paper_ids=next_state.surfaced_paper_ids,
                selected_evidence_ids=next_state.selected_evidence_ids,
            )
        )
        state = next_state
        if state.done or state.invalid_reason:
            break
    else:
        return (
            RealizationTrajectory(
                innovation=innovation,
                steps=tuple(steps),
                result=None,
                invalid_reason="max_policy_turns_exceeded",
            ),
            [],
        )

    trajectory = RealizationTrajectory(
        innovation=innovation,
        steps=tuple(steps),
        result=strict_result_from_state(state),
        invalid_reason=state.invalid_reason,
    )
    paper_lookup = {paper.paper_id: paper for paper in papers}
    selected_ids = (
        trajectory.result.selected_evidence_ids if trajectory.result is not None else ()
    )
    selected_evidence = [
        paper_lookup[paper_id] for paper_id in selected_ids if paper_id in paper_lookup
    ]
    return trajectory, selected_evidence
