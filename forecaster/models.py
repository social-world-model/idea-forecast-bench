"""Domain models for the forecaster package.

All models are frozen (immutable) dataclasses representing the factorized
latent variable model: p(Y|X) ≈ Π_j Σ_z p(y_j|z_j,X) p(z_j|X).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

INNOVATION_SCHEMA_VERSION = 1
STRICT_SEARCH_ACTION_SCHEMA_VERSION = 1
STRICT_TRAJECTORY_SCHEMA_VERSION = 2
STRICT_RUNTIME_MANIFEST_VERSION = 3
STRICT_REWARD_CONTRACT_VERSION = 1
STRICT_SCORE_CONTRACT_VERSION = 1
ALLOWED_INNOVATION_OPERATORS: tuple[str, ...] = (
    "extend",
    "transfer",
    "compose",
    "benchmark",
    "analyze",
    "simplify",
    "scale",
    "adapt",
)
ALLOWED_SEARCH_ACTION_TYPES: tuple[str, ...] = ("search", "select", "finish")
STRICT_SEARCH_ENV_DEFAULTS: dict[str, int] = {
    "max_search_steps": 3,
    "top_k": 5,
    "max_selected_evidence": 5,
}


@dataclass(frozen=True)
class Innovation:
    """Latent innovation variable z = {base_direction, operator, gap}."""

    base_direction: str  # The foundational research direction being built upon
    operator: str  # The methodological operator (e.g. "extend", "transfer", "compose")
    gap: str  # The specific research gap being addressed


@dataclass(frozen=True)
class MemoryEntry:
    """A single entry in the memory inventory with metadata."""

    innovation: Innovation
    source_paper_id: str
    timestamp_month: str  # Format: "YYYY-MM"
    frequency: int = 1
    recency_score: float = 1.0
    utility_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryInventory:
    """Serializable container of memory entries."""

    entries: tuple[MemoryEntry, ...]  # tuple (immutable) not list
    last_updated_month: str
    version: int = 1


@dataclass(frozen=True)
class HindsightSample:
    """One (context, z) training example from hindsight extraction."""

    context_paper_ids: tuple[str, ...]
    cutoff_month: str
    future_paper_id: str
    future_paper_published_date: str
    innovation: Innovation

    @property
    def future_paper_month(self) -> str:
        """Month bucket for the future paper that produced this hindsight label."""
        return self.future_paper_published_date[:7]


@dataclass(frozen=True)
class SearchObservation:
    """Strict search observation exposed to the realization policy."""

    paper_id: str
    title: str
    month: str
    summary: str


@dataclass(frozen=True)
class SearchAction:
    """One policy action in the strict search environment."""

    action_type: str
    query: str = ""
    paper_id: str = ""
    proposal_text: str = ""

    def __post_init__(self) -> None:
        if self.action_type not in ALLOWED_SEARCH_ACTION_TYPES:
            raise ValueError(
                f"Unsupported search action_type={self.action_type!r}. "
                f"Allowed types: {ALLOWED_SEARCH_ACTION_TYPES}."
            )
        if self.action_type == "search" and not self.query.strip():
            raise ValueError("SearchAction(type='search') requires a non-empty query.")
        if self.action_type == "select" and not self.paper_id.strip():
            raise ValueError("SearchAction(type='select') requires a surfaced paper_id.")
        if self.action_type == "finish" and not self.proposal_text.strip():
            raise ValueError("SearchAction(type='finish') requires a non-empty proposal_text.")


@dataclass(frozen=True)
class SearchState:
    """Mutable search state represented as an immutable dataclass snapshot."""

    innovation: Innovation
    step_index: int = 0
    action_history: tuple[SearchAction, ...] = ()
    last_observation: tuple[SearchObservation, ...] = ()
    observation_history: tuple[tuple[SearchObservation, ...], ...] = ()
    surfaced_paper_ids: tuple[str, ...] = ()
    selected_evidence_ids: tuple[str, ...] = ()
    search_queries: tuple[str, ...] = ()
    proposal_text: str = ""
    done: bool = False
    invalid_reason: str | None = None


@dataclass(frozen=True)
class StrictRealizationResult:
    """Strict realization completion shared by training and inference."""

    selected_evidence_ids: tuple[str, ...]
    proposal_text: str
    search_queries: tuple[str, ...]


@dataclass(frozen=True)
class RealizationTrajectoryStep:
    """Serializable one-step trace of the strict search environment."""

    action: SearchAction
    step_index: int = 0
    prompt_system: str = ""
    prompt_user: str = ""
    prior_observation: tuple[SearchObservation, ...] = ()
    observation: tuple[SearchObservation, ...] = ()
    surfaced_paper_ids: tuple[str, ...] = ()
    selected_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class JointCandidate:
    """Intermediate: innovation + scores before final ranking."""

    innovation: Innovation
    prior_score: float
    evidence_paper_ids: tuple[str, ...]
    proposal_text: str
    realization_score: float
    popularity_bonus: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredProposal:
    """Final output: proposal with joint score from inference."""

    innovation: Innovation
    proposal_text: str
    prior_score: float
    realization_score: float
    joint_score: float
    evidence_paper_ids: tuple[str, ...]
    rank: int = 0
    popularity_bonus: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RealizationTrajectory:
    """Strict rollout trace for the interactive realization policy."""

    innovation: Innovation
    steps: tuple[RealizationTrajectoryStep, ...]
    result: StrictRealizationResult | None = None
    invalid_reason: str | None = None
    schema_version: int = STRICT_TRAJECTORY_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def innovation_to_dict(innovation: Innovation) -> dict[str, str]:
    """Serialize an Innovation to a plain dict."""
    return {
        "base_direction": innovation.base_direction,
        "operator": innovation.operator,
        "gap": innovation.gap,
    }


def innovation_to_json(innovation: Innovation) -> str:
    """Serialize an Innovation to the frozen JSON contract."""
    return json.dumps(
        innovation_to_dict(innovation),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def innovation_schema_contract() -> dict[str, Any]:
    """Return the frozen Innovation runtime contract."""
    return {
        "schema_version": INNOVATION_SCHEMA_VERSION,
        "fields": ("base_direction", "operator", "gap"),
        "allowed_operators": list(ALLOWED_INNOVATION_OPERATORS),
    }


def strict_search_contract() -> dict[str, Any]:
    """Return the frozen strict search environment contract."""
    return {
        "action_schema_version": STRICT_SEARCH_ACTION_SCHEMA_VERSION,
        "trajectory_schema_version": STRICT_TRAJECTORY_SCHEMA_VERSION,
        "allowed_action_types": ALLOWED_SEARCH_ACTION_TYPES,
        "observation_fields": ("paper_id", "title", "month", "summary"),
        "policy_emits_one_action_per_turn": True,
        "interactive_rollout_required": True,
        "full_action_list_completion_valid": False,
        "search_env_defaults": dict(STRICT_SEARCH_ENV_DEFAULTS),
    }


def strict_runtime_manifest_contract() -> dict[str, Any]:
    """Return the frozen strict runtime manifest contract."""
    return {
        "manifest_version": STRICT_RUNTIME_MANIFEST_VERSION,
        "innovation_schema_version": INNOVATION_SCHEMA_VERSION,
        "search_action_schema_version": STRICT_SEARCH_ACTION_SCHEMA_VERSION,
        "trajectory_schema_version": STRICT_TRAJECTORY_SCHEMA_VERSION,
        "reward_contract_version": STRICT_REWARD_CONTRACT_VERSION,
        "score_contract_version": STRICT_SCORE_CONTRACT_VERSION,
        "joint_score_formula": "linear_blend(prior_score, realization_score)",
        "joint_score_components": ("prior_score", "realization_score"),
        "allows_extra_bonus_terms": False,
        "policy_emits_one_action_per_turn": True,
        "interactive_rollout_required": True,
        "full_action_list_completion_valid": False,
        "strict_realization_loop_mode": "step_interactive",
        "strict_realization_action_schema": "single_action_json",
        "strict_realization_score_factorization": "per_step_conditional",
        "search_env_defaults": dict(STRICT_SEARCH_ENV_DEFAULTS),
    }


def innovation_from_dict(d: dict[str, Any]) -> Innovation:
    """Deserialize an Innovation from a plain dict."""
    return Innovation(
        base_direction=str(d["base_direction"]),
        operator=str(d["operator"]),
        gap=str(d["gap"]),
    )


def innovation_from_json(text: str) -> Innovation:
    """Deserialize the frozen Innovation JSON contract."""
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Innovation JSON must decode to an object.")
    return innovation_from_dict(payload)


def memory_entry_to_dict(entry: MemoryEntry) -> dict[str, Any]:
    """Serialize a MemoryEntry to a plain dict."""
    return {
        "innovation": innovation_to_dict(entry.innovation),
        "source_paper_id": entry.source_paper_id,
        "timestamp_month": entry.timestamp_month,
        "frequency": entry.frequency,
        "recency_score": entry.recency_score,
        "utility_score": entry.utility_score,
        "metadata": dict(entry.metadata),
    }


def memory_entry_from_dict(d: dict[str, Any]) -> MemoryEntry:
    """Deserialize a MemoryEntry from a plain dict."""
    return MemoryEntry(
        innovation=innovation_from_dict(d["innovation"]),
        source_paper_id=str(d["source_paper_id"]),
        timestamp_month=str(d["timestamp_month"]),
        frequency=int(d.get("frequency", 1)),
        recency_score=float(d.get("recency_score", 1.0)),
        utility_score=float(d.get("utility_score", 0.0)),
        metadata=dict(d.get("metadata", {})),
    )


def memory_inventory_to_dict(inventory: MemoryInventory) -> dict[str, Any]:
    """Serialize a MemoryInventory to a plain dict."""
    return {
        "entries": [memory_entry_to_dict(e) for e in inventory.entries],
        "last_updated_month": inventory.last_updated_month,
        "version": inventory.version,
    }


def memory_inventory_from_dict(d: dict[str, Any]) -> MemoryInventory:
    """Deserialize a MemoryInventory from a plain dict."""
    entries = tuple(memory_entry_from_dict(e) for e in d.get("entries", []))
    return MemoryInventory(
        entries=entries,
        last_updated_month=str(d["last_updated_month"]),
        version=int(d.get("version", 1)),
    )


def search_observation_to_dict(observation: SearchObservation) -> dict[str, str]:
    """Serialize a strict search observation to a plain dict."""
    return {
        "paper_id": observation.paper_id,
        "title": observation.title,
        "month": observation.month,
        "summary": observation.summary,
    }


def search_observation_from_dict(payload: dict[str, Any]) -> SearchObservation:
    """Deserialize a strict search observation from a plain dict."""
    if not isinstance(payload, dict):
        raise ValueError("Search observation payload must decode to an object.")
    return SearchObservation(
        paper_id=str(payload.get("paper_id", "") or ""),
        title=str(payload.get("title", "") or ""),
        month=str(payload.get("month", "") or ""),
        summary=str(payload.get("summary", "") or ""),
    )


def search_action_to_dict(action: SearchAction) -> dict[str, str]:
    """Serialize a strict search action to a plain dict."""
    payload = {"action_type": action.action_type}
    if action.action_type == "search":
        payload["query"] = action.query
    elif action.action_type == "select":
        payload["paper_id"] = action.paper_id
    elif action.action_type == "finish":
        payload["proposal_text"] = action.proposal_text
    return payload


def search_action_from_dict(payload: dict[str, Any]) -> SearchAction:
    """Deserialize a strict search action from a plain dict."""
    if not isinstance(payload, dict):
        raise ValueError("Search action payload must decode to an object.")
    return SearchAction(
        action_type=str(payload.get("action_type", "")).strip(),
        query=str(payload.get("query", "") or ""),
        paper_id=str(payload.get("paper_id", "") or ""),
        proposal_text=str(payload.get("proposal_text", "") or ""),
    )


def strict_realization_result_to_dict(result: StrictRealizationResult) -> dict[str, Any]:
    """Serialize the strict realization completion contract."""
    return {
        "selected_evidence_ids": list(result.selected_evidence_ids),
        "proposal_text": result.proposal_text,
        "search_queries": list(result.search_queries),
    }


def strict_realization_result_from_dict(payload: dict[str, Any]) -> StrictRealizationResult:
    """Deserialize the strict realization completion contract."""
    if not isinstance(payload, dict):
        raise ValueError("Strict realization result payload must decode to an object.")
    return StrictRealizationResult(
        selected_evidence_ids=tuple(str(item) for item in payload.get("selected_evidence_ids", [])),
        proposal_text=str(payload.get("proposal_text", "") or ""),
        search_queries=tuple(str(item) for item in payload.get("search_queries", [])),
    )


def realization_trajectory_step_to_dict(step: RealizationTrajectoryStep) -> dict[str, Any]:
    """Serialize one rollout step."""
    return {
        "step_index": step.step_index,
        "prompt_system": step.prompt_system,
        "prompt_user": step.prompt_user,
        "prior_observation": [search_observation_to_dict(obs) for obs in step.prior_observation],
        "action": search_action_to_dict(step.action),
        "observation": [search_observation_to_dict(obs) for obs in step.observation],
        "surfaced_paper_ids": list(step.surfaced_paper_ids),
        "selected_evidence_ids": list(step.selected_evidence_ids),
    }


def realization_trajectory_step_from_dict(payload: dict[str, Any]) -> RealizationTrajectoryStep:
    """Deserialize one rollout step."""
    if not isinstance(payload, dict):
        raise ValueError("Realization trajectory step payload must decode to an object.")
    return RealizationTrajectoryStep(
        action=search_action_from_dict(payload.get("action", {})),
        step_index=int(payload.get("step_index", 0) or 0),
        prompt_system=str(payload.get("prompt_system", "") or ""),
        prompt_user=str(payload.get("prompt_user", "") or ""),
        prior_observation=tuple(
            search_observation_from_dict(item)
            for item in payload.get("prior_observation", [])
            if isinstance(item, dict)
        ),
        observation=tuple(
            search_observation_from_dict(item)
            for item in payload.get("observation", [])
            if isinstance(item, dict)
        ),
        surfaced_paper_ids=tuple(str(item) for item in payload.get("surfaced_paper_ids", [])),
        selected_evidence_ids=tuple(str(item) for item in payload.get("selected_evidence_ids", [])),
    )


def realization_trajectory_to_dict(trajectory: RealizationTrajectory) -> dict[str, Any]:
    """Serialize a strict realization trajectory."""
    return {
        "schema_version": trajectory.schema_version,
        "innovation": innovation_to_dict(trajectory.innovation),
        "steps": [realization_trajectory_step_to_dict(step) for step in trajectory.steps],
        "result": (
            strict_realization_result_to_dict(trajectory.result)
            if trajectory.result is not None
            else None
        ),
        "invalid_reason": trajectory.invalid_reason,
    }


def realization_trajectory_from_dict(payload: dict[str, Any]) -> RealizationTrajectory:
    """Deserialize a strict realization trajectory."""
    if not isinstance(payload, dict):
        raise ValueError("Realization trajectory payload must decode to an object.")
    result_payload = payload.get("result")
    return RealizationTrajectory(
        innovation=innovation_from_dict(payload.get("innovation", {})),
        steps=tuple(
            realization_trajectory_step_from_dict(item)
            for item in payload.get("steps", [])
            if isinstance(item, dict)
        ),
        result=(
            strict_realization_result_from_dict(result_payload)
            if isinstance(result_payload, dict)
            else None
        ),
        invalid_reason=str(payload.get("invalid_reason")) if payload.get("invalid_reason") is not None else None,
        schema_version=int(payload.get("schema_version", STRICT_TRAJECTORY_SCHEMA_VERSION) or STRICT_TRAJECTORY_SCHEMA_VERSION),
    )
