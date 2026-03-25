"""Tests for strict schema and manifest contracts."""
from __future__ import annotations

import pytest

from forecaster.models import (
    Innovation,
    STRICT_REWARD_CONTRACT_VERSION,
    STRICT_RUNTIME_MANIFEST_VERSION,
    STRICT_SCORE_CONTRACT_VERSION,
    STRICT_SEARCH_ACTION_SCHEMA_VERSION,
    STRICT_TRAJECTORY_SCHEMA_VERSION,
    SearchAction,
    innovation_to_json,
    strict_runtime_manifest_contract,
    strict_search_contract,
)


def test_innovation_to_json_uses_frozen_field_order() -> None:
    innovation = Innovation(
        base_direction="test direction",
        operator="extend",
        gap="test gap",
    )

    assert innovation_to_json(innovation) == (
        '{"base_direction":"test direction","operator":"extend","gap":"test gap"}'
    )


def test_strict_runtime_manifest_contract_freezes_versions() -> None:
    manifest = strict_runtime_manifest_contract()

    assert manifest["manifest_version"] == STRICT_RUNTIME_MANIFEST_VERSION
    assert manifest["search_action_schema_version"] == STRICT_SEARCH_ACTION_SCHEMA_VERSION
    assert manifest["trajectory_schema_version"] == STRICT_TRAJECTORY_SCHEMA_VERSION
    assert manifest["reward_contract_version"] == STRICT_REWARD_CONTRACT_VERSION
    assert manifest["score_contract_version"] == STRICT_SCORE_CONTRACT_VERSION
    assert manifest["joint_score_formula"] == "linear_blend(prior_score, realization_score)"
    assert tuple(manifest["joint_score_components"]) == ("prior_score", "realization_score")
    assert manifest["allows_extra_bonus_terms"] is False
    assert manifest["policy_emits_one_action_per_turn"] is True
    assert manifest["interactive_rollout_required"] is True
    assert manifest["full_action_list_completion_valid"] is False
    assert manifest["strict_realization_loop_mode"] == "step_interactive"
    assert manifest["strict_realization_action_schema"] == "single_action_json"
    assert manifest["strict_realization_score_factorization"] == "per_step_conditional"


def test_strict_search_contract_exposes_action_types_and_defaults() -> None:
    contract = strict_search_contract()

    assert contract["action_schema_version"] == STRICT_SEARCH_ACTION_SCHEMA_VERSION
    assert tuple(contract["allowed_action_types"]) == ("search", "select", "finish")
    assert contract["policy_emits_one_action_per_turn"] is True
    assert contract["interactive_rollout_required"] is True
    assert contract["full_action_list_completion_valid"] is False
    assert contract["search_env_defaults"]["max_search_steps"] == 3


def test_search_action_rejects_missing_payload() -> None:
    with pytest.raises(ValueError, match="non-empty query"):
        SearchAction(action_type="search")
