from __future__ import annotations

import json
from pathlib import Path

import pytest

import live_idea_bench.rl.local_generation as local_generation
import live_idea_bench.strategy.policy_rl as policy_rl_module
from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.strategy.policy_rl import PolicyRLStrategy
from live_idea_bench.strategy.registry import create_strategy


def _paper(paper_id: str, month: str) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        month=month,
        summary="summary",
        keywords=["agent"],
        source_path=f"/fake/{paper_id}.md",
        published_date=f"{month}-01",
    )


def test_create_strategy_policy_rl() -> None:
    strategy = create_strategy(
        "policy_rl",
        model_name="gpt-4o-mini",
        predictor_config="predictor.yaml",
        similarity_config="similarity.yaml",
        policy_manifest_path="/tmp/policy.json",
    )
    assert isinstance(strategy, PolicyRLStrategy)
    assert strategy.name == "policy_rl"
    assert strategy.policy_manifest_path == "/tmp/policy.json"


def test_policy_rl_strategy_uses_static_predictions_from_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "policy.json"
    manifest_path.write_text(
        json.dumps(
            {
                "policy_type": "policy_rl",
                "stage": "dpo",
                "static_predictions": {
                    "2024-06": [
                        {
                            "rank": 1,
                            "title": "Static RL idea",
                            "rationale": "from manifest",
                            "approach": "manifest approach",
                            "key_terms": ["static"],
                            "confidence": 0.9,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    strategy = PolicyRLStrategy(policy_manifest_path=str(manifest_path))

    predictions = strategy.generate(
        train_papers=[_paper("p1", "2024-05")],
        cutoff_month="2024-06",
        top_k=3,
    )

    assert len(predictions) == 1
    assert predictions[0].title == "Static RL idea"
    assert predictions[0].rank == 1


def test_policy_rl_strategy_uses_base_model_name_for_local_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selection_path = tmp_path / "selection.yaml"
    selection_path.write_text(
        "\n".join(
            [
                "candidate_pool_size: 2",
                "output_top_k: 1",
                "dedup_similarity_threshold: 0.8",
                "temperature_schedule: [0.4]",
                "top_p_schedule: [0.9]",
                "enable_context_shuffle: false",
                "relevance_frequency_weight: 0.5",
                "relevance_confidence_weight: 0.3",
                "relevance_heuristic_weight: 0.2",
                "mmr_relevance_weight: 0.7",
                "mmr_diversity_weight: 0.3",
            ]
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    manifest_path = tmp_path / "policy.json"
    manifest_path.write_text(
        json.dumps(
            {
                "policy_type": "policy_rl",
                "trainer": "grpo",
                "checkpoint_path": str(checkpoint_path),
                "base_model_name": "base-model",
                "inference_model_name": "wrong-model",
                "selection_config_path": str(selection_path),
            }
        ),
        encoding="utf-8",
    )

    captured_base_models: list[str | None] = []

    def _fake_generate_local_predictions(**kwargs):  # type: ignore[no-untyped-def]
        captured_base_models.append(kwargs.get("base_model_name"))
        return [
            IdeaPrediction(
                rank=1,
                title=f"candidate-{kwargs.get('seed')}",
                rationale="rationale",
                approach="approach",
                confidence=0.9,
            )
        ]

    monkeypatch.setattr(local_generation, "generate_local_predictions", _fake_generate_local_predictions)
    strategy = PolicyRLStrategy(policy_manifest_path=str(manifest_path))

    predictions = strategy.generate(
        train_papers=[_paper("p1", "2024-05"), _paper("p2", "2024-05")],
        cutoff_month="2024-06",
        top_k=1,
    )

    assert predictions
    assert captured_base_models == ["base-model", "base-model"]


def test_policy_rl_strategy_uses_sampling_schedule_for_candidate_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selection_path = tmp_path / "selection.yaml"
    selection_path.write_text(
        "\n".join(
            [
                "candidate_pool_size: 4",
                "output_top_k: 2",
                "dedup_similarity_threshold: 0.8",
                "temperature_schedule: [0.35, 0.85]",
                "top_p_schedule: [0.8, 0.95]",
                "enable_context_shuffle: true",
                "relevance_frequency_weight: 0.5",
                "relevance_confidence_weight: 0.3",
                "relevance_heuristic_weight: 0.2",
                "mmr_relevance_weight: 0.7",
                "mmr_diversity_weight: 0.3",
            ]
        ),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def _fake_generate_predictions(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(
            {
                "temperature": kwargs.get("temperature"),
                "top_p": kwargs.get("top_p"),
                "seed": kwargs.get("seed"),
                "train_paper_ids": [paper.paper_id for paper in kwargs["train_papers"]],
            }
        )
        seed = int(kwargs.get("seed", 0))
        return [
            IdeaPrediction(
                rank=1,
                title=f"idea-{seed}",
                rationale="rationale",
                approach="approach",
                confidence=0.8,
            )
        ]

    monkeypatch.setattr(policy_rl_module, "generate_predictions", _fake_generate_predictions)
    strategy = PolicyRLStrategy(
        model_name="gpt-4o-mini",
        selection_config=str(selection_path),
    )

    predictions = strategy.generate(
        train_papers=[_paper("p1", "2024-05"), _paper("p2", "2024-05"), _paper("p3", "2024-05")],
        cutoff_month="2024-06",
        top_k=2,
    )

    assert len(calls) == 4
    assert {call["temperature"] for call in calls} == {0.35, 0.85}
    assert {call["top_p"] for call in calls} == {0.8, 0.95}
    assert any(call["train_paper_ids"] != ["p1", "p2", "p3"] for call in calls)
    assert len(predictions) == 2
    assert all("candidate_sample_index" in prediction.metadata for prediction in predictions)
