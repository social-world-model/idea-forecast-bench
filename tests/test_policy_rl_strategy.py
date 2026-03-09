from __future__ import annotations

import json
from pathlib import Path

from live_idea_bench.models import PaperRecord
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
