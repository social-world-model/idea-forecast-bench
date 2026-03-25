from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from live_idea_bench.models import EvaluationResult, IdeaPrediction, PaperRecord, PredictionMatchDetail, ScoredPredictionList
from forecaster.models import Innovation
from forecaster.realization import RewardConfig, evaluate_rl_reward, load_ppo_train_config
from forecaster.realization.verl import dataset as verl_dataset_module
from forecaster.realization.verl.dataset import build_verl_dataset_rows, write_verl_dataset
from forecaster.realization.verl.reward_fn import compute_score


def _paper(paper_id: str, month: str, *, published_date: str, summary: str) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        month=month,
        summary=summary,
        keywords=summary.split()[:3],
        source_path=f"/fake/{paper_id}.md",
        published_date=published_date,
    )


def _mock_scored_prediction_list() -> ScoredPredictionList:
    return ScoredPredictionList(
        evaluation=EvaluationResult(
            hit_at_k=1.0,
            recall_at_k=1.0,
            precision_at_k=1.0,
            mrr=1.0,
            novelty=0.0,
            diversity=0.0,
            matched_prediction_ranks=[1],
            matched_paper_ids=["future-1"],
            lead_time=0.5,
            duplicate_rate=0.0,
        ),
        matches=[
            PredictionMatchDetail(
                prediction_rank=1,
                prediction_title="idea",
                paper_id="future-1",
                score=0.9,
                is_match=True,
                lead_time=0.5,
            )
        ],
        unmatched_future_paper_ids=[],
    )


def test_load_ppo_train_config_reads_new_yaml() -> None:
    config = load_ppo_train_config("ppo_train.yaml")

    assert config.max_prompt_length == 4096
    assert config.critic_learning_rate > 0
    assert config.dry_run is False


def test_build_verl_dataset_rows_serializes_reward_context() -> None:
    rows = build_verl_dataset_rows(
        [
            {
                "prompt": "prompt",
                "cutoff_month": "2024-06",
                "cutoff_date": "2024-06-01",
                "future_end_month": "2024-07",
                "future_end_date": "2024-07-31",
                "train_papers": [{"paper_id": "train-1"}],
                "future_papers": [{"paper_id": "future-1"}],
            }
        ]
    )

    assert len(rows) == 1
    assert rows[0]["prompt"] == "prompt"
    assert rows[0]["data_source"] == "live_idea_bench"
    payload = json.loads(rows[0]["extra_info"])
    assert payload["cutoff_date"] == "2024-06-01"
    assert payload["future_papers"] == [{"paper_id": "future-1"}]


def test_write_verl_dataset_writes_parquet_and_preview(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(verl_dataset_module, "_detect_parquet_engine", lambda: "pyarrow")

    def _fake_to_parquet(self, path, engine=None, index=False):  # type: ignore[no-untyped-def]
        Path(path).write_text(self.to_json(orient="records"), encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet)
    dataset_path = tmp_path / "trainer_dataset.parquet"

    artifacts = write_verl_dataset(
        [{"prompt": "prompt", "ground_truth": "", "data_source": "live_idea_bench", "extra_info": "{}"}],
        dataset_path,
    )

    assert artifacts.parquet_ready is True
    assert artifacts.parquet_path.exists()
    assert artifacts.preview_path.exists()
    assert artifacts.primary_path == artifacts.parquet_path


def test_compute_score_matches_rule_reward_and_handles_invalid_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forecaster.realization.reward.score_prediction_list",
        lambda **_: _mock_scored_prediction_list(),
    )
    monkeypatch.setattr(
        "forecaster.realization.verl.reward_fn.evaluate_rl_reward",
        evaluate_rl_reward,
    )
    train = [_paper("train-1", "2024-01", published_date="2024-01-01", summary="old retrieval baseline")]
    future = [_paper("future-1", "2024-02", published_date="2024-02-15", summary="retrieval agent planning")]
    innovation = {
        "base_direction": "retrieval planning",
        "operator": "compose",
        "gap": "ground long-horizon agents",
    }
    extra_info = json.dumps(
        {
            "prompt_mode": "z_conditioned_realization",
            "innovation": innovation,
            "evidence_papers": [train[0].__dict__],
            "train_papers": [train[0].__dict__],
            "future_papers": [future[0].__dict__],
            "cutoff_date": "2024-02-01",
            "future_end_date": "2024-03-31",
        }
    )
    completion = (
        "Grounded Retrieval Planning\n"
        "We compose retrieval and planning to ground long-horizon agents with evidence."
    )

    score = compute_score("live_idea_bench", completion, "", extra_info=extra_info)
    expected = evaluate_rl_reward(
        predictions=[
            IdeaPrediction(
                rank=1,
                title="Grounded Retrieval Planning",
                rationale="ground long-horizon agents",
                approach="compose: retrieval planning",
            )
        ],
        train_papers=train,
        future_papers=future,
        reward_config=RewardConfig(top_k=1),
        innovation=Innovation(**innovation),
        evidence_papers=train,
        proposal_text=completion,
        cutoff_date="2024-02-01",
        future_end_date="2024-03-31",
    )
    assert score == expected.list_reward
    assert compute_score("live_idea_bench", "", "", extra_info=extra_info) == RewardConfig().invalid_completion_reward


def test_rl_runtime_no_longer_imports_trl() -> None:
    rl_root = Path(__file__).resolve().parents[1] / "forecaster" / "realization"
    banned_patterns = (
        'import_module("trl")',
        "import trl",
        "from trl import",
        "train_rloo_with_trl",
    )

    for path in rl_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert all(pattern not in text for pattern in banned_patterns), path


def test_rl_setup_and_docs_reference_verl_only() -> None:
    project_root = Path(__file__).resolve().parents[1]
    setup_script = (project_root / "scripts" / "setup_rl_a100_env.sh").read_text(encoding="utf-8")
    assert '"trl' not in setup_script

    readme = (project_root / "rl" / "README.md").read_text(encoding="utf-8")
    assert "scripts/run_policy_rl_training.sh" in readme
    assert "--trainer ppo" in readme
    assert "--trainer grpo" in readme
    assert "--trainer rloo" in readme
