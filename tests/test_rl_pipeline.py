from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.rl import (
    CandidateListSample,
    CandidateGenerationConfig,
    DPOTrainConfig,
    EpisodeBuildConfig,
    EpisodeCandidateLists,
    GRPOTrainConfig,
    RewardConfig,
    build_dpo_pairs,
    build_grpo_advantages,
    build_rl_episodes,
    compute_reward_alignment,
    evaluate_rl_reward,
    list_small_model_specs,
    run_policy_rl_pipeline,
    train_dpo_with_trl,
    train_grpo_with_trl,
)


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


def _prediction(rank: int, title: str, rationale: str) -> IdeaPrediction:
    return IdeaPrediction(rank=rank, title=title, rationale=rationale, approach=rationale)


def test_build_rl_episodes_assigns_contiguous_train_validation_test_splits() -> None:
    papers = [
        _paper(f"p-{month:02d}", f"2024-{month:02d}", published_date=f"2024-{month:02d}-01", summary=f"summary {month}")
        for month in range(1, 13)
    ]
    config = EpisodeBuildConfig(horizon_months=1, min_train_papers=2, start_month="2024-01", end_month="2024-12")

    episodes = build_rl_episodes(papers, config)

    assert episodes
    splits = [episode.split for episode in episodes]
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits
    assert splits == sorted(splits, key=lambda split: {"train": 0, "validation": 1, "test": 2}[split])


def test_build_rl_episodes_respects_past_window_and_step_size() -> None:
    papers = [
        _paper(f"p-{month:02d}", f"2024-{month:02d}", published_date=f"2024-{month:02d}-01", summary=f"summary {month}")
        for month in range(1, 13)
    ]
    config = EpisodeBuildConfig(
        horizon_months=1,
        min_train_papers=1,
        start_month="2024-01",
        end_month="2024-12",
        past_window_months=3,
        step_months=3,
    )

    episodes = build_rl_episodes(papers, config)

    assert [episode.cutoff_month for episode in episodes] == ["2024-01", "2024-04", "2024-07", "2024-10"]
    assert episodes[1].train_paper_ids == ["p-02", "p-03", "p-04"]
    assert episodes[2].train_paper_ids == ["p-05", "p-06", "p-07"]


def test_evaluate_rl_reward_includes_duplicate_penalty() -> None:
    train = [_paper("train-1", "2024-01", published_date="2024-01-01", summary="old retrieval baseline")]
    future = [
        _paper("future-1", "2024-02", published_date="2024-02-15", summary="retrieval agent planning")
    ]
    predictions = [
        _prediction(1, "retrieval agent planning", "retrieval agent planning"),
        _prediction(2, "retrieval agent planning v2", "retrieval agent planning"),
    ]

    reward = evaluate_rl_reward(
        predictions=predictions,
        train_papers=train,
        future_papers=future,
        reward_config=RewardConfig(top_k=2),
        cutoff_date="2024-02-01",
        future_end_date="2024-03-31",
    )

    assert reward.benchmark_evaluation.matched_paper_ids == ["future-1"]
    assert len(reward.per_idea_rewards) == 2
    assert reward.per_idea_rewards[1].duplicate_penalty > 0.0
    assert reward.reward_breakdown["benchmark_score"] == reward.benchmark_score


def test_build_dpo_pairs_and_grpo_advantages_from_episode_candidates() -> None:
    episode = build_rl_episodes(
        [
            _paper("p-01", "2024-01", published_date="2024-01-01", summary="a"),
            _paper("p-02", "2024-02", published_date="2024-02-01", summary="b"),
            _paper("p-03", "2024-03", published_date="2024-03-01", summary="c"),
            _paper("p-04", "2024-04", published_date="2024-04-01", summary="d"),
        ],
        EpisodeBuildConfig(horizon_months=1, min_train_papers=1),
    )[0]
    candidate_rewards = []
    for score in (0.9, 0.7, 0.4, 0.1):
        reward = evaluate_rl_reward(
            predictions=[_prediction(1, f"idea-{score}", f"idea-{score}")],
            train_papers=[],
            future_papers=[],
            reward_config=RewardConfig(top_k=1),
        )
        reward.list_reward = score
        reward.benchmark_score = score
        candidate_rewards.append(
            CandidateListSample(predictions=[_prediction(1, f"idea-{score}", f"idea-{score}")], reward=reward)
        )

    episodes = [EpisodeCandidateLists(episode=episode, prompt="prompt", candidates=candidate_rewards)]
    dpo_pairs = build_dpo_pairs(episodes, DPOTrainConfig(quantile_fraction=0.25))
    grpo_rows = build_grpo_advantages(episodes, GRPOTrainConfig())
    alignment = compute_reward_alignment([sample.reward for sample in candidate_rewards], GRPOTrainConfig())

    assert len(dpo_pairs) == 1
    assert dpo_pairs[0]["chosen_reward"] > dpo_pairs[0]["rejected_reward"]
    assert len(grpo_rows) == 1
    assert len(grpo_rows[0]["candidates"]) == 4
    assert alignment.passed is True


def test_train_dpo_with_trl_dry_run_writes_manifest_and_dataset(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "prompt",
            "chosen": [{"title": "chosen"}],
            "rejected": [{"title": "rejected"}],
        }
    ]
    manifest = train_dpo_with_trl(
        rows,
        DPOTrainConfig(dry_run=True),
        model_name="gpt-4o-mini",
        predictor_config="predictor.yaml",
        output_dir=str(tmp_path / "policy"),
    )

    manifest_path = tmp_path / "policy" / "policy_manifest.json"
    dataset_path = tmp_path / "policy" / "dpo_dataset.jsonl"

    assert manifest["dry_run"] is True
    assert manifest_path.exists()
    assert dataset_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["stage"] == "dpo"


def test_train_grpo_with_trl_dry_run_writes_manifest_and_dataset(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "prompt",
            "cutoff_month": "2024-06",
            "cutoff_date": "2024-06-01",
            "future_end_month": "2024-07",
            "future_end_date": "2024-07-31",
            "train_papers": [
                asdict(_paper("train-1", "2024-05", published_date="2024-05-01", summary="old topic"))
            ],
            "future_papers": [
                asdict(_paper("future-1", "2024-07", published_date="2024-07-01", summary="new topic"))
            ],
        }
    ]
    manifest = train_grpo_with_trl(
        rows,
        GRPOTrainConfig(dry_run=True),
        model_name="Qwen/Qwen2.5-3B-Instruct",
        predictor_config="predictor.yaml",
        output_dir=str(tmp_path / "policy"),
        reward_config=RewardConfig(top_k=1),
    )

    manifest_path = tmp_path / "policy" / "policy_manifest.json"
    dataset_path = tmp_path / "policy" / "grpo_dataset.jsonl"

    assert manifest["dry_run"] is True
    assert manifest_path.exists()
    assert dataset_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["stage"] == "grpo"


def test_run_policy_rl_pipeline_prepare_writes_expected_artifacts(tmp_path: Path) -> None:
    papers = [
        _paper(f"p-{month:02d}", f"2024-{month:02d}", published_date=f"2024-{month:02d}-01", summary=f"summary {month}")
        for month in range(1, 13)
    ]

    manifest = run_policy_rl_pipeline(
        papers,
        model_name="Qwen/Qwen2.5-3B-Instruct",
        output_dir=str(tmp_path / "rl-run"),
        episode_config=EpisodeBuildConfig(horizon_months=1, min_train_papers=2, past_window_months=6, step_months=3),
        candidate_config=CandidateGenerationConfig(backend="heuristic", num_candidate_lists=3, ideas_per_list=4),
        reward_config=RewardConfig(top_k=2),
        dpo_config=DPOTrainConfig(dry_run=True, quantile_fraction=0.34),
        grpo_config=GRPOTrainConfig(dry_run=True),
        stage="prepare",
        split="all",
    )

    run_root = tmp_path / "rl-run"
    assert manifest["selected_episode_count"] > 0
    assert (run_root / "episodes.json").exists()
    assert (run_root / "grpo_prompts.jsonl").exists()
    assert (run_root / "candidate_rollouts.json").exists()
    assert (run_root / "dpo_pairs.jsonl").exists()
    assert (run_root / "pipeline_manifest.json").exists()
    assert manifest["dpo_pair_count"] > 0
    assert manifest["grpo_prompt_count"] == manifest["selected_episode_count"]


def test_small_model_registry_includes_requested_qwen_and_llama_candidates() -> None:
    specs = list_small_model_specs()

    assert len(specs) == 6
    assert any(spec.model_id == "Qwen/Qwen2.5-3B-Instruct" for spec in specs)
    assert any(spec.model_id == "Qwen/Qwen3-4B-Instruct-2507" for spec in specs)
    assert any(spec.model_id == "meta-llama/Llama-3.2-3B-Instruct" for spec in specs)
