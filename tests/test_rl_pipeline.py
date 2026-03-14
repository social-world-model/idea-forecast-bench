from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from live_idea_bench.models import IdeaPrediction, PaperRecord
from live_idea_bench.rl import (
    CandidateGenerationConfig,
    CandidateListSample,
    DPOTrainConfig,
    EpisodeBuildConfig,
    EpisodeCandidateLists,
    GRPOTrainConfig,
    RLOOTrainConfig,
    RewardConfig,
    SelectionConfig,
    build_dpo_pairs,
    build_grpo_advantages,
    build_rl_episodes,
    compute_reward_alignment,
    create_trainer_runner,
    evaluate_rl_reward,
    list_small_model_specs,
    prepare_common_rl_context,
    resolve_small_model,
    run_policy_rl_pipeline,
    select_top_k_predictions,
    train_dpo_with_trl,
    train_grpo_with_trl,
    train_rloo_with_trl,
)
from live_idea_bench.rl.reward import build_online_rl_reward_function


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


def test_build_rl_episodes_supports_calendar_split_boundaries() -> None:
    papers = [
        _paper(
            f"p-{year}-{month:02d}",
            f"{year}-{month:02d}",
            published_date=f"{year}-{month:02d}-01",
            summary=f"summary {year}-{month:02d}",
        )
        for year, month in (
            (2025, 10),
            (2025, 11),
            (2025, 12),
            (2026, 1),
            (2026, 2),
            (2026, 3),
        )
    ]
    config = EpisodeBuildConfig(
        horizon_months=1,
        min_train_papers=1,
        start_month="2025-10",
        end_month="2026-03",
        step_months=1,
        validation_start_month="2026-01",
    )

    episodes = build_rl_episodes(papers, config)

    assert [episode.cutoff_month for episode in episodes] == ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02"]
    assert [episode.split for episode in episodes] == ["train", "train", "train", "validation", "validation"]


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


def test_evaluate_rl_reward_is_single_idea_and_no_duplicate_penalty() -> None:
    train = [_paper("train-1", "2024-01", published_date="2024-01-01", summary="old retrieval baseline")]
    future = [_paper("future-1", "2024-02", published_date="2024-02-15", summary="retrieval agent planning")]
    predictions = [
        _prediction(1, "retrieval agent planning", "retrieval agent planning"),
        _prediction(2, "retrieval agent planning v2", "retrieval agent planning"),
    ]

    reward = evaluate_rl_reward(
        predictions=predictions,
        train_papers=train,
        future_papers=future,
        reward_config=RewardConfig(top_k=1),
        cutoff_date="2024-02-01",
        future_end_date="2024-03-31",
    )

    assert reward.benchmark_evaluation.matched_paper_ids == ["future-1"]
    assert len(reward.per_idea_rewards) == 1
    assert reward.per_idea_rewards[0].duplicate_penalty == 0.0
    assert reward.reward_breakdown["benchmark_score"] == reward.benchmark_score
    assert reward.invalid_completion is False
    assert reward.reward_breakdown["invalid_completion"] == 0.0


def test_build_online_reward_function_penalizes_invalid_completion() -> None:
    reward_func = build_online_rl_reward_function(RewardConfig(top_k=1, invalid_completion_reward=-0.05))

    rewards = reward_func(
        completions=['{"ideas":[{"title":"one"},{"title":"two"}]}'],
        train_papers=[[asdict(_paper("train-1", "2024-01", published_date="2024-01-01", summary="old topic"))]],
        future_papers=[[asdict(_paper("future-1", "2024-02", published_date="2024-02-01", summary="new topic"))]],
        cutoff_date=["2024-02-01"],
        future_end_date=["2024-03-31"],
    )

    assert rewards == [-0.05]


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
    rows = [{"prompt": "prompt", "chosen": [{"title": "chosen"}], "rejected": [{"title": "rejected"}]}]
    manifest = train_dpo_with_trl(
        rows,
        DPOTrainConfig(dry_run=True),
        model_name="gpt-4o-mini",
        predictor_config="predictor.yaml",
        output_dir=str(tmp_path / "policy"),
        trainer_config_path="dpo_train.yaml",
        selection_config=SelectionConfig(),
        selection_config_path="selection.yaml",
    )

    manifest_path = tmp_path / "policy" / "policy_manifest.json"
    dataset_path = tmp_path / "policy" / "trainer_dataset.jsonl"

    assert manifest["dry_run"] is True
    assert manifest_path.exists()
    assert dataset_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["trainer"] == "dpo"
    assert payload["inference_model_name"] == "gpt-4o-mini"


def test_train_grpo_with_trl_dry_run_writes_manifest_and_dataset(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "prompt",
            "cutoff_month": "2024-06",
            "cutoff_date": "2024-06-01",
            "future_end_month": "2024-07",
            "future_end_date": "2024-07-31",
            "train_papers": [asdict(_paper("train-1", "2024-05", published_date="2024-05-01", summary="old topic"))],
            "future_papers": [asdict(_paper("future-1", "2024-07", published_date="2024-07-01", summary="new topic"))],
        }
    ]
    manifest = train_grpo_with_trl(
        rows,
        GRPOTrainConfig(dry_run=True),
        model_name="Qwen/Qwen2.5-3B-Instruct",
        predictor_config="predictor.yaml",
        output_dir=str(tmp_path / "policy"),
        reward_config=RewardConfig(top_k=1),
        trainer_config_path="grpo_train.yaml",
        selection_config=SelectionConfig(),
        selection_config_path="selection.yaml",
    )

    manifest_path = tmp_path / "policy" / "policy_manifest.json"
    dataset_path = tmp_path / "policy" / "trainer_dataset.jsonl"

    assert manifest["dry_run"] is True
    assert manifest_path.exists()
    assert dataset_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["trainer"] == "grpo"
    assert payload["inference_model_name"] == "Qwen/Qwen2.5-3B-Instruct"


def test_train_rloo_with_trl_dry_run_writes_manifest_and_dataset(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "prompt",
            "cutoff_month": "2024-06",
            "cutoff_date": "2024-06-01",
            "future_end_month": "2024-07",
            "future_end_date": "2024-07-31",
            "train_papers": [asdict(_paper("train-1", "2024-05", published_date="2024-05-01", summary="old topic"))],
            "future_papers": [asdict(_paper("future-1", "2024-07", published_date="2024-07-01", summary="new topic"))],
        }
    ]
    manifest = train_rloo_with_trl(
        rows,
        RLOOTrainConfig(dry_run=True),
        model_name="Qwen/Qwen2.5-3B-Instruct",
        predictor_config="predictor.yaml",
        output_dir=str(tmp_path / "policy"),
        reward_config=RewardConfig(top_k=1),
        trainer_config_path="rloo_train.yaml",
        selection_config=SelectionConfig(),
        selection_config_path="selection.yaml",
    )

    manifest_path = tmp_path / "policy" / "policy_manifest.json"
    dataset_path = tmp_path / "policy" / "trainer_dataset.jsonl"

    assert manifest["dry_run"] is True
    assert manifest_path.exists()
    assert dataset_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["trainer"] == "rloo"
    assert payload["inference_model_name"] == "Qwen/Qwen2.5-3B-Instruct"


def test_prepare_common_rl_context_reuses_shared_artifacts(tmp_path: Path) -> None:
    papers = [
        _paper(f"p-{month:02d}", f"2024-{month:02d}", published_date=f"2024-{month:02d}-01", summary=f"summary {month}")
        for month in range(1, 13)
    ]
    common = prepare_common_rl_context(
        papers,
        model_name="Qwen/Qwen2.5-3B-Instruct",
        output_dir=str(tmp_path / "rl-run"),
        episode_config=EpisodeBuildConfig(horizon_months=1, min_train_papers=2, past_window_months=6, step_months=3),
        candidate_config=CandidateGenerationConfig(backend="heuristic", num_candidate_lists=3, ideas_per_list=4),
        reward_config=RewardConfig(top_k=2),
        selection_config=SelectionConfig(),
        split="all",
    )
    cached = prepare_common_rl_context(
        papers,
        model_name="Qwen/Qwen2.5-3B-Instruct",
        output_dir=str(tmp_path / "rl-run"),
        episode_config=EpisodeBuildConfig(horizon_months=1, min_train_papers=2, past_window_months=6, step_months=3),
        candidate_config=CandidateGenerationConfig(backend="heuristic", num_candidate_lists=3, ideas_per_list=4),
        reward_config=RewardConfig(top_k=2),
        selection_config=SelectionConfig(),
        split="all",
    )

    assert common.config_fingerprint == cached.config_fingerprint
    assert common.prompt_rows_path.exists()
    assert cached.shared_manifest_path.exists()


def test_run_policy_rl_pipeline_prepare_only_writes_expected_artifacts(tmp_path: Path) -> None:
    papers = [
        _paper(f"p-{month:02d}", f"2024-{month:02d}", published_date=f"2024-{month:02d}-01", summary=f"summary {month}")
        for month in range(1, 13)
    ]

    manifest = run_policy_rl_pipeline(
        papers,
        trainer="dpo",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        output_dir=str(tmp_path / "rl-run"),
        episode_config=EpisodeBuildConfig(horizon_months=1, min_train_papers=2, past_window_months=6, step_months=3),
        candidate_config=CandidateGenerationConfig(backend="heuristic", num_candidate_lists=3, ideas_per_list=4),
        reward_config=RewardConfig(top_k=2),
        selection_config=SelectionConfig(),
        trainer_config=DPOTrainConfig(dry_run=True, quantile_fraction=0.34),
        trainer_config_path="dpo_train.yaml",
        selection_config_path="selection.yaml",
        split="all",
        prepare_only=True,
    )

    run_root = tmp_path / "rl-run"
    assert manifest["selected_episode_count"] > 0
    assert (run_root / "shared" / "episodes.json").exists()
    assert (run_root / "shared" / "prompts.jsonl").exists()
    assert (run_root / "dpo" / "candidate_rollouts.json").exists()
    assert (run_root / "dpo" / "trainer_dataset.jsonl").exists()
    assert (run_root / "pipeline_manifest.json").exists()
    assert manifest["prepare_only"] is True
    assert manifest["trainer_policy_manifest_path"] == ""
    assert manifest["training_split_policy"] == "train_only"


def test_run_policy_rl_pipeline_grpo_supports_init_policy_and_skip_alignment(tmp_path: Path) -> None:
    papers = [
        _paper(f"p-{month:02d}", f"2024-{month:02d}", published_date=f"2024-{month:02d}-01", summary=f"summary {month}")
        for month in range(1, 13)
    ]
    init_path = tmp_path / "warmstart"
    init_path.mkdir()

    manifest = run_policy_rl_pipeline(
        papers,
        trainer="grpo",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        output_dir=str(tmp_path / "rl-run"),
        episode_config=EpisodeBuildConfig(horizon_months=1, min_train_papers=2, past_window_months=6, step_months=3),
        candidate_config=CandidateGenerationConfig(backend="heuristic", num_candidate_lists=2, ideas_per_list=4),
        reward_config=RewardConfig(top_k=2),
        selection_config=SelectionConfig(),
        trainer_config=GRPOTrainConfig(dry_run=True),
        trainer_config_path="grpo_train.yaml",
        selection_config_path="selection.yaml",
        split="train",
        init_policy_path=str(init_path),
        skip_alignment_check=True,
    )

    payload = json.loads((tmp_path / "rl-run" / "grpo" / "policy_manifest.json").read_text(encoding="utf-8"))
    assert manifest["trainer"] == "grpo"
    assert payload["trainer"] == "grpo"
    assert payload["init_policy_path"] == str(init_path)
    assert payload["base_model_name"] == "Qwen/Qwen2.5-3B-Instruct"
    assert payload["inference_model_name"] == "Qwen/Qwen2.5-3B-Instruct"


def test_run_policy_rl_pipeline_rejects_non_train_training_split(tmp_path: Path) -> None:
    papers = [
        _paper(f"p-{month:02d}", f"2024-{month:02d}", published_date=f"2024-{month:02d}-01", summary=f"summary {month}")
        for month in range(1, 13)
    ]

    with pytest.raises(ValueError, match="restricted to the train split"):
        run_policy_rl_pipeline(
            papers,
            trainer="grpo",
            model_name="Qwen/Qwen2.5-3B-Instruct",
            output_dir=str(tmp_path / "rl-run"),
            episode_config=EpisodeBuildConfig(horizon_months=1, min_train_papers=2, past_window_months=6, step_months=3),
            candidate_config=CandidateGenerationConfig(backend="heuristic", num_candidate_lists=2, ideas_per_list=4),
            reward_config=RewardConfig(top_k=1),
            selection_config=SelectionConfig(),
            trainer_config=GRPOTrainConfig(dry_run=True),
            trainer_config_path="grpo_train.yaml",
            selection_config_path="selection.yaml",
            split="validation",
            skip_alignment_check=True,
        )


def test_run_policy_rl_pipeline_grpo_writes_alignment_report(tmp_path: Path) -> None:
    papers = [
        _paper(f"p-{month:02d}", f"2024-{month:02d}", published_date=f"2024-{month:02d}-01", summary=f"summary {month}")
        for month in range(1, 13)
    ]

    manifest = run_policy_rl_pipeline(
        papers,
        trainer="grpo",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        output_dir=str(tmp_path / "rl-run"),
        episode_config=EpisodeBuildConfig(horizon_months=1, min_train_papers=2, past_window_months=6, step_months=3),
        candidate_config=CandidateGenerationConfig(backend="heuristic", num_candidate_lists=2, ideas_per_list=4),
        reward_config=RewardConfig(top_k=1),
        selection_config=SelectionConfig(),
        trainer_config=GRPOTrainConfig(dry_run=True, reward_alignment_threshold=0.0),
        trainer_config_path="grpo_train.yaml",
        selection_config_path="selection.yaml",
        split="train",
        skip_alignment_check=False,
    )

    report_path = tmp_path / "rl-run" / "grpo" / "alignment_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert manifest["trainer"] == "grpo"
    assert report_path.exists()
    assert report["episodes_used"] >= 1
    assert "reward_selected_avg_hit_at_1" in report
    assert "prompt_baseline_avg_mrr" in report
    assert "invalid_completion_rate" in report


def test_run_policy_rl_pipeline_requires_validation_for_alignment(tmp_path: Path) -> None:
    papers = [
        _paper(f"p-{month:02d}", f"2024-{month:02d}", published_date=f"2024-{month:02d}-01", summary=f"summary {month}")
        for month in range(1, 13)
    ]

    with pytest.raises(ValueError, match="Validation episodes are required"):
        run_policy_rl_pipeline(
            papers,
            trainer="grpo",
            model_name="Qwen/Qwen2.5-3B-Instruct",
            output_dir=str(tmp_path / "rl-run"),
            episode_config=EpisodeBuildConfig(
                horizon_months=1,
                min_train_papers=2,
                past_window_months=6,
                step_months=9,
            ),
            candidate_config=CandidateGenerationConfig(backend="heuristic", num_candidate_lists=2, ideas_per_list=4),
            reward_config=RewardConfig(top_k=1),
            selection_config=SelectionConfig(),
            trainer_config=GRPOTrainConfig(dry_run=True, reward_alignment_threshold=0.0),
            trainer_config_path="grpo_train.yaml",
            selection_config_path="selection.yaml",
            split="train",
            skip_alignment_check=False,
        )


def test_trainer_registry_supports_all_three_algorithms() -> None:
    assert create_trainer_runner("dpo").trainer_name == "dpo"
    assert create_trainer_runner("grpo").trainer_name == "grpo"
    assert create_trainer_runner("rloo").trainer_name == "rloo"


def test_select_top_k_predictions_uses_candidate_pool_dedup_and_mmr() -> None:
    train_papers = [
        _paper("p-01", "2024-01", published_date="2024-01-01", summary="retrieval planning agents"),
        _paper("p-02", "2024-02", published_date="2024-02-01", summary="vision reasoning grounded planning"),
    ]
    candidates = [
        IdeaPrediction(rank=1, title="Retrieval Agents", rationale="retrieval planning", approach="agent", confidence=0.9),
        IdeaPrediction(rank=1, title="Retrieval Agents", rationale="retrieval planning", approach="agent", confidence=0.8),
        IdeaPrediction(rank=1, title="Grounded Vision", rationale="grounded reasoning", approach="vision", confidence=0.7),
        IdeaPrediction(rank=1, title="Planning Memory", rationale="memory planning", approach="memory", confidence=0.6),
    ]

    selected = select_top_k_predictions(candidates, train_papers, SelectionConfig(candidate_pool_size=4, output_top_k=3))

    assert len(selected) == 3
    assert selected[0].rank == 1
    assert len({prediction.title for prediction in selected}) == 3
    assert all("selector_relevance" in prediction.metadata for prediction in selected)
    assert all("unique_candidate_titles" in prediction.metadata for prediction in selected)
    assert all("dedup_retention_ratio" in prediction.metadata for prediction in selected)


def test_small_model_registry_includes_requested_qwen_and_llama_candidates() -> None:
    specs = list_small_model_specs()

    assert len(specs) == 10
    assert any(spec.model_id == "Qwen/Qwen2.5-3B" for spec in specs)
    assert any(spec.model_id == "Qwen/Qwen2.5-3B-Instruct" for spec in specs)
    assert any(spec.model_id == "Qwen/Qwen2.5-7B" for spec in specs)
    assert any(spec.model_id == "Qwen/Qwen2.5-7B-Instruct" for spec in specs)
    assert any(spec.model_id == "Qwen/Qwen3-4B-Base" for spec in specs)
    assert any(spec.model_id == "Qwen/Qwen3-4B-Instruct-2507" for spec in specs)
    assert any(spec.model_id == "Qwen/Qwen3-8B-Base" for spec in specs)
    assert any(spec.model_id == "Qwen/Qwen3-8B" for spec in specs)
    assert any(spec.model_id == "meta-llama/Llama-3.2-3B-Instruct" for spec in specs)


def test_resolve_small_model_accepts_requested_qwen_short_names() -> None:
    assert resolve_small_model("qwen2.5-3b-base").model_id == "Qwen/Qwen2.5-3B"
    assert resolve_small_model("qwen2.5-3b-instruct").model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert resolve_small_model("qwen2.5-7b-base").model_id == "Qwen/Qwen2.5-7B"
    assert resolve_small_model("qwen2.5-7b-instruct").model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert resolve_small_model("qwen3-4b-base").model_id == "Qwen/Qwen3-4B-Base"
    assert resolve_small_model("qwen3-4b-instruct").model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert resolve_small_model("qwen3-8b-base").model_id == "Qwen/Qwen3-8B-Base"
    assert resolve_small_model("qwen3-8b-instruct").model_id == "Qwen/Qwen3-8B"
