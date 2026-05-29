# FILEMAP — Foresight RL recon (Phase 0)

Status: read-only recon. No code changes. Verified on branch `feature/training-trl`.

## TL;DR for the plan

- **Active GRPO loop is TRL**, not veRL. veRL code exists but is not wired into the run scripts.
- **One reward path is live in training**: `verl/reward_fn.py::compute_score → reward.py::evaluate_rl_reward → realization_reward.py::evaluate_realization_reward`. Phase 4 replaces this chain.
- **Per-rollout `extra_info` already carries `train_papers`, `future_papers`, `evidence_papers`, `innovation=(b,o,g)`, `cutoff_date`, `future_end_date`.** Future-grounding is already at row granularity — what's missing is a **vector index** over `future_papers` for retrieve-then-judge.
- **Hindsight `(base_direction, operator, gap)` extraction already exists** (`forecaster/hindsight/`). Memory builder + per-cutoff future/history vector indices do **not** yet exist as Phase 1 specifies.
- **Same embedder/judge as the eval benchmark is already reachable** via `live_idea_bench/similarity.py::score_prediction_list` — Phase 4's "reuse the benchmark's embedder + judge" is feasible without a new dependency.
- **Leakage guard:** `tests/test_backtest_leak_guard.py` checks per-paper cutoff comparison; there is **no global `max(train_cutoffs) < min(test_cutoffs)` assertion** anywhere. Plan invariant must be added in Phase 1.
- **Grouping for GRPO:** TRL's `GRPOTrainer` groups `num_generations=8` rollouts per prompt. Since each prompt today already carries a single `innovation` z, **group == (cutoff_t, z) is naturally satisfied** if we keep one row per (t,z).

## 1. GRPO training entrypoint (TRL is live; veRL is alternate)

| Role | Path | Key symbols |
|---|---|---|
| Shell entry | `scripts/run_train_and_eval.sh`, `scripts/run_policy_rl_training.sh` | invokes `examples/run_policy_rl_training.py` |
| Python entry | `examples/run_policy_rl_training.py` | `main()` ~L98 → `run_policy_rl_pipeline()` |
| Pipeline orchestrator | `forecaster/realization/pipeline.py` | `run_policy_rl_pipeline()` L1296–1412 |
| Trainer registry | `forecaster/realization/trainers/registry.py` | `create_trainer_runner("grpo") → GRPOTrainerRunner` |
| GRPO runner wrapper | `forecaster/realization/trainers/grpo.py` | `GRPOTrainerRunner.prepare/train` |
| **TRL backend (LIVE)** | `forecaster/realization/trl/runner.py` | `prepare_trl_artifacts()` L90, `train_with_trl()` L120; reward import at L160 |
| veRL backend (alt) | `forecaster/realization/verl/runner.py` | `prepare_verl_artifacts()` L55, subprocess training |
| Config | `config/forecaster/grpo_train.yaml` | `num_generations=8`, `kl_coef=0.001`, `lora_r=16`, `reward_alignment_threshold=0.5`, `use_vllm=false` |

The trainer construction lives at `trl/runner.py:245` (`GRPOTrainer(model, reward_funcs=reward_fn, ...)`); `trainer.train()` runs the actual loop.

## 2. Current reward (to be replaced in Phase 4)

| Role | Path | Function | Notes |
|---|---|---|---|
| Live entry called by TRL | `forecaster/realization/verl/reward_fn.py` | `compute_score(data_source, solution_str, ground_truth, extra_info, ...)` L43 | TRL wrapper at `trl/runner.py:160–180` calls this per completion; returns single float in [-0.05, 1] |
| Aggregator | `forecaster/realization/reward.py` | `evaluate_rl_reward()` L374–463 | Computes `score_prediction_list` (benchmark) + paper-faithful dense reward; returns `RLRewardEvaluation` with `list_reward` (used as scalar) |
| Strict-mode variant | `forecaster/realization/reward.py` | `evaluate_strict_completion_reward()` L240 | Used when `prompt_mode == "strict_interactive_realization"` |
| Paper-faithful components | `forecaster/realization/realization_reward.py` | `evaluate_realization_reward()` L200; `compute_evidence_accuracy` L88, `compute_operator_adherence` L160, `compute_coherence_score` L139 | Keyword-overlap heuristics with weights from `realization.yaml` |
| Config | `config/forecaster/reward.yaml`, `config/forecaster/realization.yaml` | weights, thresholds, `invalid_completion_reward=-0.05` |

**Existing gates today:** parse / format gate only (returns `invalid_completion_reward` on parse failure). **No grounding gate, no operator-consistency gate** — both are net-new for Phase 4.

## 3. Dataset / episode / cutoff representation

| Role | Path | Function | Notes |
|---|---|---|---|
| Episode dataclass | `forecaster/realization/episodes.py` | `RLEpisode` L13; `build_rl_episodes()` L100 | fields: `cutoff_month`, `cutoff_date`, `future_end_month`, `future_end_date`, `train_paper_ids`, `future_paper_ids`, `split` |
| Train/future split | `live_idea_bench/backtest.py` | `split_train_future_by_cutoff()` | strict published_date comparison |
| TRL/veRL dataset rows | `forecaster/realization/verl/dataset.py` | `build_verl_dataset_rows()` L21 | one row per (cutoff, innovation); writes `extra_info` JSON with `cutoff_date`, `future_end_date`, `innovation={base_direction,operator,gap}`, `train_papers`, `future_papers`, `evidence_papers`, `target_future_paper(_id)`, `prompt_mode`, `realization_config`, `search_env`, `strict_contract` |
| Episode config | `config/forecaster/episode_build.yaml` | `start_month=2023-03`, `end_month=2025-03`, `validation_start_month=2024-10`, `test_start_month=2024-10` (validation==test currently), `past_window_months=24`, `horizon_months=3`, `step_months=3` |
| Leakage test | `tests/test_backtest_leak_guard.py` | per-paper cutoff test only |

**Gap vs plan invariant:** there is no `assert max(train_cutoffs) < min(test_cutoffs)` (or non-overlap). Add in Phase 1.

## 4. Eval scorer = embedder + judge (already reusable)

| Role | Path | Function | Notes |
|---|---|---|---|
| Benchmark scorer | `live_idea_bench/similarity.py` | `score_prediction_list()` L291–450 | Already called inside `evaluate_rl_reward` (training-time ≈ eval-time objective) |
| Similarity dispatch | `live_idea_bench/similarity.py` | `compute_similarity()`, `is_match()` | routes by `similarity.engine` ∈ {heuristic, embedding, llm} |
| Embedder backends | `live_idea_bench/similarity.py` | `_embed_and_score()` L240 (SentenceTransformer), `_api_embed_pair()` L127 (Voyage / OpenAI) |
| LLM judge | `live_idea_bench/similarity.py` | `_llm_similarity()` L84–116 | parses `"Score: X.X"` from LLM output |
| Judge prompt | `live_idea_bench/prompt/similarity.yaml` | `system_prompt`, `user_prompt_template` |
| Backtest harness | `live_idea_bench/backtest.py` | top-level eval loop used by the benchmark protocol |

For Phase 4: the rubric-conditioned judge can drop in by extending `_llm_similarity` with a `rubric` kwarg, or by passing rubric into the user prompt template.

## 5. Policy / reference model wrappers

| Role | Path | Function | Notes |
|---|---|---|---|
| Model loader | `forecaster/realization/trl/runner.py` | L230–242 | `AutoModelForCausalLM.from_pretrained(..., torch_dtype=bf16, attn_implementation="sdpa")` + LoRA (`r=16, α=32, dropout=0.05, target=all-linear`) |
| Model registry | `forecaster/realization/model_zoo.py` | `SmallModelSpec` L6; `resolve_small_model(alias)` | Qwen2.5/Qwen3/Qwen3.5/Llama3.2 aliases |
| vLLM gate | `forecaster/realization/trl/runner.py` | `_vllm_available()` L28 | needs transformers <5 (Qwen3 only); disabled via `DISABLE_VLLM=1` |
| Local generation (eval/inference) | `forecaster/realization/local_generation.py` | shared HF + caching utilities |

Reference model is created implicitly by TRL's `GRPOTrainer`; LoRA on policy only.

## 6. Hindsight + memory infra (already partly built)

| Role | Path | Function | Notes |
|---|---|---|---|
| Innovation extractor | `forecaster/hindsight/extractor.py` | `extract_innovation(future_paper, context_papers, ...)` L65 | already produces `(base_direction, operator, gap)` |
| Hindsight dataset builder | `forecaster/hindsight/dataset_builder.py` | `build_hindsight_dataset()`, `load_hindsight_samples_jsonl()` L99 | JSONL schema: `{cutoff_date, future_paper_id, future_paper_published_date, innovation, context_paper_count}` |
| Hindsight prompt | `forecaster/hindsight/prompt.py`, `forecaster/prompt/hindsight.yaml` | LLM prompt builder |
| Batch + topic sampling | `forecaster/hindsight/batch.py`, `forecaster/hindsight/topic_sampling.py` | mass extraction utilities |
| HindsightSample dataclass | `forecaster/models.py` L68 | fields: `context_paper_ids, cutoff_month, future_paper_id, future_paper_published_date, innovation` |
| Memory store | `forecaster/prior/memory.py` | innovation frequency / recency / utility tracking |
| Hindsight config | `config/forecaster/hindsight.yaml` | `llm_model=gpt-4o`, `temperature=0.2`, `max_context_papers=15` |

**Phase 1 deltas to add:**
- `config/operators.yaml` (currently operator is free-text from extractor — closed inventory is not enforced).
- `build_memory(papers_before_t) -> str` that produces a compact, swappable text inventory (today's `memory.py` tracks innovations, not raw paper inventory at a cutoff).
- Per-cutoff vector indices: a `future_index[t]` and `history_index[t]` for retrieve-then-judge and the grounding gate. The data to build them is already attached to each row (`train_papers`, `future_papers`).
- Persisted `D_z` JSONL using current hindsight output but adding `memory_text` and tightening the `operator` enum.

## 7. Prior model (SFT + sampler)

| Role | Path | Function | Notes |
|---|---|---|---|
| SFT trainer | `forecaster/prior/trainer.py` | `train_prior(samples, config, output_dir)` L115; `_build_hf_dataset()` L89 | HF Trainer + LoRA; outputs checkpoint path |
| Sampler | `forecaster/prior/sampler.py` | sampler entrypoints + `_load_prior_model()` cache | local HF or SGLang remote |
| SFT dataset glue | `forecaster/prior/sft_dataset.py` | builds `{"input": memory, "target": innovation_json}` rows |
| Prompt template | `forecaster/prompt/prior_sft.yaml` | input/target format |
| Memory utilities | `forecaster/prior/memory.py` | recency/frequency/utility (not the plan's `build_memory`) |
| Config | `config/forecaster/prior.yaml` | `model_alias=qwen2.5-3b-instruct`, `num_epochs=3`, `lr=2e-5`, `lora_r=16` |

Predicts the triple `(b, o, g)` already — matches Phase 3. A `sample_z(memory, n, temperature)` thin wrapper is the missing API.

## 8. Inference / forecast()

| Role | Path | Function | Notes |
|---|---|---|---|
| Joint inference | `forecaster/inference/algorithm.py` | `run_joint_inference(innovations, papers, memory_store, ..., inference_config, realization_config, ...)` L43 | iterates innovations → evidence retrieve → realize → score → dedup → top-K |
| Scoring helpers | `forecaster/inference/scoring.py` | prior/realization/popularity scoring |
| Dedup | `forecaster/inference/deduplication.py` | similarity-threshold dedup |
| Pipeline driver | `forecaster/realization/pipeline.py` | also exposes inference entrypoints for end-to-end runs |
| Config | `config/forecaster/inference.yaml` | `num_candidates=16`, `prior_weight=0.4`, `realization_weight=0.6`, `top_k=5`, `dedup_threshold=0.8` |

Phase 7 mostly composes existing pieces; the missing call is `build_memory(X_≤t)` feeding the prior sampler.

## 9. Configs touched

| File | Purpose | Read by |
|---|---|---|
| `config/forecaster/grpo_train.yaml` | GRPO hyperparams (num_generations, lr, LoRA, vLLM) | `examples/run_policy_rl_training.py` → `train_with_trl` |
| `config/forecaster/reward.yaml` | top_k, dup threshold, invalid reward, dense weights | `load_reward_config` → `evaluate_rl_reward` |
| `config/forecaster/realization.yaml` | evidence/operator/coherence weights | `evaluate_realization_reward` |
| `config/forecaster/episode_build.yaml` | cutoff windowing, splits | `build_rl_episodes` |
| `config/forecaster/hindsight.yaml` | extractor LLM + retries | `extract_innovation` |
| `config/forecaster/prior.yaml` | prior SFT params | `train_prior` |
| `config/forecaster/inference.yaml` | joint inference weights & top-K | `run_joint_inference` |
| `config/forecaster/{ppo,rloo}_train.yaml` | alternative trainers (not live) | registry-selected |
| `live_idea_bench/prompt/similarity.yaml` | benchmark similarity engine, thresholds, judge prompt | `score_prediction_list`, `compute_similarity` |
| `live_idea_bench/prompt/predictor.yaml` | predictor LLM prompts (eval) | `live_idea_bench/predictor.py` |

## 10. Existing tests covering reward + leak guard

- `tests/test_realization_reward.py` — `evaluate_realization_reward` component bounds + `evaluate_rl_reward` ordering tests (good evidence > weak; aligned > misaligned; coherent > incoherent).
- `tests/test_strict_trajectory_reward.py` — strict-mode reward path.
- `tests/test_backtest_leak_guard.py` — per-paper temporal split correctness; **no global cutoff-disjointness assertion**.

These will need updates when Phase 4 swaps the reward chain; keep them green under `--reward legacy` for ablation.

## Risks / open questions surfaced by the recon

1. **`validation_start_month == test_start_month` (2024-10).** There is no held-out validation cutoff for the Phase 2 rubric AUC measurement — we'd have to carve one out of the train window (e.g., 2024-07..09) or out of the test window. Need a decision before Phase 2.
2. **Operator inventory is currently open / extractor-defined.** The `Innovation.operator` field is free text. Closing to the 4-operator inventory (Phase 1, 1a) will require either (a) re-running the extractor with a constrained schema, or (b) post-hoc classifying the existing `D_z` to the 4 operators (and discarding the unmappable). Either is fine but is a real call.
3. **`extra_info` payload is large.** Each row already carries train + future + evidence papers. Adding the rubric per-row is cheap; the per-cutoff vector index should be **shared across rows with the same cutoff** rather than serialized in every row.
4. **`compute_score` returns one float per completion to TRL.** That matches plan's signature, so the new `compute_reward` can drop in as the new body of `compute_score` (or a sibling) without changing the TRL wrapper.
5. **Grouping invariant for GRPO.** TRL groups `num_generations` completions per prompt. If we want **G rollouts per (cutoff_t, z)** as Phase 5 requires, we must ensure each `prompt_row` corresponds to exactly one `(t, z)` pair. The current dataset builder seems to do this (one row per innovation), but verify before Phase 5.

---

**Phase 0 deliverable complete. Stopping per plan — request go-ahead for Phase 1.**
