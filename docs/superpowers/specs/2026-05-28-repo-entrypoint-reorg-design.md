# Design: Clean up LiveIdeaBench repo entrypoints

**Goal:** Someone who reads the paper and opens the repo should immediately see how to use it.
Today `examples/` (~23 scripts) and `scripts/` (~37) are a flat dump with no "start here".

The paper has exactly **4 conceptual pieces**, and the repo should read that way:
1. **Benchmark** (§3) — run LiveIdeaBench eval (baselines, retrieve-then-judge)
2. **MDF forecaster — main experiment** (§4) — hindsight → prior SFT → realization GRPO → joint inference → eval
3. **Single-metric GRPO** (§4.3) — soft/coverage/novelty reward ablation
4. **Analysis / ablation** — supplementary (citation/coauthor/leakage, distribution metrics)

## Part A — One unified CLI front door (chosen: "deep / unified CLI")

Add `python -m live_idea_bench <command>` as the single obvious entrypoint. It is a **thin
dispatcher** (argparse subcommands) that calls the SAME underlying functions the existing
example scripts already call — no logic rewritten, so low behavioral risk. New file:
`live_idea_bench/__main__.py` (+ a small `live_idea_bench/cli/` package if it grows).

Subcommands (mapped to the 4 pieces):

```
python -m live_idea_bench benchmark   # §3: run a domain/rolling backtest + judge eval
python -m live_idea_bench hindsight   # §4 step 1: extract innovation training labels
python -m live_idea_bench train-prior # §4 step 2: SFT the innovation prior
python -m live_idea_bench train        # §4 step 3: GRPO the realization policy
python -m live_idea_bench infer        # §4 step 4: joint inference (Algorithm 1)
python -m live_idea_bench eval          # §4: eval a trained forecaster
python -m live_idea_bench ablate        # §4.3: single-metric GRPO (soft/coverage/novelty)
python -m live_idea_bench analysis      # supplementary validity analyses
```

Each subcommand wraps an existing entrypoint's `main()` (e.g. `benchmark` →
`examples/.../run_domain_backtest.py:main`, `infer` → `forecaster.inference.algorithm.run_joint_inference`).
**The existing scripts keep working** — the CLI is an addition, not a replacement, so we don't
break anyone's muscle memory or the shell wrappers. The README points at the CLI first.

> Risk note: argparse wiring only. Verified by `python -m live_idea_bench <cmd> --help`
> exiting 0 for every subcommand before merge.

## Part B — Categorize + merge the auxiliary scripts (chosen: subfolders + merge)

Reorganize `examples/` and `scripts/` into category subfolders, **merging redundant variants**
to cut file count. Moves use `git mv` (history preserved); every internal path/cross-reference
and the README get updated and re-verified.

### examples/ → grouped
```
examples/
  benchmark/      run_domain_backtest.py, run_rolling_backtest.py, run_batch_backtest.py,
                  run_new_baselines (the §3 baseline runners), llm_judge_eval.py,
                  reeval_voyage.py, rethreshold_voyage.py
  forecaster/     train.py, eval.py, train_grpo_metric.py, run_prior_sft.py,
                  run_policy_rl_training.py, run_joint_inference.py, run_prior_eval.py,
                  run_topic_hindsight.py  (drop the separate *_preview.py — fold "preview"
                  into a --preview flag on run_topic_hindsight.py; keep the manifest helper)
  analysis/       analysis_citation.py, analysis_coauthor.py, analysis_leakage.py
  data/           keyword_stats.py, organize_by_keywords.py, prepare_topic_hindsight_manifest.py,
                  run_content_matching.py, run_daily_pipeline.py, research_idea_engine.py
```

### scripts/ → grouped
```
scripts/
  setup/    setup_rl_env.sh (+ the qwen3/qwen3_5 family-specific bits folded behind a
            --model-family arg where feasible), setup_new_machine.sh, download_dataset.sh,
            check_eval_env.sh
  serve/    serve_judge.sh, launch_judge_sglang.sh  (two different judge backends — keep both
            but co-locate; merge only if truly identical, else document the difference)
  train/    run_three_grpo.sh, run_predictor_trained.sh, run_train_and_eval.sh, train.sh,
            train_vllm.sh, _trl_vllm_serve.py
  eval/     run_eval_trained.sh, run_eval_3modes.sh, run_new_baselines_batch.sh
  smoke/    phase1_spot_check.py … phase8_ablations.py, phase4_reward_live.py, health_check.sh,
            smoke_test.sh   (the §4 phase smoke harnesses, kept together, out of the main path)
  (thin one-line wrappers keyword_stats.sh / organize_by_keywords.sh / research_idea_engine.sh /
   run_content_matching.sh / run_daily_pipeline.sh / run_rolling_backtest.sh /
   run_topic_hindsight_preview.sh / prepare_topic_hindsight_manifest.sh — move next to the
   example they wrap or collapse into the CLI; decide per-wrapper)
```

### Merges to reduce count (only where safe):
- `setup_rl_env.sh` + `setup_rl_env_qwen3.sh` + `setup_rl_env_qwen3_5.sh` → already a
  dispatcher+impl set; collapse the two family files into one `setup_rl_env.sh --family {qwen3,qwen3.5}` if the bodies differ only by version pins.
- `run_topic_hindsight.py` + `run_topic_hindsight_preview.py` → one script with `--preview`
  (NOTE: tests/test_topic_hindsight_scripts.py exercises both — update the test).
- Phase smoke scripts stay as separate files (each is a distinct documented phase in
  forecaster/foresight/README.md) but move under `scripts/smoke/`.

## Part C — README rewrite

Top of README: a "Quick start" that maps the 4 pieces to the 4-ish CLI commands, then a
"Repository layout" section. Auxiliary/dev scripts get their own short "Development & reproduction"
section (NOT the front matter), and each subfolder gets a one-line README.

## Hard constraints / safety
- **frontend/ and backend/ logic untouched** (only the one dangling rl/ doc-link already fixed).
- Every `git mv` is followed by: grep for the old path across .py/.sh/.md/.yaml/CI, update refs,
  re-run `pytest tests/` to confirm == baseline (474 passed / 10 pre-existing fails), and
  `python -m live_idea_bench <cmd> --help` for each subcommand.
- Tests that reference moved scripts (test_topic_hindsight_scripts.py) updated in lockstep.
- Done on `chore/cleanup-main`; pushed as a PR; **user merges**.

## Open questions for the user before executing
1. CLI module name OK as `python -m live_idea_bench`? (vs a console_script like `lib forecast`)
2. For wrappers that just call one example (keyword_stats.sh etc.) — collapse into the CLI, or
   keep as thin wrappers moved next to their example?
3. OK to fold `*_preview` variants into `--preview` flags (touches one test)?
