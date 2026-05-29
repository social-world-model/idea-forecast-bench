# Plan: Merge 3 Feature Branches into `main` (PR-based) + Cleanup

**Date:** 2026-05-28
**Repo:** `ulab-uiuc/live-idea-bench` (shared org remote — `main == origin/main`, **never force-push main**)
**Strategy chosen by user:** go through **PRs** (one PR per branch), merge order is the safe part of the plan.

## Goal

Land the three large, long-lived feature branches onto `main` cleanly, in an order that
minimizes conflict pain, while cleaning up committed junk. Frontend/backend are **out of
scope — do not touch** `frontend/` or `backend/`.

## The three branches (verified)

| Branch | Paper section | Ahead/Behind main | merge-tree vs main | Junk |
|---|---|---|---|---|
| `feature/predictor-llm-domain-backtest` | §3 Benchmark + baselines | 13 / 0 | **CLEAN** (exit 0) | none |
| `feature/single-metric-grpo` | §4.3 single-metric GRPO | 11 / 0 | **CLEAN** (exit 0) | 6× `outputs/*.json` (~227k lines = 98% of diff) |
| `feature/foresight-judge-soft-mustnot` | §4 MDF main experiment | 50 / **2** | **CONFLICT** (exit 1, 9 files) | `paper/` COLM2026 LaTeX bundle (~3170 lines) |

The three are **independent** (not stacked); they share lineage through the
`summary-retrieval-baselines` branch, so duplicated files (baselines, registry block) auto-resolve
as identical content on the second merge.

Conflict signal is from `git merge-tree --write-tree main <branch>` (git 2.50.1), re-verified twice.

## Merge order & rationale

### PR #1 — `feature/predictor-llm-domain-backtest` → main (FIRST)
- Clean, 0-behind, no junk, purely additive §3 benchmark + baselines.
- Lands the canonical strategy registry + the two prompting baselines on main first, so
  PR #2's duplicate copies of those files become no-op/auto-resolved merges.
- Pre-merge: secret-scan done (no hardcoded keys found).
- **Action:** open PR `feature/predictor-llm-domain-backtest` → `main`, merge.

### PR #2 — `feature/single-metric-grpo` (cleaned) → main (SECOND)
- Clean vs main and 0-behind, BUT must be built from a **cleaned** branch.
- The 6 committed result dumps (~227k lines, 98% of the diff) must be removed:
  - `outputs/llm_judge_soft9b_FULL_20260524_171925.json`
  - `outputs/llm_judge_coverage9b_FULL.json`
  - `outputs/llm_judge_novelty9b_FULL.json`
  - `outputs/eval_predictor_soft_20260522_231800.json`
  - `outputs/eval_predictor_coverage_20260522_231800.json`
  - `outputs/eval_predictor_novelty_20260522_231800.json`
- `outputs/` is **not** in `.gitignore` (that's why they leaked). Add `outputs/` to `.gitignore`.
- **Cleanup approach (history rewrite on a throwaway PR branch):** create
  `pr/single-metric-grpo-clean` from the branch, use `git filter-repo --path outputs/ --invert-paths`
  to strip the JSON from that branch's history, add the `.gitignore` entry, then PR the clean branch.
  (Keeps the original `feature/single-metric-grpo` branch untouched as a backup.)
- After PR #1 is on main, the baseline/registry duplicates collapse to auto-merges.
- **Action:** build `pr/single-metric-grpo-clean`, open PR → `main`, merge.

### PR #3 — `feature/foresight-judge-soft-mustnot` (rebased + de-papered) → main (LAST)
- The MDF main experiment (§4). Only branch with **real conflicts** and the only one **behind** main
  (missing #42 `run-with-batch hindsight` and #43 `trl framework migration`).
- **Must be rebased / 3-way merged with hand resolution — never plain auto-merge**, or it will
  resurrect the pre-trl `verl` trainer stack that #43 deliberately deleted.
- **paper/ decision (user):** remove the COLM2026 `paper/` bundle **from this branch's commit
  history** via `git filter-repo --path paper/ --invert-paths` on a throwaway
  `pr/foresight-clean` branch. (Your live paper is the separate EMNLP `v1/` repo; this old bundle
  should not enter main.) Original branch kept as backup.
- **RL backend decision (user): foresight's `trl/` package wins.** When resolving conflicts:
  - Adopt foresight's `forecaster/realization/trl/` package + `trainers/` registry as the backend.
  - Reconcile main #43's flat `trl_runner.py` into it (do not keep both entrypoints).
  - Re-apply single-metric-grpo's single-metric reward dispatch (`reward_*.yaml`,
    `reward_compute`/`judge_rewards`) on top of the trl package.
  - For the 4 modify/delete conflicts on `verl`/`trainers` files, honor #43's deletion unless a
    capability is missing; if missing, port it onto the trl backend rather than resurrecting verl.
- The 9 conflicting files (from merge-tree): `examples/run_policy_rl_training.py` (modify/delete),
  `forecaster/prior/trainer.py`, `forecaster/realization/__init__.py`,
  `forecaster/realization/pipeline.py`, `forecaster/realization/trainers/__init__.py` (modify/delete),
  `forecaster/realization/trainers/grpo.py` (modify/delete),
  `forecaster/realization/verl/runner.py` (modify/delete),
  `live_idea_bench/strategy/forecaster.py`, `scripts/run_eval_trained.sh` (add/add).
- Also bring in #42's `forecaster/hindsight/batch.py` + `config/forecaster/hindsight.yaml`, reconcile
  `forecaster/hindsight/extractor.py`. Confirm nothing still imports the deleted
  `live_idea_bench/model_refs.py`.
- **Action:** build `pr/foresight-clean` (de-papered + rebased onto current main with manual
  conflict resolution), run tests, open PR → `main`, merge.

## Cross-branch hazard (why foresight goes last)

`single-metric-grpo` and `foresight` both rewrite `forecaster/realization/*` and
`config/forecaster/reward*.yaml` in **incompatible** ways (grpo edits `unsloth_runner.py`/
`reward_compute.py` in place; foresight deletes/renames them and repackages `trl/`). Putting
foresight last means it reconciles **once** against a main that already carries both #43 and
grpo's edits — instead of fighting two moving targets.

## main cleanup (separate small PR or folded into PR #1)

- Add `outputs/` to `.gitignore` (prevents re-leaking result dumps).
- Verify no other committed result artifacts exist on main (none found in current scan).
- `hindsight_*.log` at repo root are already gitignored — no action needed.
- Do **not** touch `frontend/`, `backend/`, or delete stale branches in this pass unless asked.

## Safety rules

- **Never force-push `main`** (shared remote). History rewrites happen only on new `pr/*-clean`
  branches built from the feature branches; originals are preserved as backups.
- Each PR: run the repo's test suite / pre-commit before merging.
- Squash-vs-merge-commit per PR is the maintainer's call at merge time; default to the repo's
  existing convention (the history shows PR-squash-style `feat(...)` titles with `(#NN)`).

## Open items / branch hygiene (optional, ask before doing)

The repo has ~40 stale local+remote branches (`feature/backend`, `feature/clean`, `codex/*`,
`dependabot/*`, etc.). Pruning these is a separate cleanup not required for this merge; flag to
user, do not delete unprompted.
