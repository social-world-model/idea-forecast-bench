# Foresight — future-grounded GRPO

All Phase-1..8 modules from the locked plan live under this package and
`tests/test_foresight_*.py`. The existing GRPO trainer was kept intact;
the new reward + conditioning are routed through a single
`reward_mode: legacy|foresight` switch in
`config/forecaster/grpo_train.yaml`.

## Directory layout

```
forecaster/foresight/
  operators.py          — closed 4-operator inventory + free-text → closed mapping
  memory.py             — build_memory(papers_before_t) -> str
  cutoffs.py            — TRAIN_CUTOFF_MAX / TEST_CUTOFF_MIN + leakage asserts
  indices.py            — Embedder + Future/HistoryIndex + build_cutoff_indices
  dz.py                 — augment_hindsight_rows: raw hindsight JSONL → D_z
  rubric.py             — Rubric schema + generation prompt + parser
  judge.py              — RubricJudge + StubScorer + make_live_scorer
  rubric_validation.py  — LabeledPair, compute_auc, validate_rubric (Phase 2 gate)
  prior_io.py           — D_z ↔ prior SFT bridge + RawMemoryStore adapter
  prior_api.py          — sample_z(memory_text, n, temperature)
  gates.py              — format_ok, grounded, operator_consistent
  reward.py             — compute_foresight_reward + compute_score_v2 (TRL drop-in)
  trainer_wiring.py     — make_reward_fn(config, ...) — legacy|foresight switch
  grouping.py           — assert_group_invariant + dedup penalty
  refresh.py            — Phase-6 rubric co-evolution state machine
  forecast.py           — Phase-7 forecast(papers_before_t) -> top-K
  metrics.py            — MMD + Wasserstein + impact-stratified breakdown
  ablations.py          — AblationConfig + baseline_set (Phase 8)

scripts/
  phase1_spot_check.py           — M1 spot-check on the existing D_z
  phase2_rubric_validation.py    — --mode smoke|live; writes rubrics + reports
  phase3_prior_smoke.py          — D_z → SFT JSONL + sample_z wiring
  phase4_reward_smoke.py         — end-to-end TRL-shape reward_fn
  phase5_grouping_smoke.py       — grouping assert + dedup penalty
  phase8_ablations.py            — --mode smoke|live; writes reports/results.md
```

## Quickstart by phase

```bash
# Phase 1: M1 spot-check (no LLM, runs on existing hindsight JSONL)
PYTHONPATH=. python scripts/phase1_spot_check.py

# Phase 2: rubric construction (smoke mode = no LLM)
PYTHONPATH=. python scripts/phase2_rubric_validation.py --mode smoke

# Phase 3: SFT input shape (no GPU)
PYTHONPATH=. python scripts/phase3_prior_smoke.py

# Phase 4: reward end-to-end (gates + judge stub)
PYTHONPATH=. python scripts/phase4_reward_smoke.py

# Phase 5: grouping invariant + dedup penalty
PYTHONPATH=. python scripts/phase5_grouping_smoke.py

# Phase 8: ablation table (smoke evaluator)
PYTHONPATH=. python scripts/phase8_ablations.py --mode smoke
```

## Wiring the new reward into a real GRPO run

1. Build per-cutoff indices and rubrics:

```bash
PYTHONPATH=. python scripts/phase2_rubric_validation.py --mode live
PYTHONPATH=. python - <<'PY'
from pathlib import Path
from forecaster.foresight.indices import SentenceTransformerEmbedder, build_cutoff_indices
from live_idea_bench.papers import load_papers   # adjust to your loader
papers = load_papers(...)                        # your corpus
build_cutoff_indices(
    papers=papers,
    cutoff_dates=[...],                          # training cutoffs (<= 2024-06-30)
    horizon_months=3,
    embedder=SentenceTransformerEmbedder(),
    save_dir=Path("output/foresight_artifact/indices"),
)
PY
# Move/symlink rubrics into output/foresight_artifact/rubrics/
```

2. Switch the GRPO config:

```yaml
# config/forecaster/grpo_train.yaml
reward_mode: foresight
foresight_artifact_dir: output/foresight_artifact
foresight_embedder: "sentence-transformer:all-MiniLM-L6-v2"
foresight_judge_mode: live
grouping_assert: true
dedup_penalty: 0.0           # try 0.05–0.1 if collapse appears
rubric_refresh_every: 0      # Phase 6 opt-in; 0 = static rubric
```

3. Run training as before (`scripts/run_train_and_eval.sh` etc.). The
   TRL runner now resolves the reward through `make_reward_fn` and logs
   `reward backend: foresight_reward_fn (mode=foresight)` on start.
