# Foresight — future-grounded GRPO

The foresight reward scores a rollout against the papers that actually appeared
after the cutoff: retrieve from a per-cutoff future index, then judge the
rollout against each retrieved paper with a topic rubric. It is selected with
the `reward_mode: legacy|foresight` switch in `config/forecaster/grpo_train.yaml`
and needs two prebuilt artifacts, indices and rubrics, described below.

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
  rubric_validation.py  — LabeledPair, compute_auc, validate_rubric
  prior_io.py           — D_z ↔ prior SFT bridge + RawMemoryStore adapter
  prior_api.py          — sample_z(memory_text, n, temperature)
  gates.py              — format_ok, grounded, operator_consistent
  reward.py             — compute_foresight_reward + compute_score_v2 (TRL drop-in)
  trainer_wiring.py     — make_reward_fn(config, ...) — legacy|foresight switch
  grouping.py           — assert_group_invariant + dedup penalty
  refresh.py            — rubric co-evolution state machine (opt-in)
  forecast.py           — forecast(papers_before_t) -> top-K
  metrics.py            — MMD + Wasserstein + impact-stratified breakdown

examples/forecaster/
  build_indices.py               — D_z from the hindsight labels + per-cutoff indices
  build_rubrics.py               — --mode smoke|live; writes rubrics + validation reports
```

`reward.py` scores the retrieved candidates concurrently; set
`FORESIGHT_JUDGE_WORKERS=1` to make the judge calls serial again.

## Wiring the new reward into a real GRPO run

1. Build per-cutoff indices and rubrics:

```bash
PYTHONPATH=. python examples/forecaster/build_rubrics.py --mode live
PYTHONPATH=. python - <<'PY'
from pathlib import Path
from forecaster.foresight.indices import SentenceTransformerEmbedder, build_cutoff_indices
from idea_forecast_bench.papers import load_papers   # adjust to your loader
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
foresight_artifact_dir: output/foresight_artifacts
foresight_embedder: "sentence-transformer:sentence-transformers/allenai-specter"
foresight_judge_mode: live
grouping_assert: true
dedup_penalty: 0.0           # try 0.05–0.1 if collapse appears
rubric_refresh_every: 0      # Phase 6 opt-in; 0 = static rubric
```

3. Run training as before (`scripts/run_train_and_eval.sh` etc.). The
   TRL runner now resolves the reward through `make_reward_fn` and logs
   `reward backend: foresight_reward_fn (mode=foresight)` on start.
