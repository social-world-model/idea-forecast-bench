# Concept vocabulary: runbook

The `idea_forecast_bench.vocab` experiment (v2) replaces combinatorial's single
flat element type with three per-paper slots that mirror what a title-level
research idea is made of:

- **object** — WHAT the work is done on (model family, task, data setting,
  system component).
- **mechanism** — HOW: the technique proposed or relied on.
- **problem** — WHY: the concrete difficulty being addressed.

Each extracted item is a specific `term` plus a broader `parent`
(`"expert offloading"` → `"memory offloading"`). Terms are embedded and
greedily leader-clustered per slot into **fine concepts** as of a training
cutoff (`vocab.build._fine_cluster`, the same leader-clustering pattern as
`combinatorial.canonicalize.merge_elements`); the fine clusters' majority
parent labels are separately leader-clustered into a **coarse** layer. An
optional **hybrid ("promote") level** folds any fine cluster with fewer than
`tag.promote_min_count` training papers into the concept named by its merged
parent label (`vocab.build._fold_weak_clusters`), so a paper's idea is
described at the specificity its own evidence supports instead of always at
the finest or the coarsest grain — `promote_min_count: 0` disables it and
keeps every fine cluster as its own node.

Every concept is tagged at build time:

- **background** — appears in ≥ `tag.background_doc_frac` of training
  papers: this is the topic itself, not forecastable material, so it is
  excluded from `Vocabulary.combinable()` but kept for reporting.
- **emerging** — first seen within `tag.emerging_months` months of the
  cutoff, kept even at count 1.

## Lock-in checks

`vocab.checks.run_checks` assigns every future (post-cutoff) record's terms
back onto the pre-cutoff vocabulary (`vocab.build.assign_record`: exact
normalized-text match first, else the nearest same-slot fine-cluster leader
at cosine ≥ `checks.assign_threshold`; a miss on the term itself falls back
to trying the term's own parent) and compares pre- vs. post-cutoff traffic.

| metric | definition |
|---|---|
| `coverage_object` / `coverage_mechanism` | Share of `ok` future records whose object / mechanism slot hits a non-background pre-cutoff concept. |
| `coverage_both` | Share of `ok` future records that hit on **both** object and mechanism. |
| `coverage_any_term` | Same hit test, counted per term occurrence (all three slots) rather than per record. |
| `spearman_pre_post` | Spearman correlation, over non-background concepts with training `count ≥ 2`, between each concept's pre-cutoff training count and how many future records mapped onto it — whether the ranking of "hot concepts" survives into the future. |
| `stability` | Jaccard of non-background concept **label** sets between this cutoff's vocabulary and the previous cutoff's — how much of the vocabulary a re-build reuses vs. throws away. |
| `mid_layer_share` | Of every non-background term occurrence in training, the share whose concept has `count ≥ checks.mid_layer_min_papers` — how much of the vocabulary sits in well-attested concepts rather than singletons. |
| `single_token_share_object/mechanism/problem` | Share of that slot's raw term occurrences that are single-token (e.g. `"quantization"`). |
| `n_slot_conflicts` | Count of texts whose dominant slot wins with less than `cluster.slot_majority_min` of that text's training occurrences. |
| `background_count` / `emerging_count` / `emerging_multi_count` / `combinable_count` | Vocabulary sizes: background concepts, emerging concepts, emerging concepts with `count ≥ 2`, and everything a sampler could draw. |
| `future_new_terms_share` | Share of unique future term texts that never mapped to any pre-cutoff concept. |

## Iteration protocol

Every version is eyeballed on the same **fixed 30-paper probe set** — 10
papers each from `llm_long_context`, `quantization`, `moe`, chosen once by
`vocab-probe-select` (3 mainstream + 3 niche + 2 new-concept + 2
survey/benchmark, deterministic given `--seed`) from the v1 element cache —
so a human reviewer never compares extractions on different papers across
versions.

Change **exactly one variable per version** (a prompt edit, a threshold, a
config knob) and give it its own `--tag`. `vocab-build` appends one row per
`(tag, prompt, config)` to `output/vocab/ledger.md`, averaged (nan-aware)
over every topic × cutoff window that ran.

A sweep of the clustering thresholds must lower `checks.assign_threshold`
**together with** `cluster.fine_threshold` (and keep `cluster.parent_threshold`
equal to `fine_threshold`, as every ledger row so far does): assignment
matches a future term against fine-cluster leaders, so loosening
`fine_threshold` while leaving `assign_threshold` at the old, tighter bar
starves coverage of the very effect the sweep is trying to measure. The
ledger's `_fine0.80` → `_fine0.80_assign` progression isolates that
assign-threshold change; the `_pf` suffix marks a rerun of the same config
(same `config_sha`) after `assign_record`'s parent-fallback path — try the
term's own text first, then the parent the extractor gave it — was added, so
`_pf` rows are the ones comparable to the current checkout.

## Environment

- `DASHSCOPE_API_KEY` — routes the default `--model-name deepseek-v4-flash`
  extraction call through DashScope (`idea_forecast_bench/llm.py`,
  `combinatorial/llm_caller.py`), same routing as the combinatorial
  extraction.
- `VOYAGE_API_KEY` — needed for `cluster.embed_model: voyage-3-large`.
  Extraction and embedding are the only billed calls; the lock-in checks and
  report rendering are pure offline computation over what is already on
  disk.
- Offline / dry runs: `--dry-run` (a deterministic fake extractor — title
  bigrams as terms — instead of an LLM call) plus `--embed-backend hash`
  (deterministic hash embeddings, `HASH_BACKEND` in
  `combinatorial.embeddings`) or `--skip-embed` to reuse whatever vectors are
  already cached. Neither touches the network.

## Commands, in order

```bash
# 1. Pick the fixed probe set once per topic set (no LLM calls).
idea-forecast-bench vocab-probe-select \
  --topics llm_long_context,quantization,moe

# 2. Cheap sanity pass on a new prompt/config: extract + probe report only,
#    no lock-in checks, no ledger row.
idea-forecast-bench vocab-build --only-probe \
  --topics llm_long_context,quantization,moe --tag mytag

# 3. Full run: extract (or reuse), embed, build+check every topic x cutoff
#    window, append the ledger row.
idea-forecast-bench vocab-build \
  --topics llm_long_context,quantization,moe --tag mytag

# 4. Offline re-build under a different config, reusing an already
#    extracted+embedded store (a pure clustering/tagging sweep).
idea-forecast-bench vocab-build \
  --topics llm_long_context,quantization,moe \
  --reuse-store <fingerprint> --skip-embed \
  --config config/vocab_alt.yaml --tag mytag_alt

# 5. Baseline row: push the v1 (combinatorial) element cache through the
#    v2 schema/pipeline once, then build it like any other store.
idea-forecast-bench vocab-v1-import
idea-forecast-bench vocab-build \
  --topics llm_long_context,quantization,moe \
  --reuse-store v1import --skip-embed --tag v1

# 6. Render the review page across topics and versions.
idea-forecast-bench vocab-html \
  --topics llm_long_context,quantization,moe --cutoff 2025-06 \
  --versions "v1=v1import:config/vocab.yaml,mytag=<fingerprint>:config/vocab.yaml"
```

## Output layout

```
output/vocab/
├── cache/<fingerprint>/          # one ConceptStore per (prompt, model, schema_version, temperature)
│   ├── records.jsonl             #   append-only ConceptRecords, paper_id keyed
│   ├── manifest.json             #   model/prompt/config_sha/corpus_fingerprint
│   └── vectors/<embed_model>.json
├── configs/                      # config YAML snapshots vocab-html reads by path
├── <tag>/<topic_id>/<cutoff>.md    # human report: metrics, background, concept tree, emerging, slot conflicts, unmapped terms, probe papers
├── <tag>/<topic_id>/<cutoff>.json  # same window's numbers, machine-readable
├── <tag>/summary.json            # nan-aware mean over every topic x cutoff window run under this tag
├── ledger.md                     # one row per (tag, prompt, config) -- the version history
└── vocab_review.html             # vocab-html output
```

## Results so far

From `output/vocab/ledger.md` (3 topics, 12 cutoffs each):

| tag | cov_obj | cov_mech | cov_both | spearman | stability | mid_layer | single_tok_obj | single_tok_mech |
|---|---|---|---|---|---|---|---|---|
| v1 | 0.687 | 0.694 | 0.477 | 0.658 | 0.655 | 0.447 | 0.087 | 0.144 |
| v2 | 0.489 | 0.218 | 0.107 | 0.559 | 0.572 | 0.156 | 0.001 | 0.005 |
| v2parent | 0.733 | 0.710 | 0.526 | 0.626 | 0.668 | 0.463 | 0.074 | 0.157 |
| v2_fine0.80_pf | 0.764 | 0.701 | 0.563 | 0.534 | 0.593 | 0.294 | 0.001 | 0.005 |
| v2_promote3_pf | 0.823 | 0.758 | 0.638 | 0.546 | 0.665 | 0.489 | 0.001 | 0.005 |
| v2_promote5_pf | 0.818 | 0.756 | 0.634 | 0.580 | 0.664 | 0.489 | 0.001 | 0.005 |

Reading:

- **Fine-only is too specific.** Plain `v2` (fine clusters only, no
  promotion) has the lowest coverage and `mid_layer_share` of any row: with
  no folding, a future paper's paraphrase of a one-off fine concept rarely
  re-hits it.
- **Parent-only is too coarse.** `v2parent` (concepts collapsed to the
  parent label) recovers coverage close to v1's, but at the cost of the most
  `n_slot_conflicts` of any row (41.8) — the coarse label loses the
  granularity that makes a concept combinable/forecastable rather than a
  restatement of the field.
- **The hybrid (`promote`) level locked on 2026-09-04.** `promote_min_count:
  5` was chosen over `3` for its better `spearman_pre_post` (0.580 vs.
  0.546) at essentially the same `coverage_both` (0.634 vs. 0.638) and
  identical `mid_layer_share` (0.489) — see the comment in
  `config/vocab.yaml`.
