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

## 20-topic run (2026-09-04, 12-month window, hybrid promote5)

Twenty topics (8 LLM-side, 12 other fields), corpus 2024-10 .. 2025-09,
cutoffs 2025-01 .. 2025-06, 19,183 papers extracted (0.13% unparseable),
116,604 concept texts embedded. Built in four topic shards with
`--reuse-store b493410c0021 --skip-embed` (two shards at a time: every
process loads the 2.6 GB vector file, and the machine has 16 GB). Per-topic
means over the six cutoffs are in `output/vocab/v2_20topics/<topic>/*.json`;
the review page is `output/vocab/vocab_explainer.html` (`vocab-explainer`).

| topic | cov_both | cov_obj | cov_mech | spearman | stability | mid_layer |
|---|---|---|---|---|---|---|
| reinforcement_learning | 0.847 | 0.915 | 0.924 | 0.59 | 0.67 | 0.62 |
| llm_alignment_rlhf | 0.783 | 0.901 | 0.859 | 0.55 | 0.65 | 0.55 |
| quantization | 0.776 | 0.905 | 0.852 | 0.59 | 0.66 | 0.53 |
| graph_gnn | 0.773 | 0.874 | 0.880 | 0.58 | 0.68 | 0.58 |
| time_series | 0.735 | 0.838 | 0.872 | 0.57 | 0.67 | 0.60 |
| federated_learning | 0.732 | 0.833 | 0.866 | 0.59 | 0.69 | 0.64 |
| molecular_graph | 0.722 | 0.912 | 0.772 | 0.56 | 0.67 | 0.45 |
| medical_imaging | 0.691 | 0.863 | 0.787 | 0.54 | 0.66 | 0.49 |
| image_gen_diffusion | 0.656 | 0.830 | 0.780 | 0.55 | 0.68 | 0.47 |
| llm_long_context | 0.633 | 0.795 | 0.770 | 0.56 | 0.62 | 0.44 |
| continual_learning | 0.594 | 0.737 | 0.806 | 0.50 | 0.67 | 0.45 |
| llm_agents | 0.576 | 0.732 | 0.775 | 0.47 | 0.63 | 0.35 |
| llm_reasoning_math | 0.575 | 0.857 | 0.640 | 0.54 | 0.61 | 0.36 |
| speech_audio | 0.561 | 0.811 | 0.692 | 0.44 | 0.63 | 0.30 |
| autonomous_driving | 0.561 | 0.816 | 0.682 | 0.48 | 0.65 | 0.33 |
| moe | 0.503 | 0.717 | 0.674 | 0.49 | 0.62 | 0.40 |
| recommendation | 0.495 | 0.669 | 0.723 | 0.46 | 0.65 | 0.35 |
| rag_retrieval | 0.485 | 0.637 | 0.727 | 0.50 | 0.61 | 0.36 |
| protein_structure | 0.466 | 0.784 | 0.567 | 0.43 | 0.62 | 0.30 |
| code_llm | 0.400 | 0.591 | 0.629 | 0.41 | 0.61 | 0.25 |

Reading: 13 of 20 topics clear `cov_both >= 0.5 and mid_layer >= 0.35`.
Coverage tracks topic size: the six largest topics (2,000+ papers in the
window) all score 0.73 or better, while the smallest (code_llm 382,
protein_structure 208 papers) fail, because `promote_min_count: 5` cannot
be met when a cutoff has only a hundred training papers. The next
vocabulary change is therefore to scale the fold threshold with the
topic's training size (or use a longer window for small topics), not to
touch the prompt or the merge thresholds.
