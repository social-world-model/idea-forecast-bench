#!/usr/bin/env python3
"""Zero-API "combination-level" forward/backward oracle test on the v2
concept vocabulary.

No LLM, no judge. For each topic x cutoff: build the vocabulary from
training papers only (``idea_forecast_bench.vocab.build.build_vocabulary``),
sample k (object, mechanism) concept pairs under several samplers, and check
whether each pair co-occurs in a FUTURE paper (forward) and in a PAST paper
of an equal-length window (backward, via
``idea_forecast_bench.backtest.split_backward_target``). Also reports the
"collision" statistic: how many training papers already contain both
concepts of a sampled pair.

This is a cheap sanity check on the vocabulary itself, independent of the
combinatorial forecaster's LLM realization step: if a rule like `full`
(heat x rising x unpaired-bonus) cannot beat `random`/`copy` at predicting
which pairs a future paper will actually combine, no amount of prompting
downstream will fix that.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy import stats

from idea_forecast_bench.atomic import atomic_write_text
from idea_forecast_bench.backtest import (
    split_backward_target,
    split_train_future_by_cutoff,
)
from idea_forecast_bench.combinatorial.config import (
    CombinatorialConfig,
    load_combinatorial_config,
)
from idea_forecast_bench.combinatorial.embeddings import VectorStore
from idea_forecast_bench.models import PaperRecord
from idea_forecast_bench.paper_cache import load_papers_and_topics
from idea_forecast_bench.papers import (
    date_to_ordinal,
    get_paper_published_date,
    month_start_date,
)
from idea_forecast_bench.vocab.build import assign_record, build_vocabulary
from idea_forecast_bench.vocab.config import VocabConfig, load_vocab_config
from idea_forecast_bench.vocab.store import ConceptStore
from idea_forecast_bench.vocab.types import (
    SLOTS,
    Concept,
    ConceptRecord,
    Vocabulary,
    concept_key,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = "output/vocab/cache"
SAMPLERS = ("random", "heat", "copy", "full", "new_heat")
PairKey = tuple[str, str]
_BOOTSTRAP_RESAMPLES = 2000


# ---- CLI --------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/hf_full/raw_markdown")
    parser.add_argument("--start-month", required=True)
    parser.add_argument("--end-month", required=True)
    parser.add_argument("--topics", required=True, help="Comma-separated topic ids.")
    parser.add_argument("--cutoffs", required=True, help="Comma-separated YYYY-MM.")
    parser.add_argument("--store", required=True, help="Concept-store fingerprint.")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--config", default=None, help="vocab.yaml override")
    parser.add_argument(
        "--combinatorial-config",
        default=None,
        help="combinatorial.yaml override (state/sampler numeric knobs)",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--output-dir", default=None, help="Default: output/vocab/oracle/<tag>"
    )
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--top-m", type=int, default=40)
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ---- deterministic seeding ---------------------------------------------


def _seed(topic_id: str, cutoff: str, sampler: str, rep: int) -> int:
    digest = hashlib.sha256(f"{topic_id}|{cutoff}|{sampler}|{rep}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


# ---- concept mapping ----------------------------------------------------


def _paper_concept_ids(record: ConceptRecord, vocab: Vocabulary) -> set[str]:
    """All concept ids a training paper's record maps to, trying all three
    slots per text (the dominant-slot remap means at most one hits), with
    background concepts dropped."""
    texts = {term.text for _slot, term in record.terms() if term.text}
    ids: set[str] = set()
    for text in texts:
        for slot in SLOTS:
            cid = vocab.member_of.get(concept_key(slot, text))
            if cid is not None:
                ids.add(cid)
                break
    return {cid for cid in ids if not vocab.concepts[cid].background}


def _train_concept_map(
    train_ok: Sequence[tuple[PaperRecord, ConceptRecord]], vocab: Vocabulary
) -> dict[str, set[str]]:
    return {
        paper.paper_id: _paper_concept_ids(record, vocab) for paper, record in train_ok
    }


def _future_concept_map(
    future_ok: Sequence[tuple[PaperRecord, ConceptRecord]],
    vocab: Vocabulary,
    vectors: Mapping[str, Sequence[float]],
    threshold: float,
) -> dict[str, set[str]]:
    """Object + mechanism concept ids assigned to each future paper via
    ``assign_record`` (exact text, else nearest same-slot leader)."""
    out: dict[str, set[str]] = {}
    for paper, record in future_ok:
        assigned = assign_record(record, vocab, vectors, threshold)
        ids = {cid for cid in assigned["object"] if cid is not None}
        ids.update(cid for cid in assigned["mechanism"] if cid is not None)
        out[paper.paper_id] = ids
    return out


def _reverse_index(concept_map: Mapping[str, set[str]]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for paper_id, cids in concept_map.items():
        for cid in cids:
            index[cid].add(paper_id)
    return index


def _hits(index: Mapping[str, set[str]], o: str, m: str) -> bool:
    return bool(index.get(o, set()) & index.get(m, set()))


# ---- heat / rising / unpaired-bonus (compact reimplementation of
# combinatorial.state, over concept ids instead of combinatorial elements) --

DAYS_PER_MONTH = 30.44


def _months_before(cutoff_ord: int, published_date: str) -> float:
    return max(0.0, (cutoff_ord - date_to_ordinal(published_date)) / DAYS_PER_MONTH)


def _decay_weight(age_months: float, half_life_months: float) -> float:
    return float(2.0 ** (-age_months / half_life_months))


class _WindowStats:
    """Heat, pair co-occurrence, and the rising/unpaired-bonus signals for
    one topic x cutoff, restricted to candidate (non-background, count>=2)
    object and mechanism concepts."""

    def __init__(
        self,
        train_concept_map: Mapping[str, set[str]],
        paper_dates: Mapping[str, str],
        cutoff_ord: int,
        obj_ids: set[str],
        mech_ids: set[str],
        combo_cfg: CombinatorialConfig,
    ) -> None:
        half_life = combo_cfg.state.half_life_months
        recent_months = combo_cfg.state.recent_months
        alpha = combo_cfg.state.smoothing_alpha
        self.clip = combo_cfg.sampler.rising_log_clip
        self.lambda_rising = combo_cfg.sampler.lambda_rising
        self.lambda_unpaired = combo_cfg.sampler.lambda_unpaired

        self.heat: dict[str, float] = defaultdict(float)
        pair_count: dict[PairKey, int] = defaultdict(int)
        pair_recent: dict[PairKey, int] = defaultdict(int)
        pair_older: dict[PairKey, int] = defaultdict(int)
        max_age = 0.0
        for paper_id, cids in train_concept_map.items():
            age = _months_before(cutoff_ord, paper_dates[paper_id])
            max_age = max(max_age, age)
            weight = _decay_weight(age, half_life)
            is_recent = age <= recent_months
            for cid in cids:
                self.heat[cid] += weight
            objs_here = [c for c in cids if c in obj_ids]
            mechs_here = [c for c in cids if c in mech_ids]
            for o in objs_here:
                for m in mechs_here:
                    key = (o, m)
                    pair_count[key] += 1
                    if is_recent:
                        pair_recent[key] += 1
                    else:
                        pair_older[key] += 1
        self.pair_count = dict(pair_count)
        recent_span = max(recent_months, 1e-9)
        older_span = max(1.0, max_age - recent_months)
        self._pair_recent_rate = {k: v / recent_span for k, v in pair_recent.items()}
        self._pair_older_rate = {k: v / older_span for k, v in pair_older.items()}
        self._alpha = alpha

        pool_heat = [self.heat.get(cid, 0.0) for cid in (obj_ids | mech_ids)]
        self.max_heat = max(pool_heat) if pool_heat else 0.0
        self.hot_threshold = (
            float(np.quantile(np.asarray(pool_heat), combo_cfg.state.hot_quantile))
            if pool_heat
            else 0.0
        )

    def rising(self, o: str, m: str) -> float:
        recent = self._pair_recent_rate.get((o, m), 0.0)
        older = self._pair_older_rate.get((o, m), 0.0)
        value = math.log2((recent + self._alpha) / (older + self._alpha))
        return max(-self.clip, min(self.clip, value))

    def unpaired_bonus(self, o: str, m: str) -> float:
        if self.pair_count.get((o, m), 0) > 0 or self.max_heat <= 0:
            return 0.0
        ho, hm = self.heat.get(o, 0.0), self.heat.get(m, 0.0)
        if ho < self.hot_threshold or hm < self.hot_threshold:
            return 0.0
        return min(ho, hm) / self.max_heat

    def full_score(self, o: str, m: str) -> float:
        r = self.rising(o, m)
        u = self.unpaired_bonus(o, m)
        return (
            self.heat.get(o, 0.0)
            * self.heat.get(m, 0.0)
            * (1.0 + self.lambda_rising * max(0.0, r) + self.lambda_unpaired * u)
        )


# ---- sampling ------------------------------------------------------------


def _weighted_sample_without_replacement(
    rng: random.Random, items: Sequence[PairKey], weights: Sequence[float], k: int
) -> list[PairKey]:
    pool = list(items)
    w = [float(x) for x in weights]
    chosen: list[PairKey] = []
    for _ in range(min(k, len(pool))):
        total = sum(w)
        if total <= 0:
            pick = rng.randrange(len(pool))
        else:
            r = rng.uniform(0.0, total)
            acc = 0.0
            pick = len(pool) - 1
            for i, wt in enumerate(w):
                acc += wt
                if acc >= r:
                    pick = i
                    break
        chosen.append(pool.pop(pick))
        w.pop(pick)
    return chosen


_SamplerPools = dict[str, tuple[list[PairKey], list[float] | None]]


def _sampler_pools(
    obj_ids: list[str],
    mech_ids: list[str],
    stats_: _WindowStats,
    top_m: int,
) -> tuple[_SamplerPools, list[PairKey]]:
    top_objs = sorted(obj_ids, key=lambda c: -stats_.heat.get(c, 0.0))[:top_m]
    top_mechs = sorted(mech_ids, key=lambda c: -stats_.heat.get(c, 0.0))[:top_m]
    top_pairs = [(o, m) for o in top_objs for m in top_mechs]
    all_pairs = [(o, m) for o in obj_ids for m in mech_ids]
    seen_pairs = [p for p in all_pairs if stats_.pair_count.get(p, 0) > 0]
    new_pairs = [p for p in top_pairs if stats_.pair_count.get(p, 0) == 0]

    heat_w = [stats_.heat.get(o, 0.0) * stats_.heat.get(m, 0.0) for o, m in top_pairs]
    full_w = [stats_.full_score(o, m) for o, m in top_pairs]
    copy_w = [float(stats_.pair_count[p]) for p in seen_pairs]
    new_heat_w = [
        stats_.heat.get(o, 0.0) * stats_.heat.get(m, 0.0) for o, m in new_pairs
    ]

    pools: _SamplerPools = {
        "random": (all_pairs, None),
        "heat": (top_pairs, heat_w),
        "copy": (seen_pairs, copy_w),
        "full": (top_pairs, full_w),
        "new_heat": (new_pairs, new_heat_w),
    }
    return pools, top_pairs


def _nanmean(values: Sequence[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else math.nan


# ---- per-window run -------------------------------------------------------


def _process_window(
    *,
    topic_id: str,
    cutoff: str,
    topic_papers: list[PaperRecord],
    records: Mapping[str, ConceptRecord],
    vectors: Mapping[str, Sequence[float]],
    cfg: VocabConfig,
    combo_cfg: CombinatorialConfig,
    reps: int,
    top_m: int,
    k: int,
) -> list[dict[str, object]] | None:
    train, future, _end_month, _end_date = split_train_future_by_cutoff(
        topic_papers, cutoff_month=cutoff, horizon_months=cfg.checks.horizon_months
    )
    if not train or not future:
        print(
            f"  WARNING: skipping {topic_id}@{cutoff} (train={len(train)}, "
            f"future={len(future)})",
            file=sys.stderr,
        )
        return None
    _remaining, backward, _start_date, _cutoff_date = split_backward_target(
        topic_papers, cutoff_month=cutoff, horizon_months=cfg.checks.horizon_months
    )

    cutoff_date = month_start_date(cutoff)
    cutoff_ord = date_to_ordinal(cutoff_date)
    vocab = build_vocabulary(
        topic_id=topic_id,
        cutoff_month=cutoff,
        cutoff_date=cutoff_date,
        train_papers=train,
        records=records,
        vectors=vectors,
        cfg=cfg,
    )

    train_ok = [
        (p, records[p.paper_id])
        for p in train
        if p.paper_id in records and records[p.paper_id].ok
    ]
    future_ok = [
        (p, records[p.paper_id])
        for p in future
        if p.paper_id in records and records[p.paper_id].ok
    ]
    if not train_ok or not future_ok:
        print(
            f"  WARNING: skipping {topic_id}@{cutoff} (no ok records "
            f"train={len(train_ok)}, future={len(future_ok)})",
            file=sys.stderr,
        )
        return None

    train_concept_map = _train_concept_map(train_ok, vocab)
    future_concept_map = _future_concept_map(
        future_ok, vocab, vectors, cfg.checks.assign_threshold
    )
    backward_ids = {p.paper_id for p in backward}
    backward_concept_map = {
        pid: cids for pid, cids in train_concept_map.items() if pid in backward_ids
    }
    future_index = _reverse_index(future_concept_map)
    backward_index = _reverse_index(backward_concept_map)

    candidate_objects: list[Concept] = [
        c for c in vocab.combinable() if c.slot == "object" and c.count >= 2
    ]
    candidate_mechs: list[Concept] = [
        c for c in vocab.combinable() if c.slot == "mechanism" and c.count >= 2
    ]
    obj_ids = [c.id for c in candidate_objects]
    mech_ids = [c.id for c in candidate_mechs]
    if len(obj_ids) * len(mech_ids) < k:
        print(
            f"  WARNING: skipping {topic_id}@{cutoff} (too few candidate "
            f"pairs: {len(obj_ids)} objects x {len(mech_ids)} mechanisms)",
            file=sys.stderr,
        )
        return None

    paper_dates = {p.paper_id: get_paper_published_date(p) for p, _r in train_ok}
    stats_ = _WindowStats(
        train_concept_map,
        paper_dates,
        cutoff_ord,
        set(obj_ids),
        set(mech_ids),
        combo_cfg,
    )
    pools, top_pairs = _sampler_pools(obj_ids, mech_ids, stats_, top_m)

    base_future = _nanmean(
        [1.0 if _hits(future_index, o, m) else 0.0 for o, m in top_pairs]
    )
    base_backward = _nanmean(
        [1.0 if _hits(backward_index, o, m) else 0.0 for o, m in top_pairs]
    )

    rows: list[dict[str, object]] = []
    for sampler in SAMPLERS:
        pool, weights = pools[sampler]
        fwd_reps: list[float] = []
        bwd_reps: list[float] = []
        delta_reps: list[float] = []
        new_share_reps: list[float] = []
        collisions: list[int] = []
        n_drawn_total = 0
        for rep in range(reps):
            rng = random.Random(_seed(topic_id, cutoff, sampler, rep))
            if not pool:
                draw: list[PairKey] = []
            elif weights is None:
                draw = rng.sample(pool, min(k, len(pool)))
            else:
                draw = _weighted_sample_without_replacement(rng, pool, weights, k)
            n = len(draw)
            n_drawn_total += n
            if n == 0:
                fwd_reps.append(math.nan)
                bwd_reps.append(math.nan)
                delta_reps.append(math.nan)
                new_share_reps.append(math.nan)
                continue
            hits_f = sum(1 for o, m in draw if _hits(future_index, o, m))
            hits_b = sum(1 for o, m in draw if _hits(backward_index, o, m))
            f_rate, b_rate = hits_f / n, hits_b / n
            fwd_reps.append(f_rate)
            bwd_reps.append(b_rate)
            delta_reps.append(f_rate - b_rate)
            new_share_reps.append(
                sum(1 for p in draw if stats_.pair_count.get(p, 0) == 0) / n
            )
            collisions.extend(stats_.pair_count.get(p, 0) for p in draw)

        collision_arr = np.asarray(collisions, dtype=float) if collisions else None
        rows.append(
            {
                "topic_id": topic_id,
                "cutoff_month": cutoff,
                "sampler": sampler,
                "forward_p5": _nanmean(fwd_reps),
                "backward_p5": _nanmean(bwd_reps),
                "delta": _nanmean(delta_reps),
                "new_pair_share": _nanmean(new_share_reps),
                "collision_median": float(np.median(collision_arr))
                if collision_arr is not None
                else math.nan,
                "collision_p90": float(np.percentile(collision_arr, 90))
                if collision_arr is not None
                else math.nan,
                "collision_max": float(np.max(collision_arr))
                if collision_arr is not None
                else math.nan,
                "pool_size": len(pool),
                "n_drawn_total": n_drawn_total,
                "reps": reps,
                "k": k,
                "n_train": len(train),
                "n_future": len(future),
                "n_backward": len(backward),
                "n_candidate_objects": len(obj_ids),
                "n_candidate_mechanisms": len(mech_ids),
                "base_rate_future": base_future,
                "base_rate_backward": base_backward,
                "raw_collisions": [int(x) for x in collisions],
            }
        )
    return rows


# ---- summary.md -----------------------------------------------------------


def _bootstrap_ci_mean(values: Sequence[float], seed: int) -> tuple[float, float]:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return math.nan, math.nan
    arr = np.asarray(finite, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(_BOOTSTRAP_RESAMPLES)
    for i in range(_BOOTSTRAP_RESAMPLES):
        sample = arr[rng.integers(0, n, size=n)]
        means[i] = float(np.mean(sample))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _sign_test_pvalue(deltas: Sequence[float]) -> tuple[float, int, int]:
    nonzero = [d for d in deltas if not math.isnan(d) and d != 0.0]
    n = len(nonzero)
    if n == 0:
        return math.nan, 0, 0
    positive = sum(1 for d in nonzero if d > 0)
    pvalue = stats.binomtest(positive, n, 0.5, alternative="two-sided").pvalue
    return float(pvalue), positive, n


def _render_summary(all_rows: list[dict[str, object]], args: argparse.Namespace) -> str:
    by_sampler: dict[str, dict[str, list[float]]] = {
        s: {"forward": [], "backward": [], "delta": [], "new_pair_share": []}
        for s in SAMPLERS
    }
    collisions_by_sampler: dict[str, list[int]] = {s: [] for s in SAMPLERS}
    base_by_window: dict[tuple[str, str], tuple[float, float]] = {}
    delta_by_topic_sampler: dict[tuple[str, str], list[float]] = defaultdict(list)

    for row in all_rows:
        sampler = str(row["sampler"])
        by_sampler[sampler]["forward"].append(float(row["forward_p5"]))
        by_sampler[sampler]["backward"].append(float(row["backward_p5"]))
        by_sampler[sampler]["delta"].append(float(row["delta"]))
        by_sampler[sampler]["new_pair_share"].append(float(row["new_pair_share"]))
        collisions_by_sampler[sampler].extend(row["raw_collisions"])  # type: ignore[arg-type]
        key = (str(row["topic_id"]), str(row["cutoff_month"]))
        base_by_window[key] = (
            float(row["base_rate_future"]),
            float(row["base_rate_backward"]),
        )
        if sampler in ("full", "copy"):
            delta_by_topic_sampler[(str(row["topic_id"]), sampler)].append(
                float(row["delta"])
            )

    lines: list[str] = []
    lines.append(f"# vocab-oracle: {args.tag}\n")
    lines.append(
        f"topics={args.topics}  cutoffs={args.cutoffs}  reps={args.reps}  "
        f"top_m={args.top_m}  k={args.k}  store={args.store}\n"
    )
    lines.append(
        "Forward/backward P@k and the collision statistic, per sampler, "
        "averaged over reps then over topic x cutoff windows. `delta` = "
        "forward - backward; a sign test and a bootstrap 95% CI are over "
        "windows (n = one row per topic x cutoff).\n"
    )

    lines.append(
        "| sampler | forward | backward | delta | sign-test p | delta 95% CI | "
        "new_pair_share | collision median/p90/max |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for sampler in SAMPLERS:
        vals = by_sampler[sampler]
        pvalue, n_pos, n_nonzero = _sign_test_pvalue(vals["delta"])
        seed = int.from_bytes(hashlib.sha256(sampler.encode()).digest()[:8], "big")
        lo, hi = _bootstrap_ci_mean(vals["delta"], seed)
        coll = collisions_by_sampler[sampler]
        coll_str = (
            f"{np.median(coll):.1f}/{np.percentile(coll, 90):.1f}/{max(coll)}"
            if coll
            else "n/a"
        )
        p_str = (
            f"{pvalue:.4f} ({n_pos}/{n_nonzero})" if not math.isnan(pvalue) else "n/a"
        )
        lines.append(
            f"| {sampler} | {_nanmean(vals['forward']):.4f} | "
            f"{_nanmean(vals['backward']):.4f} | {_nanmean(vals['delta']):+.4f} | "
            f"{p_str} | [{lo:+.4f}, {hi:+.4f}] | "
            f"{_nanmean(vals['new_pair_share']):.4f} | {coll_str} |"
        )
    lines.append("")

    lines.append(
        f"## Window base rates (top-{args.top_m}x{args.top_m} candidate pairs)\n"
    )
    lines.append("| topic | cutoff | base_rate_future | base_rate_backward |")
    lines.append("|---|---|---|---|")
    for (topic_id, cutoff), (bf, bb) in sorted(base_by_window.items()):
        lines.append(f"| {topic_id} | {cutoff} | {bf:.4f} | {bb:.4f} |")
    lines.append("")

    lines.append("## Per-topic delta (`full` vs `copy`, mean over cutoffs)\n")
    topics = sorted({t for t, _s in delta_by_topic_sampler})
    lines.append("| topic | full delta | copy delta |")
    lines.append("|---|---|---|")
    for topic_id in topics:
        full_delta = _nanmean(delta_by_topic_sampler.get((topic_id, "full"), []))
        copy_delta = _nanmean(delta_by_topic_sampler.get((topic_id, "copy"), []))
        lines.append(f"| {topic_id} | {full_delta:+.4f} | {copy_delta:+.4f} |")
    lines.append("")

    return "\n".join(lines)


# ---- main -----------------------------------------------------------------


def _safe_vector_filename(embed_model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in embed_model)


def main() -> int:
    args = parse_args()
    input_dir = _resolve(args.input_dir)
    papers, _topics, grouped = load_papers_and_topics(
        input_dir, args.start_month, args.end_month
    )

    topic_ids = [t.strip() for t in args.topics.split(",") if t.strip()]
    unknown = [t for t in topic_ids if t not in grouped]
    if unknown:
        print(f"Unknown topic id(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    cutoffs = [c.strip() for c in args.cutoffs.split(",") if c.strip()]
    if not topic_ids or not cutoffs:
        print(
            "--topics and --cutoffs must each name at least one value", file=sys.stderr
        )
        return 2

    cfg = load_vocab_config(args.config)
    combo_cfg = load_combinatorial_config(args.combinatorial_config)

    store = ConceptStore(_resolve(args.cache_dir), args.store)
    print(f"concept store: {store.dir}", flush=True)
    records = store.load()
    print(f"{len(records)} concept records loaded", flush=True)

    vector_name = _safe_vector_filename(cfg.cluster.embed_model)
    vector_path = store.vectors_dir / f"{vector_name}.json"
    print(f"loading vectors from {vector_path} ...", flush=True)
    vectors = VectorStore(vector_path).view()
    print(f"{len(vectors)} vectors loaded", flush=True)

    output_dir = (
        _resolve(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "output" / "vocab" / "oracle" / args.tag
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    n_windows = 0
    for topic_id in topic_ids:
        topic_papers = grouped.get(topic_id, [])
        for cutoff in cutoffs:
            rows = _process_window(
                topic_id=topic_id,
                cutoff=cutoff,
                topic_papers=topic_papers,
                records=records,
                vectors=vectors,
                cfg=cfg,
                combo_cfg=combo_cfg,
                reps=args.reps,
                top_m=args.top_m,
                k=args.k,
            )
            if rows is None:
                continue
            n_windows += 1
            all_rows.extend(rows)
            print(f"  {topic_id}@{cutoff}: ok", flush=True)

    if not all_rows:
        print("no window produced any rows; nothing to write", file=sys.stderr)
        return 2

    jsonl_lines = []
    for row in all_rows:
        payload = {k: v for k, v in row.items() if k != "raw_collisions"}
        jsonl_lines.append(json.dumps(payload, sort_keys=True))
    atomic_write_text(output_dir / "windows.jsonl", "\n".join(jsonl_lines) + "\n")

    summary_text = _render_summary(all_rows, args)
    atomic_write_text(output_dir / "summary.md", summary_text)

    print(f"\n{n_windows} window(s) -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
