from __future__ import annotations

import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from idea_forecast_bench.combinatorial.canonicalize import split_key
from idea_forecast_bench.combinatorial.config import (
    CombinatorialConfig,
    PromptPair,
    StateConfig,
    load_combinatorial_config,
    load_prompt_pair,
)
from idea_forecast_bench.combinatorial.embeddings import VectorStore
from idea_forecast_bench.combinatorial.evidence import retrieve_evidence
from idea_forecast_bench.combinatorial.llm_caller import (
    TextCaller,
    caller_for_model,
    callers_for_base_urls,
)
from idea_forecast_bench.combinatorial.realize import TEMPLATE_MODEL, realize_combos
from idea_forecast_bench.combinatorial.sampler import (
    VARIANT_FULL,
    VARIANTS,
    sample_combos,
    window_seed,
)
from idea_forecast_bench.combinatorial.state import decay_weight, months_before
from idea_forecast_bench.combinatorial.types import (
    MOVES,
    Combo,
    CommunityState,
    Element,
    ElementType,
    PairKey,
    pair_key,
)
from idea_forecast_bench.models import IdeaPrediction, PaperRecord
from idea_forecast_bench.papers import (
    date_to_ordinal,
    get_paper_published_date,
    month_start_date,
)
from idea_forecast_bench.strategy.base import IdeaStrategy
from idea_forecast_bench.vocab.build import build_vocabulary
from idea_forecast_bench.vocab.config import VocabConfig, load_vocab_config
from idea_forecast_bench.vocab.store import ConceptStore
from idea_forecast_bench.vocab.types import (
    SLOTS,
    Concept,
    ConceptRecord,
    Slot,
    Vocabulary,
    concept_key,
)

_DEFAULT_MODEL = "gpt-4o-qwen35"
_HORIZON_MONTHS = 3
_VOCAB_TOPIC_ID = "window"

#: A v2 concept's slot maps onto the element type the sampler/realiser know
#: about. "frame" has no v2 counterpart, so it never appears here -- any
#: sampler.type_pattern that needs a frame simply has no usable vocabulary.
_SLOT_TO_ELEMENT_TYPE: Mapping[Slot, ElementType] = {
    "object": "domain",
    "mechanism": "method",
    "problem": "theme",
}


def _canonical_id(concept: Concept) -> str:
    return f"{_SLOT_TO_ELEMENT_TYPE[concept.slot]}:{concept.label}"


def _lookup_concept(vocab: Vocabulary, text: str) -> Concept | None:
    """Mirrors ``vocab.checks._lookup_concept``: try every slot's key, since
    a term's own recorded slot can be the losing side of a slot conflict."""
    for slot in SLOTS:
        concept_id = vocab.member_of.get(concept_key(slot, text))
        if concept_id is None:
            continue
        concept = vocab.concepts.get(concept_id)
        if concept is not None:
            return concept
    return None


def _paper_canonical_ids(record: ConceptRecord, vocab: Vocabulary) -> set[str]:
    """Canonical element ids for one paper: background concepts and terms
    with no concept at all are dropped, so they can never be sampled."""
    ids: set[str] = set()
    bg_labels = _background_labels(vocab)
    for _slot, term in record.terms():
        if not term.text:
            continue
        concept = _lookup_concept(vocab, term.text)
        if concept is None or concept.background or concept.label in bg_labels:
            continue
        ids.add(_canonical_id(concept))
    return ids


def _background_labels(vocab: Vocabulary) -> frozenset[str]:
    """A label that is background in any slot is background in every slot:
    the topic name must not re-enter through a second slot."""
    return frozenset(c.label for c in vocab.concepts.values() if c.background)


def _build_state(
    train_papers: Sequence[PaperRecord],
    records: Mapping[str, ConceptRecord],
    vocab: Vocabulary,
    cutoff_date: str,
    state_cfg: StateConfig,
    min_count: int,
) -> CommunityState:
    """Same community-state formulas as
    ``idea_forecast_bench.combinatorial.state.build_state``, but sourced
    from a v2 :class:`Vocabulary` instead of raw ``ExtractionRecord``
    elements: a paper's canonical ids are its v2 concepts (background
    excluded), and moves are uniform because v2 records carry no move."""
    cutoff_ord = date_to_ordinal(cutoff_date)
    usable: list[tuple[PaperRecord, ConceptRecord, float]] = []
    for paper in train_papers:
        published = get_paper_published_date(paper)
        if date_to_ordinal(published) > cutoff_ord:
            raise ValueError(
                f"paper {paper.paper_id} dated {published} is after cutoff {cutoff_date}"
            )
        record = records.get(paper.paper_id)
        if record is None or not record.ok:
            continue
        usable.append((paper, record, months_before(cutoff_ord, published)))

    concept_by_canonical: dict[str, Concept] = {}
    bg_labels = _background_labels(vocab)
    for concept in vocab.concepts.values():
        if concept.background or concept.label in bg_labels:
            continue
        concept_by_canonical[_canonical_id(concept)] = concept

    paper_ids: dict[str, set[str]] = defaultdict(set)
    first_seen: dict[str, str] = {}
    heat: dict[str, float] = defaultdict(float)
    pair_count: dict[PairKey, int] = defaultdict(int)
    pair_heat: dict[PairKey, float] = defaultdict(float)
    pair_recent: dict[PairKey, int] = defaultdict(int)
    pair_older: dict[PairKey, int] = defaultdict(int)
    max_age = 0.0

    for paper, record, age in usable:
        canonical = sorted(_paper_canonical_ids(record, vocab))
        weight = decay_weight(age, state_cfg.half_life_months)
        published = get_paper_published_date(paper)
        max_age = max(max_age, age)
        is_recent = age <= state_cfg.recent_months
        for cid in canonical:
            paper_ids[cid].add(paper.paper_id)
            heat[cid] += weight
            if cid not in first_seen or published < first_seen[cid]:
                first_seen[cid] = published
        for i, a in enumerate(canonical):
            for b in canonical[i + 1 :]:
                key = pair_key(a, b)
                pair_count[key] += 1
                pair_heat[key] += weight
                if is_recent:
                    pair_recent[key] += 1
                else:
                    pair_older[key] += 1

    elements: dict[str, Element] = {}
    for cid, ids in paper_ids.items():
        element_type, label = split_key(cid)
        matched = concept_by_canonical.get(cid)
        variants = matched.variants if matched is not None else (label,)
        elements[cid] = Element(
            id=cid,
            type=element_type,
            label=label,
            variants=variants,
            paper_ids=tuple(sorted(ids)),
            first_seen=first_seen[cid],
            count=len(ids),
            heat=heat[cid],
        )

    recent_span = max(state_cfg.recent_months, 1e-9)
    older_span = max(1.0, max_age - state_cfg.recent_months)
    pair_recent_rate = {k: v / recent_span for k, v in pair_recent.items()}
    pair_older_rate = {k: v / older_span for k, v in pair_older.items()}

    heats = [e.heat for e in elements.values() if e.count >= min_count]
    max_heat = max(heats) if heats else 0.0
    hot_threshold = (
        float(np.quantile(np.asarray(heats), state_cfg.hot_quantile)) if heats else 0.0
    )

    return CommunityState(
        cutoff_date=cutoff_date,
        n_train=len(train_papers),
        n_with_records=len(usable),
        elements=elements,
        pair_count=dict(pair_count),
        pair_heat=dict(pair_heat),
        pair_recent_rate=pair_recent_rate,
        pair_older_rate=pair_older_rate,
        move_dist=dict.fromkeys(MOVES, 1.0 / len(MOVES)),
        max_heat=max_heat,
        hot_threshold=hot_threshold,
    )


def _stamp_vocab_metadata(
    predictions: Sequence[IdeaPrediction], vocab: Vocabulary
) -> list[IdeaPrediction]:
    return [
        IdeaPrediction(
            rank=p.rank,
            title=p.title,
            rationale=p.rationale,
            approach=p.approach,
            score=p.score,
            confidence=p.confidence,
            key_terms=p.key_terms,
            metadata={**p.metadata, "vocab_version": vocab.config_sha, "vocab": "v2"},
        )
        for p in predictions
    ]


class VocabCombinatorialStrategy(IdeaStrategy):
    """Ramon-Llull-style forecaster, same sampler/realiser as
    ``CombinatorialStrategy``, but its community state is built from the v2
    concept vocabulary (``idea_forecast_bench.vocab``) instead of the v1
    per-paper element cache. One instance is shared across topic threads, so
    everything loaded here is read-only after construction."""

    name = "vocab_combinatorial"

    def __init__(
        self,
        model_name: str | None = None,
        *,
        variant: str = VARIANT_FULL,
        vocab_store_dir: str | None = None,
        vocab_config_path: str | None = None,
        config_path: str | None = None,
        temperature: float | None = None,
        base_urls: Sequence[str] | None = None,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose one of {VARIANTS}")
        if not vocab_store_dir:
            raise ValueError(
                "vocab_combinatorial strategies need --vocab-store "
                "(a dir under output/vocab/cache/<fingerprint>, run "
                "examples/benchmark/vocab_build.py first)."
            )
        self.model_name = model_name or _DEFAULT_MODEL
        self.variant = variant
        self.temperature = temperature
        self.config: CombinatorialConfig = load_combinatorial_config(config_path)
        self.vocab_cfg: VocabConfig = load_vocab_config(vocab_config_path)

        store_dir = Path(vocab_store_dir)
        store = ConceptStore(store_dir.parent, store_dir.name)
        self.records: Mapping[str, ConceptRecord] = store.load()
        if not self.records:
            raise ValueError(f"vocab store {store.dir} holds no records")

        safe_name = "".join(
            ch if ch.isalnum() or ch in "-._" else "_"
            for ch in self.vocab_cfg.cluster.embed_model
        )
        self.vectors = VectorStore(store.vectors_dir / f"{safe_name}.json").view()
        if not self.vectors:
            print(
                f"[vocab_combinatorial WARNING] no concept vectors under "
                f"{store.dir}; near-synonyms will not be merged.",
                file=sys.stderr,
                flush=True,
            )

        self.realize_prompt: PromptPair = load_prompt_pair(self.config.realize.prompt)
        self._caller: TextCaller | None
        if self.model_name == TEMPLATE_MODEL:
            self._caller = None
        elif base_urls:
            self._caller = callers_for_base_urls(self.model_name, base_urls)
        else:
            self._caller = caller_for_model(self.model_name)

    # ------------------------------------------------------------------
    def _train_at_cutoff(
        self, train_papers: Sequence[PaperRecord], cutoff_date: str
    ) -> list[PaperRecord]:
        """Defensive re-filter: a no-op under run_backtest, but it makes the
        time boundary a property of the strategy, not of its caller."""
        cutoff_ord = date_to_ordinal(cutoff_date)
        return [
            p
            for p in train_papers
            if date_to_ordinal(get_paper_published_date(p)) <= cutoff_ord
        ]

    def generate(
        self,
        train_papers: list[PaperRecord],
        cutoff_month: str,
        top_k: int,
    ) -> list[IdeaPrediction]:
        cutoff_date = month_start_date(cutoff_month)
        train = self._train_at_cutoff(train_papers, cutoff_date)
        if not train:
            return []

        vocab = build_vocabulary(
            topic_id=_VOCAB_TOPIC_ID,
            cutoff_month=cutoff_month,
            cutoff_date=cutoff_date,
            train_papers=train,
            records=self.records,
            vectors=self.vectors,
            cfg=self.vocab_cfg,
        )

        state = _build_state(
            train,
            self.records,
            vocab,
            cutoff_date,
            self.config.state,
            self.config.canonicalize.min_count,
        )
        if state.coverage < self.config.realize.min_coverage_warn:
            print(
                f"[vocab_combinatorial WARNING] cutoff={cutoff_month}: only "
                f"{state.n_with_records}/{state.n_train} train papers have a "
                "concept record",
                file=sys.stderr,
                flush=True,
            )
        if not state.elements:
            print(
                f"[vocab_combinatorial WARNING] cutoff={cutoff_month}: no "
                "elements; window left empty",
                file=sys.stderr,
                flush=True,
            )
            return []

        seed = window_seed(cutoff_month, (p.paper_id for p in train), self.variant)
        rng = random.Random(seed)
        combos = sample_combos(
            state,
            self.variant,
            top_k,
            rng,
            self.config.sampler,
            self.config.state,
            self.config.canonicalize.min_count,
        )
        papers_by_id = {p.paper_id: p for p in train}
        combos = [
            Combo(
                elements=c.elements,
                move=c.move,
                sampler=c.sampler,
                score=c.score,
                components=c.components,
                evidence=retrieve_evidence(
                    c,
                    papers_by_id,
                    train,
                    n=self.config.realize.evidence_per_combo,
                    snippet_chars=self.config.realize.evidence_snippet_chars,
                ),
            )
            for c in combos
        ]
        realize_cfg = self.config.realize
        if self.temperature is not None:
            realize_cfg = type(realize_cfg)(
                prompt=realize_cfg.prompt,
                temperature=self.temperature,
                top_p=realize_cfg.top_p,
                evidence_per_combo=realize_cfg.evidence_per_combo,
                evidence_snippet_chars=realize_cfg.evidence_snippet_chars,
                fallback_template=realize_cfg.fallback_template,
                min_coverage_warn=realize_cfg.min_coverage_warn,
            )
        predictions = realize_combos(
            combos,
            state,
            caller=self._caller,
            prompt=self.realize_prompt,
            cfg=realize_cfg,
            cutoff_month=cutoff_month,
            horizon_months=_HORIZON_MONTHS,
            variant=self.variant,
            seed=seed,
        )
        predictions = _stamp_vocab_metadata(predictions, vocab)
        return predictions[:top_k]
