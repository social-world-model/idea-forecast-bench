"""Build a concept vocabulary for one topic/cutoff from training papers, and
assign new records against a built vocabulary. Fine clusters (``Concept``s,
one per slot) and their coarse parent labels are both formed by greedy
leader clustering (the pattern in ``combinatorial.canonicalize.merge_elements``)
over per-text embedding vectors."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from os.path import commonprefix

import numpy as np
from numpy.typing import NDArray

from idea_forecast_bench.models import PaperRecord
from idea_forecast_bench.papers import date_to_ordinal, get_paper_published_date
from idea_forecast_bench.vocab.config import ClusterConfig, TagConfig, VocabConfig
from idea_forecast_bench.vocab.types import (
    SLOTS,
    Concept,
    ConceptRecord,
    Slot,
    Term,
    Vocabulary,
    concept_key,
    split_concept_key,
)

DAYS_PER_MONTH = 30.44

_LeaderIndex = tuple[list[str], "NDArray[np.float64] | None"]


def all_texts(records: Iterable[ConceptRecord]) -> set[str]:
    """Every term text and parent text that needs an embedding vector."""
    texts: set[str] = set()
    for record in records:
        for _slot, term in record.terms():
            if term.text:
                texts.add(term.text)
            if term.parent:
                texts.add(term.parent)
    return texts


def _unit(vec: Sequence[float]) -> NDArray[np.float64]:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


def _is_single_token(text: str) -> bool:
    return len(text.split()) <= 1


def _usable_records(
    train_papers: Sequence[PaperRecord],
    records: Mapping[str, ConceptRecord],
    cutoff_date: str,
) -> list[tuple[PaperRecord, ConceptRecord]]:
    """Train papers with an ``ok`` record; raises if any leaks the future."""
    cutoff_ord = date_to_ordinal(cutoff_date)
    usable: list[tuple[PaperRecord, ConceptRecord]] = []
    for paper in train_papers:
        published = get_paper_published_date(paper)
        if date_to_ordinal(published) > cutoff_ord:
            raise ValueError(
                f"paper {paper.paper_id} dated {published} is after cutoff {cutoff_date}"
            )
        record = records.get(paper.paper_id)
        if record is not None and record.ok:
            usable.append((paper, record))
    return usable


def _slot_vote(
    usable: Sequence[tuple[PaperRecord, ConceptRecord]],
) -> tuple[dict[str, Slot], dict[str, dict[Slot, int]]]:
    """Per text: occurrence count in each slot across usable records."""
    counts: dict[str, dict[Slot, int]] = defaultdict(lambda: dict.fromkeys(SLOTS, 0))
    for _paper, record in usable:
        for slot, term in record.terms():
            if term.text:
                counts[term.text][slot] += 1
    dominant = {
        text: max(SLOTS, key=lambda s: by_slot[s]) for text, by_slot in counts.items()
    }
    return dominant, counts


def _dominant_shares(
    dominant: Mapping[str, Slot], counts: Mapping[str, Mapping[Slot, int]]
) -> dict[str, float]:
    shares: dict[str, float] = {}
    for text, slot in dominant.items():
        total = sum(counts[text].values())
        shares[text] = counts[text][slot] / total if total else 0.0
    return shares


def _fine_cluster(
    keys_by_slot: Mapping[Slot, Sequence[str]],
    doc_count: Mapping[str, int],
    vectors: Mapping[str, Sequence[float]],
    cfg: ClusterConfig,
) -> dict[str, str]:
    """Greedy leader clustering of remapped ``slot:text`` keys per slot, in
    ``(-doc_count, key)`` order; single-token pairs need
    ``single_token_threshold`` unless they share a long prefix."""
    leader_of: dict[str, str] = {}
    for slot, texts in keys_by_slot.items():
        ordered = sorted(texts, key=lambda t: (-doc_count[t], concept_key(slot, t)))
        leaders: list[str] = []
        matrix: NDArray[np.float64] | None = None
        for text in ordered:
            key = concept_key(slot, text)
            vec = vectors.get(text)
            if not vec:
                leader_of[key] = key
                continue
            unit = _unit(vec)
            joined = False
            if matrix is not None and matrix.shape[0] > 0:
                sims = matrix @ unit
                best = int(np.argmax(sims))
                leader_text = leaders[best]
                required = cfg.fine_threshold
                needs_prefix = _is_single_token(text) or _is_single_token(leader_text)
                prefix_len = len(commonprefix([text, leader_text]))
                if needs_prefix and prefix_len < cfg.shared_prefix_chars:
                    required = cfg.single_token_threshold
                if float(sims[best]) >= required:
                    leader_of[key] = concept_key(slot, leader_text)
                    joined = True
            if not joined:
                leader_of[key] = key
                leaders.append(text)
                row = unit[np.newaxis, :]
                matrix = row if matrix is None else np.vstack([matrix, row])
    return leader_of


def _concept_members(leader_of: Mapping[str, str]) -> dict[str, list[str]]:
    """Leader key -> member texts (the raw text half of each member key)."""
    members: dict[str, list[str]] = defaultdict(list)
    for member_key, leader_key in leader_of.items():
        members[leader_key].append(split_concept_key(member_key)[1])
    return members


def _parent_counts(
    usable: Sequence[tuple[PaperRecord, ConceptRecord]],
) -> dict[str, Counter[str]]:
    """Text -> Counter of non-empty parent strings, any original slot."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for _paper, record in usable:
        for _slot, term in record.terms():
            if term.text and term.parent:
                counts[term.text][term.parent] += 1
    return counts


def _combined_parent_counts(
    members_by_leader: Mapping[str, Sequence[str]],
    parent_counts: Mapping[str, Counter[str]],
) -> dict[str, Counter[str]]:
    combined: dict[str, Counter[str]] = {}
    for leader_key, texts in members_by_leader.items():
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(parent_counts.get(text, Counter()))
        if counter:
            combined[leader_key] = counter
    return combined


def _concept_parent_raw(combined: Mapping[str, Counter[str]]) -> dict[str, str]:
    """Majority-vote raw parent label per concept, before merging labels."""
    return {
        leader_key: sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for leader_key, counter in combined.items()
    }


def _merge_parents(
    combined: Mapping[str, Counter[str]],
    raw_parent: Mapping[str, str],
    vectors: Mapping[str, Sequence[float]],
    threshold: float,
) -> dict[str, str]:
    """Greedy leader clustering of the raw parent labels among themselves."""
    weight: dict[str, int] = defaultdict(int)
    for leader_key, label in raw_parent.items():
        weight[label] += combined[leader_key][label]
    ordered = sorted(weight, key=lambda label: (-weight[label], label))
    leader_of_label: dict[str, str] = {}
    leaders: list[str] = []
    matrix: NDArray[np.float64] | None = None
    for label in ordered:
        vec = vectors.get(label)
        if not vec:
            leader_of_label[label] = label
            continue
        unit = _unit(vec)
        if matrix is not None and matrix.shape[0] > 0:
            sims = matrix @ unit
            best = int(np.argmax(sims))
            if float(sims[best]) >= threshold:
                leader_of_label[label] = leaders[best]
                continue
        leader_of_label[label] = label
        leaders.append(label)
        row = unit[np.newaxis, :]
        matrix = row if matrix is None else np.vstack([matrix, row])
    return leader_of_label


def _final_parent(
    leader_key: str,
    label: str,
    raw_parent: Mapping[str, str],
    leader_of_label: Mapping[str, str],
) -> str:
    raw = raw_parent.get(leader_key)
    if raw is None:
        return label
    return leader_of_label.get(raw, raw)


def _fold_weak_clusters(
    members_by_leader: Mapping[str, Sequence[str]],
    raw_parent: Mapping[str, str],
    leader_of_label: Mapping[str, str],
    paper_ids_by_text: Mapping[str, set[str]],
    promote_min_count: int,
) -> tuple[dict[str, list[str]], set[str]]:
    """The hybrid level: fold every fine cluster with fewer than
    ``promote_min_count`` training papers into one concept per ``(slot,
    merged parent label)``; clusters at or above the threshold pass through
    untouched. A fold target that lands on an existing strong cluster's own
    id (that cluster's own label already equals the parent label) absorbs
    the weak siblings instead of duplicating the id. Returns the folded
    ``members_by_leader`` and the set of fold-target leader keys, so the
    caller can give just those concepts a top-level ``parent`` and a
    recomputed ``slot_share``."""
    if promote_min_count <= 0:
        return {key: list(texts) for key, texts in members_by_leader.items()}, set()

    def paper_count(texts: Sequence[str]) -> int:
        ids: set[str] = set()
        for text in texts:
            ids.update(paper_ids_by_text.get(text, ()))
        return len(ids)

    strong_keys = {
        key
        for key, texts in members_by_leader.items()
        if paper_count(texts) >= promote_min_count
    }
    folded: dict[str, list[str]] = {
        key: list(members_by_leader[key]) for key in strong_keys
    }

    weak_groups: dict[str, list[str]] = defaultdict(list)
    for key, texts in members_by_leader.items():
        if key in strong_keys:
            continue
        slot, label = split_concept_key(key)
        parent_label = _final_parent(key, label, raw_parent, leader_of_label)
        weak_groups[concept_key(slot, parent_label)].extend(texts)

    fold_targets = set(weak_groups)
    for target_key, texts in weak_groups.items():
        merged = list(folded.get(target_key, ()))
        seen = set(merged)
        for text in texts:
            if text not in seen:
                seen.add(text)
                merged.append(text)
        folded[target_key] = merged

    return folded, fold_targets


def _fold_slot_share(
    fold_targets: Iterable[str],
    members_by_leader: Mapping[str, Sequence[str]],
    dominant_shares: Mapping[str, float],
    doc_count: Mapping[str, int],
) -> dict[str, float]:
    """For each fold-target concept, the mean of its member texts' own
    dominant-slot share, weighted by each text's total occurrence count."""
    shares: dict[str, float] = {}
    for key in fold_targets:
        texts = members_by_leader[key]
        weight = sum(doc_count.get(text, 0) for text in texts)
        if weight <= 0:
            shares[key] = 0.0
            continue
        weighted = sum(
            doc_count.get(text, 0) * dominant_shares.get(text, 0.0) for text in texts
        )
        shares[key] = weighted / weight
    return shares


def _paper_ids_by_text(
    usable: Sequence[tuple[PaperRecord, ConceptRecord]],
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for paper, record in usable:
        seen = {term.text for _slot, term in record.terms() if term.text}
        for text in seen:
            out[text].add(paper.paper_id)
    return out


def _assemble_concepts(
    members_by_leader: Mapping[str, Sequence[str]],
    raw_parent: Mapping[str, str],
    leader_of_label: Mapping[str, str],
    dominant_shares: Mapping[str, float],
    paper_ids_by_text: Mapping[str, set[str]],
    paper_date: Mapping[str, str],
    slot_counts: Mapping[str, Mapping[Slot, int]],
    cutoff_ord: int,
    n_with_records: int,
    cfg: VocabConfig,
    parent_override: Mapping[str, str],
    slot_share_override: Mapping[str, float],
) -> tuple[dict[str, Concept], dict[str, str]]:
    concepts: dict[str, Concept] = {}
    member_of: dict[str, str] = {}
    window_days = cfg.tag.emerging_months * DAYS_PER_MONTH
    for leader_key, member_texts in members_by_leader.items():
        slot, label = split_concept_key(leader_key)
        paper_ids: set[str] = set()
        for text in member_texts:
            paper_ids.update(paper_ids_by_text.get(text, ()))
        sorted_ids = tuple(sorted(paper_ids))
        count = len(sorted_ids)
        doc_frac = count / n_with_records if n_with_records else 0.0
        first_seen = min((paper_date[pid] for pid in sorted_ids), default="")
        recent_count = sum(
            1
            for pid in sorted_ids
            if (cutoff_ord - date_to_ordinal(paper_date[pid])) <= window_days
        )
        background = doc_frac >= cfg.tag.background_doc_frac
        in_window = (
            bool(first_seen)
            and (cutoff_ord - date_to_ordinal(first_seen)) <= window_days
        )
        emerging = in_window and count >= cfg.tag.emerging_min_count and not background
        if not (background or count >= cfg.tag.min_count or emerging):
            continue
        if leader_key in parent_override:
            parent = parent_override[leader_key]
        else:
            parent = _final_parent(leader_key, label, raw_parent, leader_of_label)
        slot_share = slot_share_override.get(
            leader_key, dominant_shares.get(label, 0.0)
        )
        concept = Concept(
            id=leader_key,
            slot=slot,
            label=label,
            parent=parent,
            variants=tuple(sorted(member_texts)),
            paper_ids=sorted_ids,
            count=count,
            doc_frac=doc_frac,
            first_seen=first_seen,
            recent_count=recent_count,
            background=background,
            emerging=emerging,
            slot_share=slot_share,
        )
        concepts[concept.id] = concept
        for text in member_texts:
            for member_slot in SLOTS:
                if slot_counts[text][member_slot] > 0:
                    member_of[concept_key(member_slot, text)] = concept.id
    return concepts, member_of


def _extras(
    slot_counts: Mapping[str, Mapping[Slot, int]],
    leader_of: Mapping[str, str],
    concepts: Mapping[str, Concept],
    slot_conflicts: Sequence[str],
) -> dict[str, float]:
    occ_total: dict[Slot, int] = dict.fromkeys(SLOTS, 0)
    occ_single: dict[Slot, int] = dict.fromkeys(SLOTS, 0)
    for text, by_slot in slot_counts.items():
        single = _is_single_token(text)
        for slot in SLOTS:
            n = by_slot[slot]
            occ_total[slot] += n
            if single:
                occ_single[slot] += n
    n_raw_keys = sum(
        1 for by_slot in slot_counts.values() for n in by_slot.values() if n > 0
    )
    extras: dict[str, float] = {
        "n_raw_keys": float(n_raw_keys),
        "n_fine_clusters_all": float(len(set(leader_of.values()))),
        "n_kept": float(len(concepts)),
        "n_slot_conflicts": float(len(slot_conflicts)),
    }
    for slot in SLOTS:
        extras[f"single_token_share_{slot}"] = (
            occ_single[slot] / occ_total[slot] if occ_total[slot] else 0.0
        )
    return extras


def build_vocabulary(
    *,
    topic_id: str,
    cutoff_month: str,
    cutoff_date: str,
    train_papers: Sequence[PaperRecord],
    records: Mapping[str, ConceptRecord],
    vectors: Mapping[str, Sequence[float]],
    cfg: VocabConfig,
) -> Vocabulary:
    usable = _usable_records(train_papers, records, cutoff_date)
    cutoff_ord = date_to_ordinal(cutoff_date)

    dominant, slot_counts = _slot_vote(usable)
    dominant_shares = _dominant_shares(dominant, slot_counts)
    slot_conflicts = tuple(
        sorted(
            t for t, s in dominant_shares.items() if s < cfg.cluster.slot_majority_min
        )
    )
    doc_count = {t: sum(c.values()) for t, c in slot_counts.items()}
    keys_by_slot: dict[Slot, list[str]] = {s: [] for s in SLOTS}
    for text, slot in dominant.items():
        keys_by_slot[slot].append(text)
    leader_of = _fine_cluster(keys_by_slot, doc_count, vectors, cfg.cluster)
    members_by_leader = _concept_members(leader_of)

    parent_counts = _parent_counts(usable)
    combined_parents = _combined_parent_counts(members_by_leader, parent_counts)
    raw_parent = _concept_parent_raw(combined_parents)
    leader_of_label = _merge_parents(
        combined_parents, raw_parent, vectors, cfg.cluster.parent_threshold
    )

    paper_ids_by_text = _paper_ids_by_text(usable)
    paper_date = {
        paper.paper_id: get_paper_published_date(paper) for paper, _ in usable
    }

    folded_members, fold_targets = _fold_weak_clusters(
        members_by_leader,
        raw_parent,
        leader_of_label,
        paper_ids_by_text,
        cfg.tag.promote_min_count,
    )
    parent_override = {key: split_concept_key(key)[1] for key in fold_targets}
    slot_share_override = _fold_slot_share(
        fold_targets, folded_members, dominant_shares, doc_count
    )

    concepts, member_of = _assemble_concepts(
        folded_members,
        raw_parent,
        leader_of_label,
        dominant_shares,
        paper_ids_by_text,
        paper_date,
        slot_counts,
        cutoff_ord,
        len(usable),
        cfg,
        parent_override,
        slot_share_override,
    )

    return Vocabulary(
        topic_id=topic_id,
        cutoff_month=cutoff_month,
        cutoff_date=cutoff_date,
        n_train=len(train_papers),
        n_with_records=len(usable),
        concepts=concepts,
        member_of=member_of,
        slot_conflicts=slot_conflicts,
        config_sha=cfg.sha,
        extras=_extras(slot_counts, leader_of, concepts, slot_conflicts),
    )


def _leader_index_by_slot(
    vocab: Vocabulary, vectors: Mapping[str, Sequence[float]]
) -> dict[Slot, _LeaderIndex]:
    by_slot: dict[Slot, _LeaderIndex] = {}
    for slot in SLOTS:
        ids: list[str] = []
        rows: list[NDArray[np.float64]] = []
        for concept in vocab.concepts.values():
            if concept.slot != slot:
                continue
            vec = vectors.get(concept.label)
            if not vec:
                continue
            ids.append(concept.id)
            rows.append(_unit(vec))
        by_slot[slot] = (ids, np.vstack(rows) if rows else None)
    return by_slot


def _assign_term(
    text: str,
    slot: Slot,
    vocab: Vocabulary,
    vectors: Mapping[str, Sequence[float]],
    leaders: _LeaderIndex,
    threshold: float,
) -> str | None:
    if not text:
        return None
    exact = vocab.member_of.get(concept_key(slot, text))
    if exact is not None:
        return exact
    for other in SLOTS:
        if other == slot:
            continue
        exact = vocab.member_of.get(concept_key(other, text))
        if exact is not None:
            return exact
    vec = vectors.get(text)
    ids, matrix = leaders
    if not vec or matrix is None:
        return None
    sims = matrix @ _unit(vec)
    best = int(np.argmax(sims))
    return ids[best] if float(sims[best]) >= threshold else None


def assign_record(
    record: ConceptRecord,
    vocab: Vocabulary,
    vectors: Mapping[str, Sequence[float]],
    threshold: float,
) -> dict[Slot, list[str | None]]:
    """For each slot, one entry per term of the record: the concept id it
    maps to, or None. Exact match first, else the nearest concept leader
    vector of the same slot if cosine >= threshold."""
    slot_terms: dict[Slot, tuple[Term, ...]] = {
        "object": record.objects,
        "mechanism": record.mechanisms,
        "problem": record.problems,
    }
    leader_index = _leader_index_by_slot(vocab, vectors)

    def assign(term: Term, slot: Slot) -> str | None:
        # The term itself first; if the specific phrasing is unknown, fall
        # back to the parent the extractor named for it. Both come from the
        # same extraction, so no future information enters the vocabulary.
        hit = _assign_term(
            term.text, slot, vocab, vectors, leader_index[slot], threshold
        )
        if hit is not None or not term.parent:
            return hit
        return _assign_term(
            term.parent, slot, vocab, vectors, leader_index[slot], threshold
        )

    return {slot: [assign(term, slot) for term in slot_terms[slot]] for slot in SLOTS}


_Row = tuple[str, str, tuple[Term, ...], tuple[Term, ...], tuple[Term, ...]]


def _toy(row: _Row) -> tuple[PaperRecord, ConceptRecord]:
    pid, date, obj, mech, prob = row
    paper = PaperRecord(
        paper_id=pid,
        title=pid,
        month=date[:7],
        summary="",
        keywords=[],
        source_path="",
        published_date=date,
    )
    rec = ConceptRecord(
        paper_id=pid,
        published_date=date,
        status="ok",
        objects=obj,
        mechanisms=mech,
        problems=prob,
    )
    return paper, rec


def selfcheck() -> None:  # pragma: no cover - manual invocation only
    """Hand-made unit vectors on a toy 6-paper corpus; not run on import."""
    cfg = VocabConfig()
    cutoff = "2025-06-30"
    t = Term

    # "cat"/"dog" (single-token) and "cat x"/"dog y" (multiword) are each a
    # cosine-0.92 pair; "rare" is mostly a mechanism but sometimes an object;
    # "common" is a mechanism in most papers; "new term" appears once, late.
    rows: list[_Row] = [
        ("p1", "2025-01-10", (t("cat", ""),), (t("common", ""), t("rare", "")), ()),
        ("p2", "2025-01-20", (t("cat", ""),), (t("common", ""), t("rare", "")), ()),
        ("p3", "2025-02-10", (t("dog", ""),), (t("common", ""), t("rare", "")), ()),
        ("p4", "2025-02-20", (t("dog", ""), t("rare", "")), (t("common", ""),), ()),
        ("p5", "2025-03-05", (t("cat x", ""), t("dog y", "")), (), ()),
        (
            "p6",
            "2025-06-20",
            (t("cat x", ""), t("dog y", "")),
            (),
            (t("new term", ""),),
        ),
    ]
    built = [_toy(row) for row in rows]
    papers = [p for p, _r in built]
    recs = {p.paper_id: r for p, r in built}

    # Leak guard: a train paper dated after the cutoff must raise.
    future_paper, future_rec = _toy(
        ("p7", "2025-07-15", (t("x", ""),), (t("y", ""),), ())
    )
    try:
        build_vocabulary(
            topic_id="toy",
            cutoff_month="2025-06",
            cutoff_date=cutoff,
            train_papers=[future_paper],
            records={"p7": future_rec},
            vectors={},
            cfg=cfg,
        )
        raise AssertionError("expected ValueError for a leaking train paper")
    except ValueError:
        pass

    off = 0.3919183588453085
    vectors: dict[str, Sequence[float]] = {
        "cat": (1.0, 0.0, 0.0, 0.0, 0.0),
        "dog": (0.92, off, 0.0, 0.0, 0.0),
        "cat x": (0.0, 0.0, 1.0, 0.0, 0.0),
        "dog y": (0.0, 0.0, 0.92, off, 0.0),
        "far term": (0.0, 0.0, 0.0, 0.0, 1.0),
        "common": (1.0, 0.0),
        "rare": (0.0, 1.0),
    }
    vocab = build_vocabulary(
        topic_id="toy",
        cutoff_month="2025-06",
        cutoff_date=cutoff,
        train_papers=papers,
        records=recs,
        vectors=vectors,
        cfg=cfg,
    )
    assert vocab.n_train == 6
    assert vocab.n_with_records == 6

    # Single-token pair at cosine 0.92 does not merge; multiword pair does.
    assert "object:cat" in vocab.concepts
    assert "object:dog" in vocab.concepts
    assert "object:cat x" in vocab.concepts
    assert "object:dog y" not in vocab.concepts
    assert "dog y" in vocab.concepts["object:cat x"].variants

    # Slot majority remap yields one concept per text.
    rare_id = vocab.member_of["mechanism:rare"]
    assert vocab.member_of["object:rare"] == rare_id
    assert vocab.concepts[rare_id].slot == "mechanism"
    assert vocab.concepts[rare_id].count == 4

    # Background flag when doc_frac >= threshold.
    common = vocab.concepts["mechanism:common"]
    assert common.background is True
    assert common.doc_frac == 4 / 6

    # Emerging flag for a term first seen in the last month.
    new_term = vocab.concepts["problem:new term"]
    assert new_term.emerging is True
    assert new_term.background is False

    # assign_record maps exact text and rejects a far vector.
    query = _toy(
        ("q1", cutoff, (t("cat", ""), t("far term", "")), (t("common", ""),), ())
    )[1]
    assigned = assign_record(query, vocab, vectors, threshold=0.90)
    assert assigned["object"][0] == "object:cat"
    assert assigned["object"][1] is None
    assert assigned["mechanism"][0] == "mechanism:common"

    # Hybrid level (tag.promote_min_count). "widget" is a well-attested
    # object (3 papers, no parent of its own -- self-parented); "gadget" is
    # a 1-paper object whose parent is "widget". Neither has an embedding
    # vector, so fine clustering and parent merging can never accidentally
    # join them -- any folding below is purely the promote_min_count path.
    hybrid_rows: list[_Row] = [
        ("h1", "2025-01-10", (t("widget", ""),), (), ()),
        ("h2", "2025-01-20", (t("widget", ""),), (), ()),
        ("h3", "2025-02-01", (t("widget", ""),), (), ()),
        ("h4", "2025-06-20", (t("gadget", "widget"),), (), ()),
    ]
    hybrid_built = [_toy(row) for row in hybrid_rows]
    hybrid_papers = [p for p, _r in hybrid_built]
    hybrid_recs = {p.paper_id: r for p, r in hybrid_built}

    # promote_min_count=0 (default): folding is off, both concepts stand.
    vocab_off = build_vocabulary(
        topic_id="hybrid",
        cutoff_month="2025-06",
        cutoff_date=cutoff,
        train_papers=hybrid_papers,
        records=hybrid_recs,
        vectors={},
        cfg=VocabConfig(),
    )
    assert vocab_off.concepts["object:widget"].count == 3
    assert vocab_off.concepts["object:widget"].parent == "widget"
    assert vocab_off.concepts["object:gadget"].count == 1
    assert vocab_off.concepts["object:gadget"].parent == "widget"
    assert vocab_off.member_of[concept_key("object", "gadget")] == "object:gadget"

    # promote_min_count=3: "gadget" (1 paper < 3) folds into "widget" (3
    # papers >= 3, already the parent label) instead of staying its own
    # node; "widget"'s count grows to include the folded paper.
    vocab_on = build_vocabulary(
        topic_id="hybrid",
        cutoff_month="2025-06",
        cutoff_date=cutoff,
        train_papers=hybrid_papers,
        records=hybrid_recs,
        vectors={},
        cfg=VocabConfig(tag=TagConfig(promote_min_count=3)),
    )
    assert "object:gadget" not in vocab_on.concepts
    widget_concept = vocab_on.concepts["object:widget"]
    assert widget_concept.count == 4
    assert widget_concept.parent == "widget"
    assert "gadget" in widget_concept.variants
    assert vocab_on.member_of[concept_key("object", "gadget")] == "object:widget"

    print("selfcheck OK")
