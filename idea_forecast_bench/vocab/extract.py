"""Per-paper concept extraction: one LLM call -> one ConceptRecord."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from idea_forecast_bench.combinatorial.canonicalize import normalize_text
from idea_forecast_bench.combinatorial.config import PromptPair
from idea_forecast_bench.combinatorial.extraction import (
    extract_json_object,
    is_transient_error,
)
from idea_forecast_bench.combinatorial.llm_caller import TextCaller
from idea_forecast_bench.models import PaperRecord
from idea_forecast_bench.papers import get_paper_published_date
from idea_forecast_bench.similarity import _sanitize
from idea_forecast_bench.vocab.config import ExtractionConfig
from idea_forecast_bench.vocab.types import (
    RECORD_FAILED,
    RECORD_OK,
    SLOTS,
    ConceptRecord,
    Slot,
    Term,
)

FAKE_MODEL = "fake"
_SLOT_FIELD: dict[Slot, str] = {
    "object": "objects",
    "mechanism": "mechanisms",
    "problem": "problems",
}
#: Bare evaluative words the prompt forbids; a term that is exactly one of
#: these is dropped rather than cached as a concept.
_BANNED_SINGLE = frozenset(
    [
        "efficient",
        "efficiency",
        "adaptive",
        "scalable",
        "scalability",
        "robust",
        "robustness",
        "novel",
        "general",
        "generalization",
        "fast",
        "accurate",
        "accuracy",
        "lightweight",
        "simple",
        "effective",
        "performance",
        "improvement",
    ]
)
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "for",
        "to",
        "in",
        "on",
        "with",
        "via",
        "and",
        "or",
        "from",
        "by",
        "at",
        "as",
        "is",
        "are",
        "its",
        "their",
        "towards",
        "toward",
        "using",
        "based",
        "over",
        "under",
        "into",
    ]
)
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")


def extraction_fingerprint(
    prompt: PromptPair, model: str, schema_version: int, temperature: float
) -> str:
    h = hashlib.sha256()
    for part in (
        prompt.system_prompt,
        prompt.user_template,
        model,
        str(schema_version),
        repr(temperature),
    ):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def build_user_message(paper: PaperRecord, prompt: PromptPair, max_chars: int) -> str:
    abstract = _sanitize(paper.summary or "")[:max_chars]
    return prompt.user_template.format(
        title=_sanitize(paper.title or ""), abstract=abstract
    )


def _clean_term(raw: object, aliases: Mapping[str, str]) -> str:
    text = normalize_text(str(raw or ""), aliases)
    tokens = _WORD_RE.findall(text)
    if not tokens or len(tokens) > 7:
        return ""
    if len(tokens) == 1 and tokens[0] in _BANNED_SINGLE:
        return ""
    return " ".join(tokens)


def _parse_items(raw: object, cap: int, aliases: Mapping[str, str]) -> tuple[Term, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[Term] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, dict):
            text = _clean_term(item.get("term"), aliases)
            parent = _clean_term(item.get("parent"), aliases)
        else:
            text, parent = _clean_term(item, aliases), ""
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(Term(text=text, parent=parent))
        if len(out) >= cap:
            break
    return tuple(out)


def parse_extraction(
    raw_text: str, caps: Mapping[str, int], aliases: Mapping[str, str]
) -> dict[str, tuple[Term, ...]] | None:
    payload = extract_json_object(raw_text)
    if not isinstance(payload, dict):
        return None
    parsed = {
        _SLOT_FIELD[slot]: _parse_items(payload.get(slot), caps.get(slot, 3), aliases)
        for slot in SLOTS
    }
    # A record without an object or a mechanism cannot form a title-level
    # idea, so it is treated as unparseable rather than half-cached.
    if not parsed["objects"] or not parsed["mechanisms"]:
        return None
    return parsed


def _record(
    paper: PaperRecord,
    parsed: Mapping[str, tuple[Term, ...]] | None,
    *,
    model: str,
    fingerprint: str,
    error: str = "",
) -> ConceptRecord:
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    published = get_paper_published_date(paper)
    if parsed is None:
        return ConceptRecord(
            paper_id=paper.paper_id,
            published_date=published,
            status=RECORD_FAILED,
            model=model,
            fingerprint=fingerprint,
            extracted_at=stamp,
            error=error[:300],
        )
    return ConceptRecord(
        paper_id=paper.paper_id,
        published_date=published,
        status=RECORD_OK,
        objects=parsed["objects"],
        mechanisms=parsed["mechanisms"],
        problems=parsed["problems"],
        model=model,
        fingerprint=fingerprint,
        extracted_at=stamp,
    )


def extract_paper(
    paper: PaperRecord,
    caller: TextCaller,
    prompt: PromptPair,
    cfg: ExtractionConfig,
    fingerprint: str,
) -> ConceptRecord | None:
    """One paper -> one record, or None when the service was busy (the paper
    is left uncached so a later run picks it up). An answer that cannot be
    parsed twice IS cached as failed: retrying costs money and rarely helps."""
    user = build_user_message(paper, prompt, cfg.max_summary_chars)
    last_error = ""
    temperatures = (cfg.temperature, cfg.temperature + cfg.retry_temperature_bump)
    for temperature in temperatures:
        try:
            raw = caller.complete(
                prompt.system_prompt, user, temperature=temperature, seed=0
            )
        except Exception as exc:  # noqa: BLE001 - classified, not swallowed
            if is_transient_error(exc):
                return None
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        parsed = parse_extraction(raw, cfg.max_terms_per_slot, cfg.aliases)
        if parsed is not None:
            return _record(
                paper, parsed, model=caller.model_name, fingerprint=fingerprint
            )
        last_error = f"unparseable: {raw[:120]!r}"
    return _record(
        paper, None, model=caller.model_name, fingerprint=fingerprint, error=last_error
    )


def fake_extraction(
    paper: PaperRecord, aliases: Mapping[str, str], fingerprint: str
) -> ConceptRecord:
    """Deterministic stand-in for a dry run: title bigrams as terms, the
    first title word as parent. Never use its output for reported numbers."""
    words = [
        w
        for w in normalize_text(paper.title, aliases).split(" ")
        if w and w not in _STOPWORDS
    ]
    bigrams = [" ".join(words[i : i + 2]) for i in range(0, max(len(words) - 1, 0), 2)]
    parent = words[0] if words else "untitled"

    def terms(chunk: list[str]) -> tuple[Term, ...]:
        return tuple(Term(text=t, parent=parent) for t in chunk if t)

    parsed: dict[str, tuple[Term, ...]] = {
        "objects": terms(bigrams[:1]) or (Term(text="untitled object", parent=parent),),
        "mechanisms": terms(bigrams[1:3])
        or (Term(text="untitled mechanism", parent=parent),),
        "problems": terms(bigrams[3:4]),
    }
    return _record(paper, parsed, model=FAKE_MODEL, fingerprint=fingerprint)


def record_to_dict(record: ConceptRecord) -> dict[str, Any]:
    def dump(items: tuple[Term, ...]) -> list[dict[str, str]]:
        return [{"term": t.text, "parent": t.parent} for t in items]

    return {
        "paper_id": record.paper_id,
        "published_date": record.published_date,
        "status": record.status,
        "objects": dump(record.objects),
        "mechanisms": dump(record.mechanisms),
        "problems": dump(record.problems),
        "model": record.model,
        "fingerprint": record.fingerprint,
        "extracted_at": record.extracted_at,
        "error": record.error,
    }


def record_from_dict(raw: Mapping[str, Any]) -> ConceptRecord | None:
    paper_id = str(raw.get("paper_id") or "").strip()
    if not paper_id:
        return None

    def load(key: str) -> tuple[Term, ...]:
        items = raw.get(key)
        if not isinstance(items, list):
            return ()
        out: list[Term] = []
        for item in items:
            if isinstance(item, dict) and str(item.get("term", "")).strip():
                out.append(
                    Term(
                        text=str(item.get("term", "")).strip(),
                        parent=str(item.get("parent", "")).strip(),
                    )
                )
        return tuple(out)

    status = str(raw.get("status") or RECORD_FAILED)
    return ConceptRecord(
        paper_id=paper_id,
        published_date=str(raw.get("published_date") or ""),
        status=RECORD_OK if status == RECORD_OK else RECORD_FAILED,
        objects=load("objects"),
        mechanisms=load("mechanisms"),
        problems=load("problems"),
        model=str(raw.get("model") or ""),
        fingerprint=str(raw.get("fingerprint") or ""),
        extracted_at=str(raw.get("extracted_at") or ""),
        error=str(raw.get("error") or ""),
    )


def dumps_record(record: ConceptRecord) -> str:
    return json.dumps(record_to_dict(record), ensure_ascii=False)
