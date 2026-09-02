from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from idea_forecast_bench.combinatorial.canonicalize import normalize_text
from idea_forecast_bench.combinatorial.config import ExtractionConfig, PromptPair
from idea_forecast_bench.combinatorial.llm_caller import TextCaller
from idea_forecast_bench.combinatorial.types import (
    MOVES,
    RECORD_FAILED,
    RECORD_OK,
    UNKNOWN_MOVE,
    ElementType,
    ExtractionRecord,
)
from idea_forecast_bench.models import PaperRecord
from idea_forecast_bench.papers import get_paper_published_date
from idea_forecast_bench.similarity import _sanitize

FAKE_MODEL = "fake"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_STOPWORDS = frozenset(
    [
        "a", "an", "the", "of", "for", "to", "in", "on", "with", "via", "and",
        "or", "from", "by", "at", "as", "is", "are", "its", "their", "towards",
        "toward", "using", "based", "over", "under", "into",
    ]
)  # fmt: skip


@dataclass(frozen=True)
class ParsedExtraction:
    themes: tuple[str, ...]
    domains: tuple[str, ...]
    methods: tuple[str, ...]
    frames: tuple[str, ...]
    template: str
    move: str


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


def extract_json_object(raw: str) -> dict[str, object] | None:
    """Best-effort recovery of the single JSON object an LLM was asked for."""
    text = _THINK_RE.sub("", raw or "").strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return {str(k): v for k, v in payload.items()}
    return None


def _string_list(raw: object) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def _clean_elements(
    raw: object, limit: int, aliases: Mapping[str, str]
) -> tuple[str, ...]:
    seen: list[str] = []
    for item in _string_list(raw):
        value = normalize_text(item, aliases)
        if not value or len(value) > 60 or value in seen:
            continue
        seen.append(value)
        if len(seen) >= limit:
            break
    return tuple(seen)


def parse_extraction(
    raw: str,
    limits: Mapping[ElementType, int],
    aliases: Mapping[str, str],
) -> ParsedExtraction | None:
    payload = extract_json_object(raw)
    if payload is None:
        return None
    themes = _clean_elements(payload.get("A"), limits["theme"], aliases)
    domains = _clean_elements(payload.get("B"), limits["domain"], aliases)
    methods = _clean_elements(payload.get("C"), limits["method"], aliases)
    frames = _clean_elements(payload.get("Frame"), limits["frame"], aliases)
    if not themes or not methods:
        return None
    move = str(payload.get("move") or "").strip().lower()
    if move not in MOVES:
        move = UNKNOWN_MOVE
    template = str(payload.get("Template") or "").strip()[:200]
    return ParsedExtraction(
        themes=themes,
        domains=domains,
        methods=methods,
        frames=frames,
        template=template,
        move=move,
    )


def build_user_message(
    paper: PaperRecord, prompt: PromptPair, max_summary_chars: int
) -> str:
    abstract = _sanitize(paper.summary)[:max_summary_chars].replace("\n", " ")
    return prompt.user_template.format(
        title=_sanitize(paper.title).replace("\n", " "), abstract=abstract
    )


def _record(
    paper: PaperRecord,
    parsed: ParsedExtraction | None,
    *,
    model: str,
    fingerprint: str,
    error: str = "",
) -> ExtractionRecord:
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if parsed is None:
        return ExtractionRecord(
            paper_id=paper.paper_id,
            published_date=get_paper_published_date(paper),
            status=RECORD_FAILED,
            model=model,
            fingerprint=fingerprint,
            extracted_at=stamp,
            error=error[:300],
        )
    return ExtractionRecord(
        paper_id=paper.paper_id,
        published_date=get_paper_published_date(paper),
        status=RECORD_OK,
        themes=parsed.themes,
        domains=parsed.domains,
        methods=parsed.methods,
        frames=parsed.frames,
        template=parsed.template,
        move=parsed.move,
        model=model,
        fingerprint=fingerprint,
        extracted_at=stamp,
    )


def extract_paper(
    paper: PaperRecord,
    caller: TextCaller,
    prompt: PromptPair,
    cfg: ExtractionConfig,
    aliases: Mapping[str, str],
    fingerprint: str,
) -> ExtractionRecord:
    """One paper -> one record. Retries once at a higher temperature when the
    first answer cannot be parsed; a second failure is recorded, not raised,
    so a bad paper never stalls the run."""
    user = build_user_message(paper, prompt, cfg.max_summary_chars)
    last_error = ""
    temperatures = (cfg.temperature, cfg.temperature + cfg.retry_temperature_bump)
    for temperature in temperatures:
        try:
            raw = caller.complete(
                prompt.system_prompt, user, temperature=temperature, seed=0
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        parsed = parse_extraction(raw, cfg.max_elements_per_type, aliases)
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
) -> ExtractionRecord:
    """Deterministic stand-in for a dry run: elements are title bigrams and
    arXiv categories, the move is a hash of the paper id."""
    words = [
        w
        for w in normalize_text(paper.title, aliases).split(" ")
        if w and w not in _STOPWORDS
    ]
    bigrams = [" ".join(words[i : i + 2]) for i in range(0, max(len(words) - 1, 0), 2)]
    themes = tuple(bigrams[:2]) or tuple(words[:1]) or ("untitled",)
    methods = tuple(bigrams[2:4]) or tuple(words[-2:]) or ("method",)
    domains = tuple(normalize_text(k, aliases) for k in paper.keywords[:1] if k)
    digest = int(hashlib.sha1(paper.paper_id.encode()).hexdigest()[:8], 16)
    parsed = ParsedExtraction(
        themes=themes,
        domains=domains or ("general",),
        methods=methods,
        frames=(),
        template="A1 application of C1 to B1",
        move=MOVES[digest % len(MOVES)],
    )
    return _record(paper, parsed, model=FAKE_MODEL, fingerprint=fingerprint)
