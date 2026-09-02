from __future__ import annotations

import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from idea_forecast_bench.combinatorial.cache import ElementCache
from idea_forecast_bench.combinatorial.config import (
    CombinatorialConfig,
    PromptPair,
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
from idea_forecast_bench.combinatorial.state import build_state
from idea_forecast_bench.combinatorial.types import Combo, ExtractionRecord
from idea_forecast_bench.models import IdeaPrediction, PaperRecord
from idea_forecast_bench.papers import (
    date_to_ordinal,
    get_paper_published_date,
    month_start_date,
)
from idea_forecast_bench.strategy.base import IdeaStrategy

_DEFAULT_MODEL = "gpt-4o-qwen35"
_HORIZON_MONTHS = 3


class CombinatorialStrategy(IdeaStrategy):
    """Ramon-Llull-style forecaster: elements -> community state -> sampled
    combinations -> LLM realisation. One instance is shared across topic
    threads, so everything loaded here is read-only after construction."""

    name = "combinatorial"

    def __init__(
        self,
        model_name: str | None = None,
        *,
        variant: str = VARIANT_FULL,
        element_cache_path: str | None = None,
        config_path: str | None = None,
        temperature: float | None = None,
        base_urls: Sequence[str] | None = None,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose one of {VARIANTS}")
        if not element_cache_path:
            raise ValueError(
                "combinatorial strategies need --element-cache (run "
                "`idea-forecast-bench extract-elements` first)."
            )
        self.model_name = model_name or _DEFAULT_MODEL
        self.variant = variant
        self.temperature = temperature
        self.config: CombinatorialConfig = load_combinatorial_config(config_path)
        self.cache = ElementCache.locate(Path(element_cache_path))
        self.records: Mapping[str, ExtractionRecord] = self.cache.load()
        if not self.records:
            raise ValueError(f"element cache {self.cache.directory} holds no records")
        self.vectors = VectorStore(
            self.cache.vectors_path(self.config.canonicalize.embed_model)
        ).view()
        if not self.vectors:
            print(
                f"[combinatorial WARNING] no element vectors under "
                f"{self.cache.directory}; near-synonyms will not be merged. "
                "Run `extract-elements --embed`.",
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

        state = build_state(
            train,
            self.records,
            cutoff_date,
            self.config.state,
            self.config.canonicalize,
            self.vectors,
        )
        if state.coverage < self.config.realize.min_coverage_warn:
            print(
                f"[combinatorial WARNING] cutoff={cutoff_month}: only "
                f"{state.n_with_records}/{state.n_train} train papers have an "
                "extraction record",
                file=sys.stderr,
                flush=True,
            )
        if not state.elements:
            print(
                f"[combinatorial WARNING] cutoff={cutoff_month}: no elements; "
                "window left empty",
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
        return predictions[:top_k]
