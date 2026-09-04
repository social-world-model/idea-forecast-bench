"""Concept vocabulary experiment: extract specific object / mechanism /
problem terms per paper, cluster them into a two-level (parent -> concept)
vocabulary per topic and cutoff, tag background and emerging concepts, and
score the vocabulary offline. Independent of ``combinatorial``; the old
module is left untouched for comparison."""

from idea_forecast_bench.vocab.types import (
    SLOTS,
    Concept,
    ConceptRecord,
    Term,
    Vocabulary,
)

__all__ = ["SLOTS", "Concept", "ConceptRecord", "Term", "Vocabulary"]
