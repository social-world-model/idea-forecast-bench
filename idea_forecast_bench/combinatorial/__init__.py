"""Combinatorial (Ramon-Llull-style) forecaster.

Papers before the cutoff are decomposed into reusable elements (theme /
domain / method / frame); the community state at the cutoff is the decayed
frequency of each element plus their co-occurrence preferences; combinations
are sampled from that state and realised into ideas by an LLM.
"""

from idea_forecast_bench.combinatorial.cache import ElementCache
from idea_forecast_bench.combinatorial.config import (
    CombinatorialConfig,
    load_combinatorial_config,
)

__all__ = ["CombinatorialConfig", "ElementCache", "load_combinatorial_config"]
