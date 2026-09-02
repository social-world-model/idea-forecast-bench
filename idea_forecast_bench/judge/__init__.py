"""Retrieve-then-judge scoring: the protocol the benchmark reports against.

This used to live entirely inside examples/benchmark/llm_judge_eval.py, a
1,207-line entry script. The protocol is not an example -- it is what a
IdeaForecastBench number means -- so it lives in the package, and the script is the
CLI over it.
"""

from idea_forecast_bench.judge.topics import process_topic
from idea_forecast_bench.judge.windows import process_window

__all__ = ["process_topic", "process_window"]
