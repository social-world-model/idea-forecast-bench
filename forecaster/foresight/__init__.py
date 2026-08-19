"""Foresight: future-grounded conditioning + reward utilities.

This package is the Phase 1+ home for the new conditioning pipeline that
sits between the existing hindsight extractor and the GRPO trainer:

  * operators  — closed 4-class operator inventory + free-text mapping
  * memory     — build_memory(papers_before_t) -> str
  * cutoffs    — window boundary constants + leakage assertions
  * indices    — per-cutoff future/history vector indices
  * dz         — D_z augmentation (memory_text + operator_closed + source_future_id)

The existing GRPO loop is untouched until Phase 4. Importing this package
must not introduce side effects on the live trainer.
"""

from forecaster.foresight.cutoffs import (
    FUTURE_WINDOW_HARD_LIMIT,
    TEST_CUTOFF_MIN,
    TRAIN_CUTOFF_MAX,
    assert_no_test_window_leakage,
    assert_train_test_disjoint,
)
from forecaster.foresight.memory import build_memory
from forecaster.foresight.operators import (
    OperatorInventory,
    load_operator_inventory,
    map_free_text_operator,
)

__all__ = [
    "OperatorInventory",
    "load_operator_inventory",
    "map_free_text_operator",
    "build_memory",
    "TRAIN_CUTOFF_MAX",
    "TEST_CUTOFF_MIN",
    "FUTURE_WINDOW_HARD_LIMIT",
    "assert_train_test_disjoint",
    "assert_no_test_window_leakage",
]
