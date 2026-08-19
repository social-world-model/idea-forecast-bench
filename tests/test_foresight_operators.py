"""Tests for the closed operator inventory + free-text mapping."""
from __future__ import annotations

from forecaster.foresight.operators import (
    CLOSED_OPERATORS,
    UNMAPPABLE_BUCKET,
    load_operator_inventory,
    map_free_text_operator,
    operator_distribution,
)


def test_inventory_loads_and_has_four_closed_ops():
    inv = load_operator_inventory()
    assert set(inv.closed_ids) == set(CLOSED_OPERATORS)
    assert len(inv.operators) == 4
    for op in inv.operators:
        assert op.one_line, f"operator {op.id} missing one-liner"
        assert op.example, f"operator {op.id} missing example"


def test_existing_free_text_operators_map_cleanly():
    """The 6 free-text verbs covered by config map to closed ids."""
    inv = load_operator_inventory()
    expectations = {
        "extend": "limitation_extension",
        "transfer": "cross_domain_transfer",
        "adapt": "cross_domain_transfer",
        "compose": "method_composition",
        "simplify": "method_composition",
        "benchmark": "benchmark_proposal",
    }
    for free_text, expected in expectations.items():
        assert map_free_text_operator(free_text, inv) == expected


def test_unmapped_free_text_falls_into_other():
    inv = load_operator_inventory()
    assert map_free_text_operator("analyze", inv) == UNMAPPABLE_BUCKET
    assert map_free_text_operator("scale", inv) == UNMAPPABLE_BUCKET
    assert map_free_text_operator("nonsense_verb", inv) == UNMAPPABLE_BUCKET
    assert map_free_text_operator("", inv) == UNMAPPABLE_BUCKET


def test_closed_ids_pass_through_unchanged():
    inv = load_operator_inventory()
    for op_id in CLOSED_OPERATORS:
        assert map_free_text_operator(op_id, inv) == op_id


def test_mapping_is_case_insensitive_and_trims():
    inv = load_operator_inventory()
    assert map_free_text_operator("  Extend ", inv) == "limitation_extension"
    assert map_free_text_operator("TRANSFER", inv) == "cross_domain_transfer"


def test_distribution_counts_other_bucket():
    inv = load_operator_inventory()
    counts = operator_distribution(
        ["extend", "extend", "transfer", "analyze", "scale", "nonsense"], inv
    )
    assert counts["limitation_extension"] == 2
    assert counts["cross_domain_transfer"] == 1
    assert counts[UNMAPPABLE_BUCKET] == 3
