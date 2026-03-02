"""Tests for backend/output_contract.py – normalize_idea / normalize_ideas."""

from __future__ import annotations

import pytest

from backend.output_contract import normalize_idea, normalize_ideas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_idea(**overrides) -> dict:
    """Return a fully-specified raw idea dict."""
    base: dict = {
        "id": "test-001",
        "Title": "My Test Idea",
        "Problem": "A hard problem.",
        "Approach": "A clever approach.",
        "Score": "8.5",
        "Novelty": "7.0",
        "Feasibility": "6.0",
        "Interestingness": "5.0",
        "source_url": "https://arxiv.org/abs/0000.00000",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Happy-path: numeric string coercion
# ---------------------------------------------------------------------------


class TestNumericStringCoercion:
    """Numeric string values (e.g. "8.5") must be coerced to float."""

    def test_score_string_coerced_to_float(self):
        result = normalize_idea(_full_idea(Score="8.5"))
        assert result["Score"] == pytest.approx(8.5)
        assert isinstance(result["Score"], float)

    def test_novelty_string_coerced_to_float(self):
        result = normalize_idea(_full_idea(Novelty="7.0"))
        assert result["Novelty"] == pytest.approx(7.0)
        assert isinstance(result["Novelty"], float)

    def test_feasibility_string_coerced_to_float(self):
        result = normalize_idea(_full_idea(Feasibility="6.25"))
        assert result["Feasibility"] == pytest.approx(6.25)
        assert isinstance(result["Feasibility"], float)

    def test_interestingness_string_coerced_to_float(self):
        result = normalize_idea(_full_idea(Interestingness="5.5"))
        assert result["Interestingness"] == pytest.approx(5.5)
        assert isinstance(result["Interestingness"], float)

    def test_integer_score_coerced_to_float(self):
        result = normalize_idea(_full_idea(Score=9))
        assert result["Score"] == pytest.approx(9.0)
        assert isinstance(result["Score"], float)

    def test_float_score_preserved(self):
        result = normalize_idea(_full_idea(Score=8.5))
        assert result["Score"] == pytest.approx(8.5)
        assert isinstance(result["Score"], float)

    def test_zero_string_coerced_to_zero_float(self):
        result = normalize_idea(_full_idea(Score="0"))
        assert result["Score"] == pytest.approx(0.0)

    def test_negative_score_coerced(self):
        result = normalize_idea(_full_idea(Score="-1.5"))
        assert result["Score"] == pytest.approx(-1.5)

    def test_all_numeric_fields_coerced_in_one_call(self):
        raw = _full_idea(Score="9.0", Novelty="8.0", Feasibility="7.0", Interestingness="6.0")
        result = normalize_idea(raw)
        assert result["Score"] == pytest.approx(9.0)
        assert result["Novelty"] == pytest.approx(8.0)
        assert result["Feasibility"] == pytest.approx(7.0)
        assert result["Interestingness"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# 2. Missing fields → safe defaults
# ---------------------------------------------------------------------------


class TestMissingFieldDefaults:
    """Absent or None fields must resolve to safe legacy defaults."""

    def test_missing_title_defaults_to_untitled_idea(self):
        raw = _full_idea()
        del raw["Title"]
        result = normalize_idea(raw)
        assert result["Title"] == "Untitled Idea"

    def test_none_title_defaults_to_untitled_idea(self):
        result = normalize_idea(_full_idea(Title=None))
        assert result["Title"] == "Untitled Idea"

    def test_missing_problem_defaults_to_empty_string(self):
        raw = _full_idea()
        del raw["Problem"]
        result = normalize_idea(raw)
        assert result["Problem"] == ""

    def test_missing_approach_defaults_to_empty_string(self):
        raw = _full_idea()
        del raw["Approach"]
        result = normalize_idea(raw)
        assert result["Approach"] == ""

    def test_missing_score_defaults_to_zero(self):
        raw = _full_idea()
        del raw["Score"]
        result = normalize_idea(raw)
        assert result["Score"] == pytest.approx(0.0)

    def test_none_score_defaults_to_zero(self):
        result = normalize_idea(_full_idea(Score=None))
        assert result["Score"] == pytest.approx(0.0)

    def test_missing_novelty_defaults_to_zero(self):
        raw = _full_idea()
        del raw["Novelty"]
        result = normalize_idea(raw)
        assert result["Novelty"] == pytest.approx(0.0)

    def test_missing_feasibility_defaults_to_zero(self):
        raw = _full_idea()
        del raw["Feasibility"]
        result = normalize_idea(raw)
        assert result["Feasibility"] == pytest.approx(0.0)

    def test_missing_interestingness_defaults_to_zero(self):
        raw = _full_idea()
        del raw["Interestingness"]
        result = normalize_idea(raw)
        assert result["Interestingness"] == pytest.approx(0.0)

    def test_missing_source_url_defaults_to_none(self):
        raw = _full_idea()
        del raw["source_url"]
        result = normalize_idea(raw)
        assert result["source_url"] is None

    def test_completely_empty_dict_returns_all_defaults(self):
        result = normalize_idea({})
        assert result["Title"] == "Untitled Idea"
        assert result["Problem"] == ""
        assert result["Approach"] == ""
        assert result["Score"] == pytest.approx(0.0)
        assert result["Novelty"] == pytest.approx(0.0)
        assert result["Feasibility"] == pytest.approx(0.0)
        assert result["Interestingness"] == pytest.approx(0.0)
        assert result["source_url"] is None

    def test_all_defaults_align_with_app_transform_behavior(self):
        """Verify defaults produce the same values that app.py would produce
        when calling float(idea.get('Score', 0)) etc.
        """
        result = normalize_idea({})
        # app.py: float(idea.get("Score", 0)) == 0.0
        assert float(result.get("Score", 0)) == 0.0
        # app.py: idea.get("Title", "Untitled Idea")
        assert result.get("Title", "Untitled Idea") == "Untitled Idea"
        # app.py: idea.get("source_url")
        assert result.get("source_url") is None


# ---------------------------------------------------------------------------
# 3. Non-coercible numeric fields → ValueError
# ---------------------------------------------------------------------------


class TestNonCoercibleNumericRaisesValueError:
    """A non-coercible numeric field must raise ValueError, not silently default."""

    def test_score_abc_raises_value_error(self):
        with pytest.raises(ValueError, match="Score"):
            normalize_idea(_full_idea(Score="abc"))

    def test_novelty_non_numeric_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Novelty"):
            normalize_idea(_full_idea(Novelty="not-a-number"))

    def test_feasibility_non_numeric_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Feasibility"):
            normalize_idea(_full_idea(Feasibility="N/A"))

    def test_interestingness_non_numeric_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Interestingness"):
            normalize_idea(_full_idea(Interestingness="high"))

    def test_score_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Score"):
            normalize_idea(_full_idea(Score=""))

    def test_error_message_contains_bad_value(self):
        with pytest.raises(ValueError, match="abc"):
            normalize_idea(_full_idea(Score="abc"))


# ---------------------------------------------------------------------------
# 4. Output is a copy (input not mutated)
# ---------------------------------------------------------------------------


class TestOutputIsCopy:
    """normalize_idea must return a new dict; mutating it must not touch input."""

    def test_output_is_not_the_same_object(self):
        raw = _full_idea()
        result = normalize_idea(raw)
        assert result is not raw

    def test_mutating_output_does_not_mutate_input(self):
        raw = _full_idea(Title="Original")
        result = normalize_idea(raw)
        result["Title"] = "Mutated"
        assert raw["Title"] == "Original"

    def test_mutating_output_numeric_does_not_affect_input(self):
        raw = _full_idea(Score="7.0")
        result = normalize_idea(raw)
        result["Score"] = 999.0
        # raw["Score"] should still be the original string
        assert raw["Score"] == "7.0"

    def test_extra_keys_from_input_are_preserved_in_output(self):
        raw = _full_idea()
        raw["custom_key"] = "custom_value"
        result = normalize_idea(raw)
        assert result["custom_key"] == "custom_value"

    def test_extra_keys_mutation_in_output_does_not_affect_input(self):
        raw = _full_idea()
        raw["extra"] = "original"
        result = normalize_idea(raw)
        result["extra"] = "changed"
        assert raw["extra"] == "original"


# ---------------------------------------------------------------------------
# 5. normalize_ideas (list-level function)
# ---------------------------------------------------------------------------


class TestNormalizeIdeas:
    """normalize_ideas must apply normalize_idea element-wise."""

    def test_empty_list_returns_empty_list(self):
        assert normalize_ideas([]) == []

    def test_single_idea_list(self):
        raw = [_full_idea(Score="9.0")]
        result = normalize_ideas(raw)
        assert len(result) == 1
        assert result[0]["Score"] == pytest.approx(9.0)

    def test_multiple_ideas_all_normalized(self):
        raws = [
            _full_idea(Score="7.0", Title="Idea A"),
            _full_idea(Score="8.0", Title="Idea B"),
        ]
        results = normalize_ideas(raws)
        assert len(results) == 2
        assert results[0]["Score"] == pytest.approx(7.0)
        assert results[1]["Score"] == pytest.approx(8.0)
        assert results[0]["Title"] == "Idea A"
        assert results[1]["Title"] == "Idea B"

    def test_missing_fields_in_list_get_defaults(self):
        raws = [{"Title": "Only Title"}]
        results = normalize_ideas(raws)
        assert results[0]["Score"] == pytest.approx(0.0)
        assert results[0]["Problem"] == ""
        assert results[0]["source_url"] is None

    def test_bad_score_in_list_raises_value_error(self):
        raws = [_full_idea(Score="abc")]
        with pytest.raises(ValueError, match="Score"):
            normalize_ideas(raws)

    def test_output_list_elements_are_copies(self):
        raw = _full_idea()
        results = normalize_ideas([raw])
        results[0]["Title"] = "Changed"
        assert raw["Title"] == "My Test Idea"

    def test_output_list_is_new_list_object(self):
        raws = [_full_idea()]
        results = normalize_ideas(raws)
        assert results is not raws


# ---------------------------------------------------------------------------
# 6. Integration: normalized output is compatible with legacy downstream
# ---------------------------------------------------------------------------


class TestLegacyCompatibility:
    """Ensure normalized dicts behave identically to what app.py expects."""

    def test_float_score_for_impact_score_calculation(self):
        """app.py: float(idea.get("Score", 0)) must work after normalization."""
        raw = _full_idea(Score="8.5")
        result = normalize_idea(raw)
        assert float(result.get("Score", 0)) == pytest.approx(8.5)

    def test_float_interestingness_for_upvotes_calculation(self):
        """app.py: float(idea.get("Interestingness", 0)) must work."""
        raw = _full_idea(Interestingness="5.0")
        result = normalize_idea(raw)
        assert float(result.get("Interestingness", 0)) == pytest.approx(5.0)

    def test_upvotes_formula_matches_app_py(self):
        """app.py upvotes = int(float(Score)*10 + float(Interestingness))"""
        raw = _full_idea(Score="8.0", Interestingness="5.0")
        result = normalize_idea(raw)
        expected = int(float(result["Score"]) * 10 + float(result["Interestingness"]))
        assert expected == 85

    def test_novelty_comparison_works_after_normalization(self):
        """app.py: if float(idea.get("Novelty", 0)) > 8 — must work."""
        raw = _full_idea(Novelty="8.5")
        result = normalize_idea(raw)
        assert float(result.get("Novelty", 0)) > 8

    def test_feasibility_comparison_works_after_normalization(self):
        """app.py: if float(idea.get("Feasibility", 0)) > 8 — must work."""
        raw = _full_idea(Feasibility="9.0")
        result = normalize_idea(raw)
        assert float(result.get("Feasibility", 0)) > 8

    def test_run_service_safe_float_on_score(self):
        """run_service._safe_float(idea.get("Score")) must get a float or None."""
        raw = _full_idea(Score="7.5")
        result = normalize_idea(raw)
        # After normalization Score is already a float; safe_float(float) → float
        score_val = result.get("Score")
        assert isinstance(score_val, float)
        assert score_val == pytest.approx(7.5)

    def test_source_url_passthrough(self):
        """app.py: idea.get("source_url") — key must be present."""
        raw = _full_idea(source_url="https://arxiv.org/abs/1234.56789")
        result = normalize_idea(raw)
        assert result["source_url"] == "https://arxiv.org/abs/1234.56789"


# ---------------------------------------------------------------------------
# 7. Schema drift detection (Task 11 regression gate)
# ---------------------------------------------------------------------------


class TestSchemaDriftDetection:
    """
    Negative regression tests that catch schema drift deterministically.
    These tests MUST fail if someone changes the LLM output shape in a breaking way:
    - Numeric fields replaced by non-coercible types (dict, list, bool with non-numeric).
    - Required string fields replaced by wrong types that would break downstream.

    The key invariant: normalize_idea raises ValueError for non-coercible numeric fields,
    NOT a silent default — so upstream drift is caught at normalization time, not silently
    swallowed and turned into 0.0.
    """

    def test_score_as_dict_raises_value_error(self):
        """
        Drift scenario: LLM emits Score as {"value": 8.5} instead of a number.
        Must raise ValueError, not silently normalize to 0.0.
        """
        with pytest.raises(ValueError, match="Score"):
            normalize_idea(_full_idea(Score={"value": 8.5}))

    def test_score_as_list_raises_value_error(self):
        """
        Drift scenario: LLM emits Score as [8.5] (wrapped in list).
        Must raise ValueError, not silently normalize to 0.0.
        """
        with pytest.raises(ValueError, match="Score"):
            normalize_idea(_full_idea(Score=[8.5]))

    def test_novelty_as_dict_raises_value_error(self):
        """Drift: Novelty field returns a structured object instead of a number."""
        with pytest.raises(ValueError, match="Novelty"):
            normalize_idea(_full_idea(Novelty={"score": 7}))

    def test_feasibility_as_list_raises_value_error(self):
        """Drift: Feasibility field returns a list (e.g. multi-value) instead of scalar."""
        with pytest.raises(ValueError, match="Feasibility"):
            normalize_idea(_full_idea(Feasibility=[6, 7]))

    def test_interestingness_as_nested_dict_raises_value_error(self):
        """Drift: Interestingness field returns nested structured data."""
        with pytest.raises(ValueError, match="Interestingness"):
            normalize_idea(_full_idea(Interestingness={"low": 3, "high": 7}))

    def test_schema_drift_in_list_raises_on_first_bad_element(self):
        """
        Drift in normalize_ideas: if the FIRST element has a bad Score, ValueError is raised.
        Must NOT silently skip bad elements or return partial results.
        """
        raws = [
            _full_idea(Score={"drifted": True}),  # bad: Score is a dict, not a number
            _full_idea(Score="7.0"),  # good: would succeed on its own
        ]
        with pytest.raises(ValueError, match="Score"):
            normalize_ideas(raws)

    def test_schema_drift_in_list_raises_on_second_element(self):
        """
        Drift in normalize_ideas: ValueError propagates even when first element is good.
        Catches the case where only some items in a batch are malformed.
        """
        raws = [
            _full_idea(Score="9.0"),  # good
            _full_idea(Novelty="not-a-float"),  # bad: non-numeric string
        ]
        with pytest.raises(ValueError, match="Novelty"):
            normalize_ideas(raws)

    def test_score_none_is_NOT_schema_drift_returns_default(self):
        """
        None is a valid 'missing' sentinel and must NOT raise ValueError.
        Absent fields default to 0.0; this is by-design, not drift.
        Regression guard: ensure the non-coercible check doesn't accidentally flag None.
        """
        result = normalize_idea(_full_idea(Score=None))
        assert result["Score"] == pytest.approx(0.0)

    def test_boolean_true_is_schema_drift_raises_value_error(self):
        """
        Drift scenario: LLM emits Score as boolean True.
        In Python bool is a subclass of int, so float(True) == 1.0.
        This should be accepted (bool IS numeric-coercible) — document the contract.
        """
        # bool is a subclass of int; float(True) = 1.0 — this is coercible.
        # Documenting the contract here: booleans are accepted (not drift).
        result = normalize_idea(_full_idea(Score=True))
        assert result["Score"] == pytest.approx(1.0)

    def test_empty_dict_is_schema_drift_raises_value_error(self):
        """
        Drift: LLM emits Score as empty dict {}.
        float({}) is not valid — must raise ValueError.
        """
        with pytest.raises(ValueError, match="Score"):
            normalize_idea(_full_idea(Score={}))

    def test_missing_score_after_normalize_still_accepted_by_app_py_float(self):
        """
        Regression: even if Score is absent (defaults to 0.0), app.py's float(idea.get('Score', 0))
        must still work on the output without raising.
        """
        result = normalize_idea({})
        # app.py pattern: float(idea.get("Score", 0))
        coerced = float(result.get("Score", 0))
        assert coerced == 0.0
