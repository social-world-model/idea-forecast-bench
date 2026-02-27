"""Contract tests for the legacy generated-idea output shape.

These tests lock the behaviour of `_transform_ideas_for_dashboard` as used by
``GET /api/research-ideas`` and consumed by the dashboard and run reporting.

Keyed fields locked here:
  Input  (PascalCase)   → Output (snake_case / dashboard shape)
  Title                 → title
  Problem + Approach    → description  (markdown-formatted)
  Score                 → impact_score (float), upvotes (derived)
  Interestingness       → upvotes      (additive term)
  Novelty               → tags         (High Novelty when > 8)
  Feasibility           → tags         (High Feasibility when > 8)
  source_url            → url
  id (if present)       → id           (passthrough, no random generation)
"""

from __future__ import annotations

import pytest

from backend.app import _transform_ideas_for_dashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_idea(**overrides) -> dict:
    """Return a minimal, fully-specified legacy idea dict."""
    base = {
        "id": "test-id-001",
        "Title": "Test Idea Title",
        "Problem": "A tricky problem in ML.",
        "Approach": "A novel transformer approach.",
        "Score": "8.5",
        "Novelty": "7.0",
        "Feasibility": "6.0",
        "Interestingness": "5.0",
        "source_url": "https://arxiv.org/abs/1234.56789",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Core field mapping
# ---------------------------------------------------------------------------

class TestCoreFieldMapping:
    """Locks the 1-to-1 field mapping between legacy input and dashboard output."""

    def test_title_maps_from_Title(self):
        idea = _make_idea(Title="My Research Idea")
        result = _transform_ideas_for_dashboard([idea])
        assert len(result) == 1
        assert result[0]["title"] == "My Research Idea"

    def test_title_defaults_to_untitled_when_missing(self):
        idea = _make_idea()
        del idea["Title"]
        result = _transform_ideas_for_dashboard([idea])
        assert result[0]["title"] == "Untitled Idea"

    def test_description_includes_Problem(self):
        idea = _make_idea(Problem="The core problem statement.")
        result = _transform_ideas_for_dashboard([idea])
        assert "The core problem statement." in result[0]["description"]

    def test_description_includes_Approach(self):
        idea = _make_idea(Approach="The proposed approach.")
        result = _transform_ideas_for_dashboard([idea])
        assert "The proposed approach." in result[0]["description"]

    def test_description_format_problem_and_approach(self):
        """Exact markdown format: **Problem:** ... \\n\\n**Approach:** ..."""
        idea = _make_idea(Problem="P1", Approach="A1")
        result = _transform_ideas_for_dashboard([idea])
        expected = "**Problem:** P1\n\n**Approach:** A1"
        assert result[0]["description"] == expected

    def test_url_maps_from_source_url(self):
        idea = _make_idea(source_url="https://example.com/paper")
        result = _transform_ideas_for_dashboard([idea])
        assert result[0]["url"] == "https://example.com/paper"

    def test_url_is_none_when_source_url_absent(self):
        idea = _make_idea()
        del idea["source_url"]
        result = _transform_ideas_for_dashboard([idea])
        assert result[0]["url"] is None

    def test_id_passthrough_when_present(self):
        idea = _make_idea(id="stable-id-xyz")
        result = _transform_ideas_for_dashboard([idea])
        assert result[0]["id"] == "stable-id-xyz"

    def test_id_generated_when_absent(self):
        idea = _make_idea()
        del idea["id"]
        result = _transform_ideas_for_dashboard([idea])
        # ID must be present and non-empty; it starts with "gen_"
        assert result[0]["id"]
        assert result[0]["id"].startswith("gen_")


# ---------------------------------------------------------------------------
# Numeric field mapping
# ---------------------------------------------------------------------------

class TestNumericFieldMapping:
    """Locks numeric derivations: impact_score and upvotes."""

    def test_impact_score_equals_float_Score(self):
        idea = _make_idea(Score="7.25")
        result = _transform_ideas_for_dashboard([idea])
        assert result[0]["impact_score"] == pytest.approx(7.25)

    def test_impact_score_from_integer_Score(self):
        idea = _make_idea(Score=9)
        result = _transform_ideas_for_dashboard([idea])
        assert result[0]["impact_score"] == pytest.approx(9.0)

    def test_impact_score_defaults_to_zero_when_Score_missing(self):
        idea = _make_idea()
        del idea["Score"]
        result = _transform_ideas_for_dashboard([idea])
        assert result[0]["impact_score"] == pytest.approx(0.0)

    def test_upvotes_formula_score_times_10_plus_interestingness(self):
        """upvotes = int(float(Score) * 10 + float(Interestingness))"""
        idea = _make_idea(Score="8.0", Interestingness="5.0")
        result = _transform_ideas_for_dashboard([idea])
        expected_upvotes = int(8.0 * 10 + 5.0)  # 85
        assert result[0]["upvotes"] == expected_upvotes

    def test_upvotes_formula_truncates_via_int(self):
        """int() truncates, not rounds."""
        idea = _make_idea(Score="8.3", Interestingness="4.9")
        result = _transform_ideas_for_dashboard([idea])
        expected_upvotes = int(8.3 * 10 + 4.9)  # int(87.9) = 87
        assert result[0]["upvotes"] == expected_upvotes

    def test_upvotes_zero_when_both_missing(self):
        idea = _make_idea()
        del idea["Score"]
        del idea["Interestingness"]
        result = _transform_ideas_for_dashboard([idea])
        assert result[0]["upvotes"] == 0

    def test_upvotes_only_score_when_interestingness_missing(self):
        idea = _make_idea(Score="6.0")
        del idea["Interestingness"]
        result = _transform_ideas_for_dashboard([idea])
        assert result[0]["upvotes"] == int(6.0 * 10 + 0.0)  # 60


# ---------------------------------------------------------------------------
# Tag rules
# ---------------------------------------------------------------------------

class TestTagRules:
    """Locks tag generation based on Novelty and Feasibility thresholds."""

    def test_base_tags_always_present(self):
        idea = _make_idea(Novelty="5.0", Feasibility="5.0")
        result = _transform_ideas_for_dashboard([idea])
        assert "AI Generated" in result[0]["tags"]
        assert "ICLR 2025" in result[0]["tags"]

    def test_high_novelty_tag_when_Novelty_above_8(self):
        idea = _make_idea(Novelty="8.5")
        result = _transform_ideas_for_dashboard([idea])
        assert "High Novelty" in result[0]["tags"]

    def test_high_novelty_tag_NOT_added_when_Novelty_equals_8(self):
        idea = _make_idea(Novelty="8.0")
        result = _transform_ideas_for_dashboard([idea])
        assert "High Novelty" not in result[0]["tags"]

    def test_high_novelty_tag_NOT_added_when_Novelty_below_8(self):
        idea = _make_idea(Novelty="7.9")
        result = _transform_ideas_for_dashboard([idea])
        assert "High Novelty" not in result[0]["tags"]

    def test_high_feasibility_tag_when_Feasibility_above_8(self):
        idea = _make_idea(Feasibility="9.0")
        result = _transform_ideas_for_dashboard([idea])
        assert "High Feasibility" in result[0]["tags"]

    def test_high_feasibility_tag_NOT_added_when_Feasibility_equals_8(self):
        idea = _make_idea(Feasibility="8.0")
        result = _transform_ideas_for_dashboard([idea])
        assert "High Feasibility" not in result[0]["tags"]

    def test_high_feasibility_tag_NOT_added_when_Feasibility_below_8(self):
        idea = _make_idea(Feasibility="7.5")
        result = _transform_ideas_for_dashboard([idea])
        assert "High Feasibility" not in result[0]["tags"]

    def test_both_high_tags_when_both_above_8(self):
        idea = _make_idea(Novelty="9.0", Feasibility="9.5")
        result = _transform_ideas_for_dashboard([idea])
        assert "High Novelty" in result[0]["tags"]
        assert "High Feasibility" in result[0]["tags"]

    def test_no_high_tags_when_both_at_or_below_8(self):
        idea = _make_idea(Novelty="8.0", Feasibility="8.0")
        result = _transform_ideas_for_dashboard([idea])
        assert "High Novelty" not in result[0]["tags"]
        assert "High Feasibility" not in result[0]["tags"]

    def test_no_tag_duplication_on_multiple_calls(self):
        """Tags list is created fresh per idea, not accumulated."""
        idea = _make_idea(Novelty="9.0", Feasibility="9.0")
        result1 = _transform_ideas_for_dashboard([idea])
        result2 = _transform_ideas_for_dashboard([idea])
        assert result1[0]["tags"].count("High Novelty") == 1
        assert result2[0]["tags"].count("High Novelty") == 1


# ---------------------------------------------------------------------------
# Multi-idea and edge cases
# ---------------------------------------------------------------------------

class TestMultiIdeaAndEdgeCases:
    """Locks behaviour across multiple ideas and edge-case inputs."""

    def test_empty_list_returns_empty_list(self):
        result = _transform_ideas_for_dashboard([])
        assert result == []

    def test_multiple_ideas_all_transformed(self):
        ideas = [
            _make_idea(id="id-1", Title="Idea One", Score="7.0", Interestingness="3.0"),
            _make_idea(id="id-2", Title="Idea Two", Score="9.0", Interestingness="1.0"),
        ]
        result = _transform_ideas_for_dashboard(ideas)
        assert len(result) == 2
        assert result[0]["title"] == "Idea One"
        assert result[1]["title"] == "Idea Two"

    def test_each_idea_has_independent_tags(self):
        """Tags of one idea must not bleed into the next."""
        ideas = [
            _make_idea(id="id-1", Novelty="9.0", Feasibility="4.0"),
            _make_idea(id="id-2", Novelty="4.0", Feasibility="9.0"),
        ]
        result = _transform_ideas_for_dashboard(ideas)
        assert "High Novelty" in result[0]["tags"]
        assert "High Feasibility" not in result[0]["tags"]
        assert "High Novelty" not in result[1]["tags"]
        assert "High Feasibility" in result[1]["tags"]

    def test_static_fields_always_present(self):
        idea = _make_idea()
        result = _transform_ideas_for_dashboard([idea])
        item = result[0]
        assert item["author"] == "AI Researcher"
        assert item["institution"] == "Live Idea Bench"
        assert item["citations"] == 0
        assert "created_at" in item
        assert "updated_at" in item

    def test_score_as_numeric_float_input(self):
        """Score may arrive as float, not only string."""
        idea = _make_idea(Score=8.5, Interestingness=2.5)
        result = _transform_ideas_for_dashboard([idea])
        assert result[0]["impact_score"] == pytest.approx(8.5)
        assert result[0]["upvotes"] == int(8.5 * 10 + 2.5)  # 87


# ---------------------------------------------------------------------------
# Task 11 – Regression: legacy run flow contract via normalize_idea pipeline
# ---------------------------------------------------------------------------


class TestLegacyRunFlowRegression:
    """
    Regression tests: lock that the legacy generate_ideas → _transform_ideas_for_dashboard
    pipeline still works when normalize_idea is applied upstream (prompt-only migration).
    These tests would catch breakage if someone changed field names or removed coercion.
    """

    def test_normalized_idea_still_maps_through_dashboard_transform(self):
        """
        Ideas that have already been through normalize_idea (floats, not strings)
        must pass cleanly through _transform_ideas_for_dashboard.
        Catches breakage if normalize changes types that app.py float()-casts again.
        """
        from backend.output_contract import normalize_idea

        raw = _make_idea(Score="8.0", Novelty="9.0", Feasibility="7.5", Interestingness="4.0")
        normalized = normalize_idea(raw)
        # Score is now a float after normalize_idea
        result = _transform_ideas_for_dashboard([normalized])
        assert len(result) == 1
        assert result[0]["impact_score"] == pytest.approx(8.0)
        assert result[0]["upvotes"] == int(8.0 * 10 + 4.0)  # 84
        assert "High Novelty" in result[0]["tags"]
        assert "High Feasibility" not in result[0]["tags"]  # 7.5 is not > 8

    def test_run_flow_with_fake_generator_hits_dashboard_route(self, monkeypatch, tmp_path):
        """
        Regression: the full /api/runs/start -> RunService -> fake_generate_ideas path
        must still produce a valid run with persisted ideas in the legacy PascalCase shape.
        Catches breakage if RunService or run_service wiring is disrupted.
        """
        import time
        from backend import app as app_module
        from backend.services.run_service import RunService, RunStatus

        def _fake_generator(keywords, n):
            return [
                {"Title": "Idea From Legacy", "Score": 9, "Novelty": 9, "Feasibility": 9,
                 "Problem": "P", "Approach": "A", "id": "legacy-test-001"},
            ]

        monkeypatch.setattr(
            app_module, "run_service",
            RunService(str(tmp_path), idea_generator=_fake_generator),
        )

        client = app_module.app.test_client()
        start_resp = client.post("/api/runs/start", json={"keywords": ["regression"], "n": 1})
        assert start_resp.status_code == 202
        run_id = start_resp.get_json()["run"]["run_id"]

        # Poll for completion
        deadline = time.time() + 5.0
        while time.time() < deadline:
            run = app_module.run_service.get_run(run_id)
            if run and run.get("status") == RunStatus.SUCCESS.value:
                break
            time.sleep(0.05)

        detail_resp = client.get(f"/api/runs/{run_id}?includeIdeas=true")
        assert detail_resp.status_code == 200
        run_payload = detail_resp.get_json()["run"]
        assert run_payload["ideas_count"] == 1
        assert len(run_payload["ideas"]) == 1
        # Legacy PascalCase fields must still be in persisted ideas
        assert run_payload["ideas"][0]["Title"] == "Idea From Legacy"
        assert run_payload["ideas"][0]["Score"] == 9

    def test_transform_with_missing_problem_and_approach_uses_empty_string(self):
        """
        Regression: _transform_ideas_for_dashboard must not crash when Problem/Approach are absent.
        Catches breakage if upstream normalize stops defaulting these to empty string.
        """
        idea = {"id": "t-001", "Title": "Bare Idea", "Score": 5}
        result = _transform_ideas_for_dashboard([idea])
        assert result[0]["description"] == "**Problem:** \n\n**Approach:** "

    def test_generate_ideas_api_validation_rejects_empty_keywords(self, monkeypatch, tmp_path):
        """
        Regression: POST /api/generate-ideas with empty keywords list must return 400.
        Catches breakage if the validation guard is accidentally removed.
        """
        from backend import app as app_module
        from backend.services.run_service import RunService

        monkeypatch.setattr(app_module, "run_service", RunService(str(tmp_path)))
        client = app_module.app.test_client()

        resp = client.post("/api/generate-ideas", json={"keywords": [], "n": 1})
        assert resp.status_code == 400
        assert resp.get_json()["error"]["code"] == "bad_request"

    def test_generate_ideas_api_validation_rejects_missing_keywords(self, monkeypatch, tmp_path):
        """
        Regression: POST /api/generate-ideas without 'keywords' key must return 400.
        The 'keywords' field must be a list; a missing key yields None which is not a list.
        """
        from backend import app as app_module
        from backend.services.run_service import RunService

        monkeypatch.setattr(app_module, "run_service", RunService(str(tmp_path)))
        client = app_module.app.test_client()

        resp = client.post("/api/generate-ideas", json={"n": 1})
        assert resp.status_code == 400
        assert resp.get_json()["error"]["code"] == "bad_request"
