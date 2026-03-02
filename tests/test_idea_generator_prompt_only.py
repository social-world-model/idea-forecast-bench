from __future__ import annotations

from typing import Any

from backend import idea_generator


def _fake_policy(model_id: str = "policy-model") -> dict[str, Any]:
    return {
        "prompt_id": "llm_baseline",
        "version": "v1",
        "template": "You are a test system prompt.",
        "model_id": model_id,
        "temperature": 0.2,
    }


def test_generate_ideas_prompt_only_normalizes_and_adds_metadata(monkeypatch) -> None:
    papers = [
        {
            "title": "Paper A",
            "abstract": "Abstract A",
            "url": "https://openreview.net/forum?id=a",
            "id": "a",
        }
    ]
    monkeypatch.setattr(
        idea_generator,
        "fetch_papers_from_openreview",
        lambda keywords, limit: papers,
    )
    monkeypatch.setattr(
        "backend.prompt_registry.get_prompt_policy",
        lambda prompt_id, version: _fake_policy(model_id="policy-model"),
    )

    observed: dict[str, Any] = {}

    def fake_create_client(model_id: str):
        observed["model_id"] = model_id
        return object(), "resolved-model"

    def fake_get_response_from_llm(**kwargs):
        assert kwargs["model"] == "resolved-model"
        return (
            "Intro\n```json\n"
            '{"Title":"Prompt Idea","Problem":"P","Approach":"A","Score":"8.5","Novelty":"7.5","Feasibility":"6","Interestingness":"4"}'
            "\n```\nFooter",
            [],
        )

    monkeypatch.setattr("backend.llm_utils.create_client", fake_create_client)
    monkeypatch.setattr("backend.llm_utils.get_response_from_llm", fake_get_response_from_llm)
    monkeypatch.setattr(idea_generator.config, "MODEL", "config-model")

    ideas = idea_generator.generate_ideas(keywords=["vision"], n=1)

    assert observed["model_id"] == "config-model"
    assert len(ideas) == 1
    idea = ideas[0]
    assert idea["Title"] == "Prompt Idea"
    assert idea["Score"] == 8.5 and isinstance(idea["Score"], float)
    assert idea["Novelty"] == 7.5 and isinstance(idea["Novelty"], float)
    assert idea["Feasibility"] == 6.0 and isinstance(idea["Feasibility"], float)
    assert idea["Interestingness"] == 4.0 and isinstance(idea["Interestingness"], float)
    assert idea["source_paper"] == "Paper A"
    assert idea["source_url"] == "https://openreview.net/forum?id=a"
    assert idea["id"].startswith("idea_0_")


def test_generate_ideas_uses_policy_model_when_config_model_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        idea_generator,
        "fetch_papers_from_openreview",
        lambda keywords, limit: [
            {
                "title": "Paper B",
                "abstract": "Abstract B",
                "url": "https://openreview.net/forum?id=b",
                "id": "b",
            }
        ],
    )
    monkeypatch.setattr(
        "backend.prompt_registry.get_prompt_policy",
        lambda prompt_id, version: _fake_policy(model_id="policy-fallback-model"),
    )

    observed: dict[str, Any] = {}

    def fake_create_client(model_id: str):
        observed["model_id"] = model_id
        return object(), "resolved-model"

    monkeypatch.setattr("backend.llm_utils.create_client", fake_create_client)
    monkeypatch.setattr(
        "backend.llm_utils.get_response_from_llm",
        lambda **kwargs: ('{"Title":"T","Problem":"P","Approach":"A","Score":9}', []),
    )
    monkeypatch.setattr(idea_generator.config, "MODEL", "")

    ideas = idea_generator.generate_ideas(keywords=["vision"], n=1)

    assert observed["model_id"] == "policy-fallback-model"
    assert len(ideas) == 1


def test_generate_ideas_skips_malformed_or_invalid_outputs(monkeypatch) -> None:
    papers = [
        {
            "title": "Paper malformed",
            "abstract": "Abstract malformed",
            "url": "https://openreview.net/forum?id=m1",
            "id": "m1",
        },
        {
            "title": "Paper invalid",
            "abstract": "Abstract invalid",
            "url": "https://openreview.net/forum?id=m2",
            "id": "m2",
        },
        {
            "title": "Paper valid",
            "abstract": "Abstract valid",
            "url": "https://openreview.net/forum?id=m3",
            "id": "m3",
        },
    ]
    monkeypatch.setattr(
        idea_generator,
        "fetch_papers_from_openreview",
        lambda keywords, limit: papers,
    )
    monkeypatch.setattr(
        "backend.prompt_registry.get_prompt_policy",
        lambda prompt_id, version: _fake_policy(),
    )
    monkeypatch.setattr(
        "backend.llm_utils.create_client",
        lambda model_id: (object(), "resolved-model"),
    )

    responses = iter(
        [
            "not-json-at-all",
            '{"Title":"Bad","Problem":"P","Approach":"A","Score":"abc"}',
            '{"Title":"Good","Problem":"P","Approach":"A","Score":"7.1"}',
        ]
    )

    def fake_get_response_from_llm(**kwargs):
        return next(responses), []

    monkeypatch.setattr("backend.llm_utils.get_response_from_llm", fake_get_response_from_llm)
    monkeypatch.setattr(idea_generator.config, "MODEL", "config-model")

    ideas = idea_generator.generate_ideas(keywords=["vision"], n=3)

    assert len(ideas) == 1
    assert ideas[0]["Title"] == "Good"
    assert ideas[0]["source_paper"] == "Paper valid"
    assert ideas[0]["Score"] == 7.1
