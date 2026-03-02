from __future__ import annotations

import json

from src.backtest.models import IdeaPrediction, PaperRecord
from src.strategy.prompt_llm import PromptLLMStrategy


def _paper(paper_id: str, month: str, title: str, keywords: list[str]) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        month=month,
        summary="summary",
        keywords=keywords,
        source_path=f"/fake/{paper_id}.md",
    )


def test_prompt_llm_strategy_generate_happy_path(monkeypatch) -> None:
    from backend import llm_utils, prompt_registry

    calls: dict[str, object] = {}
    fake_client = object()

    def _fake_get_prompt_policy(prompt_id: str, version: str) -> dict[str, object]:
        calls["prompt_id"] = prompt_id
        calls["prompt_version"] = version
        return {
            "prompt_id": prompt_id,
            "version": version,
            "template": "SYSTEM TEMPLATE",
            "model_id": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 1024,
            "timeout_seconds": 30,
        }

    def _fake_create_client(model_id: str) -> tuple[object, str]:
        calls["create_client_model_id"] = model_id
        return fake_client, model_id

    def _fake_get_response_from_llm(**kwargs):  # type: ignore[no-untyped-def]
        calls["llm_kwargs"] = kwargs
        return (
            json.dumps(
                {
                    "ideas": [
                        {
                            "title": "Causal adapters for sparse domains",
                            "rationale": "Improve transfer with causal bottlenecks.",
                            "key_terms": ["causal", "adapter", "causal"],
                            "confidence": 0.86,
                        },
                        {
                            "Title": "Continual retrieval alignment",
                            "Problem": "Current retrievers drift over time.",
                            "Approach": "Jointly train memory and retriever updates.",
                            "Keywords": ["retrieval", "continual-learning"],
                            "Score": 8,
                        },
                        {
                            "title": "Should be clipped by top_k",
                            "rationale": "extra",
                            "key_terms": ["clip"],
                            "confidence": 0.9,
                        },
                    ]
                }
            ),
            [],
        )

    monkeypatch.setattr(prompt_registry, "get_prompt_policy", _fake_get_prompt_policy)
    monkeypatch.setattr(llm_utils, "create_client", _fake_create_client)
    monkeypatch.setattr(llm_utils, "get_response_from_llm", _fake_get_response_from_llm)

    strategy = PromptLLMStrategy(
        model_id="gpt-4o-mini",
        prompt_id="llm_baseline",
        prompt_version="v1",
    )
    train_papers = [
        _paper("p1", "2024-04", "Title A", ["causal", "adapter"]),
        _paper("p2", "2024-05", "Title B", ["retrieval", "alignment"]),
    ]

    predictions = strategy.generate(
        train_papers=train_papers,
        cutoff_month="2024-06",
        top_k=2,
    )

    assert len(predictions) == 2
    assert all(isinstance(item, IdeaPrediction) for item in predictions)
    assert [item.rank for item in predictions] == [1, 2]
    assert predictions[0].key_terms == ["causal", "adapter"]
    assert predictions[1].confidence == 0.8

    assert calls["prompt_id"] == "llm_baseline"
    assert calls["prompt_version"] == "v1"
    assert calls["create_client_model_id"] == "gpt-4o-mini"
    llm_kwargs = calls["llm_kwargs"]
    assert isinstance(llm_kwargs, dict)
    assert llm_kwargs["client"] is fake_client
    assert llm_kwargs["model"] == "gpt-4o-mini"
    assert llm_kwargs["system_message"] == "SYSTEM TEMPLATE"


def test_prompt_llm_strategy_generate_malformed_output_returns_empty(monkeypatch) -> None:
    from backend import llm_utils, prompt_registry

    monkeypatch.setattr(
        prompt_registry,
        "get_prompt_policy",
        lambda *_args, **_kwargs: {
            "template": "SYSTEM TEMPLATE",
            "model_id": "gpt-4o-mini",
            "temperature": 0.7,
        },
    )
    monkeypatch.setattr(llm_utils, "create_client", lambda _model_id: (object(), _model_id))
    monkeypatch.setattr(
        llm_utils,
        "get_response_from_llm",
        lambda **_kwargs: ("this is not valid json", []),
    )

    strategy = PromptLLMStrategy(
        model_id="gpt-4o-mini",
        prompt_id="llm_baseline",
        prompt_version="v1",
    )

    predictions = strategy.generate(
        train_papers=[_paper("p1", "2024-04", "Title A", ["a"])],
        cutoff_month="2024-06",
        top_k=3,
    )
    assert predictions == []
