"""Tests for the LLM-judge score parsing + fingerprint (C2/C7/NEW-2 fixes).

examples/live-idea-bench/llm_judge_eval.py is a path-run script (not an
importable package), so it is loaded by path. These tests cover the hardened
SCORE_RE, the partial-parse fail-loud behavior (no silent backfill to 1), the
<think>-stripping, and the decode-config-sensitive judge fingerprint — without
any network call.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_JUDGE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "live-idea-bench" / "llm_judge_eval.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("llm_judge_eval", _JUDGE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeMsg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeMsg(content)]


class _FakeClient:
    """Returns a fixed completion string for chat.completions.create."""

    def __init__(self, content):
        self._content = content
        self.calls = 0
        self.chat = type("C", (), {"completions": self})()

    def create(self, **_kwargs):
        self.calls += 1
        return _FakeResp(self._content)


def _pred(mod):
    return mod.IdeaPrediction(rank=1, title="t", rationale="r", approach="a", key_terms=["k"])


def test_score_re_anchored_and_rejects_inline_and_slash():
    mod = _load()
    rx = mod.SCORE_RE
    # start-of-line label captured
    assert {m.group(1).upper(): m.group(2) for m in rx.finditer(
        "PROBLEM_MATCH: 3\nMETHOD_MATCH: 2\nSPECIFICITY: 2")} == {
        "PROBLEM_MATCH": "3", "METHOD_MATCH": "2", "SPECIFICITY": "2"}
    # a stray "...: 3" mid-prose (e.g. injected) is NOT captured as a score line
    assert not list(rx.finditer("the method match is good: 3 out of nowhere"))
    # "METHOD_MATCH: 2/3" must not be misread as 2 (word boundary after digit)
    assert [m.group(2) for m in rx.finditer("METHOD_MATCH: 2/3")] == []


def test_full_response_parsed():
    mod = _load()
    client = _FakeClient("PROBLEM_MATCH: 3\nMETHOD_MATCH: 3\nSPECIFICITY: 3\nREASONING: identical")
    d = mod._call_judge(_pred(mod), "title", "abstract", client, "gpt-4.1-mini")
    assert d["parse_failed"] is False
    assert (d["problem_score"], d["method_score"], d["specificity_score"]) == (3, 3, 3)
    assert d["match"] is True


def test_partial_parse_fails_loud_not_silent_one():
    mod = _load()
    # SPECIFICITY missing -> must NOT silently become (3,3,1); must record parse_failed.
    client = _FakeClient("PROBLEM_MATCH: 3\nMETHOD_MATCH: 3\nREASONING: oops truncated")
    d = mod._call_judge(_pred(mod), "title", "abstract", client, "gpt-4.1-mini")
    assert d["parse_failed"] is True
    assert d["match"] is False
    assert d["problem_score"] is None and d["specificity_score"] is None
    # retried MAX_JUDGE_RETRY times before giving up
    assert client.calls == mod.MAX_JUDGE_RETRY


def test_think_block_stripped_before_parse():
    mod = _load()
    content = (
        "<think>METHOD_MATCH: 0 this is my scratchpad, ignore</think>\n"
        "PROBLEM_MATCH: 2\nMETHOD_MATCH: 3\nSPECIFICITY: 2\nREASONING: ok"
    )
    d = mod._call_judge(_pred(mod), "title", "abstract", judge_client=_FakeClient(content),
                        judge_model="qwen3.5-2b")
    # the <think> METHOD_MATCH: 0 must be stripped; real scores used
    assert d["parse_failed"] is False
    assert (d["problem_score"], d["method_score"], d["specificity_score"]) == (2, 3, 2)


def test_judge_fingerprint_sensitive_to_decode_config(monkeypatch):
    mod = _load()
    fp_default = mod._judge_fingerprint("gpt-4.1-mini")
    # same model+rubric+decode -> stable
    assert mod._judge_fingerprint("gpt-4.1-mini") == fp_default
    # qwen flips enable_thinking -> different fingerprint than a non-qwen model
    assert mod._judge_fingerprint("qwen3.5-2b") != mod._judge_fingerprint("gpt-4.1-mini")
    # changing max_tokens changes the fingerprint
    monkeypatch.setattr(mod, "JUDGE_MAX_TOKENS", 384)
    assert mod._judge_fingerprint("gpt-4.1-mini") != fp_default


def test_in_prompt_match_formula_removed():
    mod = _load()
    # C2: the gameable "A prediction MATCHES if (P+M>=5)..." line must be gone
    assert "A prediction MATCHES if" not in mod.JUDGE_SYSTEM
    assert "Score each dimension independently" in mod.JUDGE_SYSTEM
