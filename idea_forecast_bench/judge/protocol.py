from __future__ import annotations

import time
from typing import Any

import openai

from idea_forecast_bench.judge.config import (
    JUDGE_MAX_TOKENS,
    JUDGE_SYSTEM,
    JUDGE_TEMPERATURE,
    JUDGE_USER_TMPL,
    MATCH_PM_THRESHOLD,
    MATCH_S_THRESHOLD,
    MAX_JUDGE_RETRY,
    REQUIRED_DIMS,
    SCORE_RE,
    THINK_RE,
)
from idea_forecast_bench.models import IdeaPrediction


def call_judge(
    pred: IdeaPrediction,
    paper_title: str,
    paper_abstract: str,
    judge_client: openai.OpenAI,
    judge_model: str,
) -> dict[str, Any]:
    user_msg = JUDGE_USER_TMPL.format(
        pred_title=pred.title,
        pred_rationale=pred.rationale[:600],
        pred_approach=pred.approach[:400],
        pred_terms=", ".join(pred.key_terms),
        paper_title=paper_title,
        paper_abstract=paper_abstract[:800],
    )
    last_problem = ""
    for attempt in range(MAX_JUDGE_RETRY):
        try:
            create_kwargs: dict[str, Any] = {
                "model": judge_model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": JUDGE_TEMPERATURE,
                "max_tokens": JUDGE_MAX_TOKENS,
            }
            # Reasoning judges default to "thinking" mode, which burns the
            # 256-token budget on the reasoning trace before the
            # PROBLEM_MATCH/... lines and fails to parse on every call. The
            # switch is spelled differently per backend: vLLM/SGLang take a
            # chat-template kwarg, DashScope takes a top-level field.
            extra_body: dict[str, Any] = {}
            if "qwen" in judge_model.lower():
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            if "dashscope" in str(getattr(judge_client, "base_url", "")).lower():
                extra_body["enable_thinking"] = False
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            resp = judge_client.chat.completions.create(**create_kwargs)
            content = resp.choices[0].message.content or ""
            # Strip any chain-of-thought block before parsing so a reasoning
            # model's <think> ... </think> can't shadow the real score lines.
            parse_target = THINK_RE.sub("", content)
            scores: dict[str, int] = {}
            for m in SCORE_RE.finditer(parse_target):
                scores[m.group(1).upper()] = int(m.group(2))

            missing = [d for d in REQUIRED_DIMS if d not in scores]
            if missing:
                # Do NOT silently backfill to 1 — that would mark a truncated or
                # malformed response as a real low score. Treat as a parse
                # failure and retry; only after retries give up explicitly.
                last_problem = f"missing dims {missing}"
                if attempt < MAX_JUDGE_RETRY - 1:
                    print(
                        f"\n  [judge parse-retry {attempt + 1}] {last_problem}",
                        flush=True,
                    )
                    continue
                print(
                    f"\n  [judge PARSE-FAILED] {last_problem} — recording parse_failed",
                    flush=True,
                )
                return {
                    "match": False,
                    "problem_score": None,
                    "method_score": None,
                    "specificity_score": None,
                    "reasoning": "",
                    "raw": content,
                    "parse_failed": True,
                }

            problem = scores["PROBLEM_MATCH"]
            method = scores["METHOD_MATCH"]
            specificity = scores["SPECIFICITY"]
            match_val = (problem + method >= MATCH_PM_THRESHOLD) and (
                specificity >= MATCH_S_THRESHOLD
            )

            reasoning = ""
            for line in parse_target.splitlines():
                if line.strip().upper().startswith("REASONING"):
                    reasoning = line.split(":", 1)[-1].strip()
                    break

            return {
                "match": match_val,
                "problem_score": problem,
                "method_score": method,
                "specificity_score": specificity,
                "reasoning": reasoning,
                "raw": content,
                "parse_failed": False,
            }
        except Exception as exc:
            wait = 2**attempt
            if attempt < MAX_JUDGE_RETRY - 1:
                print(f"\n  [judge retry {attempt + 1}] {exc}", flush=True)
                time.sleep(wait)
            else:
                print(f"\n  [judge FAILED] {exc} — treating as NO", flush=True)
                return {
                    "match": False,
                    "problem_score": 0,
                    "method_score": 0,
                    "specificity_score": 0,
                    "reasoning": f"error: {exc}",
                    "raw": "",
                    "parse_failed": True,
                }
    return {
        "match": False,
        "problem_score": None,
        "method_score": None,
        "specificity_score": None,
        "reasoning": "max retries exhausted",
        "raw": "",
        "parse_failed": True,
    }
