from __future__ import annotations

import logging
import re
from dataclasses import asdict

logger = logging.getLogger(__name__)
from difflib import SequenceMatcher
from typing import Iterable, Optional

from live_idea_bench.config import Config, SimilarityConfig, load_runtime_config, load_similarity_config
from live_idea_bench.llm import create_client, get_response_from_llm
from live_idea_bench.models import (
    EvaluationResult,
    IdeaPrediction,
    MatchResult,
    PaperRecord,
    PredictionMatchDetail,
    ScoredPredictionList,
)
from live_idea_bench.papers import (
    date_to_ordinal,
    get_paper_published_date,
)

# Embedding engine is Voyage-only by design — no local/hybrid fallback. Mixing
# Voyage cosine, local cosine, and lexical hybrid under one threshold within a
# run destroys cross-run comparability, so a misconfigured/unavailable Voyage
# endpoint must fail loud rather than silently degrade.
VOYAGE_BASE_URL = "https://api.voyageai.com/v1"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def idea_text(prediction: IdeaPrediction | dict[str, object] | str) -> str:
    if isinstance(prediction, str):
        return prediction
    if isinstance(prediction, dict):
        title = str(prediction.get("title", ""))
        rationale = str(prediction.get("rationale", ""))
        approach = str(prediction.get("approach", ""))
        key_terms = prediction.get("key_terms") or prediction.get("keywords") or []
        key_terms_text = " ".join(str(term).strip() for term in key_terms if str(term).strip())
        return f"{title} {rationale} {approach} {key_terms_text}".strip()
    key_terms_text = " ".join(term.strip() for term in prediction.key_terms if term.strip())
    return f"{prediction.title} {prediction.rationale} {prediction.approach} {key_terms_text}".strip()


def paper_text(paper: PaperRecord) -> str:
    keywords = ", ".join(paper.keywords)
    return f"{paper.title}\n\nKeywords: {keywords}\n\n{paper.summary}".strip()


def _token_jaccard(a: str, b: str) -> float:
    sa = set(_tokenize(a))
    sb = set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _keyword_overlap(a: str, b: str) -> float:
    sa = set(_tokenize(a))
    sb = set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def _hybrid_similarity(a: str, b: str) -> float:
    jac = _token_jaccard(a, b)
    seq = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return (0.65 * jac) + (0.35 * seq)


def _llm_similarity(
    idea: str,
    context: str,
    similarity_config: SimilarityConfig,
    runtime_config: Config,
    *,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
) -> MatchResult:
    resolved_model = model_name or runtime_config.model_name
    client, resolved_model = create_client(resolved_model)
    max_chars = runtime_config.embedding.max_context_chars
    clean_idea = _sanitize(idea)
    clean_context = _sanitize(context[:max_chars])
    raw, _ = get_response_from_llm(
        msg=similarity_config.user_prompt_template.format(idea=clean_idea, context=clean_context),
        client=client,
        model=resolved_model,
        system_message=similarity_config.system_prompt,
        temperature=runtime_config.temperature,
        reasoning_effort=reasoning_effort,
    )

    score = 0.0
    reasoning = raw.strip()
    score_match = re.search(r"Score:\s*([0-1](?:\.\d+)?)", raw)
    if score_match:
        score = float(score_match.group(1))
    return MatchResult(
        score=max(0.0, min(1.0, score)),
        reasoning=reasoning,
        engine_name=f"llm:{resolved_model}",
    )


def _sanitize(text: str) -> str:
    """Remove null bytes and non-printable control characters that break JSON serialization."""
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or (ord(ch) >= 32 and ord(ch) != 127))


def _embedding_similarity(
    idea: str,
    context: str,
    runtime_config: Config,
) -> MatchResult:
    """Score an (idea, context) pair via the Voyage embedding API.

    Voyage-only by design: there is no local or lexical fallback. A missing key
    or an unreachable endpoint raises rather than silently degrading to a weaker
    engine (which would corrupt cross-run score comparability).
    """
    import math
    import os
    import time

    import openai

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Embedding engine requires VOYAGE_API_KEY (Voyage-only, no fallback). "
            "Set it, or switch the engine in similarity.yaml."
        )
    base_url = runtime_config.embedding.embedding_base_url or VOYAGE_BASE_URL
    model = runtime_config.embedding.api_model
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    clean_idea = _sanitize(idea)
    truncated_context = _sanitize(context[: runtime_config.embedding.max_context_chars])

    max_retries = 7
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=[clean_idea, truncated_context])
            a, b = resp.data[0].embedding, resp.data[1].embedding
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            score = max(0.0, min(1.0, dot / (na * nb))) if na and nb else 0.0
            return MatchResult(score=score, engine_name=f"embedding:voyage:{model}")
        except Exception as exc:  # noqa: BLE001 — retried below, re-raised on exhaustion
            last_exc = exc
            wait = 2 ** attempt  # 1, 2, 4, 8, 16, 32, 64 seconds
            if attempt < max_retries - 1:
                logger.warning(
                    f"Voyage embedding call failed (attempt {attempt+1}/{max_retries}), retrying in {wait}s. Error: {exc}"
                )
                time.sleep(wait)
    raise RuntimeError(
        f"Voyage embedding call failed after {max_retries} attempts; refusing to fall back "
        f"to a weaker engine (would corrupt comparability). Last error: {last_exc}"
    )


def compute_similarity(
    idea: str,
    context: str,
    similarity_config: SimilarityConfig | None = None,
    runtime_config: Config | None = None,
    *,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
) -> MatchResult:
    resolved_similarity = similarity_config or load_similarity_config()
    resolved_runtime = runtime_config or load_runtime_config()

    engine = resolved_similarity.engine.lower().strip()
    if engine == "llm":
        return _llm_similarity(idea, context, resolved_similarity, resolved_runtime, model_name=model_name, reasoning_effort=reasoning_effort)
    if engine == "embedding":
        return _embedding_similarity(idea, context, resolved_runtime)

    semantic = _hybrid_similarity(idea, context)
    keyword = _keyword_overlap(idea, context)
    return MatchResult(
        score=max(semantic, keyword),
        reasoning=f"hybrid semantic={semantic:.3f}, keyword={keyword:.3f}",
        engine_name="hybrid",
    )


def is_match(
    result: MatchResult,
    idea: str,
    context: str,
    similarity_config: SimilarityConfig,
) -> bool:
    engine = similarity_config.engine.lower().strip()
    if engine == "llm":
        return result.score >= similarity_config.llm_match_threshold
    if engine == "embedding":
        return result.score >= similarity_config.embedding_threshold
    # hybrid (default)
    semantic = _hybrid_similarity(idea, context)
    keyword = _keyword_overlap(idea, context)
    return (
        semantic >= similarity_config.semantic_threshold
        or keyword >= similarity_config.keyword_threshold
    )


def _prefilter_future_papers(
    prediction_text: str,
    future_papers: Iterable[PaperRecord],
    *,
    candidate_limit: int | None,
) -> list[PaperRecord]:
    pool = list(future_papers)
    if candidate_limit is None or candidate_limit <= 0 or len(pool) <= candidate_limit:
        return pool

    ranked = sorted(
        pool,
        key=lambda paper: max(
            _hybrid_similarity(prediction_text, paper_text(paper)),
            _keyword_overlap(prediction_text, paper_text(paper)),
        ),
        reverse=True,
    )
    return ranked[:candidate_limit]


def _lead_time_fraction(
    paper: PaperRecord,
    *,
    cutoff_date: str | None,
    future_end_date: str | None,
) -> float:
    if not cutoff_date or not future_end_date:
        return 0.0
    cutoff_ord = date_to_ordinal(cutoff_date)
    future_end_ord = date_to_ordinal(future_end_date)
    horizon = max(1, future_end_ord - cutoff_ord)
    lead_time_days = max(0, date_to_ordinal(get_paper_published_date(paper)) - cutoff_ord)
    return max(0.0, min(1.0, lead_time_days / horizon))


def score_prediction_list(
    predictions: list[IdeaPrediction],
    train_papers: list[PaperRecord],
    future_papers: list[PaperRecord],
    k: int,
    *,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
    model_name: str | None = None,
    cutoff_date: str | None = None,
    future_end_date: str | None = None,
    candidate_limit: int | None = None,
    popularity_weights: dict[str, float] | None = None,
    reasoning_effort: str | None = None,
) -> ScoredPredictionList:
    similarity_config = load_similarity_config(similarity_config_path)
    runtime_config = load_runtime_config(runtime_config_path)
    top_preds = predictions[:k]

    matched_ranks: list[int] = []
    matched_paper_ids: list[str] = []
    matched_lead_times: list[float] = []
    matches: list[PredictionMatchDetail] = []
    used_paper_ids: set[str] = set()
    duplicate_blocked = 0

    for pred in top_preds:
        pred_text = idea_text(pred)
        candidate_papers = _prefilter_future_papers(
            pred_text,
            future_papers,
            candidate_limit=candidate_limit,
        )
        scored_candidates: list[tuple[PaperRecord, MatchResult, bool]] = []

        def _eval_paper(paper: PaperRecord) -> tuple[PaperRecord, MatchResult, bool]:
            body = paper_text(paper)
            res = compute_similarity(
                pred_text,
                body,
                similarity_config,
                runtime_config,
                model_name=model_name,
                reasoning_effort=reasoning_effort,
            )
            res.paper_id = paper.paper_id
            return paper, res, is_match(res, pred_text, body, similarity_config)

        from concurrent.futures import ThreadPoolExecutor, as_completed
        _EVAL_WORKERS = 8
        with ThreadPoolExecutor(max_workers=_EVAL_WORKERS) as pool:
            futures = {pool.submit(_eval_paper, p): p for p in candidate_papers}
            for fut in as_completed(futures):
                scored_candidates.append(fut.result())

        scored_candidates.sort(key=lambda item: (item[1].score, item[0].paper_id), reverse=True)

        duplicate_candidate_ids = [
            paper.paper_id
            for paper, result, matched in scored_candidates
            if matched and paper.paper_id in used_paper_ids
        ]

        selected: tuple[PaperRecord, MatchResult, bool] | None = None
        for paper, result, matched in scored_candidates:
            if not matched or paper.paper_id in used_paper_ids:
                continue
            selected = (paper, result, matched)
            break

        if selected is None:
            if duplicate_candidate_ids:
                duplicate_blocked += 1
            matches.append(
                PredictionMatchDetail(
                    prediction_rank=pred.rank,
                    prediction_title=pred.title,
                    score=0.0,
                    is_match=False,
                    duplicate_candidate_paper_ids=duplicate_candidate_ids,
                )
            )
            continue

        paper, result, _ = selected
        used_paper_ids.add(paper.paper_id)
        matched_ranks.append(pred.rank)
        matched_paper_ids.append(paper.paper_id)
        lead_time = _lead_time_fraction(
            paper,
            cutoff_date=cutoff_date,
            future_end_date=future_end_date,
        )
        matched_lead_times.append(lead_time)
        paper_popularity = popularity_weights.get(paper.paper_id, 0.0) if popularity_weights else 0.0
        matches.append(
            PredictionMatchDetail(
                prediction_rank=pred.rank,
                prediction_title=pred.title,
                paper_id=paper.paper_id,
                score=round(result.score, 4),
                is_match=True,
                lead_time=round(lead_time, 4),
                matched_reasoning=result.reasoning,
                duplicate_candidate_paper_ids=duplicate_candidate_ids,
                matched_paper_popularity=round(paper_popularity, 4),
            )
        )

    hit_at_k = 1.0 if matched_ranks else 0.0
    recall_at_k = (len(matched_paper_ids) / len(future_papers)) if future_papers else 0.0
    precision_at_k = (len(matched_paper_ids) / max(1, min(k, len(top_preds)))) if top_preds else 0.0
    mrr = (1.0 / min(matched_ranks)) if matched_ranks else 0.0
    novelty = _novelty_at_k(top_preds, [paper_text(paper) for paper in train_papers], k)
    diversity = _diversity_at_k(top_preds, k)
    lead_time = sum(matched_lead_times) / len(matched_lead_times) if matched_lead_times else 0.0
    duplicate_rate = (duplicate_blocked / len(top_preds)) if top_preds else 0.0

    # Popularity-weighted metrics — only non-zero when popularity_weights are provided
    weighted_hit = 0.0
    weighted_precision = 0.0
    weighted_mrr_val = 0.0
    popularity_recall = 0.0
    if popularity_weights:
        matched_popularities = [
            popularity_weights.get(pid, 0.0) for pid in matched_paper_ids
        ]
        if matched_popularities:
            weighted_hit = max(matched_popularities)
            weighted_precision = sum(matched_popularities) / max(1, min(k, len(top_preds)))
            # weighted MRR: 1/rank of first matched * that match's popularity
            first_match = next(
                (m for m in matches if m.is_match),
                None,
            )
            if first_match is not None:
                weighted_mrr_val = (1.0 / first_match.prediction_rank) * first_match.matched_paper_popularity
        total_pop_mass = sum(popularity_weights.get(p.paper_id, 0.0) for p in future_papers)
        matched_pop_mass = sum(matched_popularities)
        popularity_recall = matched_pop_mass / total_pop_mass if total_pop_mass > 0 else 0.0

    return ScoredPredictionList(
        evaluation=EvaluationResult(
            hit_at_k=round(hit_at_k, 4),
            recall_at_k=round(recall_at_k, 4),
            precision_at_k=round(precision_at_k, 4),
            mrr=round(mrr, 4),
            novelty=round(novelty, 4),
            diversity=round(diversity, 4),
            matched_prediction_ranks=matched_ranks,
            matched_paper_ids=matched_paper_ids,
            lead_time=round(lead_time, 4),
            duplicate_rate=round(duplicate_rate, 4),
            weighted_hit_at_k=round(weighted_hit, 4),
            weighted_precision_at_k=round(weighted_precision, 4),
            weighted_mrr=round(weighted_mrr_val, 4),
            popularity_recall_at_k=round(popularity_recall, 4),
        ),
        matches=matches,
        unmatched_future_paper_ids=[
            paper.paper_id for paper in future_papers if paper.paper_id not in used_paper_ids
        ],
    )


def best_paper_match(
    prediction: IdeaPrediction,
    future_papers: Iterable[PaperRecord],
    similarity_config: SimilarityConfig | None = None,
    runtime_config: Config | None = None,
    *,
    model_name: str | None = None,
) -> MatchResult | None:
    resolved_similarity = similarity_config or load_similarity_config()
    resolved_runtime = runtime_config or load_runtime_config()
    pred_text = idea_text(prediction)
    best: MatchResult | None = None
    for paper in _prefilter_future_papers(pred_text, future_papers, candidate_limit=None):
        result = compute_similarity(
            pred_text,
            paper_text(paper),
            resolved_similarity,
            resolved_runtime,
            model_name=model_name,
        )
        if best is None or result.score > best.score:
            from dataclasses import replace as _dc_replace
            best = _dc_replace(result, paper_id=paper.paper_id)
    return best


def _novelty_at_k(
    predictions: list[IdeaPrediction],
    reference_pool: list[str],
    k: int,
) -> float:
    if not predictions or k <= 0:
        return 0.0
    if not reference_pool:
        return 1.0
    scores: list[float] = []
    for pred in predictions[:k]:
        pred_text = idea_text(pred)
        max_ref_sim = max(_hybrid_similarity(pred_text, ref) for ref in reference_pool)
        scores.append(1.0 - max_ref_sim)
    return sum(scores) / len(scores)


def _diversity_at_k(predictions: list[IdeaPrediction], k: int) -> float:
    top_preds = predictions[:k]
    if len(top_preds) < 2:
        return 0.0
    distances: list[float] = []
    for i in range(len(top_preds)):
        for j in range(i + 1, len(top_preds)):
            distances.append(1.0 - _hybrid_similarity(idea_text(top_preds[i]), idea_text(top_preds[j])))
    return sum(distances) / len(distances)


def evaluate_predictions(
    predictions: list[IdeaPrediction],
    train_papers: list[PaperRecord],
    future_papers: list[PaperRecord],
    k: int,
    *,
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
    model_name: str | None = None,
    cutoff_date: str | None = None,
    future_end_date: str | None = None,
    candidate_limit: int | None = None,
    popularity_weights: dict[str, float] | None = None,
) -> EvaluationResult:
    return score_prediction_list(
        predictions=predictions,
        train_papers=train_papers,
        future_papers=future_papers,
        k=k,
        similarity_config_path=similarity_config_path,
        runtime_config_path=runtime_config_path,
        model_name=model_name,
        cutoff_date=cutoff_date,
        future_end_date=future_end_date,
        candidate_limit=candidate_limit,
        popularity_weights=popularity_weights,
    ).evaluation


def serialize_evaluation(result: EvaluationResult) -> dict[str, object]:
    return asdict(result)
