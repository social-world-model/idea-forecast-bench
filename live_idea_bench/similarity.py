from __future__ import annotations

import re
from dataclasses import asdict
from difflib import SequenceMatcher
from typing import Iterable, Optional

from live_idea_bench.config import Config, SimilarityConfig, load_runtime_config, load_similarity_config
from live_idea_bench.llm import create_client, get_response_from_llm
from live_idea_bench.models import EvaluationResult, IdeaPrediction, MatchResult, PaperRecord
from live_idea_bench.papers import clean_paper_content, read_file_content

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    _EMBEDDING_AVAILABLE = True
except ImportError:  # pragma: no cover
    _EMBEDDING_AVAILABLE = False


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
) -> MatchResult:
    resolved_model = model_name or runtime_config.model_name
    client, resolved_model = create_client(resolved_model)
    raw, _ = get_response_from_llm(
        msg=similarity_config.user_prompt_template.format(idea=idea, context=context),
        client=client,
        model=resolved_model,
        system_message=similarity_config.system_prompt,
        temperature=runtime_config.temperature,
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


_EMBEDDING_MODEL: SentenceTransformer | None = None


def _embedding_similarity(
    idea: str,
    context: str,
    runtime_config: Config,
) -> MatchResult:
    global _EMBEDDING_MODEL
    if not _EMBEDDING_AVAILABLE:
        return MatchResult(score=_hybrid_similarity(idea, context), engine_name="hybrid-fallback")

    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(runtime_config.embedding.model_name)

    idea_vec = _EMBEDDING_MODEL.encode(idea, convert_to_numpy=True, normalize_embeddings=True)
    context_vec = _EMBEDDING_MODEL.encode(
        context[: runtime_config.embedding.max_context_chars],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    score = float(cosine_similarity(idea_vec.reshape(1, -1), context_vec.reshape(1, -1))[0, 0])
    return MatchResult(score=max(0.0, min(1.0, score)), engine_name=f"embedding:{runtime_config.embedding.model_name}")


def compute_similarity(
    idea: str,
    context: str,
    similarity_config: SimilarityConfig | None = None,
    runtime_config: Config | None = None,
    *,
    model_name: str | None = None,
) -> MatchResult:
    resolved_similarity = similarity_config or load_similarity_config()
    resolved_runtime = runtime_config or load_runtime_config()

    engine = resolved_similarity.engine.lower().strip()
    if engine == "llm":
        return _llm_similarity(idea, context, resolved_similarity, resolved_runtime, model_name=model_name)
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
    if similarity_config.engine.lower().strip() == "llm":
        return result.score >= similarity_config.llm_match_threshold
    semantic = _hybrid_similarity(idea, context)
    keyword = _keyword_overlap(idea, context)
    return (
        semantic >= similarity_config.semantic_threshold
        or keyword >= similarity_config.keyword_threshold
        or result.score >= similarity_config.llm_match_threshold
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
    for paper in future_papers:
        result = compute_similarity(
            pred_text,
            paper_text(paper),
            resolved_similarity,
            resolved_runtime,
            model_name=model_name,
        )
        result.paper_id = paper.paper_id
        if best is None or result.score > best.score:
            best = result
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
) -> EvaluationResult:
    similarity_config = load_similarity_config(similarity_config_path)
    runtime_config = load_runtime_config(runtime_config_path)
    top_preds = predictions[:k]

    matched_ranks: list[int] = []
    matched_paper_ids: list[str] = []
    seen_paper_ids: set[str] = set()

    for pred in top_preds:
        match = best_paper_match(
            pred,
            future_papers,
            similarity_config=similarity_config,
            runtime_config=runtime_config,
            model_name=model_name,
        )
        if match is None or not match.paper_id:
            continue
        future_paper = next((paper for paper in future_papers if paper.paper_id == match.paper_id), None)
        if future_paper is None:
            continue
        pred_text = idea_text(pred)
        paper_body = paper_text(future_paper)
        if is_match(match, pred_text, paper_body, similarity_config):
            matched_ranks.append(pred.rank)
            if future_paper.paper_id not in seen_paper_ids:
                seen_paper_ids.add(future_paper.paper_id)
                matched_paper_ids.append(future_paper.paper_id)

    hit_at_k = 1.0 if matched_ranks else 0.0
    recall_at_k = (len(matched_paper_ids) / len(future_papers)) if future_papers else 0.0
    precision_at_k = (len(matched_ranks) / max(1, min(k, len(top_preds)))) if top_preds else 0.0
    mrr = (1.0 / min(matched_ranks)) if matched_ranks else 0.0
    novelty = _novelty_at_k(top_preds, [paper_text(paper) for paper in train_papers], k)
    diversity = _diversity_at_k(top_preds, k)

    return EvaluationResult(
        hit_at_k=round(hit_at_k, 4),
        recall_at_k=round(recall_at_k, 4),
        precision_at_k=round(precision_at_k, 4),
        mrr=round(mrr, 4),
        novelty=round(novelty, 4),
        diversity=round(diversity, 4),
        matched_prediction_ranks=matched_ranks,
        matched_paper_ids=matched_paper_ids,
    )


def serialize_evaluation(result: EvaluationResult) -> dict[str, object]:
    return asdict(result)
