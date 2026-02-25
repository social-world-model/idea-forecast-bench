from typing import List, Set

from src.backtest.models import EvaluationResult, IdeaPrediction, PaperRecord


def _collect_keyword_set(papers: List[PaperRecord]) -> Set[str]:
    values: Set[str] = set()
    for paper in papers:
        for key in paper.keywords:
            clean = key.strip().lower()
            if clean:
                values.add(clean)
    return values


def evaluate_predictions(
    predictions: List[IdeaPrediction],
    train_papers: List[PaperRecord],
    future_papers: List[PaperRecord],
    k: int,
) -> EvaluationResult:
    top_preds = predictions[:k]
    future_keywords = _collect_keyword_set(future_papers)
    train_keywords = _collect_keyword_set(train_papers)

    matched_ranks: List[int] = []
    matched_terms: Set[str] = set()

    for pred in top_preds:
        pred_terms = [t.strip().lower() for t in pred.key_terms if t.strip()]
        overlap = [t for t in pred_terms if t in future_keywords]
        if overlap:
            matched_ranks.append(pred.rank)
            matched_terms.update(overlap)

    hit_at_k = 1.0 if matched_ranks else 0.0
    recall_at_k = (
        float(len(matched_terms)) / float(len(future_keywords))
        if future_keywords
        else 0.0
    )
    precision_at_k = (
        float(len(matched_ranks)) / float(max(1, min(k, len(top_preds))))
        if top_preds
        else 0.0
    )
    mrr = 1.0 / float(min(matched_ranks)) if matched_ranks else 0.0

    novelty_parts: List[float] = []
    for pred in top_preds:
        terms = [t.strip().lower() for t in pred.key_terms if t.strip()]
        if not terms:
            novelty_parts.append(0.0)
            continue
        unseen = [t for t in terms if t not in train_keywords]
        novelty_parts.append(float(len(unseen)) / float(len(terms)))
    novelty = sum(novelty_parts) / float(len(novelty_parts)) if novelty_parts else 0.0

    lead_terms = [pred.key_terms[0].strip().lower() for pred in top_preds if pred.key_terms]
    diversity = (
        float(len(set(lead_terms))) / float(len(lead_terms))
        if lead_terms
        else 0.0
    )

    return EvaluationResult(
        hit_at_k=round(hit_at_k, 4),
        recall_at_k=round(recall_at_k, 4),
        precision_at_k=round(precision_at_k, 4),
        mrr=round(mrr, 4),
        novelty=round(novelty, 4),
        diversity=round(diversity, 4),
        matched_prediction_ranks=matched_ranks,
        matched_terms=sorted(matched_terms),
    )

