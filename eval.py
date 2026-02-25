from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _token_jaccard(a: str, b: str) -> float:
    sa = set(_tokenize(a))
    sb = set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _semantic_score(a: str, b: str) -> float:
    jac = _token_jaccard(a, b)
    seq = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return 0.65 * jac + 0.35 * seq


def _keyword_overlap(a: str, b: str) -> float:
    sa = set(_tokenize(a))
    sb = set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


@dataclass
class MatchDecision:
    semantic: float
    keyword: float

    def is_match(self, semantic_threshold: float, keyword_threshold: float) -> bool:
        return self.semantic >= semantic_threshold or self.keyword >= keyword_threshold


def _idea_text(pred_item: dict[str, Any]) -> str:
    title = str(pred_item.get("title", ""))
    rationale = str(pred_item.get("rationale", ""))
    approach = str(pred_item.get("approach", ""))
    return f"{title} {rationale} {approach}".strip()


def _match_prediction_to_truth(
    pred_text: str,
    truth_text: str,
    semantic_threshold: float,
    keyword_threshold: float,
) -> MatchDecision:
    semantic = _semantic_score(pred_text, truth_text)
    keyword = _keyword_overlap(pred_text, truth_text)
    return MatchDecision(semantic=semantic, keyword=keyword)


def _novelty_at_k(pred_texts: list[str], reference_pool: list[str], k: int) -> float:
    if not pred_texts or k <= 0:
        return 0.0
    if not reference_pool:
        return 1.0

    top_preds = pred_texts[:k]
    scores: list[float] = []
    for pred in top_preds:
        max_ref_sim = max(_semantic_score(pred, ref) for ref in reference_pool)
        scores.append(1.0 - max_ref_sim)
    return sum(scores) / len(scores)


def _diversity_at_k(pred_texts: list[str], k: int) -> float:
    top_preds = pred_texts[:k]
    if len(top_preds) < 2:
        return 0.0

    distances: list[float] = []
    for i in range(len(top_preds)):
        for j in range(i + 1, len(top_preds)):
            distances.append(1.0 - _semantic_score(top_preds[i], top_preds[j]))

    return sum(distances) / len(distances)


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def _evaluate_domain(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    k_values: list[int],
    semantic_threshold: float,
    keyword_threshold: float,
) -> dict[str, Any]:
    pred_items: list[dict[str, Any]] = list(predictions.get("ideas", []))
    pred_texts = [_idea_text(item) for item in pred_items]
    truth_items: list[str] = [str(x) for x in truth.get("future_ideas", [])]
    reference_pool: list[str] = [str(x) for x in truth.get("historical_ideas", [])]

    per_truth_rr: list[float] = []
    matched_truth_count_at_k: dict[int, int] = {k: 0 for k in k_values}
    hit_at_k: dict[int, int] = {k: 0 for k in k_values}

    for truth_text in truth_items:
        first_rank: int | None = None
        matched_at_k_flags = {k: False for k in k_values}

        for rank, pred_text in enumerate(pred_texts, start=1):
            decision = _match_prediction_to_truth(
                pred_text,
                truth_text,
                semantic_threshold=semantic_threshold,
                keyword_threshold=keyword_threshold,
            )
            if decision.is_match(semantic_threshold, keyword_threshold):
                if first_rank is None:
                    first_rank = rank
                for k in k_values:
                    if rank <= k:
                        matched_at_k_flags[k] = True

        if first_rank is None:
            per_truth_rr.append(0.0)
        else:
            per_truth_rr.append(1.0 / first_rank)

        for k in k_values:
            if matched_at_k_flags[k]:
                matched_truth_count_at_k[k] += 1
                hit_at_k[k] = 1

    domain_metrics: dict[str, Any] = {
        "domain": predictions.get("domain", truth.get("domain", "unknown")),
        "n_predictions": len(pred_texts),
        "n_truth": len(truth_items),
        "mrr": _safe_div(sum(per_truth_rr), len(per_truth_rr)),
        "metrics": {},
    }

    for k in k_values:
        recall_k = _safe_div(matched_truth_count_at_k[k], len(truth_items))
        novelty_k = _novelty_at_k(pred_texts, reference_pool, k)
        diversity_k = _diversity_at_k(pred_texts, k)
        domain_metrics["metrics"][f"hit@{k}"] = float(hit_at_k[k])
        domain_metrics["metrics"][f"recall@{k}"] = recall_k
        domain_metrics["metrics"][f"novelty@{k}"] = novelty_k
        domain_metrics["metrics"][f"diversity@{k}"] = diversity_k

    return domain_metrics


def _aggregate(domain_results: list[dict[str, Any]], k_values: list[int]) -> dict[str, float]:
    if not domain_results:
        return {}

    metrics: dict[str, float] = {}
    mrr = sum(d["mrr"] for d in domain_results) / len(domain_results)
    metrics["mrr"] = mrr

    keys = [f"hit@{k}" for k in k_values] + [f"recall@{k}" for k in k_values] + [f"novelty@{k}" for k in k_values] + [f"diversity@{k}" for k in k_values]
    for key in keys:
        metrics[key] = sum(float(d["metrics"][key]) for d in domain_results) / len(domain_results)

    return metrics


def evaluate(
    predictions_path: Path,
    ground_truth_path: Path,
    k_values: list[int],
    semantic_threshold: float,
    keyword_threshold: float,
) -> dict[str, Any]:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))

    if isinstance(predictions, dict):
        predictions = [predictions]
    if isinstance(ground_truth, dict):
        ground_truth = [ground_truth]

    truth_by_domain = {str(item.get("domain", "")).lower(): item for item in ground_truth}

    domain_results: list[dict[str, Any]] = []
    for pred in predictions:
        domain = str(pred.get("domain", "")).lower()
        truth_item = truth_by_domain.get(domain)
        if truth_item is None:
            continue

        domain_results.append(
            _evaluate_domain(
                predictions=pred,
                truth=truth_item,
                k_values=k_values,
                semantic_threshold=semantic_threshold,
                keyword_threshold=keyword_threshold,
            )
        )

    return {
        "config": {
            "k_values": k_values,
            "semantic_threshold": semantic_threshold,
            "keyword_threshold": keyword_threshold,
        },
        "aggregate": _aggregate(domain_results, k_values),
        "domains": domain_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluator core for prediction quality")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", default="1,3,5", help="Comma-separated k values")
    parser.add_argument("--semantic-threshold", type=float, default=0.50)
    parser.add_argument("--keyword-threshold", type=float, default=0.30)
    args = parser.parse_args()

    k_values = sorted({max(1, int(k.strip())) for k in args.k.split(",") if k.strip()})

    result = evaluate(
        predictions_path=Path(args.predictions),
        ground_truth_path=Path(args.ground_truth),
        k_values=k_values,
        semantic_threshold=args.semantic_threshold,
        keyword_threshold=args.keyword_threshold,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
