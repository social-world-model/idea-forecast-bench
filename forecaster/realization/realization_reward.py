"""Extended reward function for the realization GRPO training."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from forecaster.config import RealizationConfig
from forecaster.models import Innovation, RealizationTrajectory
from live_idea_bench.models import PaperRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RealizationRewardBreakdown:
    evidence_quality: float
    operator_adherence: float
    coherence: float
    total_reward: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class StrictTrajectoryRewardBreakdown:
    evidence_quality: float
    operator_adherence: float
    proposal_coherence: float
    total_reward: float
    invalid_completion: bool = False
    invalid_reason: str | None = None

    def to_dict(self) -> dict[str, float | bool | str | None]:
        return asdict(self)


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _keyword_overlap(a: str, b: str) -> float:
    """Overlap coefficient: |A ∩ B| / min(|A|, |B|)."""
    sa = set(_tokenize_words(a))
    sb = set(_tokenize_words(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def _hybrid_similarity(a: str, b: str) -> float:
    """Weighted blend of Jaccard and sequence similarity."""
    sa = set(_tokenize_words(a))
    sb = set(_tokenize_words(b))
    jaccard = len(sa & sb) / len(sa | sb) if sa or sb else 0.0
    seq = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return (0.65 * jaccard) + (0.35 * seq)


# Keywords associated with each operator type
_OPERATOR_KEYWORDS: dict[str, list[str]] = {
    "extend": [
        "extend",
        "extension",
        "improvement",
        "building upon",
        "build upon",
        "improves",
        "enhance",
        "enhanced",
    ],
    "transfer": [
        "transfer",
        "adaptation",
        "adapt",
        "domain",
        "new domain",
        "cross-domain",
        "applied to",
    ],
    "compose": [
        "combination",
        "compose",
        "composition",
        "integration",
        "integrate",
        "combined",
        "combining",
    ],
    "benchmark": [
        "benchmark",
        "evaluation",
        "evaluate",
        "dataset",
        "assess",
        "assessment",
        "test suite",
    ],
    "analyze": [
        "analysis",
        "analyze",
        "study",
        "investigation",
        "investigate",
        "examine",
        "empirical",
    ],
    "simplify": [
        "simplif",
        "efficient",
        "efficiency",
        "lightweight",
        "streamline",
        "compact",
        "reduction",
    ],
    "scale": [
        "scaling",
        "scale",
        "large-scale",
        "capacity",
        "billion",
        "massive",
        "scal",
    ],
    "adapt": [
        "adaptation",
        "adapt",
        "fine-tuning",
        "finetun",
        "customiz",
        "personaliz",
    ],
}

# Minimum proposal length to be considered coherent
_MIN_COHERENT_CHARS = 100
# Technical vocabulary indicators
_TECHNICAL_TOKENS = {
    "model",
    "training",
    "method",
    "approach",
    "architecture",
    "evaluation",
    "dataset",
    "baseline",
    "performance",
    "experiment",
    "results",
    "accuracy",
    "loss",
    "gradient",
    "layer",
    "attention",
    "transformer",
    "encoder",
    "decoder",
    "embedding",
    "token",
    "sequence",
    "benchmark",
    "metric",
    "objective",
}


def compute_evidence_accuracy(
    evidence: list[PaperRecord],
    innovation: Innovation,
    *,
    threshold: float = 0.3,
) -> float:
    """Score how accurately the retrieved evidence matches the innovation.

    Returns float in [0, 1]. Uses keyword overlap between innovation query
    and retrieved paper summaries.
    """
    if not evidence:
        return 0.0

    query = f"{innovation.base_direction} {innovation.operator} {innovation.gap}"
    scores: list[float] = []
    for paper in evidence:
        try:
            context = f"{paper.title} {paper.summary}"
            score = max(
                _keyword_overlap(query, context),
                _hybrid_similarity(query, context),
            )
            scores.append(score)
        except Exception as exc:
            logger.warning("Failed to score evidence paper %s: %s", paper.paper_id, exc)

    if not scores:
        return 0.0

    avg_score = sum(scores) / len(scores)
    return round(max(0.0, min(1.0, avg_score)), 4)


def compute_operator_adherence(
    proposal_text: str,
    innovation: Innovation,
) -> float:
    """Score how well the proposal follows the innovation operator.

    Checks if the proposal text semantically reflects the operator by
    looking for operator-associated keywords in the proposal.

    Returns float in [0, 1].
    """
    operator = innovation.operator.lower().strip()
    keywords = _OPERATOR_KEYWORDS.get(operator, [])

    if not keywords:
        # Unknown operator: score based on whether the operator word itself appears
        return 1.0 if operator in proposal_text.lower() else 0.0

    proposal_lower = proposal_text.lower()
    matches = sum(1 for kw in keywords if kw in proposal_lower)
    score = matches / len(keywords)
    return round(max(0.0, min(1.0, score)), 4)


def compute_coherence_score(
    proposal_text: str,
    innovation: Innovation,
) -> float:
    """Score the scientific coherence of the proposal.

    Heuristic: checks proposal length (too short = incoherent),
    presence of technical vocabulary, and overlap with innovation terms.

    Returns float in [0, 1].
    """
    if not proposal_text.strip():
        return 0.0

    # Length score: rewards proposals that are at least _MIN_COHERENT_CHARS long
    length_score = min(1.0, len(proposal_text.strip()) / _MIN_COHERENT_CHARS)

    # Technical vocabulary score: fraction of proposal tokens that are technical
    tokens = set(_tokenize_words(proposal_text))
    if not tokens:
        return 0.0
    tech_count = len(tokens & _TECHNICAL_TOKENS)
    vocab_score = min(1.0, tech_count / max(1, min(5, len(_TECHNICAL_TOKENS))))

    # Innovation overlap score: how much of the innovation terms appear in proposal
    innovation_text = f"{innovation.base_direction} {innovation.gap}"
    overlap = _keyword_overlap(proposal_text, innovation_text)

    score = (0.4 * length_score) + (0.3 * vocab_score) + (0.3 * overlap)
    return round(max(0.0, min(1.0, score)), 4)


def compute_realization_reward(
    proposal_text: str,
    innovation: Innovation,
    evidence: list[PaperRecord],
    config: RealizationConfig,
) -> float:
    """Compute weighted reward for the realization module.

    reward = evidence_accuracy_weight * evidence_accuracy
           + operator_adherence_weight * operator_adherence
           + coherence_weight * coherence_score

    Returns float in [0, 1].
    """
    return evaluate_realization_reward(
        proposal_text,
        innovation,
        evidence,
        config,
    ).total_reward


def evaluate_realization_reward(
    proposal_text: str,
    innovation: Innovation,
    evidence: list[PaperRecord],
    config: RealizationConfig,
) -> RealizationRewardBreakdown:
    """Return the paper-faithful realization reward breakdown.

    This is the frozen reward contract used by both realization training and
    inference-time reward fallback scoring.
    """
    evidence_quality = compute_evidence_accuracy(evidence, innovation)
    operator_adherence = compute_operator_adherence(proposal_text, innovation)
    coherence = compute_coherence_score(proposal_text, innovation)

    reward = (
        config.evidence_accuracy_weight * evidence_quality
        + config.operator_adherence_weight * operator_adherence
        + config.coherence_weight * coherence
    )
    total_reward = round(max(0.0, min(1.0, reward)), 4)
    return RealizationRewardBreakdown(
        evidence_quality=round(evidence_quality, 4),
        operator_adherence=round(operator_adherence, 4),
        coherence=round(coherence, 4),
        total_reward=total_reward,
    )


def build_invalid_trajectory_reward(reason: str) -> StrictTrajectoryRewardBreakdown:
    """Return the frozen invalid-completion reward for strict trajectory scoring."""
    return StrictTrajectoryRewardBreakdown(
        evidence_quality=0.0,
        operator_adherence=0.0,
        proposal_coherence=0.0,
        total_reward=0.0,
        invalid_completion=True,
        invalid_reason=reason,
    )


def evaluate_strict_trajectory_reward(
    trajectory: RealizationTrajectory,
    papers: list[PaperRecord],
    config: RealizationConfig,
) -> StrictTrajectoryRewardBreakdown:
    """Score a strict realization trajectory using only policy-surfaced evidence."""
    if trajectory.invalid_reason:
        return build_invalid_trajectory_reward(str(trajectory.invalid_reason))
    if trajectory.result is None:
        return build_invalid_trajectory_reward("missing_finish_result")

    surfaced_ids = {
        observation.paper_id
        for step in trajectory.steps
        for observation in step.observation
    }
    paper_lookup = {paper.paper_id: paper for paper in papers}
    selected_evidence: list[PaperRecord] = []
    for paper_id in trajectory.result.selected_evidence_ids:
        if paper_id not in surfaced_ids:
            return build_invalid_trajectory_reward("selected_paper_id_not_surfaced")
        paper = paper_lookup.get(paper_id)
        if paper is None:
            return build_invalid_trajectory_reward(
                "selected_paper_id_missing_from_corpus"
            )
        selected_evidence.append(paper)

    proposal_text = trajectory.result.proposal_text.strip()
    if not proposal_text:
        return build_invalid_trajectory_reward("missing_proposal_text")

    evidence_quality = (
        compute_evidence_accuracy(selected_evidence, trajectory.innovation)
        if selected_evidence
        else 0.0
    )
    operator_adherence = compute_operator_adherence(
        proposal_text, trajectory.innovation
    )
    proposal_coherence = compute_coherence_score(proposal_text, trajectory.innovation)
    reward = (
        config.evidence_accuracy_weight * evidence_quality
        + config.operator_adherence_weight * operator_adherence
        + config.coherence_weight * proposal_coherence
    )
    return StrictTrajectoryRewardBreakdown(
        evidence_quality=round(evidence_quality, 4),
        operator_adherence=round(operator_adherence, 4),
        proposal_coherence=round(proposal_coherence, 4),
        total_reward=round(max(0.0, min(1.0, reward)), 4),
    )
