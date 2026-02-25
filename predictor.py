from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it",
    "of", "on", "or", "that", "the", "their", "this", "to", "via", "with", "we", "our", "using",
    "based", "towards", "toward", "across", "new", "study", "paper", "method", "methods", "results",
}


@dataclass
class Idea:
    title: str
    rationale: str
    approach: str
    score: float = 0.0


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _top_terms(abstracts: list[str], limit: int = 20) -> list[str]:
    counts: dict[str, int] = {}
    for abstract in abstracts:
        for tok in _tokenize(abstract):
            if len(tok) < 4 or tok in STOPWORDS:
                continue
            counts[tok] = counts.get(tok, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [term for term, _ in ordered[:limit]]


def _jaccard(a: str, b: str) -> float:
    sa = set(_tokenize(a))
    sb = set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _idea_text(idea: Idea) -> str:
    return f"{idea.title} {idea.rationale} {idea.approach}"


def _base_score(idea: Idea, signal_terms: list[str]) -> float:
    idea_tokens = set(_tokenize(_idea_text(idea)))
    if not signal_terms:
        return 0.0
    coverage = sum(1 for t in signal_terms if t in idea_tokens) / len(signal_terms)
    specificity = min(len(idea_tokens) / 30.0, 1.0)
    return 0.75 * coverage + 0.25 * specificity


def _dedup_ideas(candidates: list[Idea], threshold: float = 0.80) -> list[Idea]:
    deduped: list[Idea] = []
    title_keys: set[str] = set()

    for candidate in candidates:
        key = re.sub(r"\s+", " ", candidate.title.lower()).strip()
        if key in title_keys:
            continue
        if any(_jaccard(_idea_text(candidate), _idea_text(kept)) >= threshold for kept in deduped):
            continue
        title_keys.add(key)
        deduped.append(candidate)
    return deduped


def _rank_ideas(candidates: list[Idea], signal_terms: list[str], top_k: int) -> list[Idea]:
    for idea in candidates:
        idea.score = _base_score(idea, signal_terms)

    pool = sorted(candidates, key=lambda i: (-i.score, i.title.lower()))
    selected: list[Idea] = []

    while pool and len(selected) < top_k:
        if not selected:
            selected.append(pool.pop(0))
            continue

        best_idx = 0
        best_mmr = float("-inf")
        for idx, cand in enumerate(pool):
            similarity = max(_jaccard(_idea_text(cand), _idea_text(chosen)) for chosen in selected)
            novelty = 1.0 - similarity
            mmr = 0.65 * cand.score + 0.35 * novelty
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        chosen = pool.pop(best_idx)
        chosen.score = best_mmr
        selected.append(chosen)

    return selected


def _load_prompt_config(config_path: Path) -> dict[str, str]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid prompt config: {config_path}")
    return {
        "system_prompt": str(data.get("system_prompt", "You generate forward-looking research ideas.")),
        "user_template": str(data.get("user_template", "")),
    }


def _heuristic_candidates(
    domain: str,
    horizon: str,
    abstracts: list[str],
    n_candidates: int,
    rng: random.Random,
) -> list[Idea]:
    terms = _top_terms(abstracts, limit=max(10, n_candidates * 3))
    if len(terms) < 3:
        terms.extend(["retrieval", "alignment", "reasoning"])

    templates = [
        "{a}-{b} co-training for robust {domain}",
        "Data-centric {a} pipelines with uncertainty-aware {b}",
        "Continual {a} adaptation for long-horizon {b}",
        "Resource-efficient {a} with sparse {b} routing",
        "Cross-modal {a} planning with verifier-guided {b}",
        "Benchmark-driven {a} generalization beyond static {b}",
    ]

    ideas: list[Idea] = []
    for idx in range(n_candidates * 2):
        a = terms[idx % len(terms)]
        b = terms[(idx * 3 + 1) % len(terms)]
        if a == b:
            continue

        title = templates[idx % len(templates)].format(a=a, b=b, domain=domain)
        rationale = (
            f"Recent abstracts in {domain} repeatedly emphasize {a} and {b}; "
            f"this suggests a likely next stage in {horizon}."
        )
        approach = (
            f"Build a shared training/evaluation pipeline that couples {a} modules with {b} signals, "
            "then run ablations on compute cost, robustness, and transfer." 
        )

        ideas.append(Idea(title=title, rationale=rationale, approach=approach))

    rng.shuffle(ideas)
    return ideas[: n_candidates * 2]


def _llm_candidates(
    domain: str,
    horizon: str,
    abstracts: list[str],
    n_candidates: int,
    model: str,
    prompt_config: dict[str, str],
    deterministic: bool,
) -> list[Idea]:
    if openai is None:
        raise RuntimeError("openai package is not installed")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = openai.OpenAI(api_key=api_key)
    context = "\n\n---\n\n".join(abstracts[:20])
    prompt = prompt_config["user_template"].format(
        domain=domain,
        horizon=horizon,
        n_ideas=n_candidates,
        abstracts=context,
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.0 if deterministic else 0.7,
        messages=[
            {"role": "system", "content": prompt_config["system_prompt"]},
            {"role": "user", "content": prompt},
        ],
    )

    content = (response.choices[0].message.content or "").strip()
    blocks = [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]
    ideas: list[Idea] = []

    for block in blocks:
        title_match = re.search(r"Title:\s*(.+)", block, re.IGNORECASE)
        rationale_match = re.search(r"Rationale:\s*(.+)", block, re.IGNORECASE)
        approach_match = re.search(r"Approach:\s*(.+)", block, re.IGNORECASE)
        if title_match and rationale_match and approach_match:
            ideas.append(
                Idea(
                    title=title_match.group(1).strip(),
                    rationale=rationale_match.group(1).strip(),
                    approach=approach_match.group(1).strip(),
                )
            )
    if not ideas:
        raise RuntimeError("LLM output format could not be parsed into ideas")

    return ideas


def generate_ideas(
    domain: str,
    abstracts: list[str],
    horizon: str = "next 6 months",
    n_ideas: int = 5,
    deterministic: bool = True,
    model: str = "gpt-4o-mini",
    prompt_config_path: str = "prompt/predictor.yaml",
) -> dict[str, Any]:
    seed_src = f"{domain}|{horizon}|{n_ideas}|{'|'.join(abstracts[:8])}"
    seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)

    prompt_config = _load_prompt_config(Path(prompt_config_path))
    signal_terms = _top_terms(abstracts, limit=12)

    candidates: list[Idea]
    try:
        candidates = _llm_candidates(
            domain=domain,
            horizon=horizon,
            abstracts=abstracts,
            n_candidates=max(n_ideas * 2, 8),
            model=model,
            prompt_config=prompt_config,
            deterministic=deterministic,
        )
        generation_mode = "llm"
    except Exception:
        candidates = _heuristic_candidates(
            domain=domain,
            horizon=horizon,
            abstracts=abstracts,
            n_candidates=max(n_ideas * 2, 8),
            rng=rng,
        )
        generation_mode = "heuristic"

    deduped = _dedup_ideas(candidates)
    ranked = _rank_ideas(deduped, signal_terms=signal_terms, top_k=n_ideas)

    return {
        "domain": domain,
        "horizon": horizon,
        "deterministic": deterministic,
        "seed": seed,
        "generation_mode": generation_mode,
        "signal_terms": signal_terms,
        "ideas": [asdict(idea) for idea in ranked],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_prediction_input(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("Prediction input must be a JSON object or list of objects")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Next-stage predictor core")
    parser.add_argument("--input", required=True, help="JSON with domain + abstracts")
    parser.add_argument("--output", required=True, help="Where to write predictions JSON")
    parser.add_argument("--horizon", default="next 6 months")
    parser.add_argument("--n-ideas", type=int, default=5)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--prompt-config", default="prompt/predictor.yaml")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic generation")
    args = parser.parse_args()

    records = _load_prediction_input(Path(args.input))
    outputs: list[dict[str, Any]] = []

    for record in records:
        domain = str(record.get("domain", "unknown domain"))
        abstracts = [str(x) for x in record.get("abstracts", []) if str(x).strip()]
        if not abstracts:
            continue

        result = generate_ideas(
            domain=domain,
            abstracts=abstracts,
            horizon=str(record.get("horizon", args.horizon)),
            n_ideas=int(record.get("n_ideas", args.n_ideas)),
            deterministic=bool(record.get("deterministic", args.deterministic)),
            model=str(record.get("model", args.model)),
            prompt_config_path=str(record.get("prompt_config", args.prompt_config)),
        )
        outputs.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
