#!/usr/bin/env python3
"""Render one self-contained HTML explainer of how the concept vocabulary is
built: definitions first, then the data. For a research lead: unlike
vocab_html.py this has no topic/version switcher, just one build per topic
at the given cutoff (background, tree, emerging) plus a multi-cutoff mean
read from already-written ``<results-dir>/<topic>/<cutoff>.json`` files.
Offline: same build_vocabulary/run_checks calls as vocab_html.py and
vocab_build.py. Sibling imports between examples/ scripts don't work under
``python -m idea_forecast_bench`` (``runpy``), so the helpers shared with
vocab_html.py are copied here rather than imported."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from idea_forecast_bench.atomic import atomic_write_text
from idea_forecast_bench.backtest import split_train_future_by_cutoff
from idea_forecast_bench.combinatorial.embeddings import VectorStore
from idea_forecast_bench.models import PaperRecord
from idea_forecast_bench.paper_cache import load_papers_and_topics
from idea_forecast_bench.papers import corpus_fingerprint, month_start_date
from idea_forecast_bench.vocab.build import build_vocabulary
from idea_forecast_bench.vocab.checks import run_checks, stability
from idea_forecast_bench.vocab.config import VocabConfig, load_vocab_config
from idea_forecast_bench.vocab.store import ConceptStore
from idea_forecast_bench.vocab.types import Concept, ConceptRecord, Vocabulary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = "data/hf_full/raw_markdown"
DEFAULT_CACHE_DIR = "output/vocab/cache"
DEFAULT_LEDGER = "output/vocab/ledger.md"
DEFAULT_OUTPUT = "output/vocab/vocab_explainer.html"
DEFAULT_MAX_PARENTS = 12
DEFAULT_MAX_CHILDREN = 6
DEFAULT_MAX_EMERGING = 15
DEFAULT_MAX_VARIANTS = 3

_DATA_TOKEN = "__VOCAB_EXPLAINER_DATA__"

#: The window-level fields this page means, nan-aware, per topic. Same names
#: vocab_build.py writes (plus ``stability``) to ``<tag>/<topic>/<cutoff>.json``.
MEAN_METRIC_KEYS: tuple[str, ...] = (
    "coverage_both",
    "coverage_object",
    "coverage_mechanism",
    "spearman_pre_post",
    "stability",
    "mid_layer_share",
    "background_count",
    "emerging_multi_count",
    "combinable_count",
)

#: Ledger rows the version-comparison section reads, oldest to newest, each
#: with a one-line takeaway. See docs/vocab-runbook.md.
VERSION_ROWS: tuple[tuple[str, str], ...] = (
    ("v1", "旧词表"),
    ("v2", "细粒度 0.90"),
    ("v2parent", "只用父概念"),
    ("v2_fine0.80_pf", "细粒度 0.80"),
    ("v2_promote3_pf", "混合层 3 篇"),
    ("v2_promote5_pf", "混合层 5 篇，已锁定"),
)
LOCKED_TAG = "v2_promote5_pf"

#: All fixed prose. Kept in one place so the HTML template below stays pure
#: layout/JS and nobody has to hunt for a sentence across two files.
COPY: dict[str, object] = {
    "subtitle": "用来做 idea 预测的概念词表：它是什么、怎么造、怎么判断好坏，以及 20 个 topic 上的结果。",
    "one_minute": [
        "我们把每篇论文拆成三类概念：在什么上做（对象）、怎么做（机制）、为什么做（问题）。",
        "把措辞不同但意思相同的概念合并，太碎的并入它的父概念，太泛的标成背景词排除，最近才出现的标成新兴概念。",
        "词表好不好不靠感觉：用 cutoff 之前的论文造词表，用之后 3 个月的论文考它，四个数字达标才算合格。",
    ],
    "definitions": [
        (
            "对象 / 机制 / 问题",
            "论文的三个要素。对象是研究对象或场景（如 long-context language model inference），机制是技术手段（如 query-aware kv cache selection），问题是要解决的困难（如 kv cache loading overhead）。每篇论文由模型（deepseek-v4-flash）读标题和摘要后给出，要求 2 到 5 个词、不许是形容词、模型名或数据集名。",
        ),
        (
            "父概念",
            "每个概念同时给出它属于的更宽一级的概念，例如 kv cache quantization 的父概念是 quantization。父概念用来把碎片收拢，也是词表的粗粒度层。",
        ),
        (
            "归一化",
            "小写、缩写展开（RLHF 写成全称）、单复数统一，保证同一个词只有一种写法。",
        ),
        (
            "同义合并",
            "每个概念用 voyage-3-large 算成向量，同一类型里余弦相似度达到 0.90 的合成一个概念，出现最多的措辞当名字。单个词的概念要求 0.95，因为 adaptive 和 efficient 这类词天然相近，这是上一版词表出问题的原因。",
        ),
        (
            "混合层（promote_min_count = 5）",
            "合并后仍不足 5 篇论文支持的概念，并入它的父概念，成为父概念的一个变体；5 篇以上的独立成节点。最终词表 = 父概念 + 站得住的子概念。",
        ),
        (
            "背景词",
            "在这个 topic 里超过 20% 的论文都有的概念（例如 moe 里的 mixture-of-experts）。它是 topic 本身，不是预测，从可组合的词表里排除，但保留展示。",
        ),
        (
            "新兴概念",
            "cutoff 前 3 个月内第一次出现的概念，哪怕只有 1 篇也保留并标记，因为新方向一开始就是少数几篇。",
        ),
        (
            "cutoff、训练集、未来集",
            "cutoff 是一个月份。造词表只用这个月之前的论文（训练集）；检查用这个月之后 3 个月的论文（未来集）。词表看不到未来，这是所有数字可信的前提。",
        ),
        (
            "覆盖度",
            "未来集里的论文，它的对象和机制能在词表（背景词除外）里找到对应概念的比例。分别报对象、机制、两者都命中。太低说明词表表达不了未来，接近 1 说明太泛，目标 0.6 到 0.8。匹配先按原文精确匹配，再按向量相似度 0.90，再退回父概念。",
        ),
        (
            "秩相关",
            "概念在 cutoff 前的论文数和 cutoff 后的论文数的 Spearman 相关。它回答「现在热的以后是不是还热」。背景词永远高频，在这一项上没有信息。",
        ),
        (
            "稳定性",
            "相邻两个 cutoff 的词表（背景词除外）的 Jaccard 重叠度。词表不应该每月换掉一半。",
        ),
        (
            "中间层占比",
            "训练集里的概念出现次数中，落在 3 篇以上论文支持的概念里的比例。它衡量词表是不是由「站得住的中间层概念」构成，而不是一次性碎片。",
        ),
        ("形容词占比", "单个词的概念占比，越低越好；上一版词表在主题槽位上是 55%。"),
        (
            "探针集",
            "固定的 30 篇论文（3 个 topic 各 10 篇：3 篇主流、3 篇冷门、2 篇带新概念、2 篇 survey），每一版词表都在这 30 篇上人工看提取结果。",
        ),
        (
            "版本台账",
            "每一版词表一行，记录 prompt 指纹、配置 sha、阈值和全部检查数字，保证任何数字都能追溯到产生它的配置。",
        ),
    ],
    "pipeline_steps": [
        "提取三要素与父概念",
        "归一化",
        "向量合并",
        "混合层折叠 + 背景/新兴标记",
        "四项检查",
    ],
    "pipeline_note": "步骤 1 花钱（每篇约 1,300 token），其余全部离线。",
    "version_reading": [
        "细粒度太碎，父概念偏泛。",
        "混合层在覆盖、稳定、中间层上都超过旧词表，形容词问题消失。",
        "秩相关略低于旧词表，是因为旧词表的这一项被背景词撑高。",
    ],
    "threshold_note": "黄底：覆盖-两者 < 0.5 或 中间层 < 0.35——词表可能还不够用来预测这个 topic。",
    "conclusion": [
        "词表已于 2026-09-04 锁定为 promote5（混合层折叠阈值 5 篇论文）。",
        "下一步：用这份词表构建概念图，做组合层的零成本评测，共五个 arm，包含一个复述基线。",
        "引用图数据留到组合层结果出来之后再爬取。",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", required=True, help="Comma-separated topic ids.")
    parser.add_argument("--cutoff", required=True, help="YYYY-MM cutoff month.")
    parser.add_argument("--store", required=True, help="ConceptStore fingerprint.")
    parser.add_argument("--config", default=None, help="Vocab config YAML path.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--start-month", default="2024-04")
    parser.add_argument("--end-month", default="2025-09")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Root of <tag>/<topic>/<cutoff>.json files averaged for the "
        "20-topic table. A topic with none of its own falls back to the "
        "single cutoff built here.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-parents", type=int, default=DEFAULT_MAX_PARENTS)
    parser.add_argument("--max-children", type=int, default=DEFAULT_MAX_CHILDREN)
    parser.add_argument("--max-emerging", type=int, default=DEFAULT_MAX_EMERGING)
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _safe_name(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in model)


def _open_store(cache_dir: Path, fingerprint: str, input_dir: Path) -> ConceptStore:
    """Copied from vocab_html.py: warns, does not fail, if the store's corpus
    snapshot differs from the one loaded here."""
    store = ConceptStore(cache_dir, fingerprint)
    manifest_path = store.dir / "manifest.json"
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "corpus_fingerprint"
        )
        current = corpus_fingerprint(input_dir)
        if stored and stored != current:
            print(
                f"WARNING: store {fingerprint} built from corpus {stored}, now "
                f"{current}",
                file=sys.stderr,
            )
    return store


def _emerging_dict(concept: Concept) -> dict[str, object]:
    return {
        "label": concept.label,
        "slot": concept.slot,
        "count": concept.count,
        "first_seen": concept.first_seen,
    }


def _concept_dict(concept: Concept, max_variants: int) -> dict[str, object]:
    variants = [v for v in sorted(concept.variants) if v != concept.label]
    return {
        "label": concept.label,
        "slot": concept.slot,
        "count": concept.count,
        "doc_frac": concept.doc_frac,
        "emerging": concept.emerging,
        "variants": variants[:max_variants],
    }


def _concept_tree(
    vocab: Vocabulary, *, max_parents: int, max_children: int, max_variants: int
) -> list[dict[str, object]]:
    by_parent: dict[str, list[Concept]] = {}
    for concept in vocab.combinable():
        by_parent.setdefault(concept.parent, []).append(concept)
    groups: list[dict[str, object]] = []
    for parent, children in by_parent.items():
        ranked = sorted(children, key=lambda c: (-c.count, c.label))
        groups.append(
            {
                "parent": parent,
                "total": sum(c.count for c in children),
                "child_count": len(children),
                "children": [
                    _concept_dict(c, max_variants) for c in ranked[:max_children]
                ],
            }
        )
    groups.sort(key=lambda g: (-g["total"], g["parent"]))
    return groups[:max_parents]


def _parse_ledger(path: Path) -> dict[str, object]:
    """Copied from vocab_html.py's ``_parse_ledger``."""
    if not path.exists():
        return {"columns": [], "rows": []}
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        return {"columns": [], "rows": []}

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    columns = cells(lines[0])
    rows = [
        {col: (values[i] if i < len(values) else "") for i, col in enumerate(columns)}
        for values in (cells(line) for line in lines[2:])
    ]
    return {"columns": columns, "rows": rows}


def _nanmean(values: Sequence[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else math.nan


def _topic_window_metrics(
    results_dir: Path | None, topic_id: str
) -> tuple[dict[str, float], int]:
    """Nan-aware mean of MEAN_METRIC_KEYS over every ``<cutoff>.json`` under
    ``results_dir/topic_id``. ``({}, 0)`` means fall back to the single build."""
    if results_dir is None:
        return {}, 0
    topic_dir = results_dir / topic_id
    json_paths = sorted(topic_dir.glob("*.json")) if topic_dir.is_dir() else []
    if not json_paths:
        return {}, 0
    per_key: dict[str, list[float]] = {key: [] for key in MEAN_METRIC_KEYS}
    for path in json_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in MEAN_METRIC_KEYS:
            value = payload.get(key)
            is_number = isinstance(value, int | float)
            per_key[key].append(float(value) if is_number else math.nan)
    means = {key: _nanmean(vals) for key, vals in per_key.items()}
    return means, len(json_paths)


@dataclass(frozen=True)
class TopicBuild:
    vocab: Vocabulary
    single_window: Mapping[str, float]  # MEAN_METRIC_KEYS from this one cutoff


def _build_topic(
    *,
    topic_id: str,
    topic_papers: Sequence[PaperRecord],
    cfg: VocabConfig,
    records: Mapping[str, ConceptRecord],
    vectors: Mapping[str, Sequence[float]],
    cutoff: str,
) -> TopicBuild:
    train, future, _end_month, _end_date = split_train_future_by_cutoff(
        list(topic_papers),
        cutoff_month=cutoff,
        horizon_months=cfg.checks.horizon_months,
    )
    cutoff_date = month_start_date(cutoff)
    vocab = build_vocabulary(
        topic_id=topic_id,
        cutoff_month=cutoff,
        cutoff_date=cutoff_date,
        train_papers=train,
        records=records,
        vectors=vectors,
        cfg=cfg,
    )
    train_records = [records[p.paper_id] for p in train if p.paper_id in records]
    future_records = [records[p.paper_id] for p in future if p.paper_id in records]
    checks = run_checks(
        vocab=vocab,
        train_records=train_records,
        future_records=future_records,
        vectors=vectors,
        cfg=cfg,
    )
    single_window = {**checks.values, "stability": stability(None, vocab)}
    print(
        f"  {topic_id}: n_train={vocab.n_train} concepts={len(vocab.concepts)} "
        f"combinable={len(vocab.combinable())} n_future={len(future)}",
        flush=True,
    )
    return TopicBuild(vocab=vocab, single_window=single_window)


def _nan_safe(value: object) -> object:
    """Recursively turn float NaN into JSON ``null``: a bare ``NaN`` token
    fails ``JSON.parse`` in the browser, and ``json.dumps`` emits one by default."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Mapping):
        return {k: _nan_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_nan_safe(v) for v in value]
    return value


def main() -> int:
    args = parse_args()
    input_dir = _resolve(args.input_dir)
    cache_dir = _resolve(DEFAULT_CACHE_DIR)
    topic_ids = [t.strip() for t in args.topics.split(",") if t.strip()]
    if not topic_ids:
        print("--topics must name at least one topic id", file=sys.stderr)
        return 2
    cfg = load_vocab_config(args.config)
    if not (cache_dir / args.store / "records.jsonl").exists():
        print(f"store {args.store!r} not found under {cache_dir}", file=sys.stderr)
        return 2

    papers, _topics, grouped = load_papers_and_topics(
        input_dir, args.start_month, args.end_month
    )
    unknown = [t for t in topic_ids if t not in grouped]
    if unknown:
        print(f"Unknown topic id(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    store = _open_store(cache_dir, args.store, input_dir)
    records = store.load()
    embed_model = cfg.cluster.embed_model
    vec_path = cache_dir / args.store / "vectors" / f"{_safe_name(embed_model)}.json"
    vectors = VectorStore(vec_path).view()
    results_dir = _resolve(args.results_dir) if args.results_dir else None

    by_topic: dict[str, object] = {}
    topic_summaries: list[dict[str, object]] = []
    for topic_id in topic_ids:
        build = _build_topic(
            topic_id=topic_id,
            topic_papers=grouped.get(topic_id, []),
            cfg=cfg,
            records=records,
            vectors=vectors,
            cutoff=args.cutoff,
        )
        window_means, n_windows = _topic_window_metrics(results_dir, topic_id)
        source = "results_dir"
        if not window_means:
            window_means = dict(build.single_window)
            n_windows = 1
            source = "single_cutoff"

        by_topic[topic_id] = {
            "background": [
                _concept_dict(c, DEFAULT_MAX_VARIANTS) for c in build.vocab.background()
            ],
            "tree": _concept_tree(
                build.vocab,
                max_parents=args.max_parents,
                max_children=args.max_children,
                max_variants=DEFAULT_MAX_VARIANTS,
            ),
            "emerging": [
                _emerging_dict(c) for c in build.vocab.emerging()[: args.max_emerging]
            ],
        }
        topic_summaries.append(
            {
                "topic_id": topic_id,
                "n_train": build.vocab.n_train,
                "n_windows": n_windows,
                "source": source,
                **{k: window_means.get(k, math.nan) for k in MEAN_METRIC_KEYS},
            }
        )

    ledger = _parse_ledger(_resolve(args.ledger))
    ledger_by_tag = {row.get("tag"): row for row in ledger["rows"]}
    version_rows = [
        {
            "tag": tag,
            "label": label,
            "locked": tag == LOCKED_TAG,
            "row": ledger_by_tag.get(tag),
        }
        for tag, label in VERSION_ROWS
    ]

    data = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "cutoff": args.cutoff,
        "topics": topic_ids,
        "topic_count": len(topic_ids),
        "version_name": f"promote{cfg.tag.promote_min_count}",
        "config_sha": cfg.sha,
        "version_rows": version_rows,
        "topic_summaries": topic_summaries,
        "by_topic": by_topic,
    }

    html = _render_page(data)
    output_path = _resolve(args.output)
    atomic_write_text(output_path, html)
    print(f"wrote {output_path} ({output_path.stat().st_size} bytes)", flush=True)
    del papers  # loaded for grouped/cache warm-up only
    return 0


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _one_minute_html() -> str:
    return "".join(f"<li>{_esc(s)}</li>" for s in COPY["one_minute"])


def _defs_html() -> str:
    return "".join(
        f"<dt>{_esc(term)}</dt><dd>{_esc(desc)}</dd>"
        for term, desc in COPY["definitions"]
    )


def _pipeline_html() -> str:
    steps = COPY["pipeline_steps"]
    parts: list[str] = []
    for i, step in enumerate(steps, start=1):
        parts.append(
            f'<div class="pipe-step"><span class="pipe-num mono">{i}</span>'
            f'<span class="pipe-label">{_esc(step)}</span></div>'
        )
        if i < len(steps):
            parts.append('<div class="pipe-arrow" aria-hidden="true"></div>')
    return "".join(parts)


def _version_reading_html() -> str:
    return " ".join(_esc(s) for s in COPY["version_reading"])


def _conclusion_html() -> str:
    return "".join(f"<li>{_esc(s)}</li>" for s in COPY["conclusion"])


def _render_page(data: Mapping[str, object]) -> str:
    blob = json.dumps(_nan_safe(data)).replace("</", "<\\/")
    page = _TEMPLATE
    page = page.replace("__SUBTITLE__", _esc(str(COPY["subtitle"])))
    page = page.replace("__ONE_MINUTE_HTML__", _one_minute_html())
    page = page.replace("__DEFS_HTML__", _defs_html())
    page = page.replace("__PIPELINE_HTML__", _pipeline_html())
    page = page.replace("__PIPELINE_NOTE__", _esc(str(COPY["pipeline_note"])))
    page = page.replace("__VERSION_READING__", _version_reading_html())
    page = page.replace("__THRESHOLD_NOTE__", _esc(str(COPY["threshold_note"])))
    page = page.replace("__CONCLUSION_HTML__", _conclusion_html())
    return page.replace(_DATA_TOKEN, blob)


_TEMPLATE = r"""<title>概念词表构建说明</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#F5F6F8;--surface:#FFFFFF;--ink:#1B2430;--muted:#5B6B7C;--line:#D8DEE6;
  --accent:#0E7C7B;--accent-soft:#D5F0EE;--warn:#A85A12;--warn-soft:#FBEBDD;
  --object:#2456C9;--object-soft:#DCE6FB;--mechanism:#6D3FC4;--mechanism-soft:#E8E0FA;
  --problem:#C2313A;--problem-soft:#F9DEE0;
  --font-display:"IBM Plex Sans Condensed",system-ui,-apple-system,"Segoe UI",sans-serif;
  --font-body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --font-mono:"IBM Plex Mono","SFMono-Regular",Consolas,monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#10161C;--surface:#171E26;--ink:#E8ECF0;--muted:#9AA8B6;--line:#2B3540;
  --accent:#3FC1BE;--accent-soft:rgba(63,193,190,.16);--warn:#E5A254;--warn-soft:rgba(229,162,84,.16);
  --object:#7FA0F0;--object-soft:rgba(127,160,240,.16);--mechanism:#B79AF0;
  --mechanism-soft:rgba(183,154,240,.16);--problem:#E8757D;--problem-soft:rgba(232,117,125,.16);
} }
:root[data-theme="dark"]{
  --bg:#10161C;--surface:#171E26;--ink:#E8ECF0;--muted:#9AA8B6;--line:#2B3540;
  --accent:#3FC1BE;--accent-soft:rgba(63,193,190,.16);--warn:#E5A254;--warn-soft:rgba(229,162,84,.16);
  --object:#7FA0F0;--object-soft:rgba(127,160,240,.16);--mechanism:#B79AF0;
  --mechanism-soft:rgba(183,154,240,.16);--problem:#E8757D;--problem-soft:rgba(232,117,125,.16);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font-body);line-height:1.5}
h1,h2,h3{font-family:var(--font-display);font-weight:600;margin:0;text-wrap:balance}
.mono,.num{font-family:var(--font-mono);font-variant-numeric:tabular-nums}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.wrap{max-width:1040px;margin:0 auto;padding:40px 20px 64px}
header.top{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:32px}
header.top h1{font-size:28px}
header.top .subtitle{color:var(--muted);font-size:16px;margin-top:8px;max-width:72ch;line-height:1.65}
.meta-line{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--muted);font-size:13px;margin-top:14px}
.meta-line .mono{color:var(--ink)}
section{margin:40px 0}
section h2{font-size:20px;margin-bottom:14px;border-left:4px solid var(--accent);padding-left:10px}
.section-hint{color:var(--muted);font-size:14px;margin:-6px 0 16px;max-width:72ch}
ul.minute-list{margin:0;padding-left:0;list-style:none;display:flex;flex-direction:column;gap:10px;max-width:72ch}
ul.minute-list li{padding-left:26px;position:relative;font-size:16px;line-height:1.65}
ul.minute-list li::before{content:"";position:absolute;left:0;top:9px;width:8px;height:8px;border-radius:50%;background:var(--accent)}
dl.defs{display:grid;grid-template-columns:200px 1fr;column-gap:24px;row-gap:16px;margin:0}
dl.defs dt{font-weight:600;font-size:14px;color:var(--ink)}
dl.defs dd{margin:0;font-size:15px;line-height:1.65;color:var(--ink);max-width:62ch}
@media (max-width:640px){dl.defs{grid-template-columns:1fr}dl.defs dt{margin-top:14px}}
.pipeline{display:flex;flex-wrap:wrap;align-items:stretch;gap:0;margin-top:8px}
.pipe-step{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px;display:flex;flex-direction:column;gap:6px;min-width:120px;flex:1}
.pipe-num{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:50%;background:var(--accent-soft);color:var(--accent);font-size:12px}
.pipe-label{font-size:13px;line-height:1.4}
.pipe-arrow{width:24px;height:2px;background:var(--line);position:relative;align-self:center;flex:0 0 auto;margin:0 2px}
.pipe-arrow::after{content:"";position:absolute;right:-1px;top:50%;transform:translateY(-50%);border-style:solid;border-width:5px 0 5px 7px;border-color:transparent transparent transparent var(--line)}
@media (max-width:680px){
  .pipeline{flex-direction:column}
  .pipe-arrow{width:2px;height:18px;margin:2px 0 2px 20px}
  .pipe-arrow::after{right:auto;left:50%;top:auto;bottom:-1px;transform:translateX(-50%);border-width:7px 5px 0 5px;border-color:var(--line) transparent transparent transparent}
}
.pipeline-note{color:var(--muted);font-size:13px;margin-top:10px}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.02em}
tr.locked td{background:var(--accent-soft)}
.ver-label{font-size:11px;color:var(--muted);white-space:normal}
td.warn-cell{background:var(--warn-soft);color:var(--warn)}
.reading{font-size:14px;color:var(--muted);margin-top:12px;max-width:72ch;line-height:1.65}
.table-footer{font-size:12px;color:var(--muted);margin-top:8px}
.table-summary{font-size:14px;margin-top:12px;line-height:1.7}
details.topic{border:1px solid var(--line);border-radius:10px;background:var(--surface);padding:12px 16px;margin:12px 0}
details.topic summary{cursor:pointer;font-size:15px;list-style:revert}
details.topic summary .topic-nums{color:var(--muted);font-size:13px;margin-left:8px}
details.topic[open] summary{margin-bottom:14px;border-bottom:1px solid var(--line);padding-bottom:10px}
.topic-note{font-size:12px;color:var(--warn);margin:6px 0 12px}
h3.sub{font-size:14px;margin:18px 0 8px}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{display:inline-flex;align-items:baseline;gap:6px;padding:5px 10px;border-radius:999px;font-size:13px;background:var(--warn-soft);color:var(--warn)}
.chip-meta{font-family:var(--font-mono);font-size:11px;opacity:.85}
.tree-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.parent-panel{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.parent-header{display:flex;justify-content:space-between;align-items:baseline;gap:8px;border-bottom:1px solid var(--line);padding-bottom:6px;margin-bottom:6px}
.parent-label{font-weight:600;font-size:13px}
.parent-meta{font-size:11px;color:var(--muted);font-family:var(--font-mono);white-space:nowrap}
.child-row{padding:5px 0;border-bottom:1px dashed var(--line)}
.child-row:last-child{border-bottom:none}
.child-top{display:flex;align-items:center;gap:7px}
.slot-chip{font-size:10px;padding:2px 7px;border-radius:999px;white-space:nowrap}
.slot-object{background:var(--object-soft);color:var(--object)}
.slot-mechanism{background:var(--mechanism-soft);color:var(--mechanism)}
.slot-problem{background:var(--problem-soft);color:var(--problem)}
.child-label{flex:1;font-size:12px;min-width:0;overflow-wrap:anywhere}
.child-label.emerging::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent);margin-right:5px}
.child-count{font-size:11px;color:var(--muted);width:2.4em;text-align:right}
.df-bar-track{height:3px;background:var(--line);border-radius:2px;margin-top:4px;overflow:hidden}
.df-bar-fill{height:100%;border-radius:2px}
.df-bar-fill.slot-object{background:var(--object)}
.df-bar-fill.slot-mechanism{background:var(--mechanism)}
.df-bar-fill.slot-problem{background:var(--problem)}
.variants{font-size:10px;color:var(--muted);margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
ol.conclusion{margin:0;padding-left:20px;max-width:72ch;font-size:16px;line-height:1.65}
ol.conclusion li{margin-bottom:8px}
</style>
<div class="wrap">
<header class="top">
  <h1>概念词表构建说明</h1>
  <p class="subtitle">__SUBTITLE__</p>
  <div class="meta-line mono" id="metaLine"></div>
</header>

<section id="one-minute">
  <h2>一分钟看懂</h2>
  <ul class="minute-list">__ONE_MINUTE_HTML__</ul>
</section>

<section id="defs">
  <h2>定义</h2>
  <dl class="defs">__DEFS_HTML__</dl>
</section>

<section id="pipeline">
  <h2>流程</h2>
  <div class="pipeline">__PIPELINE_HTML__</div>
  <p class="pipeline-note">__PIPELINE_NOTE__</p>
</section>

<section id="version-compare">
  <h2>3 个 topic 上的版本对比</h2>
  <p class="section-hint">llm_long_context / quantization / moe，每行是该版本在 12 个 cutoff 上的均值。</p>
  <div class="table-wrap"><table id="versionTable"></table></div>
  <p class="reading">__VERSION_READING__</p>
</section>

<section id="topic-results">
  <h2>20 个 topic 的结果</h2>
  <p class="section-hint">每个 topic 在全部 cutoff 上的均值；只有单个 cutoff 结果的 topic 行首用 * 标出。</p>
  <div class="table-wrap"><table id="topicTable"></table></div>
  <p class="table-footer">__THRESHOLD_NOTE__</p>
  <p class="table-summary" id="topicSummary"></p>
</section>

<section id="per-topic">
  <h2>逐 topic 词表</h2>
  <div id="topicDetails"></div>
</section>

<section id="conclusion">
  <h2>结论与下一步</h2>
  <ol class="conclusion">__CONCLUSION_HTML__</ol>
</section>
</div>
<script>
const DATA = __VOCAB_EXPLAINER_DATA__;
const SLOT_LABEL = {object:"对象", mechanism:"机制", problem:"问题"};
function esc(s){
  return String(s).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
}
function el(id){ return document.getElementById(id); }
function fmtNum(v, digits){ return (v === null || v === undefined) ? "n/a" : Number(v).toFixed(digits); }
function fmtInt(v){ return (v === null || v === undefined) ? "n/a" : Math.round(v).toString(); }
function fmtPct(v){ return (v === null || v === undefined) ? "n/a" : (v * 100).toFixed(1) + "%"; }

function renderMeta(){
  const date = DATA.generated_at.slice(0, 10);
  const items = [`生成日期 ${esc(date)}`, `cutoff ${esc(DATA.cutoff)}`, `${DATA.topic_count} 个 topic`,
    `版本 ${esc(DATA.version_name)}`, `config sha ${esc(DATA.config_sha)}`];
  el("metaLine").innerHTML = items.map(s => `<span>${s}</span>`).join("");
}

const VERSION_COLS = [
  {key:"cov_both", label:"覆盖-两者"}, {key:"cov_obj", label:"覆盖-对象"}, {key:"cov_mech", label:"覆盖-机制"},
  {key:"spearman", label:"秩相关"}, {key:"stability", label:"稳定性"}, {key:"mid_layer", label:"中间层"},
];
function renderVersionTable(){
  const thead = `<thead><tr><th>版本</th>${VERSION_COLS.map(c => `<th>${c.label}</th>`).join("")}` +
    `<th>形容词(对象/机制)</th></tr></thead>`;
  const body = DATA.version_rows.map(v => {
    const row = v.row || {};
    const cells = VERSION_COLS.map(c => `<td class="mono">${esc(row[c.key] ?? "n/a")}</td>`).join("");
    const adj = `${esc(row.single_tok_obj ?? "n/a")} / ${esc(row.single_tok_mech ?? "n/a")}`;
    return `<tr class="${v.locked ? "locked" : ""}"><td><div class="mono">${esc(v.tag)}</div>` +
      `<div class="ver-label">${esc(v.label)}</div></td>${cells}<td class="mono">${adj}</td></tr>`;
  }).join("");
  el("versionTable").innerHTML = thead + `<tbody>${body}</tbody>`;
}

const TOPIC_COLS = [
  {key:"n_train", label:"训练论文数", int:true}, {key:"combinable_count", label:"可组合概念数", int:true},
  {key:"coverage_both", label:"覆盖-两者", warn:v => v < 0.5}, {key:"coverage_object", label:"覆盖-对象"},
  {key:"coverage_mechanism", label:"覆盖-机制"}, {key:"spearman_pre_post", label:"秩相关"},
  {key:"stability", label:"稳定性"}, {key:"mid_layer_share", label:"中间层", warn:v => v < 0.35},
  {key:"background_count", label:"背景词数", int:true}, {key:"emerging_multi_count", label:"新兴(≥ 2篇)", int:true},
];
function fmtCell(v, col){
  if (v === null || v === undefined) return "n/a";
  return col.int ? Math.round(v).toString() : Number(v).toFixed(3);
}
function renderTopicTable(){
  const thead = `<thead><tr><th>topic</th>${TOPIC_COLS.map(c => `<th>${c.label}</th>`).join("")}</tr></thead>`;
  const body = DATA.topic_summaries.map(t => {
    const mark = t.source === "single_cutoff" ? ' <span title="仅单个 cutoff">*</span>' : "";
    const cells = TOPIC_COLS.map(c => {
      const v = t[c.key];
      const warn = c.warn && v !== null && v !== undefined && c.warn(v);
      return `<td class="mono${warn ? " warn-cell" : ""}">${fmtCell(v, c)}</td>`;
    }).join("");
    return `<tr><td>${esc(t.topic_id)}${mark}</td>${cells}</tr>`;
  }).join("");
  el("topicTable").innerHTML = thead + `<tbody>${body}</tbody>`;
}

function renderTopicSummary(){
  const rows = DATA.topic_summaries;
  const ok = rows.filter(t => (t.coverage_both ?? 0) >= 0.5 && (t.mid_layer_share ?? 0) >= 0.35);
  const withCov = rows.filter(t => t.coverage_both !== null && t.coverage_both !== undefined);
  let extremes = "";
  if (withCov.length){
    const lowest = withCov.reduce((a, b) => (b.coverage_both < a.coverage_both ? b : a));
    const highest = withCov.reduce((a, b) => (b.coverage_both > a.coverage_both ? b : a));
    extremes = `覆盖-两者最低：${esc(lowest.topic_id)}（${fmtNum(lowest.coverage_both, 3)}）；` +
      `最高：${esc(highest.topic_id)}（${fmtNum(highest.coverage_both, 3)}）。`;
  }
  el("topicSummary").innerHTML =
    `达标 topic 数（覆盖-两者 ≥ 0.5 且 中间层 ≥ 0.35）：${ok.length} / ${rows.length}。<br>${extremes}`;
}

function childRow(c, maxDf){
  const width = maxDf > 0 ? Math.max(2, (c.doc_frac / maxDf) * 100) : 0;
  const variants = c.variants.length ? `<div class="variants">变体：${esc(c.variants.join("、"))}</div>` : "";
  return `<div class="child-row"><div class="child-top">` +
    `<span class="slot-chip slot-${c.slot}">${SLOT_LABEL[c.slot] || c.slot}</span>` +
    `<span class="child-label${c.emerging ? " emerging" : ""}">${esc(c.label)}</span>` +
    `<span class="child-count mono">${c.count}</span></div>` +
    `<div class="df-bar-track"><div class="df-bar-fill slot-${c.slot}" style="width:${width}%"></div></div>${variants}</div>`;
}

function topicDetail(t, isFirst){
  const payload = DATA.by_topic[t.topic_id];
  const allDf = payload.tree.flatMap(g => g.children.map(c => c.doc_frac));
  const maxDf = allDf.length ? Math.max(...allDf) : 1;
  const tree = payload.tree.length ? payload.tree.map(g =>
    `<div class="parent-panel"><div class="parent-header">` +
    `<span class="parent-label">${esc(g.parent)}</span>` +
    `<span class="parent-meta">${g.total} 篇 · ${g.child_count} 个子概念</span></div>` +
    `<div>${g.children.map(c => childRow(c, maxDf)).join("")}</div></div>`
  ).join("") : '<p class="section-hint">（无）</p>';
  const bg = payload.background.length ? payload.background.map(c =>
    `<span class="chip">${esc(c.label)} <span class="chip-meta">${c.count} · ${fmtPct(c.doc_frac)}</span></span>`
  ).join("") : '<p class="section-hint">（无）</p>';
  const emergingRows = payload.emerging.length ? payload.emerging.map(c =>
    `<tr><td>${esc(c.label)}</td><td><span class="slot-chip slot-${c.slot}">${SLOT_LABEL[c.slot] || c.slot}</span></td>` +
    `<td class="mono">${c.count}</td><td class="mono">${esc(c.first_seen)}</td></tr>`
  ).join("") : '<tr><td colspan="4" class="section-hint">（无）</td></tr>';
  const note = t.source === "single_cutoff" ?
    '<p class="topic-note">* 该 topic 只有一个 cutoff 的结果，均值仅基于它。</p>' : "";
  const nums = `覆盖-两者 ${fmtNum(t.coverage_both, 3)} · 中间层 ${fmtNum(t.mid_layer_share, 3)} · ` +
    `背景词 ${fmtInt(t.background_count)} · 可组合概念 ${fmtInt(t.combinable_count)}`;
  return `<details class="topic"${isFirst ? " open" : ""}>` +
    `<summary>${esc(t.topic_id)}<span class="topic-nums mono">${nums}</span></summary>${note}` +
    `<h3 class="sub">背景词</h3><div class="chips">${bg}</div>` +
    `<h3 class="sub">概念树</h3><div class="tree-grid">${tree}</div>` +
    `<h3 class="sub">新兴概念</h3><div class="table-wrap"><table><thead><tr>` +
    `<th>label</th><th>slot</th><th>count</th><th>first_seen</th></tr></thead>` +
    `<tbody>${emergingRows}</tbody></table></div></details>`;
}

function renderTopicDetails(){
  el("topicDetails").innerHTML = DATA.topic_summaries.map((t, i) => topicDetail(t, i === 0)).join("");
}

renderMeta();
renderVersionTable();
renderTopicTable();
renderTopicSummary();
renderTopicDetails();
</script>
"""


if __name__ == "__main__":
    raise SystemExit(main())
