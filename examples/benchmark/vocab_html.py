#!/usr/bin/env python3
"""Render one self-contained HTML review page of the concept vocabulary, so a
human can judge concept granularity across topics, versions, and against the
version ledger. Offline: builds each (topic, version) vocabulary from the
already-extracted+embedded concept store exactly the way ``vocab_build.py``
does (same ``build_vocabulary`` / ``run_checks`` calls), then embeds the
result as one JSON blob rendered by vanilla JS. Makes no LLM or embedding
calls of its own."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
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
from idea_forecast_bench.vocab.checks import CheckResult, run_checks
from idea_forecast_bench.vocab.config import VocabConfig, load_vocab_config
from idea_forecast_bench.vocab.store import ConceptStore
from idea_forecast_bench.vocab.types import Concept, ConceptRecord, Vocabulary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = "output/vocab/cache"
DEFAULT_CONFIGS_DIR = "output/vocab/configs"
DEFAULT_LEDGER = "output/vocab/ledger.md"
DEFAULT_OUTPUT = "output/vocab/vocab_review.html"

#: Tags this experiment stages from $TMPDIR into output/vocab/configs/ the
#: first time they are needed, because the sweep that produces them writes
#: there before the ledger row for the tag exists. Not configurable: it names
#: exactly the two fine-threshold probes this run tracks.
_TMP_FINE_CONFIGS: dict[str, str] = {
    "v2_fine0.85": "vocab_fine0.85.yaml",
    "v2_fine0.80": "vocab_fine0.80.yaml",
}

_DATA_TOKEN = "__VOCAB_HTML_DATA__"


@dataclass(frozen=True)
class VersionSpec:
    tag: str
    fingerprint: str
    config_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", required=True, help="Comma-separated topic ids.")
    parser.add_argument("--cutoff", required=True, help="YYYY-MM cutoff month.")
    parser.add_argument(
        "--versions",
        required=True,
        help="Comma-separated tag=fingerprint:config_path entries.",
    )
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--input-dir", default="data/hf_full/raw_markdown")
    parser.add_argument("--start-month", default="2024-04")
    parser.add_argument("--end-month", default="2025-09")
    parser.add_argument(
        "--embed-model",
        default=None,
        help="Override every version's cfg.cluster.embed_model when locating "
        "its cached vector file.",
    )
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_versions(spec: str) -> list[VersionSpec]:
    out: list[VersionSpec] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        tag, sep, rest = part.partition("=")
        fingerprint, sep2, config_path = rest.partition(":")
        tag, fingerprint, config_path = (
            tag.strip(),
            fingerprint.strip(),
            config_path.strip(),
        )
        if not sep or not sep2 or not tag or not fingerprint or not config_path:
            raise ValueError(
                f"malformed --versions entry {part!r}; want tag=fingerprint:config_path"
            )
        out.append(
            VersionSpec(tag=tag, fingerprint=fingerprint, config_path=config_path)
        )
    return out


def _stage_tmp_fine_configs(configs_dir: Path) -> None:
    """Copy the two known fine-threshold configs out of $TMPDIR the first
    time output/vocab/configs/ does not already have them. Best-effort: a
    missing tmp file just means that probe was not produced on this box."""
    tmp_dir = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    for filename in _TMP_FINE_CONFIGS.values():
        dest = configs_dir / filename
        if dest.exists():
            continue
        src = tmp_dir / filename
        if src.exists():
            shutil.copy2(src, dest)
            print(f"staged {src} -> {dest}", file=sys.stderr)


def _safe_name(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in model)


def _open_store(cache_dir: Path, fingerprint: str, input_dir: Path) -> ConceptStore:
    """Mirrors vocab_build.py's ``_open_store``: warns (not fails) if the
    store was built from a different corpus snapshot than the one loaded
    here. Only called once we already know records.jsonl exists."""
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


def _concept_dict(concept: Concept) -> dict[str, object]:
    variants = [v for v in sorted(concept.variants) if v != concept.label][:4]
    return {
        "id": concept.id,
        "label": concept.label,
        "slot": concept.slot,
        "parent": concept.parent,
        "count": concept.count,
        "doc_frac": concept.doc_frac,
        "first_seen": concept.first_seen,
        "emerging": concept.emerging,
        "variants": variants,
    }


def _concept_tree(vocab: Vocabulary) -> list[dict[str, object]]:
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
                "children": [_concept_dict(c) for c in ranked],
            }
        )
    groups.sort(key=lambda g: (-g["total"], g["parent"]))
    return groups


def _topic_version_payload(
    vocab: Vocabulary,
    checks: CheckResult,
    n_future: int,
    n_future_with_records: int,
) -> dict[str, object]:
    metrics: dict[str, object] = dict(checks.values)
    metrics.update(
        {
            "n_train": vocab.n_train,
            "n_with_records": vocab.n_with_records,
            "combinable_count": len(vocab.combinable()),
            "n_future": n_future,
            "n_future_with_records": n_future_with_records,
        }
    )
    unmapped = checks.details.get("unmapped_future_terms", ())
    growth = checks.details.get("top_post_growth", ())
    return {
        "metrics": metrics,
        "background": [_concept_dict(c) for c in vocab.background()],
        "tree": _concept_tree(vocab),
        "emerging": [
            {
                "label": c.label,
                "slot": c.slot,
                "count": c.count,
                "first_seen": c.first_seen,
                "parent": c.parent,
            }
            for c in vocab.emerging()
        ],
        "slot_conflicts": list(vocab.slot_conflicts),
        "unmapped_terms": [{"text": t, "count": n} for t, n in unmapped],
        "top_post_growth": [
            {"label": label, "pre": pre, "post": post} for label, pre, post in growth
        ],
    }


def _parse_ledger(path: Path) -> dict[str, object]:
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


def _load_topic_version(
    *,
    topic_id: str,
    topic_papers: Sequence[PaperRecord],
    version: VersionSpec,
    cfg: VocabConfig,
    records: Mapping[str, ConceptRecord],
    vectors: Mapping[str, Sequence[float]],
    cutoff: str,
) -> dict[str, object]:
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
    future_records = [records[p.paper_id] for p in future if p.paper_id in records]
    train_records = [records[p.paper_id] for p in train if p.paper_id in records]
    checks = run_checks(
        vocab=vocab,
        train_records=train_records,
        future_records=future_records,
        vectors=vectors,
        cfg=cfg,
    )
    print(
        f"  {topic_id} x {version.tag}: n_train={vocab.n_train} "
        f"n_with_records={vocab.n_with_records} concepts={len(vocab.concepts)} "
        f"combinable={len(vocab.combinable())} n_future={len(future)}",
        flush=True,
    )
    return _topic_version_payload(vocab, checks, len(future), len(future_records))


def main() -> int:
    args = parse_args()
    input_dir = _resolve(args.input_dir)
    cache_dir = _resolve(DEFAULT_CACHE_DIR)
    configs_dir = _resolve(DEFAULT_CONFIGS_DIR)
    configs_dir.mkdir(parents=True, exist_ok=True)
    _stage_tmp_fine_configs(configs_dir)

    topic_ids = [t.strip() for t in args.topics.split(",") if t.strip()]
    if not topic_ids:
        print("--topics must name at least one topic id", file=sys.stderr)
        return 2

    try:
        versions = _parse_versions(args.versions)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not versions:
        print("--versions must name at least one version", file=sys.stderr)
        return 2

    papers, _topics, grouped = load_papers_and_topics(
        input_dir, args.start_month, args.end_month
    )
    unknown = [t for t in topic_ids if t not in grouped]
    if unknown:
        print(f"Unknown topic id(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    kept: list[VersionSpec] = []
    cfg_by_tag: dict[str, VocabConfig] = {}
    for version in versions:
        config_path = _resolve(version.config_path)
        if not config_path.exists():
            print(
                f"WARNING: version {version.tag!r} config not found at "
                f"{config_path}; skipping",
                file=sys.stderr,
            )
            continue
        store_dir = cache_dir / version.fingerprint
        if not (store_dir / "records.jsonl").exists():
            print(
                f"WARNING: version {version.tag!r} store not found at "
                f"{store_dir} (fingerprint {version.fingerprint!r}); skipping",
                file=sys.stderr,
            )
            continue
        cfg_by_tag[version.tag] = load_vocab_config(str(config_path))
        kept.append(version)

    if not kept:
        print("no versions available; nothing to render", file=sys.stderr)
        return 2

    record_cache: dict[str, Mapping[str, ConceptRecord]] = {}
    vector_cache: dict[tuple[str, str], Mapping[str, Sequence[float]]] = {}
    by_topic: dict[str, dict[str, object]] = {}

    for topic_id in topic_ids:
        topic_papers = grouped.get(topic_id, [])
        by_topic[topic_id] = {}
        for version in kept:
            cfg = cfg_by_tag[version.tag]
            if version.fingerprint not in record_cache:
                store = _open_store(cache_dir, version.fingerprint, input_dir)
                record_cache[version.fingerprint] = store.load()
            records = record_cache[version.fingerprint]

            embed_model = args.embed_model or cfg.cluster.embed_model
            vec_key = (version.fingerprint, embed_model)
            if vec_key not in vector_cache:
                vec_path = (
                    cache_dir
                    / version.fingerprint
                    / "vectors"
                    / f"{_safe_name(embed_model)}.json"
                )
                vector_cache[vec_key] = VectorStore(vec_path).view()
            vectors = vector_cache[vec_key]

            by_topic[topic_id][version.tag] = _load_topic_version(
                topic_id=topic_id,
                topic_papers=topic_papers,
                version=version,
                cfg=cfg,
                records=records,
                vectors=vectors,
                cutoff=args.cutoff,
            )

    ledger = _parse_ledger(_resolve(args.ledger))

    data = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "cutoff": args.cutoff,
        "topics": topic_ids,
        "versions": [v.tag for v in kept],
        "version_meta": {
            v.tag: {"fingerprint": v.fingerprint, "config_path": v.config_path}
            for v in kept
        },
        "by_topic": by_topic,
        "ledger": ledger,
    }

    html = _render_page(data)
    output_path = _resolve(args.output)
    atomic_write_text(output_path, html)
    print(f"wrote {output_path} ({output_path.stat().st_size} bytes)", flush=True)
    del papers  # loaded for grouped/cache warm-up only
    return 0


def _render_page(data: Mapping[str, object]) -> str:
    blob = json.dumps(data).replace("</", "<\\/")
    return _TEMPLATE.replace(_DATA_TOKEN, blob)


_TEMPLATE = r"""<title>Concept Vocabulary Review</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#F3F5F7;--surface:#FFFFFF;--ink:#17202A;--muted:#64748B;--line:#D9DEE5;
  --accent:#0F766E;--accent-soft:#CCFBF1;--warn:#B45309;--warn-soft:#FEF3C7;
  --object:#2563EB;--object-soft:#DBEAFE;--mechanism:#7C3AED;--mechanism-soft:#EDE9FE;
  --problem:#DC2626;--problem-soft:#FEE2E2;--shadow:0 1px 2px rgba(23,32,42,.08);
  --font-display:"IBM Plex Sans Condensed",system-ui,-apple-system,"Segoe UI",sans-serif;
  --font-body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --font-mono:"IBM Plex Mono","SFMono-Regular",Consolas,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0F1418;--surface:#161C22;--ink:#E6EAEE;--muted:#94A3B8;--line:#2A333D;
    --accent:#2DD4BF;--accent-soft:rgba(45,212,191,.18);--warn:#F59E0B;
    --warn-soft:rgba(245,158,11,.18);--object:#60A5FA;--object-soft:rgba(96,165,250,.18);
    --mechanism:#A78BFA;--mechanism-soft:rgba(167,139,250,.18);--problem:#F87171;
    --problem-soft:rgba(248,113,113,.18);--shadow:0 1px 2px rgba(0,0,0,.4);
  }
}
:root[data-theme="dark"]{
  --bg:#0F1418;--surface:#161C22;--ink:#E6EAEE;--muted:#94A3B8;--line:#2A333D;
  --accent:#2DD4BF;--accent-soft:rgba(45,212,191,.18);--warn:#F59E0B;
  --warn-soft:rgba(245,158,11,.18);--object:#60A5FA;--object-soft:rgba(96,165,250,.18);
  --mechanism:#A78BFA;--mechanism-soft:rgba(167,139,250,.18);--problem:#F87171;
  --problem-soft:rgba(248,113,113,.18);--shadow:0 1px 2px rgba(0,0,0,.4);
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;font-family:var(--font-body);
  line-height:1.5}
h1,h2,h3{font-family:var(--font-display);font-weight:600;margin:0}
.mono,.num{font-family:var(--font-mono);font-variant-numeric:tabular-nums}
button,input[type=text]{font-family:inherit}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px 56px}
.topbar{position:sticky;top:0;z-index:10;background:var(--surface);
  border-bottom:1px solid var(--line);box-shadow:var(--shadow)}
.topbar-inner{max-width:1180px;margin:0 auto;padding:12px 20px;display:flex;
  flex-wrap:wrap;align-items:center;gap:14px}
.brand{font-size:16px;white-space:nowrap}
.cutoff-tag{margin-left:auto;font-family:var(--font-mono);color:var(--muted);
  font-size:13px;white-space:nowrap}
.seg{display:flex;gap:2px;background:var(--bg);border:1px solid var(--line);
  border-radius:8px;padding:2px}
.seg button{border:none;background:transparent;color:var(--ink);padding:6px 12px;
  border-radius:6px;cursor:pointer;font-size:13px}
.seg button.active{background:var(--accent);color:#fff}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));
  gap:10px;margin:22px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:12px 14px;box-shadow:var(--shadow)}
.tile-label{font-size:12px;color:var(--muted)}
.tile-value{font-size:22px;margin-top:4px}
.tile-baseline{font-size:11px;color:var(--muted);margin-top:4px}
.section{margin:32px 0}
.section h2{font-size:18px;margin-bottom:4px}
.section .hint{color:var(--muted);font-size:13px;max-width:70ch;margin:2px 0 14px}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{display:inline-flex;align-items:baseline;gap:6px;padding:5px 10px;
  border-radius:999px;font-size:13px;background:var(--warn-soft);color:var(--warn)}
.chip-meta{font-family:var(--font-mono);font-size:11px;opacity:.85}
.tree-controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin-bottom:14px}
.tree-controls input[type=text]{padding:7px 10px;border:1px solid var(--line);
  border-radius:8px;background:var(--surface);color:var(--ink);font-size:13px;
  min-width:220px}
.tree-controls label{font-size:13px;color:var(--muted);display:flex;
  align-items:center;gap:6px}
.tree-count{font-size:12px;color:var(--muted);margin-left:auto}
.tree-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));
  gap:14px}
.parent-panel{background:var(--surface);border:1px solid var(--line);
  border-radius:10px;padding:12px 14px;box-shadow:var(--shadow)}
.parent-header{display:flex;justify-content:space-between;align-items:baseline;
  gap:8px;border-bottom:1px solid var(--line);padding-bottom:8px;margin-bottom:8px}
.parent-label{font-weight:600;font-size:14px}
.parent-meta{font-size:12px;color:var(--muted);font-family:var(--font-mono);
  white-space:nowrap}
.child-row{padding:6px 0;border-bottom:1px dashed var(--line)}
.child-row:last-child{border-bottom:none}
.child-top{display:flex;align-items:center;gap:8px}
.slot-chip{font-size:10px;padding:2px 7px;border-radius:999px;white-space:nowrap}
.slot-object{background:var(--object-soft);color:var(--object)}
.slot-mechanism{background:var(--mechanism-soft);color:var(--mechanism)}
.slot-problem{background:var(--problem-soft);color:var(--problem)}
.child-label{flex:1;font-size:13px;min-width:0;overflow-wrap:anywhere}
.child-count{font-size:12px;color:var(--muted);width:2.5em;text-align:right}
.emerging-badge{font-size:11px;color:var(--accent);white-space:nowrap}
.emerging-badge::before{content:"\25cf ";font-size:8px}
.df-bar-track{height:4px;background:var(--line);border-radius:2px;margin-top:5px;
  overflow:hidden}
.df-bar-fill{height:100%;border-radius:2px;background:var(--muted)}
.df-bar-fill.slot-object{background:var(--object)}
.df-bar-fill.slot-mechanism{background:var(--mechanism)}
.df-bar-fill.slot-problem{background:var(--problem)}
.variants{font-size:11px;color:var(--muted);margin-top:3px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--surface)}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);
  white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;
  letter-spacing:.02em;position:sticky;top:0;background:var(--surface)}
tbody tr:hover{background:var(--bg)}
.empty-hint{color:var(--muted);font-size:13px}
details.collapsible{border:1px solid var(--line);border-radius:10px;
  background:var(--surface);padding:10px 14px;margin:14px 0}
details.collapsible summary{cursor:pointer;font-weight:600;font-size:14px;
  list-style:revert}
details.collapsible[open] summary{margin-bottom:10px}
ul.term-list{margin:0;padding-left:20px;columns:2;column-gap:24px;font-size:13px}
ul.term-list li{break-inside:avoid;margin-bottom:2px}
</style>
<div class="topbar">
  <div class="topbar-inner">
    <h1 class="brand">概念词表评审</h1>
    <div class="seg" id="topicBar"></div>
    <div class="seg" id="versionBar"></div>
    <span class="cutoff-tag mono" id="cutoffLabel"></span>
  </div>
</div>
<div class="wrap">
  <div class="metrics" id="metricsStrip"></div>

  <div class="section">
    <h2>背景词（已排除） <span class="hint" id="bgCount"></span></h2>
    <p class="hint">出现在训练语料中 &ge; background_doc_frac 的词——是主题本身，不是可组合的预测素材，仍保留用于核对。</p>
    <div class="chips" id="bgChips"></div>
  </div>

  <div class="section">
    <h2>概念树</h2>
    <p class="hint">按父概念分组的可组合概念（对象 / 机制 / 问题）。条形长度是相对页面内最大 doc_frac 的占比。</p>
    <div class="tree-controls">
      <input type="text" id="treeSearch" placeholder="按子概念标签过滤…">
      <label><input type="checkbox" id="treeOnly3"> 只看 &ge;3 篇</label>
      <span class="tree-count mono" id="treeCount"></span>
    </div>
    <div class="tree-grid" id="treeGrid"></div>
  </div>

  <div class="section">
    <h2>新兴概念</h2>
    <p class="hint">cutoff 前 emerging_months 个月内首次出现的概念，即使只出现一次也保留。</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>label</th><th>slot</th><th>count</th><th>first_seen</th><th>parent</th></tr></thead>
        <tbody id="emergingBody"></tbody>
      </table>
    </div>
  </div>

  <details class="collapsible">
    <summary>未匹配的未来 term（前 40）</summary>
    <ul class="term-list" id="unmappedList"></ul>
  </details>

  <details class="collapsible">
    <summary>槽位冲突（术语在训练集中跨槽位摇摆）</summary>
    <ul class="term-list" id="conflictList"></ul>
  </details>

  <details class="collapsible">
    <summary>Post-cutoff 增长 Top（post/pre 比例最高的概念）</summary>
    <ul class="term-list" id="growthList"></ul>
  </details>

  <details class="collapsible">
    <summary>版本台账</summary>
    <div class="table-wrap"><table id="ledgerTable"></table></div>
  </details>
</div>
<script>
const DATA = __VOCAB_HTML_DATA__;
const METRIC_TILES = [
  {key:"n_train", label:"训练论文数", ledgerCol:null, fmt:"int"},
  {key:"combinable_count", label:"概念数（可组合）", ledgerCol:null, fmt:"int"},
  {key:"coverage_object", label:"覆盖度-对象", ledgerCol:"cov_obj", fmt:"num"},
  {key:"coverage_mechanism", label:"覆盖度-机制", ledgerCol:"cov_mech", fmt:"num"},
  {key:"coverage_both", label:"覆盖度-两者", ledgerCol:"cov_both", fmt:"num"},
  {key:"spearman_pre_post", label:"秩相关", ledgerCol:"spearman", fmt:"num"},
  {key:"mid_layer_share", label:"中间层占比", ledgerCol:"mid_layer", fmt:"num"},
  {key:"background_count", label:"背景词数", ledgerCol:"bg_n", fmt:"int"},
  {key:"emerging_count", label:"新兴概念数", ledgerCol:"emerg_n", fmt:"int"},
];
const SLOT_LABEL = {object:"对象", mechanism:"机制", problem:"问题"};

const state = {topic: DATA.topics[0], version: DATA.versions[0]};

function esc(s){
  return String(s).replace(/[&<>"']/g, ch => (
    {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]
  ));
}
function fmtVal(v, kind){
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  return kind === "int" ? Math.round(v).toString() : Number(v).toFixed(3);
}
function fmtPct(v){
  return (v === null || v === undefined || Number.isNaN(v)) ? "n/a" : (v*100).toFixed(1)+"%";
}
function currentPayload(){
  return DATA.by_topic[state.topic][state.version];
}
function el(id){ return document.getElementById(id); }

function renderSwitchers(){
  el("topicBar").innerHTML = DATA.topics.map(t =>
    `<button data-topic="${esc(t)}" class="${t===state.topic?"active":""}">${esc(t)}</button>`
  ).join("");
  el("versionBar").innerHTML = DATA.versions.map(v =>
    `<button data-version="${esc(v)}" class="${v===state.version?"active":""}">${esc(v)}</button>`
  ).join("");
  el("cutoffLabel").textContent = `cutoff: ${DATA.cutoff}`;
}

function renderMetrics(){
  const payload = currentPayload();
  const v1 = (DATA.ledger.rows||[]).find(r => r.tag === "v1");
  el("metricsStrip").innerHTML = METRIC_TILES.map(t => {
    const val = payload.metrics[t.key];
    let baseline = "";
    if (t.ledgerCol && v1 && v1[t.ledgerCol] !== undefined && v1[t.ledgerCol] !== "") {
      baseline = `<div class="tile-baseline">v1: ${esc(v1[t.ledgerCol])}</div>`;
    }
    return `<div class="tile"><div class="tile-label">${t.label}</div>` +
      `<div class="tile-value mono">${fmtVal(val, t.fmt)}</div>${baseline}</div>`;
  }).join("");
}

function renderBackground(){
  const bg = currentPayload().background;
  el("bgCount").textContent = `${bg.length} 个`;
  el("bgChips").innerHTML = bg.length ? bg.map(c =>
    `<span class="chip" title="${esc(c.slot)} · first seen ${esc(c.first_seen)}">` +
    `${esc(c.label)} <span class="chip-meta">${c.count} · ${fmtPct(c.doc_frac)}</span></span>`
  ).join("") : '<p class="empty-hint">（无）</p>';
}

function childRow(c, maxDf){
  const width = maxDf > 0 ? Math.max(2, (c.doc_frac/maxDf)*100) : 0;
  const variants = c.variants.length ?
    `<div class="variants">variants: ${esc(c.variants.join(", "))}</div>` : "";
  const badge = c.emerging ? '<span class="emerging-badge">新</span>' : "";
  return `<div class="child-row">
    <div class="child-top">
      <span class="slot-chip slot-${c.slot}">${SLOT_LABEL[c.slot]||c.slot}</span>
      <span class="child-label">${esc(c.label)}</span>
      <span class="child-count mono">${c.count}</span>
      ${badge}
    </div>
    <div class="df-bar-track"><div class="df-bar-fill slot-${c.slot}" style="width:${width}%"></div></div>
    ${variants}
  </div>`;
}

function renderTree(){
  const groups = currentPayload().tree;
  const q = el("treeSearch").value.trim().toLowerCase();
  const only3 = el("treeOnly3").checked;
  const allDf = groups.flatMap(g => g.children.map(c => c.doc_frac));
  const maxDf = allDf.length ? Math.max(...allDf) : 1;
  let shown = 0;
  const html = groups.map(g => {
    const children = g.children.filter(c =>
      (!q || c.label.toLowerCase().includes(q)) && (!only3 || c.count >= 3)
    );
    if (!children.length) return "";
    shown++;
    return `<div class="parent-panel">
      <div class="parent-header">
        <span class="parent-label">${esc(g.parent)}</span>
        <span class="parent-meta">${g.total} 篇 · ${g.child_count} 个子概念</span>
      </div>
      <div class="children">${children.map(c => childRow(c, maxDf)).join("")}</div>
    </div>`;
  }).join("");
  el("treeGrid").innerHTML = html || '<p class="empty-hint">没有匹配的概念。</p>';
  el("treeCount").textContent = `${shown} / ${groups.length} 个父概念`;
}

function renderEmerging(){
  const rows = currentPayload().emerging;
  el("emergingBody").innerHTML = rows.length ? rows.map(c =>
    `<tr><td>${esc(c.label)}</td>` +
    `<td><span class="slot-chip slot-${c.slot}">${SLOT_LABEL[c.slot]||c.slot}</span></td>` +
    `<td class="mono">${c.count}</td><td class="mono">${esc(c.first_seen)}</td>` +
    `<td>${esc(c.parent)}</td></tr>`
  ).join("") : '<tr><td colspan="5" class="empty-hint">（无）</td></tr>';
}

function renderUnmapped(){
  const rows = currentPayload().unmapped_terms;
  el("unmappedList").innerHTML = rows.length ? rows.map(r =>
    `<li>${esc(r.text)} <span class="mono">(${r.count})</span></li>`
  ).join("") : '<li class="empty-hint">（无）</li>';
}

function renderConflicts(){
  const rows = currentPayload().slot_conflicts;
  el("conflictList").innerHTML = rows.length ? rows.map(t =>
    `<li>${esc(t)}</li>`
  ).join("") : '<li class="empty-hint">（无）</li>';
}

function renderGrowth(){
  const rows = currentPayload().top_post_growth;
  el("growthList").innerHTML = rows.length ? rows.map(r =>
    `<li>${esc(r.label)} <span class="mono">(${r.pre} &rarr; ${r.post})</span></li>`
  ).join("") : '<li class="empty-hint">（无）</li>';
}

function renderLedger(){
  const {columns, rows} = DATA.ledger;
  if (!columns.length) {
    el("ledgerTable").outerHTML = '<p class="empty-hint">（未找到台账）</p>';
    return;
  }
  const thead = `<thead><tr>${columns.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows.map(r =>
    `<tr>${columns.map(c => `<td class="mono">${esc(r[c]||"")}</td>`).join("")}</tr>`
  ).join("")}</tbody>`;
  el("ledgerTable").innerHTML = thead + tbody;
}

function renderAll(){
  renderSwitchers();
  renderMetrics();
  renderBackground();
  renderTree();
  renderEmerging();
  renderUnmapped();
  renderConflicts();
  renderGrowth();
}

el("topicBar").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  state.topic = b.dataset.topic;
  renderAll();
});
el("versionBar").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  state.version = b.dataset.version;
  renderAll();
});
el("treeSearch").addEventListener("input", renderTree);
el("treeOnly3").addEventListener("change", renderTree);

renderAll();
renderLedger();
</script>
"""


if __name__ == "__main__":
    raise SystemExit(main())
