#!/usr/bin/env python3
"""Assemble the main table from judged artifacts, at two match thresholds.

The two columns are NOT two judge runs. `S>=2` is the judge's own `is_match`,
which applies MATCH_PM_THRESHOLD=5 and MATCH_S_THRESHOLD=2 (judge/config.py).
`S>=3` re-applies `problem + method >= 5 AND specificity >= 3` to the raw
per-dimension scores already stored in `per_prediction` -- it is a recount, not
a re-judge, so it costs nothing and cannot drift from the S>=2 column. To move
the threshold again, change the comparison here; to change what a dimension
means, you have to re-run the judge.

Why the strict column exists: over 90% of matches at S>=2 sit exactly on the
threshold, and the rubric reads S=1 as "generic enough to loosely fit" -- so the
loose column rewards breadth. It does not merely rescale the table. Across all
three backbones, topic_trend and predictor_llm swap rank between the two
columns (topic_trend shrinks 15-20x, summary/memory only 4.7-8.1x), while the
backbone ordering (14B > 7B > gpt-4.1) holds under both. Report the two facts
separately.

Usage: build_main_table.py [mini|q35]
"""
import json, glob, sys, collections, math

O = '/mnt/disk1_from_server1/max7/shared_rw/output'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'mini'
SRC = {
    'mini': [('gpt-4.1',      f'{O}/gpt41_judged/*.judged.json'),
             ('Qwen2.5-7B',   f'{O}/judged/qwen7b.*.judged.json'),
             ('Qwen2.5-14B',  f'{O}/judged/qwen14b.*.judged.json'),
             ('MDF-Qwen2.5-7B', f'{O}/mdf_judged/mdf.*.judged.json')],
    'q35':  [('gpt-4.1',      f'{O}/gpt41_judged_q35/*.judged.json'),
             ('Qwen2.5-7B',   f'{O}/judged_q35/qwen7b.*.judged.json'),
             ('Qwen2.5-14B',  f'{O}/judged_q35/qwen14b.*.judged.json'),
             ('MDF-Qwen2.5-7B', f'{O}/mdf_judged_q35/mdf.*.judged.json')],
}[TAG]
ORDER = ['topic_trend','predictor_llm','summary_prompting','retrieval_prompting','memory_prompting','forecaster']

def strategy_of(fp, d):
    s = d.get('strategy')
    if s: return s
    for k in ORDER:
        if k in fp: return k
    return '?'

rows = collections.defaultdict(lambda: {'w':0,'hit2':0,'hit3':0,'np':0,'fb_prior':0,'short':0})
coverage = collections.defaultdict(set)
for backbone, pat in SRC:
    for fp in sorted(glob.glob(pat)):
        try: d = json.load(open(fp))
        except Exception: continue
        strat = strategy_of(fp, d)
        key = (backbone, strat)
        for tid, tr in d.get('topic_results', {}).items():
            bt = tr.get('backtest')
            if not bt: continue
            for w in bt.get('windows', []):
                rows[key]['w'] += 1
                coverage[key].add((tid, w.get('cutoff_month')))
                pp = w.get('per_prediction', []) or []
                # S>=2 原口径：judge 自己的 is_match
                if any(p.get('is_match') for p in pp): rows[key]['hit2'] += 1
                # S>=3 严格口径：用原始三维分数重算 (P+M>=5 且 S>=3)
                strict = any((p.get('problem_score') or 0) + (p.get('method_score') or 0) >= 5
                             and (p.get('specificity_score') or 0) >= 3 for p in pp)
                if strict: rows[key]['hit3'] += 1
                n = len(pp); rows[key]['np'] += n
                if n < 5: rows[key]['short'] += 1
                for p in pp:
                    for e in ((p.get('metadata') or {}).get('fallback_events') or []):
                        if e.get('phase') == 'prior': rows[key]['fb_prior'] += 1

print(f'===== 主表（judge = {TAG}）=====')
h = f'{"backbone":<17}{"strategy":<21}{"窗":>5}{"hit@k S>=2":>12}{"hit@k S>=3":>12}{"短窗":>6}{"prior降级":>10}'
print(h); print('-'*len(h))
for backbone, _ in SRC:
    for strat in ORDER:
        v = rows.get((backbone, strat))
        if not v or v['w'] == 0: continue
        print(f'{backbone:<17}{strat:<21}{v["w"]:>5}{v["hit2"]/v["w"]:>12.4f}{v["hit3"]/v["w"]:>12.4f}'
              f'{v["short"]:>6}{v["fb_prior"]:>10}')
tot = sum(v['w'] for v in rows.values())
# Coverage assertions. Both are needed and neither implies the other: counting
# unique topics alone passes a row whose topic ran only three of its twelve
# cutoffs, and counting unique (topic, cutoff) pairs alone is filled in by
# duplicates -- two shard sets built from different partitions once summed to
# exactly 624 windows while covering 40 of 52 topics, with file count, window
# total, error count and per-shard topic counts all looking right.
EXPECTED_TOPICS, EXPECTED_WINDOWS = 52, 624
for (backbone, strat), cov in sorted(coverage.items()):
    ts = {t for t, _ in cov}
    if len(ts) != EXPECTED_TOPICS or len(cov) != EXPECTED_WINDOWS:
        print(f'  !! {backbone}/{strat}: {len(ts)}/{EXPECTED_TOPICS} topics, '
              f'{len(cov)}/{EXPECTED_WINDOWS} windows -- incomplete, do not report')

print(f'\n合计 {len(rows)} 行 / {tot} 窗'  + ('  ✅ 16 行齐' if len(rows) == 16 else f'  ⚠️ 期望 16 行'))
