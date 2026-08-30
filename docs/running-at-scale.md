# Running at scale

Notes from a full sweep — five baselines across three backbones plus MDF, 624
windows each — on two 96GB-GPU hosts against an NFS-mounted corpus. Everything
here cost real time to find. Each entry is a failure that produced no error.

## The failures that look like success

The pipeline degrades rather than stops. Every path below finishes, writes a
well-formed artifact, and reports plausible numbers — while measuring something
other than what was asked for. None of them raise.

### A LoRA adapter records its base model as a Hugging Face id

`adapter_config.json` stores `base_model_name_or_path` as whatever the training
run was given. Train through a model-zoo alias and it records `Qwen/Qwen2.5-7B-Instruct`,
not a path. At inference `forecaster/prior/sampler.py` loads *that*, so passing
`--model-name /abs/path` on the command line changes nothing.

Offline, the id has to resolve through the HF cache. When it does not:

```
WARNING Prior sampling failed (couldn't connect to huggingface.co); using heuristic.
WARNING Prior scorer unavailable (...); falling back to heuristic memory scores.
```

`live_idea_bench/strategy/forecaster.py` catches this, logs a warning, and
substitutes `_build_heuristic_innovations`. The run still emits five predictions
per window, with no exception and no short windows. **A fully degraded MDF row
is indistinguishable from a real one except through `fallback_events`.**

Fix: rewrite `base_model_name_or_path` to an absolute path, which bypasses cache
and revision resolution entirely.

```bash
python - <<'PY'
import json
p = "<checkpoint>/adapter_config.json"
d = json.load(open(p))
d["base_model_name_or_path"] = "/abs/path/to/base"
json.dump(d, open(p, "w"), indent=2)
PY
```

GRPO writes one checkpoint per epoch and each needs the same treatment.

### `HF_HOME` pointed at a synthetic cache hides every other model

Building a cache directory holding symlinks for just the models a vLLM server
needs, then setting `HF_HOME` to it, hides everything else — here
`sentence-transformers/allenai-specter`, which the foresight reward needs for its
retrieval index. Under `HF_HUB_OFFLINE=1` the failure reads
`couldn't connect to huggingface.co`, which points at the network rather than at
the cache.

Worse, a symlinked cache whose `refs/main` is not a real 40-hex commit sha passes
`AutoConfig` and `AutoTokenizer` and then fails on **weights**, because only the
weight load verifies the revision. Partial success is the confusing part.

Point `HF_HOME` at the project's real cache and let absolute paths handle the
rest.

### An unlocked model cache stampedes under `--workers`

`forecaster/prior/sampler.py` cached loaded models in a bare dict with no lock,
while its sibling `forecaster/realization/local_generation.py` uses correct
double-checked locking. With `--workers 8`, eight threads miss the cache
together and each loads a full 7B with `device_map="auto"` — about 120GB against
a 95GB card. The OOM is swallowed by the same `except Exception` as above and
becomes a heuristic fallback.

The signature is a one-to-one match between OOM count and fallback count, per
shard. Fixed in `cf1cc39`.

**A single-window probe cannot catch this** — one window is one thread, and the
cache never stampedes. Probe with `--topics <one topic> --workers 8`: one probe
then covers both configuration (paths, permissions, adapters) and concurrency.

### `models.data[0].id` when an endpoint serves LoRA modules

`forecaster/realization/proposal_generator.py` takes the first model the endpoint
lists. An endpoint serving a base model plus a LoRA module may list the base
first, silently evaluating the untrained model. Merge the adapter and serve one
id; assert on it before launching:

```bash
curl -s :PORT/v1/models | python -c "
import json,sys; d=json.load(sys.stdin)
assert len(d['data']) == 1 and d['data'][0]['id'] == 'EXPECTED', d['data']"
```

### `SGLANG_URL` with a trailing `/v1`

`proposal_generator.py` probes `{url}/v1/models`, so a trailing `/v1` makes it
`/v1/v1/models`. The probe fails and generation falls back to HF `generate` —
about 20 minutes per window instead of 460 seconds, without an error.

### `while read` drops a file's last line

A shard launcher reading topic lists with `while IFS= read -r line; do ... done <
file` silently skips the last shard when the file lacks a trailing newline. Six
topics went unrun; the merged table would have aggregated 46 topics and looked
entirely normal. `wc -l` disagreeing with the real line count is the tell. Use
`done < <(cat file; echo)`.

### `nohup` does not survive a process-group kill

A probe launched as `nohup bash job.sh &` from a tool call that also waited for
its output vanished three minutes in: no traceback, no `Killed`, no OOM line, the
log ending mid-progress-bar and the GPU back to 0 MiB. Nothing distinguished it
from a run that finished without writing its output, which sends you looking at
the write path instead of the process.

The wait loop was what got reaped, and the reap terminates the whole process
group. `nohup` only ignores SIGHUP; it does not stop a SIGTERM addressed to the
group, so the job died with the loop that was watching it. Use `setsid` and do
not wait for a long job in the call that launches it:

```bash
setsid nohup bash job.sh > log 2>&1 < /dev/null &
```

Confirm the detachment rather than assuming it -- the job's `sid` must differ
from the launching shell's `pgid`:

```bash
ps -eo pid,pgid,sid,cmd | grep job.sh
```

To check an already-running job, walk its parents up to `ppid=1`. A wrapper
script sitting in `wait` stays alive for as long as its children do, so "the
session leader still exists" does not by itself mean the job is exposed; what
matters is whether the chain reaches an active tool-call shell or terminates at
init.

## Concurrency: hosted APIs and self-hosted endpoints tune in opposite directions

The parameters have the same names and the correct values differ by more than an
order of magnitude.

Against a hosted API, more concurrency is strictly better. Against a local
endpoint, exceeding its capacity starts a collapse: requests time out, retries
lengthen the queue, more requests time out. Throughput falls while GPU
utilisation stays pinned at 100%, so the GPU reading suggests the opposite of the
truth.

Measured here, judging 9,360 windows across six local 9B endpoints:

| | 3,840 client threads | 240 client threads |
|---|---|---|
| throughput | 11 windows/min | 39 windows/min |
| timeouts | 6,182 | 0 |
| projected | 10.5 h | 2.7 h |

**Sixteen times less concurrency ran three times faster.** GPU utilisation was
100% in both cases. The judge is what to watch instead:

```bash
curl -s :PORT/metrics | grep -E 'num_requests_(running|waiting)'
```

`waiting` climbing above zero, or `request_queue_time_seconds` spreading beyond
the sub-second buckets, means back off.

### The multiplier is not always one layer deep

`benchmark` has a single pool (`run_domain_backtest.py`, `--workers` over
topics). `judge-eval` nests two: `--topic-workers` over topics
(`llm_judge_eval.py`) and `--workers` over candidates within a window
(`judge/windows.py`). Same flag name, and in `judge-eval` the real concurrency is
their product times the process count:

```
60 processes x --workers 16 x --topic-workers 4 = 3,840
```

## Measuring progress

Four separate progress readings were wrong in one session, all from inferring
state out of logs.

| Method | Reported | Actual |
|---|---|---|
| Summing `window N/12` across topics | 100% | 20% |
| Counting unique `(topic, window)` pairs | 2% | 20% |
| Reading a log written before the fix | "fix ineffective" | fix worked |
| Rate from `PREV=0` on the first tick | 390/min | 36/min |
| Max of every `n/12` a log printed | 8 topics at 6-7/12 | 4 topics, counted twice |
| `${g%.*}` on a vLLM counter | 7 windows | 7,307,144 tokens |

The last two are worth separating from the others. `tqdm` reprints a bar in
place, so a log holds many lines per bar and none of them says which topic it
belongs to: taking the last match undercounts, taking the top-N maxima counts one
bar several times, and either way the bar advances when a window *starts*, not
when it finishes. And vLLM's `/metrics` reports counters in scientific notation
(`vllm:generation_tokens_total{...} 7.307144e+06`), which shell arithmetic and
`${var%.*}` both silently mangle -- parse it with
`python3 -c "print(int(float(...)))"`. That one was caught only because 7 was
absurd; a truncation landing on a plausible number would not have been.

Read what the code commits, not what it prints. For `judge-eval` that is
`completed_windows` in the state file:

```bash
python -c "
import glob, json
print(sum(len(json.load(open(f)).get('completed_windows') or [])
          for f in glob.glob('OUT/*.state.json')))"
```

Two things follow. Compare timestamps before trusting a log — a stale file
reports the previous run. And detect stalls explicitly: a finished run and a hung
one both go quiet, so a watcher that only reports progress cannot tell them
apart.

## Corpus loading dominates short runs

`corpus_fingerprint` walks every paper — one `rglob` plus a `stat` each. On a
108k-paper corpus over NFS that measured over 110s, and it runs *before* the
cache lookup it feeds, so a warm cache does not avoid it. Twenty sharded
processes each pay it: a 20-process sweep spent its first quarter-hour computing
one hash twenty times.

`LIVE_IDEA_CORPUS_FINGERPRINT` short-circuits it (110s to 0.000s) and is an
explicit assertion that the corpus is frozen. Derive it once:

```bash
python -c "
from live_idea_bench.papers import corpus_fingerprint
print(corpus_fingerprint('data/csml_v2/raw_markdown'))"
```

Verify it before relying on it, by re-deriving a cache filename it should
produce — a wrong fingerprint misses the cache and re-parses the whole corpus,
which is worse than the walk:

```bash
python -c "
import hashlib
from pathlib import Path
d = str(Path('data/csml_v2/raw_markdown').resolve())
print('raw_' + hashlib.sha256(f'{d}|{start}|{end}|{fp}'.encode()).hexdigest()[:16] + '.pkl')"
```

Note the key includes the month bounds, so `benchmark --start-month X` and a
run without bounds use different cache entries.

## Skipping work the judge redoes

`judge-eval` reads only `cutoff_month`, `cutoff_date`, `predictions` and
`train_papers` from a backtest artifact; it re-embeds the papers and retrieves
candidates itself, and never touches `evaluation` or `matches`. When the judge
supplies the reported numbers, the embedding match inside `benchmark` is
duplicated work — one Voyage call per (prediction, candidate) pair plus a
`SequenceMatcher` prefilter over every future paper.

`--skip-matching` drops it. On this sweep that was roughly 1M Voyage calls and
25 CPU-hours; metrics come out NaN rather than 0.0, and the artifact is stamped
`matching_skipped`.

Paper embeddings are also shareable across judge runs: the state key is
`{embed_fp}__{paper_id}` with no judge component, and the sidecar is merged
unconditionally at load. Copying a completed run's `.state.embeddings.json` into
a new run's sidecar path skips re-embedding entirely, even when the judge model
changes.

## Cheap checks worth running first

- `--topics <one topic> --workers 8` before any full sweep — covers both
  configuration and concurrency, and costs minutes.
- One live judge call, parsed with the real `SCORE_RE`, before a judging run. A
  Qwen judge left in thinking mode spends its 256-token budget on `<think>` and
  fails to parse on every call.
- `models.data[0].id` asserted before generation.
- Shard union checked against the full topic list before launch, by set
  comparison rather than by counting lines.
- `fallback_events` read from the artifact — not the absence of warnings in a
  log — before believing an MDF row.
