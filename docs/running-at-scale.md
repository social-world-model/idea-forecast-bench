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

`idea_forecast_bench/strategy/forecaster.py` catches this, logs a warning, and
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

### An absent field reads as a clean result

`fallback_events` is the only thing separating a real MDF row from a fully
degraded one, so it was checked before trusting the row — at the window level,
where windows carry `cutoff_month`, `evaluation`, `matches`, `predictions`,
`train_papers` and no `fallback_events` at all. `dict.get` returned nothing,
which was read as "no fallbacks recorded".

It is written per prediction, in `metadata` (`forecaster.py:371`). Checking the
wrong level does not error; it silently confirms whatever you hoped.

```python
# Wrong: windows have no such key, so this is always empty.
w.get("fallback_events", [])

# Right, and it fails loudly if the field ever moves:
evs = [e for w in windows for p in w["predictions"]
       for e in p["metadata"]["fallback_events"]]
```

Two kinds are recorded and only one invalidates a row:

| event | meaning |
|---|---|
| `phase="prior"`, `heuristic_innovations` | the trained prior was not used — the row is a heuristic baseline |
| `phase="runtime_boundary"`, `demo_wrapper` | `strict_eval` lacked an artifact and dropped to the flexible runtime |

Corroborate rather than relying on the absence of a warning. A real
`metadata.prior_score` (a conditional logprob) and an operator prefix on
`approach` — `"adapt: ..."`, `"compose: ..."` — both come from prior sampling;
`_build_heuristic_innovations` produces neither.

Note that `prior.yaml` points `memory_path` at `data/memory_inventory.json`,
which is not in the repository. Every run therefore takes the `demo_wrapper`
path, including the ones that produced the published checkpoints. Manufacturing
that snapshot would make a new run diverge from the published setup rather than
match it — the right move is to record the condition, not to fix it.

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

`benchmark` has a single pool (`benchmark.py`, `--workers` over
topics). `judge-eval` nests two: `--topic-workers` over topics
(`judge_eval.py`) and `--workers` over candidates within a window
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

Two later readings were wrong for a different reason: they used a signal that
lags the work rather than one that misreports it.

**Artifact mtime is not liveness.** A sharded `benchmark` run writes its
output at *topic* boundaries, so a shard mid-topic touches nothing. Eight shards
sitting at the same 36-40 minute mtime, with the window count unchanged across
two readings 40 minutes apart, read as a dead run. It was not: the serving
endpoint was still emitting 1,449 tok/s and the window count was 516/624, not the
384 the artifacts showed. Judge liveness from the endpoint's token counter or the
log's mtime; the artifact tells you about the last completed topic, which on a
slow topic is ancient.

**`pgrep -f PATTERN` matches other pgreps carrying the same pattern.** A progress
monitor polling `pgrep -f "benchmark.py.*probe"` every 180s makes the
shard count read 9 for 8 shards, and — worse — can keep a `while pgrep; do` chain
alive after the work has finished, delaying whatever it gates. `pkill -f` is
sharper still: the pattern also matches the *invoking shell's own* command line,
so `pkill -f foo.sh` from a shell whose command mentions `foo.sh` kills that
shell. Both were hit in one session, the second twice.

```bash
# Self-matching: counts pgrep itself and any sibling pgrep.
pgrep -fc "benchmark.py.*probe"

# The bracket matches "run" in a target's cmdline but not the literal
# "[r]un" in a sibling's.
pgrep -fc "[r]un_domain_backtest.py.*probe"
```

Better than either: chain on a marker the work itself prints after `wait`
returns. `pgrep` cannot distinguish *finished* from *all eight crashed*; a
completion marker can only be written by the first.

Finally, a comparison discipline. A distribution summary and a single sample are
not comparable, and mistaking one for the other manufactures findings: a median
`future_papers` of 218 next to one sampled window's 49 looked like two different
quantities being recorded in two places. A paired per-window check across all 624
windows found zero differences — the value ranges from 35 to 1,722, so 49 is
ordinary. Compare like with like, and prefer a keyed join over two summaries.

## Four failures that reported success, in order of how well they hid

Every one of these ran to completion, exited zero, and left artifacts behind.

**1. A context budget that fails every request.** `llm.py` pins
`MAX_NUM_TOKENS = 4096` as the requested output length. `summary_prompting`
prompts measure about 4,097 tokens. Served behind `--max-model-len 8192`, every
call returns

    400 ... maximum context length is 8192 tokens. However, you requested 4096
    output tokens and your prompt contains at least 4097 input tokens, for a
    total of at least 8193 tokens.

one token over. Failures are per-window, so the run finishes normally with
`total_windows=0`. Two machines burned twenty minutes each on this because the
outward signs -- live processes, advancing tqdm, growing log files, a clean exit
-- are identical to a healthy run. Give the endpoint headroom above
`MAX_NUM_TOKENS` (`scripts/serve_vllm.sh` refuses anything under
16384), and check client logs for `maximum context length` rather than
checking that processes are alive.

**2. A launch that dies before the first line of work.** Exporting
`PYTHONPATH=.` while invoking the script by absolute path from another cwd
resolves "." to the wrong directory, and every shard dies instantly with
`ModuleNotFoundError: No module named 'idea_forecast_bench'`. The log is 294
bytes, which reads as "just started". Judge a launch by whether the log contains
the first topic's paper count, not by its size.

**3. Thinking traces in the output.** Qwen3.5 defaults to emitting a reasoning
transcript. `llm.py` disables it -- but only when `OPENAI_BASE_URL` is set,
since that is what marks the request as going to a local vLLM. A client that
misses that variable produces "Titles" that are entire reasoning transcripts.
This is the likely origin of an archived run whose predicted titles had a
median length of 129 characters and a mean of 4,810, with 28.8% over 2,000
characters. Test the endpoint the way the client calls it, not with a bare curl:
a bare curl shows the thinking trace even when the pipeline is fine, and a
pipeline missing the variable looks fine until you read the titles.

**4. Duplicates filling the hole left by missing work.** Two machines split 52
topics. One used the agreed grouping; the other rebuilt its own split and reused
the same shard names. Result: 12 topics run twice, 12 never run. Every
conventional check passed --

    shard files       8            ✅
    window total      312 + 312 = 624 against a target of 624   ✅
    errors            0            ✅
    per-shard topics  13/13/5/4/4/5/4/4, all plausible          ✅

    unique topics     40 of 52     ❌
    unique windows    480 of 624   ❌

because the duplicates exactly offset the gap. This is the most concealed of the
four: the other three leave at least one signal wrong, while here even the
headline total is correct. Gate on **unique `(topic, cutoff)` pairs**, and assert
both that unique topics equals 52 and that unique pairs equals 624 -- the first
alone misses a topic that ran only some of its cutoffs, the second alone is what
gets fooled here. `main-table` performs exactly this check and refuses to
report an incomplete row.

The coordination lesson is narrower and worth stating on its own: **never pass a
group name across a machine boundary.** "Take S0+S1" was received by a peer that
had built its own S0 and S1. Pass the explicit member list, or partition by
artifact filename, which cannot be reinterpreted.
`examples/split_topics.py` is deterministic for a fixed corpus, so two
machines running it on the same corpus get the same split; still pass the list.

## Four more that reported success, from the re-judging round

**5. Judge traffic sent to a generation-only endpoint.** One endpoint was served
with `--served-model-name gpt-4.1-qwen35` alone; the others carried both that and
the judge name. A judging shard pointed at it received

    error: Error code: 404 - The model `qwen35-9b-judge` does not exist

for every pair, and `llm_judge_eval` stored each as a decision with
`problem=method=specificity=0`. The run finished, wrote a complete artifact,
reported 156/156 windows, and produced a plausible-looking hit@5. 44% of one
row's verdicts and 100% of two shards were fabricated zeros. Nothing in the
process count, log growth, exit code or window count showed it.

The signature to grep for is the reasoning text, not the score:

```bash
grep -c "Error code" OUT/*.judged.json
```

A zero score is indistinguishable from a real "no match"; the error string is
the only thing that separates them. Smoke-test the endpoint with the exact model
name the client will request -- `curl /v1/models` proves reachability, not that
the name resolves. `scripts/run_benchmark.sh` runs this grep at
the end of every sweep and warns per file.

Two caches must both be cleared to repair it. Purging `judge_decisions` alone
leaves the window in `completed_windows`, so the rerun skips it and changes
nothing; clearing `completed_windows` alone leaves the errored verdicts cached
and the rerun re-serves them.

**6. `pgrep` matching the shell that WROTE the script.** A chain script waited on
`while pgrep -fc '[l]lm_judge_eval.py'`. The bracket stops it matching itself,
but the parent that created the script via heredoc has the whole script text --
including the literal `judge_eval.py` on the launch line -- in its own
command line. The loop matched its own parent and waited forever while the work
had been finished for twenty minutes. Chain on a marker the work writes, or on a
state-file count; a process-name match has now failed in five distinct ways in
one session.

**7. Seeding an embedding sidecar from the wrong shard.** Paper vectors are keyed
by paper id, so a sidecar copied from a shard covering different topics is
almost entirely useless. The copy loop took whichever file the glob returned
first, and the shards quietly re-embedded: `[in_context_learning] embedding 2089
papers`, billed to Voyage, for vectors that already existed. Index the available
sidecars by their topic SET and copy the exact match.

**8. A harness timeout taking backgrounded children with it.** `nohup ... &`
inside a command that then hits its own timeout can still lose the children when
the parent is torn down -- the launch printed "launched 27 shards" and the
process count was zero a minute later. `setsid nohup ... < /dev/null &` detaches
the process group and survives.

## Stray environment variables redirect the judge

`JUDGE_BASE_URL`, `JUDGE_MODEL`, `OPENAI_BASE_URL` and `VOYAGE_BASE_URL` are all
honoured by `judge-eval` and by the generation client. Any of them left
exported in a shared shell -- a GRPO run sets `JUDGE_BASE_URL` to its local
vLLM, for instance -- silently sends scoring somewhere other than the model you
think you are using. The run finishes, the artifact is complete, the numbers
are plausible, and none of them were judged by gpt-4.1-mini.

Start every judging script with

```bash
unset JUDGE_BASE_URL JUDGE_MODEL JUDGE_API_KEY OPENAI_BASE_URL VOYAGE_BASE_URL
```

and choose the judge by flag (`--judge-model`, `--judge-base-url`), which is
what `scripts/run_benchmark.sh` does.

## The paper cache races when the cache is cold

`cache_key()` covers `(input_dir, start_month, end_month, topics_fp, corpus_fp)`
and nothing about sharding, so every shard of a sharded run computes the *same*
key and targets the same `.pkl`. The write used to be direct:

```python
with open(cache_path, "wb") as handle:     # eight writers, one path
    pickle.dump({...}, handle)
```

Eight interleaved `pickle.dump` calls leave a torn file that `cache_path.exists()`
then finds on the next run, which either raises inside `pickle.load` or loads a
truncated corpus and scores against it without erroring. The second is the
dangerous one: fewer papers, no error, plausible numbers. It is now written to a
pid-suffixed temp and `os.replace`d, which is atomic within a filesystem.

The exposure rule is worth stating separately, because it predicts who gets hit:
**only a cold cache races.** When the `.pkl` already exists every shard short-
circuits on `exists()` and no one enters the write branch. A run reusing the
main sweep's month range is therefore safe, while a run that changes
`--start-month` or `--end-month` changes the key, finds nothing, and puts every
shard into the write path at once. The sweep that exposed this was a
contamination probe over a different month range; the concurrent sweep on the
same machine, using the main range, never wrote at all.

## Corpus loading dominates short runs

`corpus_fingerprint` walks every paper — one `rglob` plus a `stat` each. On a
108k-paper corpus over NFS that measured over 110s, and it runs *before* the
cache lookup it feeds, so a warm cache does not avoid it. Twenty sharded
processes each pay it: a 20-process sweep spent its first quarter-hour computing
one hash twenty times.

`IDEA_FORECAST_CORPUS_FINGERPRINT` short-circuits it (110s to 0.000s) and is an
explicit assertion that the corpus is frozen. Derive it once:

```bash
python -c "
from idea_forecast_bench.papers import corpus_fingerprint
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
  log — before believing an MDF row, and read from `predictions[].metadata`
  rather than from the window, where the key does not exist.

## A note on the shape of these failures

Almost every entry above is the same mistake in a different costume: something
was inferred rather than read, and the inference happened to agree with what was
expected.

A window count summed across topics agreed that the run was finished. A missing
`fallback_events` key agreed that nothing had degraded. An endpoint at 100% GPU
agreed that concurrency was well tuned. A stale log agreed that a fix had not
worked. `except Exception: pass` around a state read agreed that progress had
stalled. In every case the signal was indirect, the reading was wrong, and
nothing raised.

The habit that catches these is cheap: after changing something, look at a
direct observable that must move — `num_requests_running` after adding an
endpoint, a `skipping` line after a resume, a `prior_score` after fixing a
checkpoint path. Not "it should be working now."
