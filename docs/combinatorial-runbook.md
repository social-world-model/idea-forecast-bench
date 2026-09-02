# Combinatorial (Ramon-Llull) forecaster: runbook

The L1 "static rule" experiment: mine theme / domain / method elements from
every pre-cutoff paper, build the community state (decayed element heat plus
co-occurrence preferences), sample five combinations per window under four
sampling rules, realise them into ideas with one LLM call, and score them with
the benchmark's frozen retrieve-then-judge protocol.

The four arms differ **only** in how combinations are sampled; extraction,
realisation prompt and judge are shared:

| strategy name               | sampler                                                |
|-----------------------------|--------------------------------------------------------|
| `combinatorial`             | heat x co-occurrence freshness (rising pairs, unpaired-hot pairs) |
| `combinatorial_frequency`   | heat only                                              |
| `combinatorial_independent` | Llull: each slot drawn by heat independently, pairs ignored |
| `combinatorial_random`      | uniform elements, uniform move                         |

The headline contrast is `combinatorial` minus `combinatorial_independent`:
whether the co-occurrence distribution carries forecasting signal at all.

Generation and judging can both run on local Qwen3.5-9B replicas, or on a
hosted OpenAI-compatible API. Voyage embeddings are needed either way
(~$5 for the full sweep): the judge retrieval is Voyage-only by design.

## Two ways to run

**A. Hosted API, no GPU** (`scripts/run_combinatorial_api.sh`) — jump to the
API section below. This is the cheap way to find out whether the idea works
at all.

**B. Local GPU replicas** (`scripts/run_combinatorial.sh`) — sections 0-5,
for the full 624-window sweep.

---

# API mode (DashScope / no GPU)

DashScope hosts third-party models under `vendor/model` ids, which collide
with Hugging Face ids. Exporting `DASHSCOPE_API_KEY` makes `llm.py` route any
id that is not a local path and not claimed by another provider to the
DashScope OpenAI-compatible endpoint. **Unset that key when you want a local
model**, or the local backend is shadowed.

```bash
export DASHSCOPE_API_KEY=...
export VOYAGE_API_KEY=...          # element merging + the judge retrieval
idea-forecast-bench fetch --from-hf --out-dir data/hf_full/raw_markdown   # once, 1.6 GB
```

### Step 1: generate only, do not score

The cheapest thing that can falsify the approach. Two topics, the last four
cutoffs, three sampler arms; no judge.

```bash
TOPICS=llm_alignment_rlhf,rag_retrieval MIN_CUTOFF_MONTH=2025-03   OUTPUT_DIR=output/pilot JUDGE=0 bash scripts/run_combinatorial_api.sh
```

Then read the output. Look at `output/pilot/logs/extract.log` for the merge
clusters (are the elements real research concepts, or arXiv boilerplate?) and
at a window of ideas (are they specific, or could they describe any paper?).
If either is bad, no amount of judging fixes it — fix the extraction prompt or
`canonicalize.merge_threshold` first.

### Step 2: add scoring

Widen to roughly 100 windows, which is where a paired difference of about
0.15 in Hit@5 separates from noise (standard error near 0.055); at 50 windows
you can only read the direction.

```bash
TOPICS=llm_alignment_rlhf,rag_retrieval,llm_reasoning_math,llm_long_context,\
llm_agents,moe,quantization,image_gen_diffusion \
  OUTPUT_DIR=output/step2 JUDGE=1 bash scripts/run_combinatorial_api.sh
```

Reads out `output/step2/table.txt` plus a per-stage token ledger from
`output/step2/usage.jsonl`, which is what sizes the full run.

### Thinking levels

Set per stage by the script; `DASHSCOPE_THINKING` overrides.

`vanchin/deepseek-v4-pro-0813` enables thinking unless told otherwise, does
not support `thinking_budget`, and promotes `reasoning_effort` low/medium to
high. There is no cheap middle setting: it is off, or unbounded. The router
therefore sends `enable_thinking: false` unless a level is requested, and it
does not forward `seed`, which this model rejects.

| Stage | Level | Why |
|---|---|---|
| Extraction | off | Strict JSON over a short abstract, tens of thousands of calls. Reasoning multiplies billed output tokens and the trace comes back in `reasoning_content`, which the client drops — billed, never seen. |
| Realisation | off, `REALIZE_THINKING=high` to try it | Only a few hundred calls, so it is the one stage cheap enough to experiment with; it applies equally to all arms, so it cannot confound the contrast. Watch for truncated JSON: an unbounded trace can consume the completion budget. |
| Judge | off (forced) | `JUDGE_MAX_TOKENS` is 256; a reasoning trace eats it and every call fails to parse. |
| Specificity | off | The reply is three integers. |

### Token budget and cost

Prompt sizes are measured; paper counts are estimated from the topic
classifier on a sample of the corpus. Prices are the Beijing off-peak tier of
`vanchin/deepseek-v4-pro-0813` (input CNY 4.5/M, output CNY 13.5/M, cache-hit
input CNY 0.45/M; peak is roughly double). TPM is 1,200,000, which is what
sets the wall clock.

| Run | Papers | Windows x arms | Tokens in / out | Cost (off-peak) | Time |
|---|---|---|---|---|---|
| Step 1, no judge | ~1,700 | 36 x 3 | 2.1M / 0.3M | ~CNY 13 | 10-20 min |
| Step 2, judged | ~14,700 | 96 x 3 | 55M / 3M | CNY 120-290 | 1.5-3 h |
| Full sweep | ~45,000 | 624 x 3 | 311M / 17M | CNY 640-1,600 | 6-10 h |

The range on cost is whether the repeated system prompts hit the context
cache. The judge dominates every judged run: 624 windows x 5 predictions x 10
retrieved candidates x 3 arms = 93,600 calls, each resending a 2,246-token
rubric. Scope it first if the budget is tight —
`ARMS="combinatorial combinatorial_independent"` drops a third and still
answers whether co-occurrence carries signal.

Voyage embeddings are billed separately: about USD 3 for step 2, USD 9 for
the full sweep.

---

# GPU mode (local Qwen3.5-9B)

## 0. Environment (once)

```bash
nvidia-smi | head -4          # driver >= 570: pip vLLM works as is. 550-569 usually also works
                              # through CUDA minor-version compatibility; if you see
                              # "CUDA driver version is insufficient", update the driver.
git clone <repo> live-idea-bench && cd live-idea-bench
git checkout feat/combinatorial-l1
conda create -n ifb python=3.11 -y && conda activate ifb
pip install poetry && poetry install            # core deps are enough
pip install -U vllm                             # >= 0.27 has Qwen3.5; wheels are built for CUDA 12.9
python -c "import vllm, torch; print(vllm.__version__, torch.cuda.device_count())"

export VOYAGE_API_KEY=...                       # the only paid service
unset OPENAI_API_KEY TOGETHER_API_KEY DEEPSEEK_API_KEY JUDGE_BASE_URL JUDGE_MODEL OPENAI_BASE_URL VOYAGE_BASE_URL
```

## 1. Corpus (once, ~1.6 GB)

`fetch --from-hf` skips month directories that already exist, and
`data/csml/raw_markdown` in a checkout may hold a small subset. Always fetch
into a fresh directory.

```bash
idea-forecast-bench fetch --from-hf --out-dir data/hf_full/raw_markdown
export INPUT_DIR=data/hf_full/raw_markdown
```

## 2. Serve Qwen3.5-9B, one replica per GPU

```bash
GPUS="0 1 2 3 4 5 6 7 8 9" BASE_PORT=31000 MAX_MODEL_LEN=16384 \
  SERVED_NAMES="gpt-4o-qwen35 qwen35-judge" EXTRA_ARGS="--reasoning-parser qwen3" \
  bash scripts/benchmark/serve_multi.sh Qwen/Qwen3.5-9B
# prints PORTS="31000 31001 ... 31009" when every replica answers /v1/models
```

Two names are served on purpose. Generation only routes to a local server
when the model name starts with `gpt-4o`, `gpt-4.1` or `gpt-5`
(`idea_forecast_bench/llm.py`); the judge takes its own name and turns
thinking off whenever the name contains `qwen`.

Smoke-test the endpoint the way the pipeline calls it (a bare `curl` shows the
thinking trace even when the pipeline is fine):

```bash
OPENAI_BASE_URL=http://127.0.0.1:31000/v1 OPENAI_API_KEY=EMPTY python -c "
from idea_forecast_bench.llm import create_client, get_response_from_llm
c, m = create_client('gpt-4o-qwen35')
print(get_response_from_llm('Return the JSON {\"ok\": 1} and nothing else.', c, m, 'Return only JSON.')[0][:200])"
```

The output must be the JSON, not a reasoning transcript.

If vLLM cannot load the model, `BACKEND=sglang` in `serve_multi.sh` runs the
same layout on SGLang (`pip install "sglang[all]"`).

## 3. Pilot: 4 topics, 48 windows (about 15 minutes)

```bash
PORTS="31000 31001 31002 31003 31004 31005 31006 31007 31008 31009" \
  TOPICS="llm_pretraining,llm_long_context,llm_alignment_rlhf,rag_retrieval" \
  OUTPUT_DIR=output/pilot bash scripts/run_combinatorial.sh
```

Read before going further:

- `output/pilot/logs/extract.log`: `failure_rate` below 0.03 and
  `unknown_move_rate` below 0.10.
- `output/pilot/logs/embed.log`: the 30 largest merge clusters. If unrelated
  labels are merged, raise `canonicalize.merge_threshold` in
  `config/combinatorial.yaml`; if obvious synonyms stay apart, lower it. Then
  re-run the same command (extraction is cached; only merging changes).
- `output/pilot/table.txt`: no `short` windows.
- `output/pilot/judged/*.judged.json`: `judge_parse_failure_rate` near 0. A
  value near 1 means the judge is in thinking mode.
- `output/pilot/specificity.json`: breadth per arm.

## 4. Full sweep: 52 topics, 624 windows (about 1.5-2 hours)

```bash
PORTS="31000 31001 31002 31003 31004 31005 31006 31007 31008 31009" \
  OUTPUT_DIR=output/sweep_combi bash scripts/run_combinatorial.sh
```

Re-running the same command resumes: extraction and judging read their
caches, generation skips finished topics. Outputs:

```
output/sweep_combi/elements/        element cache (records + vectors)
output/sweep_combi/backtest/        predictions per arm and shard
output/sweep_combi/judged/          judge decisions
output/sweep_combi/table.txt        Hit@5 / P@5 / MRR at S>=2 and S>=3
output/sweep_combi/specificity.json outcome-blind breadth per arm
```

Provenance of every prediction (elements, move, evidence paper ids, state
coverage, fallback flag) is in `predictions[].metadata` of the backtest
artifacts and is carried into `per_prediction[].metadata` of the judged ones.

## 5. Tear down

```bash
bash scripts/benchmark/serve_multi.sh stop
```

## Reading the result

Compare the four arms within the table; the backbone and judge are the same
for all of them. `combinatorial - combinatorial_independent` is the
co-occurrence signal, `combinatorial - combinatorial_frequency` is the
contribution of freshness, `combinatorial_random` is the floor. When placing
the numbers next to the paper's Qwen3.5-9B rows, state which judge and which
window set that table used; absolute levels move with both.

## Offline checks (no GPU, no keys)

```bash
idea-forecast-bench extract-elements --selfcheck
idea-forecast-bench extract-elements --dry-run --embed --embed-backend hash \
  --input-dir data/csml/raw_markdown --topics llm_pretraining --cache-dir /tmp/el_fake
idea-forecast-bench benchmark --strategy combinatorial --model-name template \
  --element-cache /tmp/el_fake --topics llm_pretraining --start-month 2024-04 \
  --end-month 2025-09 --min-cutoff-month 2024-07 --skip-matching --output /tmp/combi.json
```
