# Combinatorial (Ramon-Llull) forecaster: GPU runbook

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

Everything runs on local Qwen3.5-9B replicas (generation and judge). The only
paid call is Voyage embeddings (~$5 for the full sweep).

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
