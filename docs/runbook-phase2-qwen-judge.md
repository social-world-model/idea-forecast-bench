# Runbook — re-run Phase 2 with Qwen3.5-9B-Instruct as judge

Purpose: validate that the **soft must_not** judge (Option 2, locked at a
0.2 deduction, applied at most once, floor at 0) holds the M2 AUC ≥ 0.70
gate using the same self-hosted judge the trainer will use. **Numbers
must be reported as-is** — see the locked design memo: this gate is what
distinguishes "judge actually separates emergence from established work"
from "we tuned the scale until the gate passed".

---

## 0. Box prereqs

- 1 × GPU with ≥ 22 GB free (Qwen3.5-9B in bf16 is ~18 GB + KV cache).
- Working dir = repo root.
- the project environment activated (or point `PYTHON_BIN` at its interpreter).
- `Qwen/Qwen3.5-9B` weights either cached at `~/.cache/huggingface/hub/` or downloadable.

## 1. Pull the branch on the new box

```bash
cd /path/to/live-idea-bench
git fetch origin
git checkout feature/foresight-judge-soft-mustnot
git pull origin feature/foresight-judge-soft-mustnot
```

## 2. Launch SGLang serving the judge

```bash
# Foreground — keep an eye on it the first time.
bash scripts/launch_judge_sglang.sh

# Or background it:
mkdir -p logs
nohup bash scripts/launch_judge_sglang.sh > logs/judge_sglang.log 2>&1 &
```

Wait for the line `Uvicorn running on http://127.0.0.1:30000` (≈ 60–120 s
on a warm cache). Sanity-check the endpoint:

```bash
curl -s http://127.0.0.1:30000/v1/models | head -c 400
# Should list "qwen3.5-9b-instruct".
```

## 3. Run Phase-2 with the soft-must_not judge on Qwen

```bash
export JUDGE_BASE_URL=http://127.0.0.1:30000/v1
export JUDGE_MODEL=qwen3.5-9b-instruct
export JUDGE_API_KEY=EMPTY     # SGLang doesn't require auth; the openai SDK still needs a string.

PYTHONPATH=. python examples/forecaster/phase2_rubric_validation.py \
    --mode live \
    --n-topics 5 \
    --n-per-class 8 \
    --judge-base-url "$JUDGE_BASE_URL" \
    --model "$JUDGE_MODEL" \
    --rubrics-dir rubrics_qwen \
    --report reports/rubric_validation_qwen.md \
    --leakage-report reports/leakage_qwen.md \
    2>&1 | tee logs/phase2_qwen.log
```

Estimated cost: 5 × (rubric-gen + 16 judge calls) ≈ 85 inference calls
≈ 2–4 min on H200 / similar.

## 4. Read the result — and report it AS-IS

```bash
cat reports/rubric_validation_qwen.md
cat reports/leakage_qwen.md
ls rubrics_qwen/                       # per-topic rubrics + scored.csv
```

Per-topic score distributions:

```bash
for t in $(ls rubrics_qwen/*.json | xargs -n1 basename | sed 's/\.json$//'); do
  echo "--- $t ---"
  python3 -c "
import csv
pos, neg = [], []
with open('rubrics_qwen/${t}.scored.csv') as fh:
    for r in csv.DictReader(fh):
        (pos if r['label']=='1' else neg).append(float(r['score']))
print(f'pos n={len(pos)} mean={sum(pos)/len(pos):.3f} max={max(pos):.3f}')
print(f'neg n={len(neg)} mean={sum(neg)/len(neg):.3f} max={max(neg):.3f}')
"
done
```

## 5. What "passing" means

- Each topic's AUC ≥ 0.70 with 0 leakage hits → that topic is M2-clean.
- **N/5 passing is the result. Do not soften the judge further to get
  more passes.** That is exactly the failure mode this gate exists to
  catch. If a topic fails, the correct next moves are:
  - inspect `rubrics_qwen/<topic>.scored.csv`,
  - if the failure is *content* (must_not items that overlap with real
    positives, e.g. "must not extend 2D-VLMs" when a real positive
    extends 2D-VLMs), regenerate that topic's rubric with a stronger
    generation prompt OR drop the topic from the M2-clean set,
  - if the failure is *the judge can't distinguish anywhere* (positives
    and negatives both score near 0 or near 1), the judge or the
    negatives need re-thinking, NOT the deduction value.
- Only the topics that pass Qwen-M2 are safe to use in Phase-4 real GRPO
  training. Topics that fail can still be trained on at your own risk
  (their reward signal will be noisier).

## 6. Tear down

```bash
# If you backgrounded SGLang:
pkill -f "sglang.launch_server"
```

---

## Why these flags / why this judge

- The judge is the **base** Qwen3.5-9B Instruct (no LoRA): we evaluate
  separating capability of the JUDGE, not of the policy. Reusing the
  trained policy as the judge would be fitting-the-test-to-the-answer.
- `MUST_NOT_PENALTY = 0.2` is a constant in `forecaster/foresight/judge.py`;
  the prompt enforces "subtract once, floor at 0". Do not tune.
- `JUDGE_BASE_URL` is honored by both `make_live_scorer` (judge) and
  `generate_rubric_via_llm` (rubric author) so the entire M2 stack runs
  on the same model. The legacy gpt-4o path is the fallback when the env
  var is unset.
