#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT="${INPUT:-output/sweep/judged/summary_prompting.s0.judged.json}"
OUTPUT="${OUTPUT:-output/analysis/coauthor.json}"
S2_KEY="${S2_API_KEY:-}"

extra=()
[[ -n "$S2_KEY" ]] && extra+=(--s2-key "$S2_KEY")

python examples/analysis_coauthor.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  "${extra[@]}" \
  "$@"
