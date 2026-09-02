#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SOURCES="${SOURCES:-gpt-4.1=output/sweep/judged/*.judged.json}"
EXPECTED_TOPICS="${EXPECTED_TOPICS:-52}"
EXPECTED_WINDOWS="${EXPECTED_WINDOWS:-624}"

set -f  # SOURCES carries globs for main_table.py to expand, not the shell
sources=()
for spec in $SOURCES; do
  sources+=(--source "$spec")
done

python examples/main_table.py \
  "${sources[@]}" \
  --expected-topics "$EXPECTED_TOPICS" \
  --expected-windows "$EXPECTED_WINDOWS" \
  "$@"
