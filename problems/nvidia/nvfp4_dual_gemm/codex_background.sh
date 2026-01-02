#!/usr/bin/env bash
MAX_ITERS=30
COMPLETE="DONE"

for ((i=1; i<=MAX_ITERS; i++)); do
  out="$(
    cat skills/nvfp4-dual-gemm-optimizer/SKILL.md \
        AGENTS.md \
        FLOW.md \
        PROMPT.md \
    | script -q -c codex /dev/null
  )"

  printf '%s\n' "$out"

  if grep -q "$COMPLETE" <<< "$out"; then
    echo "Seen completion phrase, stopping at iteration $i"
    break
  fi
done
