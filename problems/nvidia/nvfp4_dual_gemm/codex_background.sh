#!/usr/bin/env bash
MAX_ITERS=30           # safety hatch like --max-iterations
COMPLETE="DONE"        # must match completion promise in PROMPT.md

for ((i=1; i<=MAX_ITERS; i++)); do
  prompt="$(
    cat skills/nvfp4-dual-gemm-optimizer/SKILL.md \
        AGENTS.md \
        FLOW.md \
        PROMPT.md
  )"

  out="$(codex -c model='"gpt-5.2-codex-xhigh"' -C . -- "$prompt")"

  printf '%s\n' "$out"

  if grep -q "$COMPLETE" <<< "$out"; then
    echo "Seen completion phrase, stopping at iteration $i"
    break
  fi
done
