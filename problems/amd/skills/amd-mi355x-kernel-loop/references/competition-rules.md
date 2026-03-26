# AMD MI355X Competition Rules

This note is the shared legality and remote-eval playbook across `mxfp4-mm`, `moe-mxfp4`, and `mixed-mla`.

## Always Legal

- Return the correct output for the public and live judge contracts.
- Use a single `submission.py` with `custom_kernel(data) -> output_t`.
- Use remote-first evaluation through `popcorn-cli` / kernelbot.
- Replace helper stages with native ownership only when the new path preserves correctness.
- Use staged evidence:
  - `test`
  - `benchmark`
  - `leaderboard` only after a measured win

## Never Legal

- Cross-call caches.
- Global state tricks.
- Replaying intermediate tensors from earlier calls.
- Benchmark-only shortcuts that do not preserve full-call correctness.
- Returning stale buffers or aliasing output illegally.
- Shape-specific cheating against only the visible benchmark cases.
- Hidden “fast path” branches that violate the real task contract.

## Purity Rules

- A kernel may specialize by exact shape, lane, or regime.
- A kernel may precompute inside the call.
- A kernel may allocate temporary tensors inside the call.
- A kernel may not persist those temporaries across calls.
- A kernel may not rely on host-global memoization to skip real work.

## Remote-First Rules

- Do not treat local import or static compile success as proof.
- `test` is the first real gate.
- `benchmark` is the first speed gate.
- `leaderboard` is a separate seeded-distribution gate, not a formality after benchmark.
- If quota is closed, queue the next legal run instead of guessing from local results.

## Spend Rules

- No new remote branch unless the deleted cost center is named up front.
- No branch whose main story is only:
  - cleanup
  - hoist
  - fast path
  - helper swap
  unless it deletes a whole bucket of work.
- For `mxfp4-mm` thin-family `A-pack`, no remote spend unless the candidate states:
  - `quant_dup_upper_bound`
  - `parallelism_floor_ratio`
  - `n_bundle_per_owner`

## Evidence Rules

- Promote measured winners, not theory winners.
- Record strong negatives so other problems do not repeat them.
- Distinguish:
  - correctness failure
  - runtime failure
  - benchmark miss
  - ranked miss

## Transfer Rule

- A legality lesson found on one problem should be reusable on the others when it is general:
  - purity rules
  - remote-eval discipline
  - staged promotion
  - anti-cheating rules
  - “delete a whole bucket” branch gate

- A performance lesson should only transfer when the cost center is actually analogous.
