# `mxfp4-mm` Exact-Shape Frontier

## Canon

- Best measured trunk:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_pyprep_v83/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_pyprep_v83/submission.py)
- Best measured rerun:
  `26.791 us`
- Best ranked anchor:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_three_regime_v76/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_three_regime_v76/submission.py)
- Best ranked score:
  `27.774 us`

## Exact Dispatch

`v83` routes these shapes explicitly:

- `m == 4`
- `m == 8`
- `m == 16`
- `m == 32`
- `m == 64`
- `m == 256`
- separate `other multiples of 32` path behind the exact wide shapes

Treat this as the active `mxfp4-mm` structure. Future branches should be shape-local unless the single hypothesis is dispatch itself.

## Public Benchmark Anchors

- `m4`: `18.8 us`
- `m16`: `40.1 us`
- `m32`: `22.6 / 21.8 us`
- `m64`: `35.3 us`
- `m256`: `28.2 us`

`m8` is visible in the public `test` set, not the public benchmark mix. Keep it test-green and shape-isolated, but do not spend benchmark budget on `m8` before `m4` or `m16` proves the tiny-path prep deletion.

## Current Cost-Center Policy

- Treat prep-only `m16` edits as plateau territory unless a new branch deletes a whole wrapper-local or direct-body cost center.
- Treat prep-only `m32` edits the same way; simple exact prep/hoist cleanup is not moving the frontier anymore.
- Spend the next budget on `m64` direct-body/feed work first, then `m4` exact-path fixed-overhead deletion, then `m256` direct-body robustness.
- Reopen `m16` or `m32` only for structural branches, not another round of tiny prep polish.

## Ranked-Seed Caveat

Leaderboard/ranked is a separate seeded-distribution gate, not a final formality after benchmark.

- `eval.py` mixes public case seeds with `POPCORN_SEED`
- leaderboard correctness reruns also bump case seeds again
- benchmark and ranked can diverge materially even when shapes are identical

Practical rule:

- require two benchmark wins before leaderboard
- require the two wins to agree within about `0.75%`
- do not assume a tiny benchmark edge will survive ranked

## Allowed Next Branches

- `m16`: treat prep-only branches as low-priority plateau work after `v85`; only revisit if a branch deletes a full wrapper-local cost center rather than shaving the existing one
- `m32`: treat prep-only branches as low-priority plateau work after `v86`; only revisit if a branch deletes a full exact-path cost center rather than trimming copies/hoists
- `m64`: exact B-prep fast path first, then pointer-hoist and cheaper-zeroing cleanup only inside the exact `m64` body
- `m4`: corrected exact wave64 path plus shape-local prep; keep full-wave B loads and the proven thin semantics
- `m8`: dedicated wrapper now, dedicated exact compute body later
- `m256`: stay on the direct-body line until two non-staged rounds plateau; only then reopen CTA-order, then 4-wave interleave, then 8-wave producer/consumer

## Plateau Update

Recent exact-shape prep-only variants did not create a real wave:

- `v85` (`m16` raw-scale wrapper isolation): effectively flat versus `v83`
- `v86` (`m32` exact B-prep fast path): effectively flat versus `v83`

Operational rule:

- stop spending benchmark slots on prep-only `m16` and prep-only `m32` edits
- move the next serious budget to `m64`, then `m4`, then `m256` direct-body robustness
- only reopen `m16` or `m32` if the new branch removes a whole cost center instead of polishing the same one

## Banned or Deferred Ideas

- smaller BF16-style MFMA shapes for `m<=16`
- broad thin rewrites that touch `m4`, `m8`, and `m16` together
- shared `m32/m64` experiments that retune the generic wide path instead of the exact paths
- broad LDS staging or ping-pong outside a later `m256`-only branch
- leaderboard spends for branches that beat benchmark by noise only

## Recent Signal

- `v85` (`m16` prep-only/raw-scale isolation) was effectively flat. Treat this as evidence that tiny `m16` prep cleanup is not the next needle.
- `v86` (`m32` prep-only exact B-prep cleanup) was also effectively flat. Treat this as evidence that tiny `m32` prep cleanup is not the next needle.
- `v88` proved the corrected exact `m4` path is runtime-correct and benchmark-stable, but it did not beat `v83` overall. The exact `m4` compute path is now a valid base, but it still needs another cost-center deletion to matter on geomean.
- `v87` is test-green and remains the next live `m64` benchmark candidate once quota resets.
