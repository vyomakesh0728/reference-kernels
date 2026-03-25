# `mxfp4-mm` Exact-Shape Frontier

## Canon

- Best measured trunk:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_v97/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_v97/submission.py)
- Best measured benchmark pair:
  `25.7488 us`, then `25.8866 us`
- Best ranked anchor:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_three_regime_v76/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_three_regime_v76/submission.py)
- Best ranked score:
  `27.774 us`

## Exact Dispatch

`v97` routes these shapes explicitly:

- `m == 4`
- `m == 8`
- `m == 16`
- `m == 32`
- `m == 64`
- `m == 256`
- separate `other multiples of 32` path behind the exact wide shapes

Treat this as the active `mxfp4-mm` structure. Future branches should be shape-local unless the single hypothesis is dispatch itself.

## Public Benchmark Anchors

- `m4`: `17.4 - 17.7 us`
- `m16`: `40.1 - 40.6 us`
- `m32`: `22.7 / 21.9 us`, then `22.8 / 22.4 us`
- `m64`: `29.6 - 29.9 us`
- `m256`: `27.7 - 28.1 us`

`m8` is visible in the public `test` set, not the public benchmark mix. Keep it test-green and shape-isolated, but do not spend benchmark budget on `m8` before `m4` or `m16` proves the tiny-path prep deletion.

## Current Cost-Center Policy

- Treat prep-only `m16` edits as plateau territory unless a new branch deletes the whole tiny-path `B-scale` materialization bucket.
- Treat prep-only `m32` edits the same way; the next legal `m32` branch is exact-wide `B-pack/repack` deletion, not another micro-cleanup.
- Treat `m64` and `m256` as closed for now: `v97` already compounds their winning `B-pack` deletions, and the refreshed profile does not isolate another legal undeleted bucket there.
- Spend the next budget on `m32`, then `m16`, then `m4` `A-pack` on the new trunk.
- Reopen `m64` or `m256` only after a newer real MI355X profile run names a new whole cost center.

## Branch Gate

Every new branch must have a Candidate Card as defined in
[/Users/v/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-cost-center-gate.md](/Users/v/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-cost-center-gate.md).

Operational rule:

- no new branch unless the deleted cost center is named up front
- if the description sounds like cleanup, hoist, fast path, prep improvement, or a cleaner version of the same path, reject it
- no branch may touch more than one exact shape
- no branch may change prep and scheduling together

## Ranked-Seed Caveat

Leaderboard/ranked is a separate seeded-distribution gate, not a final formality after benchmark.

- `eval.py` mixes public case seeds with `POPCORN_SEED`
- leaderboard correctness reruns also bump case seeds again
- benchmark and ranked can diverge materially even when shapes are identical

Practical rule:

- require two benchmark wins before leaderboard
- require the two wins to agree within about `0.75%`
- do not assume a tiny benchmark edge will survive ranked

## Profiling Lane

Use the new hardware-counter lane only after a benchmark winner or a plateau-signaling branch:

- stage: `profile_rocprof`
- transport: `popcorn-cli --mode profile`
- backend: `POPCORN_PROFILE_BACKEND=rocprofv3`
- artifacts:
  - `profile/profile_summary.json`
  - `profile/candidate_cards.json`
  - `profile/raw/...`
- exact-shape ROCTx ranges are already live in the `v97` trunk:
  - `mxfp4/custom_kernel`
  - `mxfp4/exact_m4`
  - `mxfp4/exact_m8`
  - `mxfp4/exact_m16`
  - `mxfp4/exact_m32`
  - `mxfp4/exact_m64`
  - `mxfp4/exact_m256`
  - plus `b_prep`, `a_pack`, and `kernel_launch` subranges where the Python-side exact path owns those buckets

Operational rule:

- profile one exact-shape winner at a time
- `m8` stays out of the default profile set but can be requested explicitly via `POPCORN_PROFILE_CASES=m8`
- consume the resulting Candidate Cards before opening the next branch
- if the profiler does not identify a whole cost-center deletion, do not open a polish branch anyway
- if kernelbot returns a built-in rocPROF trace zip instead of the custom PMC payload, mine that zip immediately; it is now part of the live profiling workflow

## Active Profile Prior

The current active MI355X profile prior is the compounded `v97` run:

- run dir:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile)
- raw zip:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile/profile_20260325_051711_run0.zip](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile/profile_20260325_051711_run0.zip)
- derived summary:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile/stages/01_profile_rocprof/profile/profile_summary.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile/stages/01_profile_rocprof/profile/profile_summary.json)
- derived cards:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile/stages/01_profile_rocprof/profile/candidate_cards.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile/stages/01_profile_rocprof/profile/candidate_cards.json)

The actionable bucket ratios are:

- `m4`: `a_pack` dominates; raw scale decode is already gone
- `m16`: `a_pack + b_scale_decode` still dominate
- `m32`: `a_pack`, `b_pack`, and kernel are still roughly one-third each
- `m64`: no legal new whole-bucket candidate; `a_pack`, `b_scale_decode`, and kernel split the remaining time
- `m256`: same as `m64`; no legal new whole-bucket candidate after the raw-`b_q` win

Keep the older `v83` zip as historical evidence only:

- run dir:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260324-165909-native-scaled-exact-shape-pyprep-v83-profile-rocprof](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260324-165909-native-scaled-exact-shape-pyprep-v83-profile-rocprof)
- raw zip:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260324-165909-native-scaled-exact-shape-pyprep-v83-profile-rocprof/profile_20260324_170221_run0.zip](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260324-165909-native-scaled-exact-shape-pyprep-v83-profile-rocprof/profile_20260324_170221_run0.zip)
- derived summary:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260324-165909-native-scaled-exact-shape-pyprep-v83-profile-rocprof/stages/01_profile_rocprof/profile/profile_summary.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260324-165909-native-scaled-exact-shape-pyprep-v83-profile-rocprof/stages/01_profile_rocprof/profile/profile_summary.json)
- derived cards:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260324-165909-native-scaled-exact-shape-pyprep-v83-profile-rocprof/stages/01_profile_rocprof/profile/candidate_cards.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260324-165909-native-scaled-exact-shape-pyprep-v83-profile-rocprof/stages/01_profile_rocprof/profile/candidate_cards.json)

The actionable bucket ratios are:

- `m4`: `a_pack + b_scale_decode` dominate; kernel body is small
- `m16`: same; `a_pack + b_scale_decode` dominate
- `m32`: `a_pack`, `b_pack`, and kernel are roughly one-third each
- `m64`: same split, with kernel only slightly larger
- `m256`: also roughly one-third each

## Current Zip-Derived Queue

Read [mxfp4-profile-branch-queue.md](/Users/v/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-profile-branch-queue.md) before opening the next branch.

Current required order:

1. leaderboard `v97` when the time gate reopens
2. `m32`: delete exact-wide `B-pack/repack`
3. `m16`: delete exact tiny-path `B-scale` row-major materialization
4. `m4`: reopen exact `A-pack` only because the refreshed `v97` profile now shows it dominates after the raw-scale win
5. hold `m64` and `m256` closed until a future profile run produces a stronger card

## Allowed Next Branches

- `m32`: reopen only for exact-wide `B-pack` deletion on the `v97` trunk, then exact `A-pack`
- `m16`: reopen only for exact `B-scale` materialization deletion on the `v97` trunk, then exact `A-pack`, then constant-`m16` body slimming
- `m4`: exact `A-pack` is reopened on the `v97` trunk because the refreshed profile shows `a_pack_share=0.708` after the raw-scale win; do not reuse the old `v94` negative as a permanent closure
- `m64`: closed until a future profile run identifies a new legal bucket beyond the already-solved raw-`b_q` deletion
- `m256`: closed until a future profile run identifies a new legal bucket beyond the already-solved raw-`b_q` deletion
- `m8`: keep test-green and isolated; true exact `m8` body before any benchmark budget

## Plateau Update

Recent exact-shape prep-only variants did not create a real wave:

- `v85` (`m16` raw-scale wrapper isolation): effectively flat versus `v83`
- `v86` (`m32` exact B-prep fast path): effectively flat versus `v83`

Operational rule:

- stop spending benchmark slots on prep-only `m16` and prep-only `m32` edits
- move the next serious budget to `m32`, then `m16`, then `m4`
- only reopen `m64` or `m256` when a newer profile run identifies a new legal whole-bucket deletion

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
- `v87` benchmarked cleanly at `26.992 us` with a real workflow URL and improved `m64` to `35.0 us`, but it lost overall versus `v83`. Treat this as evidence that exact `m64` prep specialization is not enough; the next `m64` branch must delete body-level work.
- `v90` was the first exact `m64` body-level deletion branch. Its first benchmark reached `26.829 us` with `m64 34.8 us`, but the rerun fell back to `27.075 us` with `m64 35.9 us`. Treat this bucket as unstable for now and move the next serious slot to `m256`.
- `v89` is the first clean `m256` direct-body candidate because it deletes Python-side exact `m256` materialization and `_MFMA_SCALE_INFLIGHT` retention instead of polishing prep.
- `v91` is the fresh `m256` wrapper-collapse candidate to use for the next real remote slot; it exists only to avoid the poisoned old `v89` ledger/source mapping.
- `v92` is the staged follow-up candidate that keeps the exact `m256` direct-entry path but deletes the generic exact-wide `B` repack bucket by reading raw `b_q` directly in an `m256`-local body.
- `v93` is the first zip-driven whole-bucket winner. It deletes generic exact-wide `B` repack on `m64`, benchmarked at `26.126 us` and `26.162 us`, and pushed `m64` down to `29.6 us` while keeping the other visible shapes in range. Treat `v93` as the new measured trunk and the proof that profile-derived whole-bucket deletions can create a real wave.
- `v91` (`m256` direct-entry/materialization deletion only) benchmarked at `26.884 us`, essentially flat versus the old `v83` frontier. Treat this as evidence that exact `m256` wrapper collapse alone is not enough; the next `m256` branch should delete the exact-wide `B` repack bucket too.
- `v94` (`m4` exact A-pack deletion only) passed remote test but benchmarked at `27.317 us` with `m4 20.4 us`. Treat this as evidence that exact `m4` A-pack deletion by itself is not the next winning tiny-shape bucket.
- `v95` (`m256` raw-`b_q` exact-wide `B` repack deletion on top of `v93`) passed remote test and benchmarked at `26.006 us`, then `26.071 us` on rerun. Treat this as a stable positive result and the new measured trunk over `v93`.
- `v96` (`m4` raw `b_scale_sh` / scale-decode deletion on top of `v93`) passed remote test and benchmarked at `25.928 us`, then `25.830 us` on rerun. Treat this as the new measured trunk and proof that the other half of the tiny-path prep bucket was the right next delete, not exact `A-pack`.
- `v97` compounds the `v95` exact `m256` raw-`b_q` deletion into the `v96` trunk. It passed remote test, benchmarked at `25.7488 us`, then `25.8866 us` on rerun, and is now the measured frontier.
- the refreshed `v97` profile reopens `m32` and `m16` as the next legal whole-bucket lanes, reopens `m4` `A-pack` because scale decode is already gone, and explicitly closes `m64` and `m256` until a future profile isolates a new legal bucket.
