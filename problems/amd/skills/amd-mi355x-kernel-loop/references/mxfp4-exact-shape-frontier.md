# `mxfp4-mm` Exact-Shape Frontier

## Canon

- Best measured trunk:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_v101/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_v101/submission.py)
- Best measured benchmark pair:
  `25.3007 us`, then `25.2188 us`
- Best ranked trunk:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_v101/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_v101/submission.py)
- Best ranked score:
  `26.2218 us`

## Exact Dispatch

`v101` routes these shapes explicitly:

- `m == 4`
- `m == 8`
- `m == 16`
- `m == 32`
- `m == 64`
- `m == 256`
- separate `other multiples of 32` path behind the exact wide shapes

Treat this as the active `mxfp4-mm` structure. Future branches should be shape-local unless the single hypothesis is dispatch itself.

## Public Benchmark Anchors

- `m4`: `17.3 - 17.8 us`
- `m16`: `38.6 - 39.1 us`
- `m32`: `21.5 / 21.6 us`
- `m64`: `29.6 - 29.7 us`
- `m256`: `27.7 - 27.8 us`

`m8` is visible in the public `test` set, not the public benchmark mix. Keep it test-green and shape-isolated, but do not spend benchmark budget on `m8` before `m4` or `m16` proves the tiny-path prep deletion.

## Current Cost-Center Policy

- Treat prep-only `m16` edits as plateau territory now that raw `b_scale_sh` is already resolved positive; the next legal `m16` lane is `A-pack` launch annihilation.
- Treat prep-only `m32` edits the same way; `v98` already resolved exact-wide `B-pack/repack` positively.
- Treat `m64` and `m256` as closed for now: `v101` already compounds their winning `B-pack` deletions, and the refreshed profile does not isolate another legal undeleted bucket there.
- Spend the next budget on `m16`/`m4` `A-pack` launch annihilation only if the design preserves amortization without shared-panel sweep serialization; otherwise move straight to a family-wide `A-pack` redesign.
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

The current active MI355X profile prior is the compounded `v101` run:

- run dir:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile)
- raw zip:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/profile_20260325_111743_run0.zip](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/profile_20260325_111743_run0.zip)
- derived summary:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/stages/01_profile_rocprof/profile/profile_summary.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/stages/01_profile_rocprof/profile/profile_summary.json)
- derived cards:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/stages/01_profile_rocprof/profile/candidate_cards.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/stages/01_profile_rocprof/profile/candidate_cards.json)

The actionable bucket ratios are:

- `m4`: `a_pack` dominates at about `71.4%`; raw scale decode is already gone
- `m16`: `a_pack` dominates at about `72.8%`; `b_scale_decode_share = 0`
- `m32`: `a_pack`, `b_scale_decode`, and kernel are still roughly one-third each
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

1. keep `v101` as the active measured and ranked trunk
2. `m16`: delete the separate exact `A-pack` launch only with a producer/consumer or other amortized ownership model; do not inline full quantization work into the MFMA threads and do not use shared-panel sweep kernels as direct replacements
3. `m4`: reopen only for the stricter “delete the launch entirely” form of `A-pack` annihilation
4. plan the next wide pass as a family-wide `A-pack` collapse for `m32/m64/m256`
5. hold `m64` and `m256` closed until a future profile run produces a stronger legal card

## Allowed Next Branches

- `m32`: reopen only as part of a family-wide wide-shape `A-pack` collapse, or after a future profile names a stronger legal bucket
- `m16`: reopen only for exact `A-pack` launch annihilation on the `v101` trunk; the old `B-scale` card is already resolved positive
- `m4`: exact `A-pack` is reopened only in the stricter “delete the launch entirely” form; do not reuse the old `v94` negative as a permanent closure, but do not repeat another helper-kernel swap either
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
- `v98` deletes exact-wide `B-pack/repack` on `m32` and benchmarked at `25.505 us`, pushing both visible `m32` cases to `21.5 / 21.6 us`.
- `v99` deletes exact `m16` tiny-path `B-scale` materialization and benchmarked at `25.644 us`, pushing `m16` to `39.1 us`.
- `v100` (`m4` exact `A-pack` revisit via a replacement helper path) benchmarked at `26.037 us` with `m4 18.8 us`. Keep that helper-swap form closed.
- `v101` compounds the `v99` exact `m16` raw-scale win into the `v98` trunk. It benchmarked at `25.3007 us`, then `25.2188 us`, and ranked at `26.2218 us`. Treat `v101` as the current measured and ranked frontier.
- the refreshed `v101` profile says the next blocker is `A-pack` as a launch family, not another `B-scale` branch: `m16 a_pack_share≈0.728`, `m4 a_pack_share≈0.714`.
- `v102` is the first strict `m16` `A-pack` launch-annihilation attempt on top of `v101`. It passed remote test but benchmarked at `37.381 us` with `m16 401.0 us`. Treat this as strong evidence that naive “inline full A quantization into the MFMA threads” is the wrong shape of `A-pack` annihilation. Keep the launch-deletion goal, but ban this specific in-kernel per-thread quantization pattern.
- `v103` is the first “prepack full exact `m16` A panel into LDS once, then sweep all `N`” attempt. It passed remote test but benchmarked at `40.421 us` with `m16 647.0 us`. Treat this as strong evidence that full-call shared-panel sweep serialization is also the wrong shape of `A-pack` annihilation.
- `v104` keeps the `v103` shared-panel ownership model but spreads `N` across a small fixed block grid. It passed remote test and improved over `v103`, but still benchmarked at `31.803 us` with `m16 156.0 us`. Treat this as evidence that “duplicate full-panel prepack per block, then sweep” is still too serial/duplicative to be a winning exact `m16` shape.
- `v105` keeps the `v101` exact `m16` compute kernel but swaps the generic `mxfp4_pack_a_fixed` launch for a shape-local four-wave producer kernel that quantizes each `1x32` `A` block exactly once. It passed remote test and benchmarked at `27.597 us` with `m16 63.1 us`. Treat this as evidence that “smaller specialized producer launch only” is also not enough; the next legal `A-pack` move has to change ownership/amortization more deeply than just shrinking the standalone producer footprint.
- `v106` is the first chunked exact `m16` producer/consumer ownership attempt: one producer wave quantizes each `K=128` `A` slice into LDS and four consumer waves reuse it across `N` tiles inside the same launch. It passed remote test but benchmarked at `37.679 us` with `m16 426.0 us`. Treat this as strong evidence that shape-local exact `m16` `A-pack` ownership experiments are now exhausted; move the next `A-pack` budget to a family-wide redesign instead of another `m16`-only ownership variant.
- `v107` is the first family-wide thin-path on-chip `A`-slice service for `m4/m8/m16`. It passed real MI355X `test`, but benchmarked at `87.946 us` with `m4 352.0 us` and `m16 3400.0 us`. Treat this as strong evidence that a fixed small CTA-count persistent sweep across the whole `N` range is the wrong thin-family ownership shape: it bounds quant duplication, but it collapses thin-shape parallelism too hard to be viable.
- `v108` is the follow-up thin-family cooperative-grid attempt that tried to preserve the `v101` one-CTA-per-tile thin grid while sharing each staged `A` slice across a resident cooperative block set. After the compile fixes landed, the real MI355X rerun still ended in `check_fail` because the self-hosted kernelbot runner lost communication and artifacts could not be downloaded. Treat this as strong evidence that thin-family grid-cooperative launch + global grid sync is not a safe execution model for the current MI355X workflow, even before we ask whether it is fast enough.
- `v109` is the first non-cooperative block-local thin ownership follow-up after `v108`: exact `m16` uses one producer wave and two consumer waves per CTA, owns only a 2-tile `N` bundle, and avoids both fixed-CTA global sweeps and whole-grid synchronization. The first real MI355X test still ended `submit_error` because kernelbot cancelled the workflow after exceeding its 12 minute timeout. Treat this as strong evidence that this 192-thread, 2-tile block-local ownership shape is still far too heavy to be a viable exact `m16` answer, even though it is execution-model-safe compared with `v108`.

## Thin `A-pack` Pre-Spend Gate

Before any new thin-family `A-pack` remote spend, the candidate must state these numbers for the public thin benchmark cases:

- `quant_dup_upper_bound`: how many times one `A 1x32` block can be quantized per call
- `parallelism_floor_ratio`: candidate active wave supply divided by the `v101` exact-path baseline wave supply
- `n_bundle_per_owner`: how many `N` tiles one producer scope serves before the work is discarded

Reject the candidate before remote spend if any of these are true:

- `parallelism_floor_ratio < 0.50` on `m4` or `m16`
- the design uses a fixed CTA sweep across the whole `N` range
- the design relies on a grid-cooperative launch / full-grid barrier across the thin-family tile grid
- one producer scope serializes a full-call `N` traversal before another independent CTA group can make progress
- the design cannot explain why quant work does not scale linearly with the number of output-column CTAs

`v107` is the concrete warning shot here:

- `m4`: `v101` launches about `180` thin CTAs; `v107` capped itself at `4`, so wave supply fell to about `11%` of the baseline
- `m16`: `v101` launches about `132` thin CTAs; `v107` capped itself at `4`, so wave supply fell to about `15%` of the baseline
- `v108`: even after preserving a near-`v101` tile grid on paper, the cooperative-grid execution model was unstable enough to kill the remote runner, so “use the whole thin grid as one cooperative group” is now banned too

The next legal thin `A-pack` branch must preserve most of the original tile-parallel grid while reducing repeated quant work. Bounding duplication alone is not enough.
