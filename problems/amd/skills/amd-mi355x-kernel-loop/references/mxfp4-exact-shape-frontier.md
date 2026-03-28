# `mxfp4-mm` Exact-Shape Frontier

## Canon

- Best measured trunk:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_m32_directentry_v116/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_m32_directentry_v116/submission.py)
- Best measured benchmark:
  `24.0287 us`
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
- `m16`: `38.6 - 39.3 us`
- `m32`: `18.8 - 21.6 us`
- `m64`: `29.7 - 29.8 us`
- `m256`: `27.3 - 27.8 us`

`m8` is visible in the public `test` set, not the public benchmark mix. Keep it test-green and shape-isolated, but do not spend benchmark budget on `m8` before `m4` or `m16` proves the tiny-path prep deletion.

## Current Cost-Center Policy

- Treat prep-only `m16` edits as plateau territory now that raw `b_scale_sh` is already resolved positive; the next legal `m16` lane is `A-pack` launch annihilation.
- Treat prep-only `m32` edits as plateau territory unless they keep the new raw shuffled-scale win intact while deleting more whole-call overhead.
- Treat exact-wide raw shuffled-scale consumption as resolved positive for `m32` and mildly positive for `m256`; these are now part of the measured frontier, not speculative side lanes.
- Treat exact `m64` raw shuffled-scale consumption as unresolved in its current form: `v114` proved the family-wide delete can win overall, but exact `m64` paid back too much of the deleted helper launch as in-kernel address work, so keep `m64` on the row-major scale path unless the next branch specifically lowers that kernel-side address cost.
- Spend the next serious budget either on a rerun/stability pass for `v118`, on a stability/profile pass for `v116`, or on a cheaper exact `m64` shuffled-scale address path that preserves the `v116` routing if the `v118` rerun stays portfolio-flat.

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

- `m32`: keep the raw shuffled-scale exact path from `v114`/`v115`; reopen only to lower its kernel-side address math further, and only after the `v118` rerun resolves whether the public-`k=512` constant-body lane is a real portfolio win or just a shape-local tie
- `m32`: `v117` proved the public-`k=512` constant-body lane is easy to make test-green but also easy to break on the actual public `m32` cases if the shuffled-scale remap math is even slightly wrong; `v118` repaired that bug and moved both visible `m32` cases, so the next `m32` spend is a rerun gate first, not an immediate sibling rewrite
- `m16`: reopen only for exact `A-pack` launch annihilation on the `v101` trunk; the old `B-scale` card is already resolved positive
- `m4`: exact `A-pack` is reopened only in the stricter “delete the launch entirely” form; do not reuse the old `v94` negative as a permanent closure, but do not repeat another helper-kernel swap either
- `m64`: reopen only for a cheaper in-kernel shuffled-scale address path or after a future profile identifies a different whole bucket; do not re-run the full `v114` raw-scale shape unchanged
- `m256`: keep the `v115` raw shuffled-scale path as part of the measured frontier; reopen only if a later profile isolates another undeleted bucket
- `m8`: keep test-green and isolated; true exact `m8` body before any benchmark budget

## Plateau Update

Recent exact-shape prep-only variants did not create a real wave:

- `v85` (`m16` raw-scale wrapper isolation): effectively flat versus `v83`
- `v86` (`m32` exact B-prep fast path): effectively flat versus `v83`

Operational rule:

- stop spending benchmark slots on prep-only `m16` and prep-only `m32` edits
- after the failed exact-`m32` follow-up line, move the next serious budget to `m16`, then `m4`, then the first family-wide wide-shape `A-pack` collapse
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
- `v109` is the first non-cooperative block-local thin ownership follow-up after `v108`: exact `m16` uses one producer wave and two consumer waves per CTA, owns only a 2-tile `N` bundle, and avoids both fixed-CTA global sweeps and whole-grid synchronization. After fixing the local-scale indexing bug, it passed remote MI355X correctness but benchmarked at `37.321 us` with `m16 399.0 us`. Treat this as strong evidence that this 192-thread, 2-tile block-local ownership shape can be made correct, but is still far too heavy to be a viable exact `m16` answer.
- `v110` is the first wide-family proof branch for exact `m32` `A-pack` collapse without an external exact `A-pack` buffer. It passed remote correctness, but benchmarked at about `36.75 us` overall with exact `m32` around `66.3 / 66.4 us`. Treat it as proof that the wide-family ownership direction can be made correct, not as a viable performance branch.
- `v111` is the narrow follow-up that stayed on the `v101` trunk and reused one on-chip exact `m32` `A` slice across two neighboring public `N` tiles without the external exact `A-pack`. It passed remote MI355X `test`, but benchmarked at `25.360 us` with `m32 21.7 / 21.8 us`, `m4 17.6 us`, and `m16 38.7 us`, so it regressed the overall geomean versus `v101`. Treat this as a clean negative on the exact-`m32`-only `bundle2` fast-path line and move the next budget back to `m16`/`m4` launch annihilation before reopening another exact-`m32` solo experiment.
- `v112a` is the follow-up exact `m16` launch-annihilation branch that kept `v101` elsewhere, staged `K=128` slices, used one producer wave plus seven consumer waves per CTA, and gave each consumer wave two neighboring `N` tiles so one owner CTA covered a 14-tile bundle. It passed remote MI355X `test`, but benchmarked at `37.766 us` with `m16 437.0 us`, while the other visible shapes stayed near trunk. Treat this as a strong negative on the “large-bundle heavy block-local exact `m16` owner” line: lowering the duplication bound on paper is still not enough when the consumer CTA becomes this heavy.
- `v114` is the first family-wide non-`A-pack` follow-up after that pivot: exact `m32/m64/m256` stopped materializing row-major `b_scale` and consumed raw `b_scale_sh` directly inside the raw-`b_q` kernels. It passed remote MI355X `test` and benchmarked at `24.3098 us`, with `m32 18.8 / 18.8 us`, `m256 27.4 us`, and `m64 30.7 us`. Treat this as proof that deleting exact-wide `B-scale` materialization is a real whole-call win, but also as evidence that exact `m64` pays too much for the current in-kernel shuffled-scale address path.
- `v115` is the measured follow-up that kept the `v114` raw shuffled-scale path for `m32` and `m256`, but restored exact `m64` to the old row-major scale materialization path. It passed remote MI355X `test` and benchmarked at `24.1372 us`, with `m32 18.8 / 18.8 us`, `m64 29.8 us`, `m256 27.3 us`, `m4 17.5 us`, and `m16 39.3 us`. Treat `v115` as the new measured frontier and the current best evidence that the real next multiplier is end-to-end contract collapse, not more `A-pack` iteration.
- `v116` is the exact-`m32` direct-entry follow-up on top of `v115`: it kept the winning raw shuffled-scale `m32` kernel contract, but routed exact `m32` through the compiled direct-entry wrapper so Python no longer owns the temp setup and inflight retention for that shape. It passed remote MI355X `test` and benchmarked at `24.0287 us`, with `m32` still at `18.8 / 18.8 us`, `m64 29.5 us`, `m16 38.5 us`, and `m4 17.5 us`. Treat `v116` as the new measured frontier, but also as evidence that the next large `m32` gain likely requires a public-`k=512` constant-body kernel rather than more wrapper collapse alone.
- `v117` is the first public-`k=512` exact `m32` constant-body clone on top of `v116`: it kept the raw `b_q + b_scale_sh` contract but specialized the hot kernel for fixed `k=512`. It passed remote MI355X `test`, but benchmark correctness failed on both visible `m32` cases with first errors starting at column `32`. Treat this as a clean negative on the buggy row-block mapping in that implementation, not as closure of the constant-body lane itself.
- `v118` is the immediate `v117` repair that fixes the specialized shuffled-scale row-block term from `row_block * 16` to `row_block * 32`. The local formula check matches the generic helper exactly, remote MI355X `test` passed, and benchmarked at `24.0398 us`, with `m32 18.5 / 18.5 us`, `m64 29.6 us`, `m16 39.0 us`, `m4 17.7 us`, and `m256 27.6 us`. Treat this as a real public-`m32` shape win over `v116` (`18.8 / 18.8 us` -> `18.5 / 18.5 us`), but only a portfolio tie because the overall geomean stayed slightly worse than `v116` (`24.0398 us` vs `24.0287 us`). Operational rule: rerun `v118` before promoting the lane or opening another `m32` sibling. If the rerun is still flat overall, pivot the next serious budget to a cheaper exact `m64` shuffled-scale address path or another non-`m32` whole-call bucket.

Operational pivot from the current evidence:

- stop spending immediate budget on `A-pack` prep/materialization as the primary target
- the `A-pack` line is not dead forever, but it is closed until a future idea changes ownership fundamentally instead of repeating the same work with a different producer/bundle shape
- move the next serious effort to end-to-end latency and non-`A-pack` data-movement work:
  - exact public-shape constant-body clones
  - wide direct-from-shuffled `B-scale` consumption or another fresh legal non-`A-pack` bucket
  - architecture-jump lanes if the non-`A-pack` body/latency work still stays flat
- `v112` is the first ISA-grounded `A-pack` implementation experiment after the ownership dead ends: it keeps `v101`’s exact-shape ownership model, but swaps the scalar `mxfp4_pack_a_fixed` inner loop for the native gfx950 FP4 scale-pack builtin path. It passed remote compile, which proves those builtins are available in the real MI355X environment, but failed correctness on every visible test shape with large mismatch counts. Treat this as evidence that a raw drop-in replacement of the hand-rolled `A-pack` quantizer with the native FP4 pack builtins is not contract-compatible yet.
- `v113` is the immediate scale-convention follow-up to `v112`: it keeps the builtin path but flips the scale argument to the dequant-scale interpretation instead of the quant-scale interpretation. It is preflight-green locally, but has not been remote-tested yet because the test-stage hourly coordinator budget was exhausted. Keep it as the next ready-to-spend validation branch if the next slot still belongs to the native-builtin `A-pack` lane.
- `v121` is the exact-`m16` global `A-pack` temp-law delete via in-kernel raw-`A` quantization on the direct-entry path. It passed remote correctness, but benchmarked at `25.1570 us` with `m16 63.0 us`. Treat this as a strong negative on “delete the external temp law by moving quantization directly into the exact `m16` compute path”: the saved temp bytes are real, but the internal quant/control cost overpays them badly.
- `v122` is the strict exact public-`m32,k=512` follow-up that tried to mirror the successful `B`-side pattern: it preserved the winning raw-`B` kernel body and only swapped the `A` feeder to CTA-local staged quantization, deleting the external `a_packed + a_scale` temp law on that path. It passed remote correctness, but benchmarked at `35.1567 us` with `m32 58.2 / 58.3 us`. Treat this as a strong negative on “per-CTA local re-quantized `A` feeder” even when the MFMA body and CTA ownership stay fixed: duplicated quant work dominates the saved global write+read traffic.
- `v125` is the stricter exact public-`m32,k=512` register-only follow-up after `v122`: it preserved the winning raw-`B` body, quantized each warp-owned `A` strip directly from BF16 into MFMA input registers, and deleted both the standalone `mxfp4_pack_a_fixed` launch and the external `a_packed + a_scale` temp law on that path. It passed remote correctness, but benchmarked at `38.6531 us` with `m32 77.9 / 77.3 us`. Treat this as a strong negative on “warp-local register-only `A` feeder swap” as well: removing LDS and global temp traffic still does not help if each output CTA re-quantizes the same logical `A` rows independently.

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

## Any `A-pack` Reopen Gate

Before any new `A-pack` remote spend on any exact shape, the candidate must also state:

- `reuse_factor`: how many output consumers reuse one quantized `A 1x32` block before it is discarded
- `duplication_factor`: how many times that same `A 1x32` block is quantized per full call
- `saved_global_bytes_per_block`: old external write+read bytes removed for that `A 1x32` block
- `why_total_quant_work_drops`: one short proof that the new law reduces total work instead of only relocating it

Reject the branch before remote spend if any of these are true:

- `duplication_factor > reuse_factor`
- the proposal quantizes `A` independently in each output-column CTA with no cross-CTA reuse story
- the proposal preserves only local CTA reuse while the duplication factor still scales with the full output-column CTA count
- the main win claim is only “remove a launch” or “remove temp bytes” without a proof that quant work is amortized

## Wide `A-pack` Reopen Gate

Before any new exact-wide `A-pack` remote spend, the candidate must state:

- `reuse_factor_per_quant`: how many output consumers reuse one quantized `A 1x32` block before it is discarded
- `quant_dup_upper_bound`: how many times one logical `A 1x32` block is quantized per call
- `saved_global_bytes_per_block`: bytes removed from global temp write+read traffic
- `new_internal_quant_scope`: where the quant work moves instead

Reject the branch before remote spend if any of these are true:

- the design re-quantizes `A` independently per output-column CTA with no cross-CTA reuse proof
- `quant_dup_upper_bound > reuse_factor_per_quant`
- the branch changes both `A` feed and CTA ownership in one step
- the branch cannot explain why total quant work drops rather than merely moving from global temp traffic into the hot path

Working warning shots:

- `v121`: deleting external exact-`m16` `A-pack` temp traffic was not enough because direct in-kernel quantization raised the internal cost more than the launch/temp law it removed
- `v122`: preserving the winning exact-`m32` body was still not enough when each CTA re-quantized its own `A` slice; the duplicate quant work drove public `m32` from `16.6 / 16.7 us` to `58.2 / 58.3 us`
- `v125`: removing the standalone exact-`m32` `A-pack` launch and even the global `a_packed/a_scale` temp itself was still not enough when the same logical `A` rows were re-quantized per output CTA; the public `m32` cases regressed further to `77.9 / 77.3 us`

## Bounded Wide-Family `A` Reuse Arithmetic

The first paper-only bounded wide-family reuse pass is now closed for the natural `C2` and `C4` cluster laws on the live `v119` grid:

- cluster law:
  - `C2`: one producer quantizes one `A` slice for `2` neighboring `N`-tile consumers
  - `C4`: one producer quantizes one `A` slice for `4` neighboring `N`-tile consumers
- preserved assumptions:
  - keep the current exact-wide raw-`B` kernels intact
  - no full-grid cooperation
  - no fixed tiny CTA sweep across all `N`
  - no per-output-CTA local feeder swap

Derived duplication counts under the current one-CTA-per-`N32`-tile wide grid:

- public `m32, n=4096`: `128` output-column CTAs
  - `C2`: `reuse_factor=2`, `duplication_factor=64`
  - `C4`: `reuse_factor=4`, `duplication_factor=32`
- public `m32, n=2880`: `90` output-column CTAs
  - `C2`: `reuse_factor=2`, `duplication_factor=45`
  - `C4`: `reuse_factor=4`, `duplication_factor=23`
- public `m64, n=7168`: `224` output-column CTAs
  - `C2`: `reuse_factor=2`, `duplication_factor=112`
  - `C4`: `reuse_factor=4`, `duplication_factor=56`
- public `m256, n=3072`: `96` output-column CTAs
  - `C2`: `reuse_factor=2`, `duplication_factor=48`
  - `C4`: `reuse_factor=4`, `duplication_factor=24`

Operational conclusion:

- bounded `C2/C4` cluster reuse does not clear the `A-pack` reopen gate on the current wide-family grid
- do not code a bounded wide-family `A` service branch from this arithmetic alone
- any future exact-wide `A-pack` reopen must change the duplication law more fundamentally than “serve 2-4 neighboring `N` tiles from one producer”

Macro-cluster threshold from the same grid law:

- if one producer span covers `S` consecutive `N32` tiles, then:
  - `reuse_factor = S`
  - `duplication_factor = ceil(num_n_tiles / S)`
- first gate (`reuse >= duplication`) only starts clearing at about:
  - `m32, n=4096`: `S >= 12`
  - `m32, n=2880`: `S >= 12`
  - `m64, n=7168`: `S >= 16`
  - `m256, n=3072`: `S >= 12`
- stronger practical gate (`reuse >= 2 * duplication`) starts at about:
  - `m32, n=4096`: `S >= 16`
  - `m32, n=2880`: `S >= 16`
  - `m64, n=7168`: `S >= 24`
  - `m256, n=3072`: `S >= 16`

Operational rule:

- do not spend another exact-wide `A-pack` slot on any ownership law whose producer span is below these thresholds
- the next plausible `A-pack` reopen, if any, must be a macro-cluster law that changes duplication over at least `12-24` `N32` tiles, not just a local `C2/C4` cluster

Mechanism check after the first serious macro-cluster pass:

- arithmetic is no longer the blocker for exact `m32`; `S=16` clears the strong practical gate on the public shapes
- the current blocker is mechanism:
  - the live `v119` exact `m32` path still has only per-CTA LDS/register scope and whole-grid global-memory scope
  - there is no documented HIP/ROCm cluster-scoped CTA primitive we can rely on here for:
    - guaranteed co-residency
    - cluster-local barrier
    - cluster-visible shared storage/direct handoff
- operational conclusion:
  - no exact `m32` macro-cluster `A-pack` branch should be coded yet
  - keep `A-pack` research paper-only until a legal cluster-scoped handoff mechanism is identified

Current active optimization split:

- spend remote budget on non-`A-pack` whole-call deletions first:
  - exact public `m32,k=512` non-`A-pack` fixed-overhead buckets
  - exact `m4` non-`A-pack` fixed-cost buckets
  - exact `m64` address-bucket cleanup only after the higher-leverage lanes
- keep `A-pack` as the strategic research target, but paper-only until the missing mechanism changes
