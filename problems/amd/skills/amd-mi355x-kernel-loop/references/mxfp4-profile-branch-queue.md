# `mxfp4-mm` Profile-Derived Branch Queue

## Source of Truth

Current queue is derived from the latest real MI355X exact-shape profile run on the `t18` benchmark winner:

- run dir:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-103934-wavepack-directentry-t18-profile-rocprof](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-103934-wavepack-directentry-t18-profile-rocprof)
- raw zip:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-103934-wavepack-directentry-t18-profile-rocprof/profile_20260401_104326_run0.zip](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-103934-wavepack-directentry-t18-profile-rocprof/profile_20260401_104326_run0.zip)
- summary:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-103934-wavepack-directentry-t18-profile-rocprof/stages/01_profile_rocprof/profile/profile_summary.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-103934-wavepack-directentry-t18-profile-rocprof/stages/01_profile_rocprof/profile/profile_summary.json)
- cards:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-103934-wavepack-directentry-t18-profile-rocprof/stages/01_profile_rocprof/profile/candidate_cards.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-103934-wavepack-directentry-t18-profile-rocprof/stages/01_profile_rocprof/profile/candidate_cards.json)

Operational rule:

- when this queue exists, follow it before opening a new exact-shape branch
- do not bypass it with intuition or polish work
- only replace it after a newer real MI355X profile run produces a stronger queue

## Queue Order

1. keep `t20` as the active measured frontier until another branch beats `14.5092 us`
2. exact `m4` non-`A-pack` fixed-cost cleanup is now near noise after `t20`; keep it but deprioritize further spend on this lane
3. prioritize exact `m64` address-law deletion (`m64-address-last`) that removes shuffled-scale address rebuild cost without reopening naive `t19` behavior
4. prioritize high-leverage `m16/m256` whole-bucket deletions before any schedule-polish work
5. keep exact `m32` closed unless a newer real profile identifies a fresh non-`A-pack` whole bucket
6. keep `m8` isolated and test-green only

Queue correction after `t20` (`m4` fixed-cost fast-dispatch):

- `t20` passed remote `test` and benchmarked at `14.5092 us`:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-122947-native-scaled-exact-shape-m4-fixedcost-fastdispatch-t20-benchmark/stages/01_benchmark/parsed_metrics.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-122947-native-scaled-exact-shape-m4-fixedcost-fastdispatch-t20-benchmark/stages/01_benchmark/parsed_metrics.json)
- The gain over `t18` is tiny (about `0.004%`) and `m4` remains `10.0 us`.
- Operational rule: keep the cleanup, but move serious budget to `m64-address-last` and larger-share lanes (`m16/m256`) instead of spending another m4 fixed-cost micro-slot.

Queue correction after the `v111` remote result:

- `v111` kept `v101` intact outside exact `m32` and tried a public-`k=512` raw-`b_q` `bundle2` fast path that reused one on-chip exact `A` slice across two neighboring `N` tiles.
- It passed remote `test`, but benchmarked at `25.360 us` with `m32 21.7 / 21.8 us`, which is worse than `v101` overall and not better on the visible exact `m32` cases either.
- Operational rule: do not spend another exact-`m32`-only remote slot until a family-wide wide-shape `A-pack` collapse has a stronger amortization story than `v110`/`v111`, or a newer real profile names a better legal bucket.

Queue correction after the `v112a` remote result:

- `v112a` was the first large-bundle exact `m16` follow-up after `v109`: one producer wave, seven consumer waves, two neighboring `N` tiles per consumer wave, and a 14-tile owner bundle with `K=128` slice staging.
- It passed remote `test`, but benchmarked at `37.766 us` with `m16 437.0 us`, so the branch is correct-but-catastrophically-slow rather than almost-good.
- Operational rule: do not spend another heavy block-local exact-`m16` owner slot. Any future `m16` `A-pack` candidate must change ownership more fundamentally than “bigger CTA, bigger bundle”; otherwise move the next thin-budget to `m4` or a truly different family-wide redesign.

Operational pivot after the current `A-pack` wave:

- stop treating exact-path `A-pack` prep/materialization deletion as the primary branch objective
- only reopen `A-pack` when the proposal changes the ownership law fundamentally, not just the producer location, block size, or bundle width
- move the next serious budget to non-`A-pack` lanes:
  - exact public-shape constant-body clones that target end-to-end latency, addressing math, and launch-side overhead
  - wide direct-from-shuffled `B-scale` consumption or other non-`A-pack` data-movement deletes with a fresh legal bucket
  - if those stay flat, open the architecture-jump lane earlier than originally planned

Queue correction after the `v114` and `v115` remote results:

- `v114` was the first family-wide exact-wide raw shuffled-scale branch after the `A-pack` pivot. It passed remote `test` and benchmarked at `24.3098 us`, with exact `m32 18.8 / 18.8 us`, exact `m256 27.4 us`, and exact `m64 30.7 us`.
- `v115` kept the `v114` raw shuffled-scale path for exact `m32` and `m256`, but restored exact `m64` to the old row-major scale materialization path. It passed remote `test` and benchmarked at `24.1372 us`, with exact `m32 18.8 / 18.8 us`, exact `m64 29.8 us`, and exact `m256 27.3 us`.
- Operational rule: direct-from-shuffled `B-scale` consumption is now a measured whole-call win on exact `m32` and a mild positive on exact `m256`; exact `m64` should stay on the row-major scale path unless the next branch specifically lowers the in-kernel shuffled-scale address cost.

Queue correction after the `v116` remote result:

- `v116` kept the `v115` wide routing, but moved exact `m32` from Python-owned temp orchestration into the compiled direct-entry wrapper while preserving the same raw shuffled-scale kernel contract.
- It passed remote `test` and benchmarked at `24.0287 us`, with exact `m32` still at `18.8 / 18.8 us`.
- Operational rule: exact `m32` direct-entry collapse is a small positive at the geomean level, but it did not move the visible public `m32` means themselves. The next serious `m32` branch should therefore target the remaining public-`k=512` kernel/address bucket, not more Python-side cleanup alone.

Queue correction after the `v117` and `v118` follow-ups:

- `v117` was the first public-`k=512` exact `m32` constant-body branch on top of `v116`. It kept the raw shuffled-scale `m32` contract but specialized the exact `m32` kernel body for fixed `k=512`, fixed scale columns, and fixed stores.
- It passed remote `test`, but benchmark correctness failed on both visible `m32` cases with first errors at column `32`. The root cause was the specialized shuffled-scale row-block term using `row_block * 16` instead of the generic helper’s effective `row_block * 32`.
- `v118` is the immediate repair branch. It fixes that row-block term, passed remote `test`, and benchmarked at `24.0398 us` with `m32 18.5 / 18.5 us`.
- Operational rule: `v118` is a real public-`m32` shape win but only a portfolio tie versus `v116` (`24.0398 us` vs `24.0287 us`). Do not open another `m32` branch yet. Rerun `v118` first because it is within the `1%` noise gate. If the rerun still does not beat `v116` overall, pivot the next serious budget to cheaper exact `m64` shuffled-scale addressing instead of more `m32` body polish.

## Candidate Cards

### `m32-Q2`

- `shape`: `m32`
- `deleted_cost_center`: `generic exact-wide B-pack/repack reused across exact m32`
- `expected_upside_source`: kernelbot profile shows exact `m32` still splits most of its CUDA time across `A-pack`, `B-pack`, and kernel, leaving a large reusable `B` materialization bucket
- `why_larger_than_noise`: both visible `m32` cases agree: `a_pack_share=0.316`, `b_pack_share=0.342`, `kernel_share=0.342`; and `a_pack_share=0.324`, `b_pack_share=0.324`, `kernel_share=0.351`
- `touched_symbols_or_regions`:
  - `mxfp4/exact_m32`
  - `mxfp4/exact_m32/b_prep`
- `forbidden_edits`:
  - do not open another prep-only micro-variant
  - do not share the branch with `m64`
- `success_gate`: both visible `m32` cases beat or match `22.6 / 21.8 us`
- `allowed implementation direction`:
  - delete the whole `b_packed` materialization path for exact `m32`
  - keep the branch shape-local and feed raw `b_q` into an `m32`-local body
  - keep `a_pack` and the current kernel body unchanged in the first branch

### `m16-Apack-Fuse`

- `shape`: `m16`
- `deleted_cost_center`: `separate mxfp4_pack_a_fixed launch on exact m16`
- `expected_upside_source`: the refreshed `v101` profile shows exact `m16` is now dominated by `A-pack`, not by `B-scale`; `a_pack_share≈0.728`, `b_scale_decode_share=0`, `kernel_share≈0.272`
- `why_larger_than_noise`: this is the last dominant tiny-shape prep launch on `m16`, and deleting the launch family is larger than any remaining wrapper polish
- `touched_symbols_or_regions`:
  - `mxfp4/exact_m16`
  - `mxfp4/exact_m16/a_pack`
- `forbidden_edits`:
  - do not reopen `B-scale` work; that bucket is already resolved positive
  - do not inline full per-thread quantization into the MFMA threads; `v102` proved that shape is catastrophic
  - do not use one-block or small-grid shared-panel sweep kernels that prepack the whole `A` panel per block; `v103` and `v104` proved those shapes are still catastrophically under-amortized or under-parallel
  - do not spend another slot on wrapper polish without deleting this bucket
- `success_gate`: materially beat the `v101` `m16` regime without regressing geomean
- `allowed implementation direction`:
  - delete the separate exact `A-pack` launch from the `m16` path
  - keep the winning raw `b_scale_sh` path intact
  - move quant/pack ownership into the exact path in a way that does not duplicate the full quantization workload across the hot MFMA threads

### `m4-Apack-Fuse`

- `shape`: `m4`
- `deleted_cost_center`: `generic mxfp4_pack_a_fixed on exact m4`
- `expected_upside_source`: the `v101` profile shows the exact `m4` path is now dominated by `A-pack`, not by scale decode or the MFMA body
- `why_larger_than_noise`: `a_pack_share≈0.714`, `b_scale_decode_share=0`, `kernel_share≈0.286`; the raw-scale win changed the balance enough that the old `v94` negative no longer closes this lane
- `touched_symbols_or_regions`:
  - `mxfp4/exact_m4`
  - `mxfp4/exact_m4/a_pack`
- `forbidden_edits`:
  - do not reopen the broken `v79` launch structure
  - do not rewrite `m8` or `m16` in the same branch
- `success_gate`: `m4 < 17.3 us`
- `allowed implementation direction`:
  - keep the proven wave64 ownership model and the winning raw `b_scale_sh` path from `v96`
  - delete the separate exact `A-pack` launch entirely, not by swapping in another pack helper

## Closed Lanes

### `m64`

- `status`: selectively reopened
- `reason`: `v114` proved the family-wide `B-scale` delete can win overall, but exact `m64` itself regressed, so the next legal `m64` branch is now “cheaper raw shuffled-scale address path” rather than “full raw-scale path again”
- `why`: the deleted helper launch was real, but the current exact `m64` kernel overpaid that win as long-`K` address math
- rule: do not reopen exact `m64` unless the branch keeps `m32`/`m256` intact and specifically lowers exact `m64` kernel-side scale address cost, or a new profile identifies another whole bucket

### `m256`

- `status`: resolved for the current lane
- `reason`: `v114`/`v115` showed the exact `m256` raw shuffled-scale path is mildly positive on the measured frontier
- `why`: exact `m256` moved from `27.8 us` on `v101` to `27.4 us` on `v114` and `27.3 us` on `v115`
- rule: keep the `v115` exact `m256` route locked unless a future profile names another undeleted whole bucket

## Recent Wins Locked Into The Trunk

- `v93`: exact `m64` raw-`b_q` deletion resolved positive and is part of the live trunk
- `v95`: exact `m256` raw-`b_q` deletion resolved positive and is now compounded into `v97`
- `v96`: exact `m4` raw `b_scale_sh` / scale-decode deletion resolved positive and is part of `v97`
- `v97`: compounded trunk benchmarked at `25.7488 us`, then `25.8866 us`
- `v98`: exact `m32` raw-`b_q` deletion benchmarked at `25.505 us`
- `v99`: exact `m16` raw `b_scale_sh` deletion benchmarked at `25.644 us`
- `v101`: compounded trunk benchmarked at `25.3007 us`, then `25.2188 us`, and ranked at `26.2218 us`; treat it as the current measured and ranked frontier
- `v102`: exact `m16` A-pack launch annihilation via naive in-kernel per-thread quantization benchmarked at `37.381 us` with `m16 401.0 us`; ban this specific implementation shape
- `v103`: exact `m16` A-pack launch annihilation via one-block full-panel shared prepack benchmarked at `40.421 us` with `m16 647.0 us`; ban this serialization-heavy shape
- `v104`: exact `m16` A-pack launch annihilation via small-grid full-panel shared prepack benchmarked at `31.803 us` with `m16 156.0 us`; treat this as better than `v103` but still far from viable, and do not reopen this family without a stronger producer/consumer or family-wide design
- `v105`: exact `m16` A-pack ownership via a narrower four-wave specialized producer launch benchmarked at `27.597 us` with `m16 63.1 us`; keep this as a clean negative on “smaller standalone producer stage only” and do not spend another slot on launch-footprint-only variants
- `v106`: exact `m16` chunked producer/consumer ownership benchmarked at `37.679 us` with `m16 426.0 us`; close shape-local `m16` `A-pack` ownership variants and move the next `A-pack` spend to a family-wide redesign
- `v107`: thin-family on-chip `A`-slice service for `m4/m8/m16` passed remote test but benchmarked at `87.946 us`, with `m4 352.0 us` and `m16 3400.0 us`; keep this as a clean negative on “fixed small CTA persistent sweep across all `N`” and ban that ownership shape
- `v108`: thin-family cooperative-grid `A`-slice service cleared the compile issues but the real MI355X rerun still ended in `check_fail` because the kernelbot self-hosted runner lost communication and no artifacts could be downloaded; keep this as a clean negative on “grid-cooperative launch across the thin tile grid” and ban that execution model for now
- `v109`: exact `m16` block-local 2-tile ownership cleared correctness after the local-scale indexing fix, but benchmarked at `37.321 us` with `m16 399.0 us`; keep this as a clean negative on “192-thread / 3-wave block-local A owner with 2-tile bundle” because it is correct-but-slow, not just timeout-prone
- `v112`: native gfx950 FP4 scale-pack builtins in `mxfp4_pack_a_fixed` compiled on the real MI355X runner, but the direct drop-in replacement failed correctness across every visible test shape; keep this as a clean negative on “assume the builtin FP4 pack path is a drop-in semantic match for our current hand-rolled quantizer”
- `v113`: the scale-convention fixup follow-up to `v112` is preflight-green and ready, but it has not spent a remote test slot yet because the hourly test-stage coordinator budget was exhausted
- `v114`: family-wide exact-wide raw shuffled-scale consumption deleted the helper `mxfp4_unshuffle_b_scale` launch and row-major temp on exact `m32/m64/m256`, passed remote `test`, and benchmarked at `24.3098 us`. Keep this as the proof branch that end-to-end exact-wide `B-scale` materialization deletion is a real win, not just a tidy cleanup.
- `v115`: hybrid wide exact routing kept the `v114` raw shuffled-scale path for exact `m32` and `m256` while restoring exact `m64` to the old row-major scale path. It passed remote `test` and benchmarked at `24.1372 us`; treat it as the new measured frontier and the current branch to rerun/profile before opening another wide exact lane.
- `v116`: exact `m32` direct-entry collapse kept the `v115` wide kernel contracts but deleted Python-owned temp orchestration for exact `m32`. It passed remote `test` and benchmarked at `24.0287 us`; treat it as the new measured frontier, but also as evidence that the next large `m32` move must be a public-`k=512` constant-body/body-address branch rather than more wrapper collapse alone.
- `v117`: the first public-`k=512` exact `m32` constant-body clone passed remote `test` but failed benchmark correctness on both visible `m32` cases because the specialized shuffled-scale row-block term was wrong. Keep this as a clean negative on that buggy implementation, not as closure of the lane.
- `v118`: the `v117` repair branch corrected the specialized shuffled-scale row-block term, passed remote `test`, and benchmarked at `24.0398 us`. It improved both visible public `m32` cases to `18.5 / 18.5 us`, but the overall geomean stayed slightly behind `v116`, so treat it as a rerun gate rather than an immediate promotion or a license to open another `m32` sibling branch.
- `v121`: exact `m16` global `A-pack` temp-law deletion via in-kernel direct quantization passed remote `test`, but benchmarked at `25.1570 us` with `m16 63.0 us`. Keep this as a clean negative on “delete exact `m16` external `A-pack` temp traffic by moving quantization into the compute path.”
- `v122`: exact public-`m32,k=512` CTA-local `A` feeder swap passed remote `test`, but benchmarked at `35.1567 us` with `m32 58.2 / 58.3 us`. Keep this as a clean negative on “preserve the winning exact `m32` body but re-quantize `A` independently per output CTA.” The saved temp bytes were real, but duplicate quant work dominated them.
- `v121`: exact `m16` global `A-pack` temp-law deletion via in-kernel raw-`A` quantization passed remote `test`, but benchmarked at `25.1570 us` with `m16 63.0 us`. Keep this as a clean negative on “delete external `A-pack` by moving quantization directly into the exact `m16` compute path”.
- `v122`: exact public `m32,k=512` CTA-local `A` feeder swap passed remote `test`, but benchmarked at `35.1567 us` with `m32 58.2 / 58.3 us`. Keep this as a clean negative on “preserve the winning raw-`B` body but re-quantize `A` independently inside each output-column CTA”.

## Thin `A-pack` Remote-Spend Gate

No new thin-family `A-pack` branch may spend remote quota unless its candidate card includes all of:

- `quant_dup_upper_bound`
- `parallelism_floor_ratio` versus the `v101` thin exact-path baseline
- `n_bundle_per_owner`
- a short proof that quant work does not scale with the full count of output-column CTAs

Reject the branch before remote spend if any of these are true:

- `parallelism_floor_ratio < 0.50` on the public `m4` or `m16` benchmark case
- the design uses a fixed CTA count to sweep the entire `N` range
- the design requires a grid-cooperative launch or full-grid barrier across the thin-family tile grid
- the design serializes a full-call `N` traversal behind one producer scope
- the main benefit is only “fewer launches” without a credible preservation of grid-level parallelism

Working example from `v107`:

- `m4`: `v101` baseline thin grid is about `180` CTAs; `v107` used `4`, so the parallelism floor was only about `0.11`
- `m16`: `v101` baseline thin grid is about `132` CTAs; `v107` used `4`, so the parallelism floor was only about `0.15`
- `v108`: the cooperative-grid follow-up preserved a high paper parallelism floor, but it still destabilized the remote runner, so thin-family full-grid synchronization is banned regardless of the nominal floor calculation

That failure means the next legal direction must preserve most of the original tile-parallel grid while reducing quant duplication. No more fixed-CTA persistent sweeps across all `N`.

## Wide `A-pack` Remote-Spend Gate

No new exact-wide `A-pack` branch may spend remote quota unless its candidate card includes all of:

- `reuse_factor_per_quant`
- `quant_dup_upper_bound`
- `saved_global_bytes_per_block`
- `new_internal_quant_scope`
- a short proof that total quant work drops rather than merely moving from global temp traffic into the hot path

Reject the branch before remote spend if any of these are true:

- the design re-quantizes `A` independently per output-column CTA
- `quant_dup_upper_bound > reuse_factor_per_quant`
- the branch changes both feeder law and CTA ownership in one step
- the claimed win is only “fewer launches” or “no temp buffer” without a quantified reuse story

Operational rule after `v122`:

- do not spend another exact-`m32`-only `A-pack` slot on CTA-local re-quant
- the only remaining legal `A-pack` lane is bounded multi-consumer reuse that lowers duplication itself, not just temp bytes

Queue correction after `v125`:

- `v125` was the stricter exact public-`m32,k=512` follow-up after `v122`: it deleted the standalone `mxfp4_pack_a_fixed` launch and the external `a_packed + a_scale` temp law by quantizing each warp-owned `A` strip directly from BF16 into MFMA input registers.
- It passed remote `test`, but benchmarked at `38.6531 us` with `m32 77.9 / 77.3 us`.
- Operational rule:
  - do not spend another slot on exact-shape local `A` feeder rewrites, even if they are register-only and avoid LDS
  - local `A-pack` deletion is now closed in all tested forms; the only remaining legal `A-pack` lane is still a true duplication-law break with a mechanism stronger than per-CTA local state

Bounded wide-family paper pass on top of `v119`:

- natural `C2`/`C4` cluster laws are now evaluated and closed on paper
- current exact-wide grid law is still one CTA per `N32` tile, so duplication stays `num_n_tiles / cluster_size`
- derived public-shape arithmetic:
  - `m32, n=4096`: `C2 -> reuse 2 / duplication 64`, `C4 -> reuse 4 / duplication 32`
  - `m32, n=2880`: `C2 -> reuse 2 / duplication 45`, `C4 -> reuse 4 / duplication 23`
  - `m64, n=7168`: `C2 -> reuse 2 / duplication 112`, `C4 -> reuse 4 / duplication 56`
  - `m256, n=3072`: `C2 -> reuse 2 / duplication 48`, `C4 -> reuse 4 / duplication 24`
- operational rule:
  - do not open a bounded exact-wide `A` service branch whose only new law is “one producer serves 2-4 neighboring `N`-tile CTAs”
  - the next legal `A-pack` reopen must change the duplication law more fundamentally than local bounded `N` clustering

Macro-cluster follow-up arithmetic:

- if a producer span covers `S` consecutive `N32` tiles, then `reuse=S` and `duplication=ceil(num_n_tiles / S)`
- first gate (`reuse >= duplication`) starts only at:
  - `m32, n=4096`: `S >= 12`
  - `m32, n=2880`: `S >= 12`
  - `m64, n=7168`: `S >= 16`
  - `m256, n=3072`: `S >= 12`
- stronger practical gate (`reuse >= 2 * duplication`) starts only at:
  - `m32, n=4096`: `S >= 16`
  - `m32, n=2880`: `S >= 16`
  - `m64, n=7168`: `S >= 24`
  - `m256, n=3072`: `S >= 16`
- operational rule:
  - do not open another exact-wide `A-pack` branch whose producer span is below these thresholds
  - any future `A-pack` reopen must be framed as a macro-cluster duplication-law change, not a local cluster-service tweak

Queue correction after the post-`v119` three-scout paper pass:

- the active exact public `m32,k=512` non-`A-pack` branch slot is now closed on paper
 
## Multi-Shape Portfolio Ladder

The active search is now portfolio-first, not `m32`-first.

Operational rule:

- choose the next branch by total geomean leverage across `m4/m16/m32/m64/m256`
- do not prefer `m32` just because it already has the cleanest direct-contract path
- prefer branches that simplify the exact-shape call law itself:
  - compiled direct-entry everywhere
  - fewer helper launches
  - fewer temp write+read laws
  - less per-call runtime shaping
  - less hot-loop address math on public fixed shapes

Current ladder order:

1. compiled direct-entry and runtime/orchestration collapse on the remaining hot exact shapes
2. whole helper-launch deletion where the deleted work does not come back as duplicate hot-path work
3. temp-law deletion outside `A-pack`, or `A-pack` only if a future branch changes total quant count instead of moving it
4. public-shape constant-body deletion of setup/addressing work
5. paper-only `A-pack` duplication-law research

Current hot-shape portfolio targets on top of `v119`:

- `m4`: exact path still pays the separate `mxfp4_pack_a_fixed` law and remains highly latency-sensitive; prefer direct-entry/orchestration and helper-law deletion before any body polish
- `m16`: exact path still pays the separate `mxfp4_pack_a_fixed` law; do not reopen local `A-pack` feeder rewrites, but keep targeting end-to-end exact-path orchestration and non-duplicating helper-law deletions
- `m32`: keep the current best raw-contract path; only reopen if the branch deletes a whole-call fixed bucket and improves portfolio geomean, not because `m32` is aesthetically clean
- `m64`: still the only hot shape with both `mxfp4_pack_a_fixed` and row-major `b_scale` repair in the default exact path, so whole-call helper/repair deletion remains legal if it does not overpay in kernel-side setup
- `m256`: keep the current raw shuffled-scale path and prefer orchestration/runtime shaping collapse over kernel-body polish
- reason:
  - `v119` already landed the exact public-`m32,k=512` shuffled-scale remap/address delete that the queue was still reserving
  - the live `v119` profile now shows exact `m32` as only `A-pack` plus the fixed public body, with `b_pack_share=0`
  - the remaining tempting `m32` edits are wrapper/view polish or sibling body micro-polish, which do not clear the whole-bucket gate
- operational rule:
  - do not open another exact `m32` branch from this queue unless a newer real MI355X profile names a fresh non-`A-pack` whole-call bucket
  - move the next real remote spend to exact `m4` non-`A-pack` fixed-cost deletion

## Current Active Program

Use the next budget in this order until a newer real MI355X profile replaces this queue:

1. `m4-fixed-cost-delete`
   - branch class: exact `m4` only
   - deleted cost center: non-`A-pack` fixed-cost overhead on the tiny path
   - allowed directions:
     - wrapper/orchestration/temp retention collapse
     - dead exact-path epilogue or addressing work
   - forbidden:
     - reopening `m4` `A-pack` by itself
     - broad thin-family ownership changes

2. `m64-address-last`
   - branch class: exact `m64` only
   - deleted cost center: in-kernel shuffled-scale address/rebuild cost
   - allowed directions:
     - only if the branch keeps `m32` and `m256` intact
     - only if the deleted bucket is specifically the exact `m64` address path, not a fresh `A-pack` story
   - note:
     - `v120` and `v123` proved this is real but portfolio-secondary, so keep it behind `m4`

3. `Apack-paper-only`
   - branch class: research only, no remote spend
   - active question:
     - can a future exact-wide `A-pack` reopen change duplication fundamentally without grid-coop, without collapsing wide parallelism, and without recreating a global temp
   - current status:
     - arithmetic threshold is known
     - implementation mechanism is missing
   - operational rule:
     - no code and no remote spend unless a new ownership law answers the missing cluster-scoped handoff mechanism first

4. `m32-nonApack-overhead`
   - status: paper-vetoed after `v119`
   - reason:
     - the live `v119` exact public-`m32,k=512` source already contains the queue’s remap/address-law delete
     - the live `v119` profile leaves no measured `m32` `b_prep` bucket and no new whole-call wrapper bucket
   - reopen only if:
     - a newer real MI355X profile names a fresh exact-`m32` non-`A-pack` whole bucket that is not the same specialized-body polish

## Any `A-pack` Remote-Spend Gate

No new `A-pack` branch on any exact shape may spend remote quota unless its Candidate Card includes all of:

- `reuse_factor`
- `duplication_factor`
- `saved_global_bytes_per_block`
- a short proof that total quant work drops instead of only relocating the old external temp bytes

Reject the branch before remote spend if any of these are true:

- `duplication_factor > reuse_factor`
- the design quantizes `A` independently in each output-column CTA with no cross-CTA reuse law
- the design keeps only local CTA reuse while duplication still scales with the full output-column CTA count
- the main benefit is only “fewer launches” or “fewer temp bytes” without a credible amortization story for quant work

## Freeze Rules

- Do not reopen `m64` or `m256` without a newer profile-backed Candidate Card.
- Do not spend benchmark budget on `m8` before the family-wide `A-pack` redesign resolves.
- Do not open a branch whose description sounds like cleanup, hoist, fast path, or prep improvement without deleting a whole bucket.
- Do not spend another slot on shape-local local-requant `A-pack` deletion; `v121`, `v122`, and `v125` already proved that deleting the external temp law is not enough when quant work still scales with output-column CTAs.
