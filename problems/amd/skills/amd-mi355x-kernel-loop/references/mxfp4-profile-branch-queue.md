# `mxfp4-mm` Profile-Derived Branch Queue

## Source of Truth

Current queue is derived from the latest real MI355X exact-shape profile run on the compounded `v101` trunk:

- run dir:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile)
- raw zip:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/profile_20260325_111743_run0.zip](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/profile_20260325_111743_run0.zip)
- summary:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/stages/01_profile_rocprof/profile/profile_summary.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/stages/01_profile_rocprof/profile/profile_summary.json)
- cards:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/stages/01_profile_rocprof/profile/candidate_cards.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/stages/01_profile_rocprof/profile/candidate_cards.json)

Operational rule:

- when this queue exists, follow it before opening a new exact-shape branch
- do not bypass it with intuition or polish work
- only replace it after a newer real MI355X profile run produces a stronger queue

## Queue Order

1. keep `v101` as the active measured and ranked trunk
2. `m16`: annihilate the separate exact `A-pack` launch only with a stronger amortized ownership model; do not inline full per-thread quantization into the MFMA threads and do not use full-panel shared-memory sweep kernels as direct replacements
3. `m4`: reopen only for the stricter “delete the launch entirely” form of `A-pack` annihilation
4. plan the next wide move as a family-wide `A-pack` collapse for `m32/m64/m256`
5. keep `m64` and `m256` closed until a future profile isolates a new legal whole-bucket deletion
6. keep `m8` isolated and test-green only

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

- `status`: closed for now
- `reason`: the `v101` profile did not isolate a legal undeleted whole bucket
- `why`: `a_pack_share=0.316`, `b_scale_decode_share=0.342`, `kernel_share=0.342`, `b_pack_share=0.000`
- rule: do not open another `m64` branch until a future profile names a new legal bucket

### `m256`

- `status`: closed for now
- `reason`: the compounded `v101` trunk already deleted the resolved exact-wide `B-pack` bucket, and the fresh profile does not isolate another legal undeleted bucket
- `why`: `a_pack_share=0.316`, `b_scale_decode_share=0.342`, `kernel_share=0.342`, `b_pack_share=0.000`
- rule: do not open another `m256` branch until a future profile names a new legal bucket

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
- `v109`: exact `m16` block-local 2-tile ownership passed the pre-spend audit on paper but still timed out badly enough that kernelbot cancelled the workflow after 12 minutes; keep this as a clean negative on “192-thread / 3-wave block-local A owner with 2-tile bundle” and do not benchmark this shape

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

## Freeze Rules

- Do not reopen `m64` or `m256` without a newer profile-backed Candidate Card.
- Do not spend benchmark budget on `m8` before the family-wide `A-pack` redesign resolves.
- Do not open a branch whose description sounds like cleanup, hoist, fast path, or prep improvement without deleting a whole bucket.
