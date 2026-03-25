# `mxfp4-mm` Profile-Derived Branch Queue

## Source of Truth

Current queue is derived from the first real MI355X exact-shape profile run on the compounded `v97` trunk:

- run dir:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile)
- raw zip:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile/profile_20260325_051711_run0.zip](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile/profile_20260325_051711_run0.zip)
- summary:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile/stages/01_profile_rocprof/profile/profile_summary.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile/stages/01_profile_rocprof/profile/profile_summary.json)
- cards:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile/stages/01_profile_rocprof/profile/candidate_cards.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-051230-compound-v97-profile/stages/01_profile_rocprof/profile/candidate_cards.json)

Operational rule:

- when this queue exists, follow it before opening a new exact-shape branch
- do not bypass it with intuition or polish work
- only replace it after a newer real MI355X profile run produces a stronger queue

## Queue Order

1. leaderboard `v97` when the time gate reopens
2. `m32`: delete generic exact-wide `B-pack/repack`
3. `m16`: delete generic tiny-path `B-scale` row-major materialization
4. `m4`: reopen exact `A-pack` only because the refreshed `v97` profile now shows it dominates after the raw-scale win
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

### `m16-Q2`

- `shape`: `m16`
- `deleted_cost_center`: `generic tiny-path B-scale row-major materialization on exact m16`
- `expected_upside_source`: kernelbot profile shows exact `m16` still spends most of its CUDA time in tiny-path scale decode/materialization rather than MFMA
- `why_larger_than_noise`: `a_pack_share=0.411`, `b_scale_decode_share=0.411`, `kernel_share=0.178`; this is a whole-bucket deletion candidate, not wrapper polish
- `touched_symbols_or_regions`:
  - `mxfp4/exact_m16`
  - `mxfp4/exact_m16/b_prep`
- `forbidden_edits`:
  - do not retry vec-load-only body rewrites
  - do not spend another slot on wrapper polish without deleting this bucket
- `success_gate`: `m16 < 39.5 us`
- `allowed implementation direction`:
  - decode final scale packets from raw `b_scale_sh` inside the exact `m16` direct path
  - keep `A-pack` and the dense exact body unchanged in the first branch

### `m4-Q2`

- `shape`: `m4`
- `deleted_cost_center`: `generic mxfp4_pack_a_fixed on exact m4`
- `expected_upside_source`: kernelbot profile shows the exact `m4` path is now dominated by `A-pack`, not by scale decode or the MFMA body
- `why_larger_than_noise`: `a_pack_share=0.708`, `b_scale_decode_share=0.000`, `kernel_share=0.292`; the raw-scale win changed the balance enough that the old `v94` negative no longer closes this lane
- `touched_symbols_or_regions`:
  - `mxfp4/exact_m4`
  - `mxfp4/exact_m4/a_pack`
- `forbidden_edits`:
  - do not reopen the broken `v79` launch structure
  - do not rewrite `m8` or `m16` in the same branch
- `success_gate`: `m4 < 17.3 us`
- `allowed implementation direction`:
  - keep the proven wave64 ownership model and the winning raw `b_scale_sh` path from `v96`
  - delete only the exact `m4` `A-pack` bucket

## Closed Lanes

### `m64`

- `status`: closed for now
- `reason`: the `v97` profile did not isolate a legal undeleted whole bucket
- `why`: `a_pack_share=0.316`, `b_scale_decode_share=0.342`, `kernel_share=0.342`, `b_pack_share=0.000`
- rule: do not open another `m64` branch until a future profile names a new legal bucket

### `m256`

- `status`: closed for now
- `reason`: the compounded `v97` trunk already deleted the resolved exact-wide `B-pack` bucket, and the fresh profile does not isolate another legal undeleted bucket
- `why`: `a_pack_share=0.316`, `b_scale_decode_share=0.342`, `kernel_share=0.342`, `b_pack_share=0.000`
- rule: do not open another `m256` branch until a future profile names a new legal bucket

## Recent Wins Locked Into The Trunk

- `v93`: exact `m64` raw-`b_q` deletion resolved positive and is part of the live trunk
- `v95`: exact `m256` raw-`b_q` deletion resolved positive and is now compounded into `v97`
- `v96`: exact `m4` raw `b_scale_sh` / scale-decode deletion resolved positive and is part of `v97`
- `v97`: compounded trunk benchmarked at `25.7488 us`, then `25.8866 us`; treat it as the current measured frontier while leaderboard waits for the time gate

## Freeze Rules

- Do not reopen `m64` or `m256` without a newer profile-backed Candidate Card.
- Do not spend benchmark budget on `m8` before `m16` resolves.
- Do not open a branch whose description sounds like cleanup, hoist, fast path, or prep improvement without deleting a whole bucket.
