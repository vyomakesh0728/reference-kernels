# MXFP4 Through V45

## Canonical Path

Treat these as the proven `mxfp4_mm` milestones:

1. Stable hybrid trunk:
   [/Users/v/reference-kernels/problems/amd/fp8-mm/hip_phase2_working.py](/Users/v/reference-kernels/problems/amd/fp8-mm/hip_phase2_working.py)
   - healthy benchmark anchor
   - `319.371 us`
   - not the endgame kernel

2. Correctness and fault-recovery phase:
   - driven by staged probes under [/Users/v/reference-kernels/problems/amd/.agent-loop/manual](/Users/v/reference-kernels/problems/amd/.agent-loop/manual)
   - especially the `native_scaled_m32_fault_probe_*` and `native_scaled_m16_fault_probe_*` lineages
   - goal was fault-free, correctness-green native scaled-MFMA

3. First big scaled-MFMA breakthrough:
   [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_compiled_a_pack_m8_v34/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_compiled_a_pack_m8_v34/submission.py)
   - `32.946 us`
   - real scaled-MFMA
   - later rejected as impure because of pointer-keyed cross-call B-side caches

4. First clean pure base:
   [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_pure_compiled_bscale_v44/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_pure_compiled_bscale_v44/submission.py)
   - `46.964 us`
   - current pure recovery anchor

5. Current promoted winner:
   [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_m32_bfrag_direct_v45/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_m32_bfrag_direct_v45/submission.py)
   - `39.355 us` benchmark
   - `40.093 us` leaderboard
   - current proven base

## What Actually Worked

- Keep the family split:
  - `16x16x128` for `m<=16`
  - `32x32x64` for `m>=32`
- Move hot A-pack logic into compiled HIP.
- Move B-scale unshuffle out of Python.
- Keep the wide `32x32x64` math stable while attacking data movement.
- For `v45`, prepack wide-path `B_q` into the direct per-lane fragment layout so the wide kernel can stop rebuilding fragments byte-by-byte at runtime.

## What Not To Re-Derive

- Broad correctness was not the lasting blocker once staged `test -> benchmark -> leaderboard` runs were in place.
- Random schedule changes were not the right next step before the direct wide contract was healthy.
- Wrapper-ownership or wrapper-fusion alone did not produce the next big win.
- Raw `B_shuffle` or raw `B_scale_sh` should not be fed into the wide `m32` ABI without proof.

## Dead Ends Before V45

- Generalized split-16 / chunked `m16` endgame ideas
- Broad m16 bring-up that reintroduced GPUVM faults without isolating the contract
- Naive shared-B or two-wave scheduling experiments before the data path was ready
- Thin/wide unification attempts that broke the wide ABI

## Read These Files First

- Handoff and early context:
  [/Users/v/reference-kernels/problems/amd/session-summary.md](/Users/v/reference-kernels/problems/amd/session-summary.md)
- First scaled-MFMA breakthrough:
  [/Users/v/reference-kernels/problems/amd/team_results/hip/2026-03-16/summary.md](/Users/v/reference-kernels/problems/amd/team_results/hip/2026-03-16/summary.md)
- Pure-path recovery:
  [/Users/v/reference-kernels/problems/amd/team_results/hip/2026-03-17/summary.md](/Users/v/reference-kernels/problems/amd/team_results/hip/2026-03-17/summary.md)
- Current wide winner:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_m32_bfrag_direct_v45/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_m32_bfrag_direct_v45/submission.py)
- Current closed-loop ledger:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/closed_loop/mxfp4_mm/experiment_ledger.jsonl](/Users/v/reference-kernels/problems/amd/.agent-loop/closed_loop/mxfp4_mm/experiment_ledger.jsonl)

## Reusable Rules

- Change one semantic axis per remote run.
- Keep `test` green before burning `benchmark`.
- Promote from the current measured winner, not from nostalgia for older branches.
- Treat `v45` as the real base until a measured successor beats it.
