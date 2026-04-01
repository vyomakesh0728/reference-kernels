# `mxfp4-mm` Multi-Shape Portfolio Ladder

## Goal

Drive the overall `mxfp4-mm` geomean down by deleting whole-call overhead across the hot exact-shape family:

- `m4`
- `m16`
- `m32`
- `m64`
- `m256`

This ladder exists because the profiled self GPU time is much smaller than wall time on the public benchmark shapes, so the frontier is primarily:

- launches
- temp write/read traffic
- setup and address shaping
- runtime/orchestration
- non-math data movement

not MFMA body throughput alone.

## Current Best Anchor

- best measured trunk:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_m32_k512_scaleaddr_v119/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_m32_k512_scaleaddr_v119/submission.py)
- best measured geomean:
  `23.0711 us`
- benchmark:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260328-003505-v119-benchmark/stages/01_benchmark/result.txt](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260328-003505-v119-benchmark/stages/01_benchmark/result.txt)
- fresh profile:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260328-010439-v119-profile-rocprof/stages/01_profile_rocprof/profile/profile_summary.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260328-010439-v119-profile-rocprof/stages/01_profile_rocprof/profile/profile_summary.json)

## Why Portfolio First

On the current `v119` path:

- `m32` self GPU is only `1.718 us`, while measured wall time is `16.6 - 16.7 us`
- `m16` self GPU is only `1.838 us`, while measured wall time is `38.7 us`
- `m256` self GPU is only `0.998 us`, while measured wall time is `27.2 us`

Operational conclusion:

- exact-shape kernel body work matters only when it deletes a whole bucket
- the next wins must be portfolio-wide whole-call deletions, not pretty single-shape kernel polish

## Current Exact-Shape Call Law

These are the active exact-shape call structures on top of `v119`.

### `m4`

- current path:
  direct-entry wrapper + exact kernel
- helper launches:
  `mxfp4_pack_a_fixed`
- extra temp law:
  global `a_packed + a_scale`
- current visible wall time:
  about `17.4 us`

### `m16`

- current path:
  direct-entry wrapper + exact kernel
- helper launches:
  `mxfp4_pack_a_fixed`
- extra temp law:
  global `a_packed + a_scale`
- current visible wall time:
  about `38.7 us`

### `m32`

- current path:
  direct-entry wrapper + exact raw-`b_q` / raw-`b_scale_sh` kernel
- helper launches:
  `mxfp4_pack_a_fixed`
- extra temp law:
  global `a_packed + a_scale`
- current visible wall time:
  about `16.6 - 16.7 us`

### `m64`

- current path:
  direct-entry wrapper + row-major `b_scale` repair + exact kernel
- helper launches:
  `mxfp4_pack_a_fixed`
  `mxfp4_unshuffle_b_scale`
- extra temp law:
  global `a_packed + a_scale`
  global row-major `b_scale`
- current visible wall time:
  about `29.7 us`

### `m256`

- current path:
  direct-entry wrapper + exact raw-`b_q` / raw-`b_scale_sh` kernel
- helper launches:
  `mxfp4_pack_a_fixed`
- extra temp law:
  global `a_packed + a_scale`
- current visible wall time:
  about `27.2 us`

## Ladder Order

Every branch must delete one whole-call bucket and justify why it helps the total portfolio geomean.

### Ladder 1: Compiled Direct-Entry Everywhere

Intent:

- eliminate Python-owned exact-path shaping as the default hot path law
- keep the exact dispatch table brutally explicit

Questions:

- which exact shapes still pay Python-owned temp shaping or fallback routing?
- can each hot exact shape use one compiled direct-entry path as the default call contract?

Success condition:

- no exact hot shape is still using Python-owned helper routing when a compiled direct-entry exists

### Ladder 2: Kill Helper Launches Before Body Polish

Intent:

- delete standalone helper launches only when the deleted work does not return as duplicate hot-path work

Priority:

1. `m64` row-major `b_scale` repair if a branch can remove the helper without paying back too much in kernel address math
2. any remaining exact-shape wrapper-owned prep/repair helper that has not already been proven plateau or catastrophic

Hard rule:

- no new `A-pack` local feeder branch
- helper deletion must lower total work, not just move it

### Ladder 3: Remove Temp Laws, Not Just Temp Tensors

Intent:

- target write-then-read temp traffic where the temp exists only to bridge helper kernels into the exact kernel

Priority:

1. `m64` row-major `b_scale` temp
2. exact-shape non-`A-pack` temps
3. `A-pack` only if a future branch changes duplication law instead of moving the same quant work into the compute path

### Ladder 4: Constant-Body Public Shapes

Intent:

- remove generic branches, tails, and repeated address shaping on the highest-leverage public shapes

Priority:

1. public `m32, k=512`
2. public `m4`
3. public `m64, k=2048` only if the branch deletes a real whole bucket and not just generic loop text

Hard rule:

- constant-body clones are allowed only when they delete setup/addressing work with a concrete portfolio story

### Ladder 5: Runtime/Orchestration Collapse

Intent:

- reduce host/runtime overhead between exact dispatch, helper setup, temp allocation, and kernel launch

Questions:

- can exact-shape direct-entry become the only hot serving contract?
- can temp sizing and helper selection become compile-time shape law instead of per-call runtime shaping?

Success condition:

- exact-path runtime logic becomes mostly dispatch selection plus one compiled call

## Current Closed Lanes

- exact-shape local `A-pack` feeder rewrites:
  `v121`, `v122`, `v125`
- bounded `C2/C4` exact-wide `A-pack` service
- macro-cluster `A-pack` paper designs without a legal handoff mechanism
- `m64` body-only unrolled specialization without a deleted whole bucket:
  `v124`

## Allowed Next Spend Types

Only these lane classes should spend remote budget right now:

- multi-shape orchestration and direct-entry collapse with one deleted whole bucket
- exact-shape non-`A-pack` helper deletion with clear portfolio leverage
- public-shape constant-body deletion of setup/addressing work
- paper-only `A-pack` duplication-law research until a legal mechanism exists

## Branch Selection Rule

Before coding a branch, answer all of:

1. which exact shape does it touch?
2. which whole-call bucket disappears?
3. does the work disappear, or only move?
4. why should it improve total geomean instead of one local shape?
5. what previous negative branch does it avoid repeating?

If any answer is weak, veto the branch on paper.
