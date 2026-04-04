# mxfp4-mm non-A-pack frontier audit

Date: 2026-04-03
Repo: `/Users/v/reference-kernels/problems/amd`
Scope: read-only research pass on the live `13.406 us` anchor, explicitly excluding new A-pack rewrites and AITER.

## Executive conclusion

I agree the next spend should move off helper A-pack patching.
But the evidence says something narrower and harsher:

- pure kernel-body tuning is not the path from `13.406 us` to `<=7 us`
- the public exact benchmark paths already deleted most old non-A-pack B-temp laws
- the only remaining non-A-pack lanes with real upside are:
  1. B-feed/layout changes, especially `m16`
  2. wide-shape B reuse/caching, especially `m256`, then `m64`
  3. second-order register/load cleanup after a B-feed win lands

If "do not touch A-pack" means "do not spend on helper A-pack code or fused-A ownership experiments", that is sensible.
If it means "do not change anything that affects the current two-launch law", then `<=7 us` looks implausible from current evidence.

## Verified anchor

- `fp8-mm/submission.py` is byte-identical to `fp8-mm/submission_anchor_13p406.py`
- anchor benchmark artifact:
  - `.agent-loop/harness_runs/mxfp4_mm/20260402-120426-p0p5-runtime-collapse-owned-scaleflat-r1-benchmark/stages/01_benchmark/parsed_metrics.json`
- fresh profile artifact:
  - `.agent-loop/harness_runs/mxfp4_mm/20260402-143032-baseline-recheck-profile-r1-harness/stages/01_profile_rocprof/profile/profile_summary.json`

## Wall vs self table

| case | bench_us | self_cuda_us | a_pack_us | kernel_us | wall_minus_self_us |
|---|---:|---:|---:|---:|---:|
| m4 k512 n2880 | 9.90 | 2.918 | 2.079 | 0.839 | 6.982 |
| m16 k7168 n2112 | 19.60 | 2.598 | 2.119 | 0.479 | 17.002 |
| m32 k512 n2880 | 9.98 | 2.438 | 1.959 | 0.479 | 7.542 |
| m32 k512 n4096 | 10.10 | 2.078 | 1.599 | 0.479 | 8.022 |
| m64 k2048 n7168 | 18.10 | 0.958 | 0.479 | 0.479 | 17.142 |
| m256 k1536 n3072 | 16.40 | 0.958 | 0.479 | 0.479 | 15.442 |

Key interpretation:

- the benchmark wall time is still far above profiled GPU self time on every scored case
- `m64` and `m256` are the most extreme: almost all wall time is outside kernel self
- zeroing kernel self alone only models to about `12.822 us` geomean, so body-only polish is a dead lane for the `<=7 us` objective

## Live exact public path audit

Exact public fast paths in `custom_kernel(...)`:

- m4: `fp8-mm/submission.py:3778-3796`
- m16: `fp8-mm/submission.py:3797-3811`
- m32: `fp8-mm/submission.py:3812-3826`
- m64: `fp8-mm/submission.py:3827-3841`
- m256: `fp8-mm/submission.py:3842-3856`

Owned-workspace wrappers:

- m16: `1700-1754`
- m4: `1799-1852`
- m32: `2986-3043`
- m64: `3097-3150`
- m256: `3208-3262`

Common exact-path facts:

- Python still allocates one BF16 workspace tensor per call for every public exact shape
- Python still carves `c` out of that workspace with `narrow(...).view(...)`
- every owned-workspace wrapper still launches:
  1. `launch_mxfp4_pack_a_fixed_raw(...)`
  2. one exact compute kernel
- exact public benchmark paths do NOT launch standalone B repack/unshuffle helpers anymore
- exact public benchmark paths consume raw `b_q` and raw shuffled `b_scale_sh` directly
- `b_shuffle` is present in the top-level contract but unused on every exact public fast path

So the old generic B-temp deletions are mostly already harvested on the live benchmark anchor.

## Dead or exhausted non-A-pack lanes

From live code + `../../HISTORY.md`:

1. Graph replay on exact public wrappers: dead
- `exact_public_graph_replay_r1` captured empty graphs on runner and exploded to ~1.37 s per shape.

2. Host-only wrapper collapse: mostly exhausted
- `owned-workspace-nocheck-r1`, `owned-workspace-cpp-return-r1`, and related exact-wrapper cleanup branches were noise-to-worse.

3. One-shot sync/service designs: dead
- `m4` producer-consumer ready-flag path regressed badly.
- serial `m32` single-CTA sweep was catastrophic (`248-314 us` on the two scored m32 cases).

4. Pure kernel-body-only tuning is too small
- current kernel self is only `0.479-0.839 us` per shape.
- even a perfect kernel body would not move the portfolio close to `7 us`.

## Non-A-pack structural opportunities that are still real

### 1. `m16` B-feed/layout rewrite is the strongest non-A-pack lane

Why `m16` first:

- the exact `m16` kernel still reads raw row-major `b_q`
- each lane walks a different B row with stride `k/2 = 3584` bytes
- loop length is longest in portfolio (`K=7168`, 56 MFMA steps)
- scale decode still does division/mod arithmetic each step in `pack_scale_e8m0x4_lane_from_shuffled_exact_m16_k7168_fast(...)`
- launch grid is only `132 x 1` blocks, so latency hiding is weak

Relevant code:

- scale decode helper: `1292-1305`
- exact kernel: `1308-1375`
- public wrapper: `1700-1754`

Deleted cost center:

- raw row-major B feed with poor inter-lane access pattern
- per-step shuffled-scale decode arithmetic in the hottest thin exact shape

Fast falsifier:

- if a replacement B contract still requires a new per-call pack kernel, kill the branch unless the feed layout is already provided by the caller

### 2. `m256` B reuse is the best wide-shape non-A-pack lane

Why `m256` next:

- exact `m256` grid is `96 x 8`
- the same logical B tile is reread across 8 y-tiles because the kernel has no B staging/reuse path
- one-shot B-direct preparation is materially more plausible here than on thin shapes because the reread factor is high

Relevant code:

- exact scale loader: `2012-2052`
- exact kernel: `2808-2876`
- public wrapper: `3208-3262`

Deleted cost center:

- repeated raw-B reread across y-tiles
- repeated shuffled-scale decode on a shape with real B reuse opportunity

Fast falsifier:

- if the candidate adds a one-shot B pack that is not amortized by the `grid.y = 8` reread law, kill it quickly

### 3. `m64` B reuse is still open, but smaller than `m256`

Why after `m256`:

- exact `m64` grid is `224 x 2`
- there is only 2x B reread across y-tiles, so the amortization story is weaker
- scale decode is already mostly closed-form shift/add, so less arithmetic headroom than `m16` or `m256`

Relevant code:

- exact scale loader: `1979-2010`
- exact kernel: `2636-2702`
- public wrapper: `3097-3150`

Deleted cost center:

- repeated raw-B reread across the two y-tiles
- remaining B-side decode/contract waste in the public exact path

Fast falsifier:

- if a one-shot B-direct path adds more traffic than the 2x reread it removes, kill it

### 4. `m4` only matters if a real caller-provided B-feed contract exists

Why not first:

- low useful-work density already hurts `m4`
- but K is short, and there is no wide-shape B reread multiplier
- the only interesting non-A-pack upside is better B-feed/coalescing, not body math

Relevant code:

- exact kernel: `1530-1613`
- public wrapper: `1799-1852`

Deleted cost center:

- raw row-major thin B feed and per-step scalar register movement

Fast falsifier:

- if the path needs new per-call B packing, it is probably too small to matter

### 5. m4/m16 register-movement cleanup is real but second-order

What is live in code:

- m4/m16 zero all 8 dwords every step and fill 16 bytes scalar-by-scalar
- m32/m64/m256 already use the better pattern: zero only upper half, then vector-load lower 16B with `i32x4`

Relevant code:

- m16 scalar copy pattern: `1347-1355`
- m4 scalar copy pattern: `1571-1585`
- better pattern in wider kernels: `2444-2453`, `2678-2684`, `2852-2858`

Deleted cost center:

- unnecessary register clearing and scalar byte moves in thin exact kernels

Expected gain:

- not a first-spend lane by itself
- use only after a B-feed win or as a piggyback on an m16/m4 B-contract rewrite

## Important `b_shuffle` finding

`b_shuffle` is not yet a drop-in win.

Why:

- top-level dispatch only checks `b_shuffle.shape[0] == b.shape[0]`
- the in-repo helper `_get_b_preshuffled_mfma_fp4(b_q)` returns shape `[n/16, k_half*16]`

Implication:

- the helper result is not shape-compatible with the current visible `b_shuffle` contract
- the live benchmark runner may provide a different opaque `b_shuffle` layout than the in-repo helper
- any non-A-pack B-feed branch must first verify the actual runner contract instead of assuming the local helper matches it
- same caution as `b_scale_sh`: treat live contract tensors as opaque until proven otherwise

## Ranked non-A-pack patch ladder

1. `probe_live_bshuffle_contract_noquota`
- deleted_cost_center: avoid blind B-feed work against the wrong contract
- touched: local contract inspection only
- why larger than noise: prevents wasting another remote slot on a dead layout assumption
- major risk: none; read-only
- fast falsifier: runtime `b_shuffle` contract does not match any directly consumable exact-kernel feed layout

2. `m16_exact_bfeed_direct_contract_k7168_n2112`
- deleted_cost_center: raw-B stride + scale decode overhead on the worst thin exact shape
- touched: `custom_kernel`, new exact m16 wrapper/kernel surface, B-feed decode helper
- why larger than noise: strongest thin-shape B bandwidth problem in live code
- major risk: caller contract may not expose a usable feed layout
- fast falsifier: needs per-call B pack kernel or shows no measurable self-time movement

3. `m256_exact_breuse_direct_contract_k1536_n3072`
- deleted_cost_center: 8x reread of raw B across y-tiles + repeated scale decode
- touched: exact m256 wrapper/kernel, possible B-direct/cache path
- why larger than noise: `grid.y = 8` gives real reuse economics
- major risk: one-shot B prep can still overpay if not truly reused
- fast falsifier: B-prep traffic is not amortized by measured reread reduction

4. `m64_exact_breuse_direct_contract_k2048_n7168`
- deleted_cost_center: 2x reread of raw B across y-tiles
- touched: exact m64 wrapper/kernel, possible B-direct/cache path
- why larger than noise: still a real reuse bucket, but smaller than m256
- major risk: weaker amortization than m256
- fast falsifier: pack/contract setup costs exceed B reread savings

5. `m4_m16_vectorload_partialzero_cleanup`
- deleted_cost_center: scalar byte copies and unnecessary reg clearing in thin exact kernels
- touched: exact m4/m16 kernel bodies only
- why larger than noise: genuine register-movement cleanup already proven better in wider kernels
- major risk: too small alone
- fast falsifier: local compile/ISA diff shows no reduction in vectorized loads or live-range pressure

6. `wide_post_bfeed_l2lds_staging_32x32x64`
- deleted_cost_center: repeated VGPR-mediated B ingress after a B-direct win exists
- touched: exact m64/m256 kernel bodies
- why larger than noise: only matters after a usable wide-shape B contract exists
- major risk: LDS staging without true reuse just adds traffic and pressure
- fast falsifier: no measured reduction in B reread or register pressure

## Bottom line

The right pivot is not "keep polishing the current kernels".
The right pivot is:

- stop spending on helper A-pack micro-variants
- stop pretending kernel-body-only work can buy `13.4 -> 7 us`
- move the next research/implementation spend to B-feed law and wide-shape B reuse
- start with `m16` for thin-shape B bandwidth/coalescing
- start with `m256` for wide-shape B reuse economics

But be explicit:

- a no-A-pack research program is reasonable
- a strict no-change-to-the-two-launch-law program probably is not enough to reach `<=7 us`
