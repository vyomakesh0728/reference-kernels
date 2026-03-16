# New Codex Session Handoff

## Start Here

- Problem: `mxfp4_mm`
- Stable best trunk: [fp8-mm/hip_phase2_working.py](/Users/v/reference-kernels/problems/amd/fp8-mm/hip_phase2_working.py)
- Best benchmark run: [.agent-loop/harness_runs/mxfp4_mm/20260314-044528-aiter-corrected-hybrid-m16-m64](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260314-044528-aiter-corrected-hybrid-m16-m64)
- Best benchmark objective: `319.371 us`

## What The Trunk Does

- Exact `m=16` uses corrected-A AITER packed FP4 path.
- Exact `m=64` uses corrected-A AITER packed FP4 path.
- Middle regimes use the custom HIP path.
- Large regime still has a safe non-native fallback.

This is a stable hybrid winner, not the endgame kernel.

## Current Frontier

- We are in CDNA4 ladder step 6, not step 7.
- The bottleneck is the direct native `32x32x64` scaled-MFMA path for the `m=32/64/256` family.
- Do **not** re-debug harness, broad correctness, or the trunk.
- Do **not** start a big multi-wave / ping-pong / occupancy campaign yet.
- Step 7 only makes sense after the direct scaled-MFMA engine is healthy and competitive.

## Main Goal

Replace the hybrid/AITER path regime by regime with native scaled-MFMA serving kernels built around:

- `V_MFMA_SCALE_F32_16X16X128_F8F6F4`
- `V_MFMA_SCALE_F32_32X32X64_F8F6F4`

## What We Already Proved

- The stable trunk is healthy and benchmark-competitive.
- The `m=16` scaled-MFMA path is comparatively healthier than the `32x32x64` family.
- The direct `32x32x64` native path still fails before it becomes a usable serving engine.
- This is not just a benchmark harness issue.
- This is not just an output-store bug.
- This is not fixed by forcing constant scales.
- This is not fixed by isolating away the later AITER serving path.

## Highest-Signal Failed Probes

- [native_scaled_m64_direct_chunk1_probe_v1/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_m64_direct_chunk1_probe_v1/submission.py)
- [native_scaled_m64_direct_chunk0_probe_v1/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_m64_direct_chunk0_probe_v1/submission.py)

Key failed runs:

- [20260315-091716-native-scaled-m64-direct-chunk1-probe-v2-const-scale](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260315-091716-native-scaled-m64-direct-chunk1-probe-v2-const-scale)
  Same GPU memory fault even with `scale_a = 127` and `scale_b = 127`.
- [20260315-093441-native-scaled-m64-direct-chunk1-probe-v3-const-scale-isolated](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260315-093441-native-scaled-m64-direct-chunk1-probe-v3-const-scale-isolated)
  Same GPU memory fault even after removing the later AITER return path.
- [20260315-110633-native-scaled-m64-direct-chunk1-probe-v4-phased](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260315-110633-native-scaled-m64-direct-chunk1-probe-v4-phased)
  Same fault before any JSON phase markers flush, so the crash is still inside or immediately around the direct scaled-MFMA feed/launch path.

## Best Public Contract References

These are the closest useful references for the operand contract. Prefer these over random GitHub search hits.

- [/tmp/aiter-inspect.vhxmO2/op_tests/opus/device/test_mxfp.cu](/tmp/aiter-inspect.vhxmO2/op_tests/opus/device/test_mxfp.cu)
- [/tmp/aiter-inspect.vhxmO2/csrc/include/opus/opus.hpp](/tmp/aiter-inspect.vhxmO2/csrc/include/opus/opus.hpp)
- [/tmp/aiter-inspect.vhxmO2/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_afp4wfp4.py](/tmp/aiter-inspect.vhxmO2/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_afp4wfp4.py)
- [/tmp/aiter-inspect.vhxmO2/csrc/ck_gemm_a4w4_blockscale/include/gemm_a4w4_blockscale_common.cuh](/tmp/aiter-inspect.vhxmO2/csrc/ck_gemm_a4w4_blockscale/include/gemm_a4w4_blockscale_common.cuh)
- [important_papers/amd-instinct-cdna4-instruction-set-architecture.pdf](/Users/v/reference-kernels/problems/amd/important_papers/amd-instinct-cdna4-instruction-set-architecture.pdf)

## What Those References Mean

- The healthy contract is much closer to the AITER/Opus/Triton preshuffle path than to our raw row-major direct feed.
- `opus::mfma<fp4_t, fp4_t, fp32_t, 32, 32, 64>` is the closest known-good direct reference for the `32x32x64` family.
- Triton/AITER preshuffle logic strongly suggests the block-scale layout and operand feed are instruction-local and nontrivial.
- The most likely remaining mismatch is still the direct operand contract for `32x32x64`, especially how packed B and scales are fed into the native kernel.

## Next Suggested Move

- Do **not** start from scratch.
- Keep the stable trunk untouched at `319.371 us`.
- Keep focusing on the direct native `32x32x64` scaled-MFMA bring-up.
- Compare our direct feed against the healthy preshuffled AITER/Triton contract more mechanically.
- If patching, prefer fixing the direct operand/feed layout over more random schedule changes.
- Only go deeper into step 7 after the native scaled engine is healthy and competitive.

## Useful Repo Breadcrumbs

- Team summary: [team_results/hip/2026-03-14/summary.md](/Users/v/reference-kernels/problems/amd/team_results/hip/2026-03-14/summary.md)
- Optimization skill: [skills/optimization-skill/SKILL.md](/Users/v/reference-kernels/problems/amd/skills/optimization-skill/SKILL.md)
- Correctness skill: [skills/amd-live-reference-correctness/SKILL.md](/Users/v/reference-kernels/problems/amd/skills/amd-live-reference-correctness/SKILL.md)
