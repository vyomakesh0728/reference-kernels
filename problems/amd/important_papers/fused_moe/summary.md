# Fused-MoE Summary

Date: 2026-03-25 UTC

## What Changed Today

- Added the fused-MoE paper pack and architectural multiplier notes under this directory.
- Added a MoE-specific closed-loop coordinator, candidate-card gates, MoE retrieval sources, and MoE quota-watch support.
- Added MoE-specific skill references for branch order, cost-center gates, sub-agent prompts, and the `<=140 us` milestone push.
- Added named native MoE candidate sources for:
  - `dispatch_pack_sparse256_hip_pack_v1`
  - `stage1_grouped_bf16_sparse256_v1`
  - `stage1_grouped_bf16_re32_de512_bs512_v1`
  - `moe_scaled_mfma_correct_v1`

## Current Baselines

- Current stable repo control path: `moe_blockm_tp8_bs512_v1`
- Current stable geomean: `182.636368 us`
- Current repo submission is still AITER-backed `fused_moe(...)`

## Native Candidate Status

- `dispatch_pack_sparse256_hip_pack_v1`
  - Passed remote `test`
  - One benchmark run hit `174.996 us`
  - Rerun came back `183.138 us`
  - Decision: not stable enough to keep as the working baseline

- `stage1_grouped_bf16_sparse256_v1`
  - Reached remote `test`
  - Did not reach correctness-green

- `stage1_grouped_bf16_re32_de512_bs512_v1`
  - Preflighted and registered
  - Not promoted past the new scaled-MFMA baseline pivot

- `moe_scaled_mfma_correct_v1`
  - New correctness-first native baseline for the active MXFP4 fused-MoE contract
  - Uses `gfx950` `load_inline`
  - Uses a native scaled-MFMA exact `m16` stage kernel over raw MXFP4 weights/scales
  - No `fused_moe` fallback in the hot path
  - Two remote `test` attempts failed on runtime ABI/layout issues, not semantic mismatches:
    - first failure: rank-2 packed/scale tensor normalization
    - second failure: raw scale layout mismatch during stage1 scale normalization
  - Latest source patches normalize packed inputs and tolerate shared-per-K-block scale layouts

## Next Step

- Re-test `moe_scaled_mfma_correct_v1` at the next MoE `test` quota window.
- If it passes, use it as the correctness baseline for:
  - stage1 sparse256 optimization
  - stage1 dense32 optimization
  - stage2 weighted-reduce specialization
  - later full-pipeline MFMA/LDS tuning toward `<=140 us`

## Important Paths

- Repo control path:
  - `/root/reference-kernels/problems/amd/moe/submission.py`
- MoE closed loop:
  - `/root/reference-kernels/problems/amd/agent_loop/moe_closed_loop.py`
- New scaled-MFMA correctness baseline:
  - `/root/reference-kernels/problems/amd/.agent-loop/manual/moe_scaled_mfma_correct_v1/submission.py`
