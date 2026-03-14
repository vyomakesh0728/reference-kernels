## 2026-03-14 MM Update

Current stable best trunk:
- `fp8-mm/hip_phase2_working.py`
- benchmark geometric mean: `319.371 us`
- source run: `.agent-loop/harness_runs/mxfp4_mm/20260314-044528-aiter-corrected-hybrid-m16-m64`

What the current best trunk does:
- corrected-A AITER packed FP4 path for exact `m=16`
- corrected-A AITER packed FP4 path for exact `m=64`
- custom HIP path for the middle regimes
- safe non-native fallback for the large regime

Why scaled-MFMA is the next frontier:
- BF16 de-AITER replacements lost for `m=16`, `m=64`, and `m=256`
- exact `m=16` native scaled-MFMA is now test+benchmark stable, but slower than trunk on its own
- exact `m=32` direct native scaled-MFMA previously hit benchmark-only GPU memory faults
- block-scaled CDNA4 semantics from the ISA are now the main guide, especially:
  - `V_MFMA_SCALE_F32_16X16X128_F8F6F4`
  - `V_MFMA_SCALE_F32_32X32X64_F8F6F4`
  - scale group over `K=32`
  - `E8M0` scale format
  - `ABID[0]` must be enabled for scale MFMA ops

What is included in this update:
- hand-rolled de-AITER / scaled-MFMA experiment orchestration changes in `agent_loop/handroll.py`
- current best trunk in `fp8-mm/hip_phase2_working.py`
- first assembled native-scaled bundle candidate at:
  - `.agent-loop/manual/native_scaled_bundle_m16_m32_m64_m256_v1/submission.py`
- dataset-mining query pack under `dataset_mining/kernelbot_data/`
- latest rules PDF, HipKittens paper, and benchmark/TFLOPS screenshots

Current assembled native-scaled direction:
- exact `m=16` on native `16x16x128` scaled-MFMA
- exact `m=32` on native `32x32x64` scaled-MFMA
- exact `m=64` as two native `32x32x64` tiles
- exact `m=256` on native HIP instead of `torch.mm`

Notes:
- the older `kernelbot-data` mining sandbox was useful for benchmark-failure memory, but it is no longer needed live for this scaled-MFMA bring-up and was shut down
- the main unresolved work is making the native scaled bundle outperform the hybrid trunk without reintroducing benchmark-only faults
