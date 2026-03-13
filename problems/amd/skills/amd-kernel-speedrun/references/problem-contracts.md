# Problem Contracts

## mxfp4_mm

- Local file: `/Users/v/reference-kernels/problems/amd/fp8-mm/task.yml`
- Local benchmark envelope:
  - `m` includes `64, 96, 128, 512, 1024, 6144`
  - `n` includes `512, 576, 1536, 3072, 4096, 4608, 7168`
  - `k` includes `128, 256, 512, 1536, 2048, 2304, 7168`
- Important nuance:
  - the live leaderboard target is `amd-mxfp4-mm`
  - local wording is older MI300/FP8-style
  - use the YAML for shape intuition, but trust the live server-side contract for semantics
- Current team rule:
  - preserve shuffled MXFP4 semantics first
  - for a true Triton candidate, `custom_kernel` must not keep the hot path on `aiter.gemm_a4w4`
  - for a true HIP candidate, `custom_kernel` must use `load_inline` on `gfx950` and stay in the Python + HIP compilation path
  - CDNA4 scaled MFMA (`V_MFMA_SCALE_F32_16X16X128_F8F6F4`, `V_MFMA_SCALE_F32_32X32X64_F8F6F4`) is the long-term throughput target

## moe_mxfp4

- Local file: `/Users/v/reference-kernels/problems/amd/moe/task.yml`
- Local tests:
  - `dhidden=7168`
  - `dexpert=2048`
  - smaller routed expert counts for tests
  - `bs=1/2`, `seqlen=512/8192`
- Local benchmarks:
  - `nroutedexperts=32`
  - `nexpertspertoken=4`
  - `bs=1`
  - `seqlen=2048/8192`
- Current team rule:
  - keep routing/top-k semantics fixed while rewriting one stage at a time
  - for a true Triton candidate, `custom_kernel` must not keep the hot path on `fused_moe`

## mixed_mla

- Local file: `/Users/v/reference-kernels/problems/amd/mla-decode/task.yml`
- Local shape envelope:
  - `batchsize=128`
  - `dim=7168`
  - `dq=1536`
  - decode only, `sq=1`
  - benchmark `prefill=4096/6144`
- Current team rule:
  - optimize decode latency, especially the `q_seq_len=1` hot path
  - for a true Triton candidate, `custom_kernel` must not keep the hot path on `mla_decode_fwd`
