# AMD Kernel Hackathon -- reference-kernels

## What This Is
AMD $100K kernel competition via GPU MODE. Team: code1 (vyom, elliot, pritam).
Target GPU: MI355X (CDNA4, gfx950). Deadline: April 7, 2026.

**Session memory:** Root **`HISTORY.md`** (append per iteration); **`AGENTS.md`** has the Popcorn **`test` → `benchmark` → `leaderboard`** loop and logging habits—read after compaction. **Codex:** **`codex.md`** → `AGENTS.md`.

## Elliot's Problem: mixed-mla (leaderboard: amd-mixed-mla)
MLA decode attention. DeepSeek R1 forward_absorb path.
- 16 query heads, 1 KV head (16:1 GQA), qk_head_dim=576, v_head_dim=512
- Decode only: q_seq_len=1, kv_seq_len up to 8192
- Input: (q bf16, kv_data dict with bf16/fp8/mxfp4, qo_indptr, kv_indptr, config)
- Output: (total_q, 16, 512) bf16
- Tolerance: rtol=1e-01, atol=1e-01

## Current Status: 66 µs geomean, rank ~38/88. #1 is 29 µs.
- **Production:** `submission.py` = aiter fp8 with aggressive kv8k config (~66 µs)
- **HIP WIP:** Dead path (load_inline >17 min compile timeout)
- **Gap to #1:** need ~2.3× speedup from current best

## 2026-04-02: tl.dot_scaled WORKS on gfx950!

**BREAKTHROUGH: Fixed tensor layout for `tl.dot_scaled`**

Previous tests failed with "Reduction dimension should pack the same number of elements".
Fixed by studying Triton's test_matmul.py:
- **B tensor**: Generate (N, K), pack along K → (N, K//2), transpose → (K//2, N)
- **B_scale**: Shape (N, K//32) - NOT transposed!
- **rhs_k_pack=True** for K-packed FP4 data

**Test result** (workflow 23896940925):
```
DOT_SCALED V3: SUCCESS! C sum: -430.52777099609375
```

**Critical insight for bf16 × mxfp4:**
- Triton uses **software emulation** (upcasts mxfp4 to bf16)
- Still get **2× bandwidth savings** from reading mxfp4
- But **no native FP4 compute** - would need fp8×mxfp4 or mxfp4×mxfp4

**Challenge for MLA:**
- QK_HEAD_DIM=576 is NOT power of 2
- `tl.dot()` on AMD needs power-of-2 dims
- aiter handles this with 18 MFMA tiles (576/32=18)

**Files:**
- `test_dotscaled_v3.py` - working dot_scaled test
- `triton_mxfp4_qk.py` - WIP MLA QK kernel

## Novel paths to #1 (pick one)
1. **Single HIP kernel, zero Python dispatch** — fuse stage1+reduce, pre-allocate everything, one `hipLaunchKernel`
2. **mxfp4 KV + `V_MFMA_SCALE_F32_16x16x128_F8F6F4`** — 2× bandwidth savings; native fp4 MFMA
3. **Single-pass (no split-K)** — eliminates 256 MiB split buffer traffic on large shapes
4. **Decode #1's "pg8"** — their filename hints at "persistent group 8" or custom tiling

## ⚠️ CRITICAL: load_inline is DEAD on gfx950
**2026-03-31 finding:** `torch.utils.cpp_extension.load_inline` compilation for gfx950 takes **>17 minutes**, exceeding the harness timeout. ANY JIT HIP kernel approach is infeasible.

**Viable paths:**
1. **Pure aiter** — stick with pre-compiled assembly (~60 µs, rank ~38)
2. **Pre-compiled HIP** — compile offline on gfx950 machine, upload .co files (complex)
3. **Triton** — may have compilation caching (untested on gfx950)

**Dead paths:**
- `load_inline` HIP/C++ — compilation timeout
- Any runtime JIT for gfx950

## Active direction: Triton + inline asm (or pure aiter)

**ISA Reference:** See **`docs/mi355x_isa_reference.md`** for comprehensive MI355X/gfx950 instruction tables.

Since HIP JIT (`load_inline`) is infeasible (>17min compile), options are:
1. **Triton + inline asm** - Triton has compilation caching, can embed GCN asm via `tl.inline_asm_elementwise`
2. **Pure aiter tuning** - num_splits, page_size, fast_mode optimization
3. **Native MXFP4 path** - `v_mfma_scale_f32_16x16x128_f8f6f4` with type=4 (FP4) - 2× BW savings

**Key optimizations from ISA doc:**
- `v_mfma_scale_f32_16x16x128_f8f6f4` - native FP4/FP8 with E8M0 scales
- `buffer_load_dwordx4 ... lds` - global→LDS bypass VGPRs
- `ds_read_b128 a[0:3]` - LDS→AGPR for MFMA feed
- `v_cvt_pk_fp8_f32` - fast softmax score→FP8 pack
- 8-wave ping-pong scheduling (HipKittens)

## MFMA Kernel Progress
- `__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8` compiles and runs on gfx950
- Set `PYTORCH_ROCM_ARCH=gfx950` for fast single-arch compilation
- Register mapping confirmed: D[lane%16, (lane/16)*4:+4], A/B[lane%16, (lane/16)*8:+8]
- MFMA REGISTER MAPPING SOLVED (from ROCm/amd_matrix_instruction_calculator):
  D = A * B (NOT A * B^T!)
  A[i][k]: lane%16=i(row), k=8*(lane/16)+4*gpr+byte
  B[k][j]: lane%16=j(col), k=8*(lane/16)+4*gpr+byte
  D[i][j]: lane%16=j(col!), i=4*(lane/16)+gpr
  Key: lane%16 = COLUMN in D output (not row as I assumed)

## aiter Assembly Reverse Engineering (KEY FINDINGS)
**Analyzed `mla_a8w8_qh128_m32x4_n16x2_msk0.co` — 7159 lines of GCN assembly**

### Critical Optimizations We Were Missing:

1. **USE ACCUMULATOR GPRs (AGPRs)**
   ```asm
   ds_read_b128 a[0:3], v29          // Load to AGPRs, not VGPRs!
   v_mfma_f32_16x16x32_fp8_fp8 v[40:43], a[72:73], a[0:1], 0
   ```
   - MFMA inputs come from `a[...]` registers, outputs go to `v[...]`
   - We were loading to VGPRs, causing register pressure

2. **BUFFER→LDS DIRECT LOADS**
   ```asm
   buffer_load_dword v28, s[16:19], 0 offen lds  // BYPASSES VGPRs
   ```
   - Uses `lds` modifier to load global→LDS directly
   - We loaded global→VGPR→LDS (wasteful)

3. **HARDWARE FP8 CONVERSION**
   ```asm
   v_cvt_pk_fp8_f32 v40, v40, v41              // 2 floats → 2 fp8
   v_cvt_pk_fp8_f32 v40, v42, v43 op_sel:[0,0,1]  // pack high
   ```
   - **This is how they convert softmax scores to FP8 for V MFMA**
   - We used software conversion — should use `__builtin_amdgcn_cvt_pk_fp8_f32`

4. **128-BIT LDS READS**
   ```asm
   ds_read_b128 a[0:3], v29 offset:0
   ds_read_b128 a[4:7], v29 offset:64
   ds_read_b128 a[8:11], v29 offset:128
   ```
   - Reads 16 bytes at a time into 4 AGPRs
   - Pattern: 9 reads × 16B = 144B per tile load

5. **HARDWARE EXP**
   ```asm
   v_exp_f32_e32 v40, v40  // base-2 exp
   ```
   - Uses `v_exp_f32_e32` (exp2) for softmax
   - 8 scores per wave in parallel

### Kernel Naming Decoded
- `mla_a8w8_qh16_qseqlen1_gqaratio16.co` = **our exact config**
  - A8W8 = FP8 activations/weights
  - QH16 = 16 query heads
  - qseqlen1 = decode (seq_len=1)
  - gqaratio16 = 16:1 GQA (16 Q heads : 1 KV head)
- `_ps` suffix = persistent mode (uses work_metadata)

### Stats from FP8 Kernel
- **816 MFMA instructions** (heavy matrix compute)
- **204 buffer→LDS loads** (direct memory→LDS)
- **696 LDS read/write ops** (LDS↔AGPR)
- **48 FP8 conversion ops** (`v_cvt_pk_fp8_f32`)
- **114 exp/rcp ops** (softmax)
- **0 atomic ops** (no inter-block sync)

### Implementation Path
To match aiter's performance, our HIP kernel needs:
1. Use inline asm for `ds_read_b128` to AGPRs
2. Use `buffer_load_dword ... lds` for direct global→LDS
3. Use `__builtin_amdgcn_cvt_pk_fp8_f32` for score→FP8
4. Use `__builtin_amdgcn_exp_f32` for exp2 (or `__expf`)
5. Tile structure: M=32×4, N=16×2 like their kernels

## Dead Ends (do NOT retry)
- **load_inline HIP on gfx950**: Compilation takes >17 minutes, exceeds harness timeout. CRITICAL BLOCKER.
- **Software mxfp4 dequant** (Triton or HIP scalar): 30-68x slower. Useless without hardware MFMA fp4.
- **aiter mxfp4 MLA path**: does not exist. "cannot get heuristic kernel kv_type:byte"
- **CUDA graphs with aiter**: memory access fault on replay. aiter asm kernels do internal allocations.
- **Direct stage1+reduce API**: same perf as mla_decode_fwd. PyTorch caching allocator already makes split buf alloc free.
- **page_size=8 with aiter**: crashes with memory fault. The paging contract (kv_indptr in pages vs tokens, kv_indices mapping) is broken for ps>1. aiter's own test only uses page_size=1.
- **Fused HIP V kernel** (probs @ dequant_V): slower than torch.einsum because serial KV loop per output dim.
- **Single-pass HIP with scalar V**: 100-300× slower than aiter due to serial V accumulation.

## What Works (current best approach)
- Hybrid a16w8/a8w8 dispatch: bf16 Q for small batches (skip fp8 quant), fp8 Q for large (save BW)
- Cached metadata, kv_indices, output buffers across calls (zero per-call allocations)
- page_size=2 + fast_mode=True for small cases, page_size=1 for large a8w8 cases
- Geomean: 69.1us. Small cases competitive (~27-40us), large cases slow (~150-303us)

## Key Profiling Data (bs=256, kv=8192)
```
Total GPU time: 24us    Total wall time: 293us
mla_decode_stage1_asm_fwd:  2.2us (9.2%)   -- the actual MLA assembly kernel
mla_a8w8_..._ps:           1.7us (7.2%)   -- MLA stage1
mla_reduce_v1:             1.6us (6.7%)   -- split-K reduce
aten::arange:              3.4us (14.2%)  -- kv_indices (should be cached)
aten::mul/amax/div/clamp:  8.0us (33.3%)  -- fp8 Q quantization (4 kernel launches)
aten::copy_:               3.0us (12.4%)  -- tensor copies
```
CPU dispatch overhead: ~270us. Each Python->C++->kernel transition costs ~10-30us.

## Roofline Analysis
```
bs=256, kv=8k: 1.2GB fp8 KV at 8 TB/s = 150us memory-bound roofline
bs=4, kv=1k:   2.4MB = 0.3us roofline
Geomean roofline across 8 shapes: ~8us
```
Current 69us = 8.6x above roofline. #1 at 29us = 3.6x above roofline.

## The HIP Kernel Architecture (from reading aiter/mla.py source)
aiter's algorithm is split-K attention with online softmax:
1. **stage1**: Grid of persistent workgroups. Each handles a KV split for a (batch, head) pair.
   Uses MFMA fp8 instructions for QK^T dot products. Writes partial output + LSE per split.
2. **reduce**: Combines partial results across splits using log-sum-exp correction.

For a custom HIP kernel:
- Single `torch.utils.cpp_extension.load_inline` C++ function
- Pre-allocate ALL buffers (splitData, splitLse, output, metadata)
- Use fp8 MFMA: V_MFMA_F32_16X16X32_FP8_FP8 (M=16 heads, K=32 fp8 dims per MFMA, N=16 KV tokens)
- 576/32 = 18 MFMA instructions per 16-KV-token tile for QK^T
- Online softmax in registers, V accumulation with same MFMA tile
- Split-K with num_splits tuned per shape
- Reduce kernel: simple log-sum-exp across splits

## MI355X Hardware (CDNA4, gfx950)
- 256 CUs, 4 SIMDs/CU, wavefront=64, 512 VGPRs + 512 AGPRs per SIMD
- 160KB LDS per CU
- HBM3E: ~8 TB/s, 288GB
- FP8 MFMA: V_MFMA_F32_16X16X32_FP8_FP8, V_MFMA_F32_32X32X16_FP8_FP8
- FP4 MFMA: V_MFMA_SCALE_F32_32X32X64_F8F6F4 (native fp4 with E8M0 scales)

## Leaderboard Intel
- #1 Danishlynx 29.0us: "submission_v75_pg8_8k.py" (pg8 = unknown, NOT aiter page_size)
- #2 Nicky Pochinkov 32.7us: "submission-v1774310923.py" (1774 iterations)
- olezhka_007 54.3us: "submission_a16w8_ps2.py" (a16w8 + ps2)
- noobmaster69_og 54.5us: "submission_bf16kv_prejit.py" (bf16 KV + pre-JIT)

## Build/Run
- `popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test/benchmark/leaderboard --no-tui problems/amd_202602/mixed-mla/submission.py`
- benchmark mode: unlimited submissions, ~10min round trip
- leaderboard mode: 1/hour rate limit
- test mode: unlimited, ~5min round trip

## Eval Harness (problems/amd_202602/eval.py)
- Timed region: just `output = custom_kernel(data)` between CUDA events
- Warmup: one run of test[0] before benchmarking
- L2 cache cleared between benchmark iterations
- Subprocess persists across calls (caches survive)
- 900s timeout for benchmark, 17min GitHub Actions workflow limit

## aiter Source (ROCm/aiter on GitHub)
- `aiter/mla.py`: Python wrapper. Two modes: persistent (with metadata) and non-persistent (auto splits).
- `csrc/cpp_itfs/mla/asm_mla_decode_fwd.cpp`: C++ stage1 binding
- `hsa/gfx950/mla/`: Pre-compiled assembly kernels (.co files)
- Non-persistent mode auto-tunes num_kv_splits via `get_meta_param()` and uses a Triton reduce kernel
- Persistent mode uses `mla_reduce_v1` (C++ reduce kernel)
