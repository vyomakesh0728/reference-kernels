# Fused A-pack in exact m16 path Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Fuse A-pack into the exact m16 kernel so raw A is quantized on-the-fly inside the MFMA kernel, eliminating the separate mxfp4_pack_a_fixed launch and temporary buffers while keeping B’s contract unchanged.

**Architecture:** Add a new HIP kernel variant that takes raw A (bf16), b_q, and b_scale_sh and, per tile, computes the 32-element A scale byte and fp4 packed bytes in registers before issuing MFMA. Guard it behind MXFP4_FUSE_A_PACK and keep existing paths for fallback. Route the m16 direct entry to the fused kernel when the flag is enabled. (Optional Phase 2: extend to m4/m8/m32/m64/m256 by adding fused variants of their kernels.)

**Tech Stack:** PyTorch + ROCm HIP (load_inline), fp8-mm/hip_phase2_working.py, agent_loop harness.

---

### Task 1: Add a fused-A selftest harness (GPU smoke test)

**Objective:** Provide a tiny m16 test that compares fused output vs the existing pack+kernel output.

**Files:**
- Modify: fp8-mm/hip_phase2_working.py (near the bottom, after custom_kernel)

**Step 1: Write failing test**

Add a selftest function and __main__ guard. The test should:
- Create random A (m=16, k multiple of 128) and B (n multiple of 16, k same).
- Produce b_q and b_scale_sh using _quant() (verify shapes; if shuffle=True is required, set it and assert dtype/shape).
- Call the fused wrapper directly (once added) and compare to baseline custom_kernel.

Example:

```python

def _selftest_fused_a_pack():
    if not torch.cuda.is_available():
        print("SKIP: no HIP GPU")
        return
    torch.manual_seed(0)
    m, k, n = 16, 128, 256
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    quant = _quant()
    # If aiter quant requires shuffle for live contract, use shuffle=True here and assert shapes.
    b_q, b_scale_sh = quant(b.contiguous(), shuffle=True)
    b_q = b_q.contiguous().view(torch.uint8)
    b_scale_sh = b_scale_sh.contiguous().view(torch.uint8)
    b_shuffle = torch.empty_like(b_q)  # unused by direct m16

    # Baseline path (pack + dense_rawscale)
    out_ref = custom_kernel((a, b, b_q, b_shuffle, b_scale_sh))

    # Fused path (direct wrapper)
    c_fused = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    _module().mxfp4_mm_hip_mfma_scale_exact_m16_dense_rawscale_fuseda(
        a, b_q, b_scale_sh, c_fused
    )
    torch.testing.assert_close(out_ref, c_fused, rtol=0, atol=0)

if __name__ == "__main__" and os.environ.get("MXFP4_FUSE_A_PACK_SELFTEST") == "1":
    _selftest_fused_a_pack()
```

**Step 2: Run test to verify failure**

Run:
- MXFP4_FUSE_A_PACK_SELFTEST=1 python3 fp8-mm/hip_phase2_working.py

Expected: FAIL (missing fused wrapper symbol) until Task 3.

**Step 3: Write minimal implementation**

None in this task.

**Step 4: Run test to verify pass**

After Task 3, re-run the same command and expect PASS.

**Step 5: Commit**

```bash
git add fp8-mm/hip_phase2_working.py
git commit -m "test: add fused A-pack smoke test"
```

---

### Task 2: Add device helpers for fused A-pack (scale + pack 32 values)

**Objective:** Create reusable device helpers for computing the scale byte and packing 32 raw bf16 values into 16 fp4 bytes.

**Files:**
- Modify: fp8-mm/hip_phase2_working.py (HIP_SRC section near mxfp4_pack_a_fixed_kernel)

**Step 1: Write failing test**

Run the selftest from Task 1 (still fails).

**Step 2: Run test to verify failure**

Same as Task 1 Step 2 (expected FAIL).

**Step 3: Write minimal implementation**

Add helpers that mirror mxfp4_pack_a_fixed_kernel logic:

```cpp
__device__ __forceinline__ uint8_t mxfp4_scale_byte_32(const __hip_bfloat16* a_src) {
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        amax = fmaxf(amax, fabsf(static_cast<float>(a_src[i])));
    }
    if (amax <= 0.0f) return 0;
    const unsigned int rounded_bits = (__builtin_bit_cast(unsigned int, amax) + 0x200000u) & 0xFF800000u;
    const unsigned int rounded_exp = (rounded_bits >> 23) & 0xFFu;
    return static_cast<uint8_t>(rounded_exp - 2u);
}

__device__ __forceinline__ int mxfp4_pack_scale_lane(uint8_t scale_byte) {
    return static_cast<int>(scale_byte) | (127 << 8) | (127 << 16) | (127 << 24);
}

__device__ __forceinline__ void mxfp4_pack_a_group32(
    const __hip_bfloat16* a_src,
    unsigned char* out_bytes,
    uint8_t* out_scale
) {
    const uint8_t scale_byte = mxfp4_scale_byte_32(a_src);
    *out_scale = scale_byte;
    const float scale_f = fp4_scale_from_e8m0(scale_byte);
    const int scale_unbiased = static_cast<int>(scale_byte) - 127;
    const float quant_scale = ldexpf(1.0f, -scale_unbiased);
    const bool use_builtin_bf16 = (USE_FP4_BUILTIN_BF16_PACK != 0) && (HAS_CVT_SCALE_FP4_BF16 != 0);
    const bool use_builtin_f32 = (USE_FP4_BUILTIN_PACK != 0) && (HAS_CVT_SCALE_FP4_F32 != 0);

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float src0 = static_cast<float>(a_src[2 * i + 0]);
        const float src1 = static_cast<float>(a_src[2 * i + 1]);
        const float q0 = src0 * quant_scale;
        const float q1 = src1 * quant_scale;
#if HAS_CVT_SCALE_FP4_BF16
        if (use_builtin_bf16) {
            const bf16x2_t src = *reinterpret_cast<const bf16x2_t*>(a_src + 2 * i);
            unsigned int packed = cvt_scalef32_pk_fp4_bf16<0>(0u, src, scale_f);
            unsigned char byte = static_cast<unsigned char>(packed & 0xFFu);
            unsigned char nib0 = apply_fixed_adjustment(fp4_extract(byte, 0), q0);
            unsigned char nib1 = apply_fixed_adjustment(fp4_extract(byte, 1), q1);
            out_bytes[i] = fp4_pack(nib0, nib1);
        } else
#endif
#if HAS_CVT_SCALE_FP4_F32
        if (use_builtin_f32) {
            unsigned int packed = cvt_scalef32_pk_fp4_f32<0>(0u, src0, src1, scale_f);
            unsigned char byte = static_cast<unsigned char>(packed & 0xFFu);
            unsigned char nib0 = apply_fixed_adjustment(fp4_extract(byte, 0), q0);
            unsigned char nib1 = apply_fixed_adjustment(fp4_extract(byte, 1), q1);
            out_bytes[i] = fp4_pack(nib0, nib1);
        } else {
            const unsigned char nib0 = apply_fixed_adjustment(quantize_fp4_scaled(q0), q0);
            const unsigned char nib1 = apply_fixed_adjustment(quantize_fp4_scaled(q1), q1);
            out_bytes[i] = fp4_pack(nib0, nib1);
        }
#else
        const unsigned char nib0 = apply_fixed_adjustment(quantize_fp4_scaled(q0), q0);
        const unsigned char nib1 = apply_fixed_adjustment(quantize_fp4_scaled(q1), q1);
        out_bytes[i] = fp4_pack(nib0, nib1);
#endif
    }
}
```

**Step 4: Run test to verify pass**

Selftest still FAILS (expected, fused kernel not wired yet).

**Step 5: Commit**

```bash
git add fp8-mm/hip_phase2_working.py
git commit -m "feat: add fused A-pack helpers"
```

---

### Task 3: Implement fused m16 dense rawscale kernel + wrapper

**Objective:** Add a fused kernel that accepts raw A and computes a_buf + scale_a inline for MFMA.

**Files:**
- Modify: fp8-mm/hip_phase2_working.py (HIP_SRC section + CPP_WRAPPER + _module())

**Step 1: Write failing test**

Run the selftest from Task 1 (still FAILS).

**Step 2: Run test to verify failure**

MXFP4_FUSE_A_PACK_SELFTEST=1 python3 fp8-mm/hip_phase2_working.py
Expected: FAIL.

**Step 3: Write minimal implementation**

Add a fused kernel near mxfp4_mm_kernel_mfma_scale_exact_m16_dense_rawscale:

```cpp
__global__ void mxfp4_mm_kernel_mfma_scale_exact_m16_dense_rawscale_fuseda(
    const __hip_bfloat16* __restrict__ a,
    const unsigned char* __restrict__ b_packed,
    const uint8_t* __restrict__ b_scale_sh,
    __hip_bfloat16* __restrict__ c,
    int n,
    int k,
    int scale_cols,
    int src_rows,
    int src_cols
) {
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 128;
    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane16 = lane & 15;
    const int group4 = lane >> 4;
    const int b_bytes_per_row = k / 2;

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) { a_buf.v[i] = 0; b_buf.v[i] = 0; }

        // Fused A pack for this 32-value group
        const __hip_bfloat16* a_row_ptr = a + lane16 * k + tile_k + group4 * 32;
        unsigned char a_bytes[16];
        uint8_t scale_byte = 0;
        mxfp4_pack_a_group32(a_row_ptr, a_bytes, &scale_byte);
        #pragma unroll
        for (int i = 0; i < 16; ++i) { a_buf.b[i] = a_bytes[i]; }
        const int scale_a = mxfp4_pack_scale_lane(scale_byte);

        // B pack + scale (existing path)
        const unsigned char* ldg_b = b_packed + (tile_col + lane16) * b_bytes_per_row + tile_k / 2 + group4 * 16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) { b_buf.b[i] = ldg_b[i]; }
        const int scale_block = tile_k / 32;
        const int scale_b = pack_scale_e8m0x4_lane_from_shuffled(
            b_scale_sh, n, scale_cols, src_rows, src_cols, tile_col + lane16, scale_block, group4
        );

        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_buf.v, b_buf.v, acc, 4, 4, 0, scale_a, 0, scale_b);
    }

    const int out_col = tile_col + lane16;
    const int out_row_base = group4 * 4;
    c[(out_row_base + 0) * n + out_col] = static_cast<__hip_bfloat16>(acc[0]);
    c[(out_row_base + 1) * n + out_col] = static_cast<__hip_bfloat16>(acc[1]);
    c[(out_row_base + 2) * n + out_col] = static_cast<__hip_bfloat16>(acc[2]);
    c[(out_row_base + 3) * n + out_col] = static_cast<__hip_bfloat16>(acc[3]);
}
```

Add wrapper:

```cpp
void mxfp4_mm_hip_mfma_scale_exact_m16_dense_rawscale_fuseda(
    torch::Tensor a,
    torch::Tensor b_q,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a.size(1));
    const int scale_cols = k / 32;
    dim3 block(64);
    dim3 grid((n + 16 - 1) / 16, 1);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m16_dense_rawscale_fuseda,
        grid,
        block,
        0,
        0,
        reinterpret_cast<const __hip_bfloat16*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned char*>(b_q.data_ptr<uint8_t>()),
        reinterpret_cast<const uint8_t*>(b_scale_sh.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        n,
        k,
        scale_cols,
        static_cast<int>(b_scale_sh.size(0)),
        static_cast<int>(b_scale_sh.size(1))
    );
}
```

Update CPP_WRAPPER and _module() functions list to include:
- mxfp4_mm_hip_mfma_scale_exact_m16_dense_rawscale_fuseda

**Step 4: Run test to verify pass**

MXFP4_FUSE_A_PACK_SELFTEST=1 python3 fp8-mm/hip_phase2_working.py
Expected: PASS.

**Step 5: Commit**

```bash
git add fp8-mm/hip_phase2_working.py
git commit -m "feat: add fused A-pack m16 kernel"
```

---

### Task 4: Wire fused path into custom_kernel (env-guarded)

**Objective:** Route the exact m16 path to the fused wrapper when MXFP4_FUSE_A_PACK=1.

**Files:**
- Modify: fp8-mm/hip_phase2_working.py (custom_kernel m16 path)

**Step 1: Write failing test**

Run:
- MXFP4_FUSE_A_PACK=1 python3 - <<'PY'
import os, torch
import fp8-mm.hip_phase2_working as mod
# build minimal m16 inputs using _quant, then call custom_kernel
PY

Expected: FAIL (still uses old path).

**Step 2: Run test to verify failure**

Expected: FAIL until wiring complete.

**Step 3: Write minimal implementation**

Add env flag near top:

```python
USE_FP4_FUSE_A_PACK = int(os.environ.get("MXFP4_FUSE_A_PACK", "0"))
```

In m16 path inside custom_kernel, branch:

```python
if m16_gate_ok and (a_cols % 128) == 0 and (b.shape[0] % 16) == 0:
    ...
    if USE_FP4_FUSE_A_PACK and a is not None:
        mod.mxfp4_mm_hip_mfma_scale_exact_m16_dense_rawscale_fuseda(
            a.contiguous(),
            b_q.contiguous().view(torch.uint8),
            b_scale_sh.contiguous().view(torch.uint8),
            c,
        )
    else:
        mod.mxfp4_mm_hip_mfma_scale_exact_m16_direct_entry(...)
```

**Step 4: Run test to verify pass**

Run:
- MXFP4_FUSE_A_PACK_SELFTEST=1 python3 fp8-mm/hip_phase2_working.py
Expected: PASS

**Step 5: Commit**

```bash
git add fp8-mm/hip_phase2_working.py
git commit -m "feat: route m16 path to fused A-pack when enabled"
```

---

### Task 5: Update candidate card + variant name for fused A-pack

**Objective:** Make the submission metadata reflect the fused A-pack change.

**Files:**
- Modify: fp8-mm/hip_phase2_working.py (Candidate Card + CONFIG.variant_name)

**Step 1: Write failing test**

None (metadata change).

**Step 2: Run test to verify failure**

None.

**Step 3: Write minimal implementation**

Update Candidate Card:
- deleted_cost_center: generic mxfp4_pack_a_fixed (fused into m16 kernel)
- expected_upside_source: pack launch + temp materialization eliminated for m16

Update CONFIG variant_name to something like:
- native_scaled_exact_shape_m16_fusedapack_t16

**Step 4: Run test to verify pass**

Run preflight:
- python3 -m agent_loop --config agent_loop.toml mxfp4-closed-loop preflight --variant <new> --source fp8-mm/hip_phase2_working.py --lane A --hypothesis "Fuse A-pack into m16 kernel" --expected-gain "m16 -0.3us; geomean -0.2us" --next-patch "if flat, extend fused A-pack to m32+" --runtime none
Expected: warn (static-only), purity ok.

**Step 5: Commit**

```bash
git add fp8-mm/hip_phase2_working.py
git commit -m "chore: update candidate card for fused A-pack"
```

---

### Task 6: Run remote test + benchmark (m16 fused enabled)

**Objective:** Validate correctness and measure gains with fused A-pack enabled.

**Files:**
- No code changes

**Step 1: Write failing test**

None.

**Step 2: Run test to verify failure**

Submit test:

```
MXFP4_FUSE_A_PACK=1 python3 -m agent_loop --config agent_loop.toml mxfp4-closed-loop submit \
  --variant <new> --source fp8-mm/hip_phase2_working.py --lane A \
  --hypothesis "Fuse A-pack into m16 kernel" \
  --expected-gain "m16 -0.3us; geomean -0.2us" \
  --next-patch "if flat, extend fused A-pack to m32+" \
  --stage test
```

Expected: status ok.

**Step 3: Write minimal implementation**

None.

**Step 4: Run test to verify pass**

Submit benchmark:

```
MXFP4_FUSE_A_PACK=1 python3 -m agent_loop --config agent_loop.toml mxfp4-closed-loop submit \
  --variant <new> --source fp8-mm/hip_phase2_working.py --lane A \
  --hypothesis "Fuse A-pack into m16 kernel" \
  --expected-gain "m16 -0.3us; geomean -0.2us" \
  --next-patch "if flat, extend fused A-pack to m32+" \
  --stage benchmark
```

Expected: status ok, record contains benchmark_geomean + per_shape_times.

**Step 5: Commit**

None.

---

### Task 7 (Phase 2, optional): Extend fused A-pack to m4/m8

**Objective:** Remove A-pack helper from m4/m8 direct entries.

**Files:**
- Modify: fp8-mm/hip_phase2_working.py (m4/m8 kernels + wrappers + custom_kernel)

**Approach:**
- Add fused m4 kernel analogous to mxfp4_mm_kernel_mfma_scale_exact_m4_dense, but with raw A and inline pack per lane16<4.
- For m8, either:
  - Add fused kernel variant of mxfp4_mm_kernel_mfma_scale_exact_m16 (general) with raw A, then call it when m==8, or
  - Add a dedicated m8 kernel if needed.

**Verification:** rerun Task 6 with MXFP4_FUSE_A_PACK=1 and compare m4/m8 times.

---

### Task 8 (Phase 3, optional): Extend fused A-pack to m32/m64/m256

**Objective:** Remove the A-pack helper for wide shapes by fusing A-pack into m32/m64/m256 kernels.

**Files:**
- Modify: fp8-mm/hip_phase2_working.py (m32/m64/m256 kernels + wrappers)

**Approach:**
- Create fused variants of:
  - mxfp4_mm_kernel_mfma_scale_exact_m32
  - mxfp4_mm_kernel_mfma_scale_exact_m32_m64plus
  - mxfp4_mm_kernel_mfma_scale_exact_m32_rawb
  - and the m64/m256 entry wrappers
- For each kernel, replace a_packed loads with:
  - raw A pointer = a + (tile_row + lane32) * k + tile_k + group * 32
  - mxfp4_pack_a_group32 to fill a_buf.b[0..15]
  - scale_a = pack_scale_e8m0x2_lane equivalent (use scale byte from pack)

**Verification:** rerun Task 6 and ensure geomean improvement is sustained.

---

### Task 9: Final sanity + reporting

**Objective:** Summarize benchmark deltas and determine next move.

**Steps:**
- Report geomean + per-shape times vs baseline.
- If improvements are flat, move to Phase 2/3.
- If regressing, revert to last-improving baseline per policy.

---
