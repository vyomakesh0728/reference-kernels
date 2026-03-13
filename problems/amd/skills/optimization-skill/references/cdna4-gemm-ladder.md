# CDNA4 GEMM Ladder

This is the compact optimization ladder distilled from AMD's CDNA4 GEMM optimization blog for MI355X / `gfx950`.

## Goal

Start from a correctness-first HIP kernel and climb toward a real CDNA4 matrix-core kernel without breaking semantics.

## Order

1. **Correctness-first HIP reference**
   - real HIP hot path
   - slow is acceptable
   - outputs must match live reference

2. **Ingress cleanup**
   - remove unnecessary host/Python overhead
   - use vectorized reads where safe
   - keep the memory path simple and measurable

3. **Global to LDS movement**
   - improve how tiles arrive in shared memory / LDS
   - make the dataflow explicit and regular

4. **LDS layout**
   - reduce bank conflicts
   - introduce swizzle only when the baseline remains correct

5. **Software pipelining**
   - add double buffering after the LDS path is stable
   - keep one change at a time

6. **CDNA4 matrix-core path**
   - replace the scalar tiled inner loop with scaled MFMA
   - key long-term targets for MXFP4:
     - `V_MFMA_SCALE_F32_16X16X128_F8F6F4`
     - `V_MFMA_SCALE_F32_32X32X64_F8F6F4`

7. **Occupancy / wave tuning**
   - adjust block shape, wave mapping, and occupancy only after the math path is right

## Guardrails

- Do not start at step 6 while step 2 is still obviously expensive.
- Do not combine swizzle + double buffering + MFMA replacement in one edit.
- Do not confuse lower measured time from benchmark artifacts with a better kernel.

## Current MM Interpretation

For `mxfp4_mm`, phase 2 starts with:
- cache or remove expensive Python-side B-side reconstruction
- keep the passing calibrated semantics
- then reduce A-side overhead
- then move the kernel body and memory hierarchy closer to the CDNA4 ladder
