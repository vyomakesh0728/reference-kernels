
<CURRENT_FLOW>
Current tcgen05 GEMM Flow
Data Preparation (Python)
A, B (FP4 packed) → uint8 tensors
SFA, SFB → to_blocked() → uint8 tensors
Kernel Flow (per CTA tile)
1. TMA Load (Global → SMEM)
Global A/B (packed FP4) → TMA → SMEM [128×128 bytes]
Global SFA/SFB (FP8, to_blocked layout) → TMA → SMEM [2048 bytes = 4×512]
2. Scale Copy (SMEM → TMEM)
tcgen05.cp × 4 iterations:
  SMEM [512 bytes] → TMEM [32 rows × 16 bytes]
  
TMEM layout:
  - Columns 0-127: Accumulator C
  - Columns 128-143: SFA scales (16 columns)
  - Columns 144-159: SFB scales (16 columns)
3. MMA (SMEM + TMEM → TMEM)
tcgen05.mma.block_scale:
  Operand A: SMEM (packed FP4)
  Operand B: SMEM (packed FP4)
  Scale A: TMEM [columns 128-143]
  Scale B: TMEM [columns 144-159]
  Output C: TMEM [columns 0-127]
4. Epilogue (TMEM → Global)
TMEM.load → Registers (FP32)
Convert FP32 → FP16
Store → Global D
Current Issues
Still getting inf/-inf errors
Using to_blocked() layout but may need atom-tiled
TMEM addressing for scales unclear (row-major vs column-major)
</CURRENT_FLOW>