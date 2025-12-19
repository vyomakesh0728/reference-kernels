Yes—**mismatching from (0,0,0)** after switching to the deep_gemm-style “explicit `tmem_addr` walk” strongly indicates your new `tmem_addr` math does **not** match how *your* tcgen05 MXF4 accumulator is laid out in TMEM. In fact, your new epilogue is currently reading TMEM as if it were a simple linear `[N]` row-major array of FP32 values, which it isn’t for the CuTe/CUTLASS-partitioned accumulator you constructed.[1][2]


## What changed and what it implies
- Before, the first mismatch at `(0,8,0)` was consistent with “coverage stops after first N8 slice”.[3]
- Now, mismatches at `(0,0,0)` and lots of `inf/-inf` suggest you are reading essentially **garbage addresses** (or wrong interpretation of what each address contains), not just missing a slice.[2][1]


## The core issue in updated epilogue
In your updated submission, the epilogue does:
- `uint32_t tmemaddr = tmemc + uint32_t(col0);`
- then `cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmemaddr, v0,v1,v2,v3)` and writes those as floats to D.[1]


This assumes:
1) `tmemc` is a base address where **column 0 corresponds to output n=0**, and  
2) `tmemc + col` is the correct address for output column `col`, and  
3) consecutive addresses correspond to consecutive output columns.[1]


But in your own previous (CuTe) path, you *explicitly avoided hardcoding TMEM address math* and used `partition_shape_C(...)` + `make_tensor(FrgTypeC)` precisely because the accumulator is **MMA-partitioned** and has non-trivial addressing.[4]


So: **the new epilogue is almost certainly wrong address mapping**, not necessarily “TMEM data is wrong”.[2][1]


## A second, even bigger correctness bomb: you changed the issuing warp
In the updated file, tcgen05.mma is now issued under:
- `if (warpid == 4) { ... tcgen05.mma ... }`[1]


But your CTA uses `kThreads=128` in tcgen05 mode, i.e. warps are `warpid=0..3`.[1]


That means **no warp ever issues tcgen05.mma** (the condition is never true), so TMEM accumulator contents are uninitialized/garbage, and any epilogue will mismatch starting at `(0,0,0)`.[2][1]


This alone can explain the giant junk values/`inf` you’re seeing.[2][1]


## What to do right now (minimal, high-signal)
- Fix the issuing condition back to an existing warp. If you intend “one warp issues”, it must be something like `warpid==0` (or whichever exists in 0..3).[1]
- Keep the explicit epilogue, but *don’t* assume `tmemaddr = tmemc + n`. Instead, use explicit TMEM loads only as a **diagnostic** once the producer is definitely writing TMEM.[1]


## Next diagnostic to validate TMEM mapping (without full CuTe epilogue)
Once tcgen05.mma is actually issuing again:
- Compare two reads for the same output element:
  1) CuTe `make_tmem_copy(...)` path (known “almost works”, at least for first N8 previously).[4]
  2) Your explicit `SM100_TMEM_LOAD_*` path at some guessed address.[1]


If (1) gives sane values and (2) gives junk, then it’s 100% address mapping. If both give junk, producer is still broken.[4][1]