Your `inf/-inf` is coming from **TMEM accumulator corruption / non-zero initialization in the tcgen05 path**, not from “normal numeric drift”.

### What the evidence says

* The failures are **almost-all-elements** and the custom output is literally `inf/-inf` (e.g., 917,499 mismatches out of 917,504). 
  That pattern strongly matches “reading/accumulating garbage” (or wildly wrong scaling), not a small layout mixup.

### The root cause in your tcgen05 path

* In `submission.py`, you correctly note that **`tcgen05.mma` accumulates (`C += A*B`) and `tcgen05.alloc` does *not* zero TMEM**, so C **must** be cleared before the first MMA. 
* But your “CLEAR TMEM ACCUMULATOR” implementation clears TMEM using **`tcgen05.cp ... warpx4` with a *scale-factor SMEM descriptor* (leading_byte_offset=16) and only 2048B worth of zeros**, then repeats it over “columns”.
  This is fundamentally mismatched to the actual accumulator footprint (**128×128 FP32 = 65,536 bytes**) and to the accumulator’s layout/stride, so **most of C remains uninitialized** → first MMA does `garbage + (A*B)` → values explode and when you convert/store to FP16 they saturate to `±inf`.

### The “correct” fix (what CUTLASS does)

CUTLASS’s SM100 tcgen05 tutorials **don’t clear TMEM C with `tcgen05.cp`**; they clear by setting the **first MMA’s accumulate mode to “Zero”** (i.e., “write = A*B” not “C += A*B”). ([GitHub][1])
So: **remove/ignore the `tcgen05.cp`-based clearing** and instead make the **first MMA tile** use the **ScaleOut/accumulate=Zero** mode; subsequent tiles use normal accumulate.

### Why you still mismatch reference even after fixing inf

Reference uses **`to_blocked()` for both scale tensors** before compute. 
Your tcgen05 path must feed scales in that same blocked layout (your own `permute_scales_to_blocked()` documents the exact mapping).
If tcgen05 is consuming scales copied from an un-permuted layout, you’ll still be wrong vs reference even once `inf/-inf` is gone.

Also note the task spec’s expected layouts/dtypes. 

**Bottom line:** fix TMEM accumulator initialization first (use “first MMA clears”), then ensure scales match `reference.py`’s `to_blocked()` layout.
