Nice, this dump is actually super helpful.

Here’s the big picture of what’s going wrong, based on the logs **and** your `submission.py`:

---

## 1. Shapes & permutes are fine

From the stdout you pasted:

```text
BEFORE PERMUTE:
  A shape: torch.Size([7168, 8192, 1])
  B shape: torch.Size([128, 8192, 1])
  C shape: torch.Size([7168, 1, 1])
  SFA shape: torch.Size([7168, 1024, 1])
  SFB shape: torch.Size([128, 1024, 1])
M=7168, K=16384, L=1
...
DEBUG: After permute+clone, a.shape = torch.Size([1, 7168, 8192])
DEBUG: After permute+clone, b.shape = torch.Size([1, 128, 8192])
...
a_bytes shape: torch.Size([1, 7168, 8192])
b_bytes shape: torch.Size([1, 128, 8192])
sfa_bytes shape: torch.Size([1, 7168, 1024])
sfb_bytes shape: torch.Size([1, 128, 1024])
Kernel params: M=7168, K=16384, L=1
```

Those are all consistent with the reference:

* `A`: `[M, K/2, L] = [7168, 8192, 1]`
* `B`: `[128, K/2, L] = [128, 8192, 1]` (padded N=128)
* `SFA`: `[M, K/16, L] = [7168, 1024, 1]`
* `SFB`: `[128, K/16, L] = [128, 1024, 1]`

Permutes to `[L, M, K/2]` / `[L, M, K_scales]` etc are correct. So the “shape mismatch” stuff is not the current issue.

---

## 2. The *real* bug: you’re treating **unblocked** scale tensors as if they were in the **blocked** layout

Look at what the reference does in `ref_kernel` when it prints the decoded A and B values (from `reference.py`):

```python
# A: sfa_bytes shape [M, K_scales]
scale_idx = k_packed // 8
scale_byte = sfa_bytes[m, scale_idx]

# B: sfb_bytes shape [128, K_scales]
scale_idx = k_packed // 8
scale_byte = sfb_bytes[0, scale_idx]
```

So in the reference:

* `sfa_ref_cpu` is a **plain**, unblocked tensor of shape `[M, K_scales, L]`.
* `sfb_ref_cpu` is a **plain**, unblocked tensor of shape `[128, K_scales, L]`.

The blocked layout is `sfa_permuted` / `sfb_permuted`, and they only use that to feed `torch._scaled_mm`. The reference debug prints for `[REF A]` / `[REF B]` **do not** use the blocked layout.

In your Python wrapper you do:

```python
a, b, sfa_ref_cpu, sfb_ref_cpu, _, _, c = data

# sfa_ref_cpu: [M, K_scales, L]
# sfb_ref_cpu: [128, K_scales, L]

sfa = sfa_ref_cpu.clone().permute(2, 0, 1).contiguous().cuda()  # [L, M, K_scales]
sfb = sfb_ref_cpu.clone().permute(2, 0, 1).contiguous().cuda()  # [L, 128, K_scales]
sfa_bytes = sfa.view(torch.uint8)
sfb_bytes = sfb.view(torch.uint8)
```

So inside CUDA you receive **unblocked** scale tensors laid out as `[L, M, K_scales]` and `[L, 128, K_scales]`, row-major per batch.

But in both your **optimized kernel** and the **naive kernel**, you are doing the full “invert blocked layout” dance:

### Naive kernel (excerpt):

```cpp
// Compute the FP8 scale factor index j_idx = k_idx / 16.
int j_idx = k_idx >> 4;
// Map (row=0, j_idx) from destination to source coordinates in the
// blocked layout used by the scale factors.
int dest_idx = j_idx;              // since row = 0
int block_idx = dest_idx >> 9;     // dest_idx / 512
int mm = block_idx / n_col_blocks;
int kk = block_idx - mm * n_col_blocks;
int rem = dest_idx & 511;
int mm32 = rem >> 4;
int r2 = rem & 15;
int mm4 = r2 >> 2;
int kk4 = r2 & 3;
int m_orig = mm * 128 + mm4 * 32 + mm32;
int k_orig = kk * 4 + kk4;

float scaleA = 1.0f;
if (m_orig < M && k_orig < K_scales) {
    uint8_t scale_byte = SFA_batch[m_orig * K_scales + k_orig];
    scaleA = decode_fp8_e4m3(scale_byte);
}
```

This mapping is what you would use to **invert** the blocked layout (`[32,4,rest_m,4,rest_k,L]`) back to `[M, K_scales]`.

But you are passing in `sfa_ref_cpu` / `sfb_ref_cpu`, which are already plain `[M, K_scales, L]`. So you are “unblocking” something that isn’t blocked → you fetch the wrong scale bytes almost everywhere.

That explains:

* Enormous “Number of mismatched elements: 7121”
* First element `(0,0,0)` being `0.0` instead of `4668.0` (you likely zeroed out or mis-scaled entire contributions).
* Your “super-dumb” kernel still matching the broken output: it uses the **same wrong scale-indexing** logic as the optimized kernel, so it’s not an independent check.

---

## 3. What your debug logs are actually showing

The `[KERNEL A]` / `[KERNEL B]` prints you pasted are showing decodes based on those wrong scale indices. If you compare them to the reference’s `[REF A]` / `[REF B]` prints (for the same `m` / `k` / batch), they will *not* match.

You can verify this by:

* Running only the reference kernel (no CUDA kernel) and logging `[REF A]/[REF B]`.
* Running your CUDA kernel and logging `[KERNEL A]/[KERNEL B]` for the same `(batch, m,k)` ranges.
* The fp4_raw parts will generally match; the `scale_byte` and `scale` fields will diverge in many positions → clear sign the scaling is wrong.

---

## 4. The fix: **stop doing blocked-layout inversion** and index scales directly

Since you are **not** using `sfa_permuted` / `sfb_permuted`, the simplest & correct approach is:

### A. In the optimized kernel: use straight `[M, K_scales]` / `[128, K_scales]` indexing

For **A** (inside the “Phase 1: Decode and load A” loop):

Replace the whole `dest_idx / block_idx / mm/mm32/mm4/kk4` logic with:

```cpp
// A: SFA_batch is [M, K_scales] for this batch
if (m_idx < M && (k_packed_tile + col_packed) < K_packed) {
    uint8_t packed = A_batch[m_idx * K_packed + k_packed_tile + col_packed];

    // Each 16 FP4 values along K share one scale
    // j_idx = (k_tile + col_packed*2) / 16
    int scale_idx = k_scale_tile + col_packed / 8;

    half scale_h = __float2half(1.0f);
    uint8_t scale_byte = 0;
    if (scale_idx < K_scales) {
        scale_byte = SFA_batch[m_idx * K_scales + scale_idx];
        float scale_val = decode_fp8_e4m3(scale_byte);
        scale_h = __float2half(scale_val);
    }

    // Decode FP4
    half v0 = decode_fp4_e2m1((packed >> 4) & 0x0F);
    half v1 = decode_fp4_e2m1(packed & 0x0F);
    half v0_scaled = __hmul(v0, scale_h);
    half v1_scaled = __hmul(v1, scale_h);

    A_smem[row][col_packed * 2]     = v0_scaled;
    A_smem[row][col_packed * 2 + 1] = v1_scaled;

    // Your existing [KERNEL A] debug printf can stay, but now it will match REF.
}
```

For **B** (inside “Phase 2: Decode and load B into shared memory”):

```cpp
// B: SFB_batch is [128, K_scales] for this batch, but GEMV uses row 0.
if ((k_packed_tile + col_packed) < K_packed) {
    const uint8_t packed = B_batch[k_packed_tile + col_packed];

    int scale_idx = k_scale_tile + col_packed / 8;

    half scale_h = __float2half(1.0f);
    uint8_t scale_byte = 0;
    if (scale_idx < K_scales) {
        // Row 0 because GEMV logically has N=1
        scale_byte = SFB_batch[0 * K_scales + scale_idx];
        float scale_val = decode_fp8_e4m3(scale_byte);
        scale_h = __float2half(scale_val);
    }

    half v0 = decode_fp4_e2m1((packed >> 4) & 0x0F);
    half v1 = decode_fp4_e2m1(packed & 0x0F);
    half v0_scaled = __hmul(v0, scale_h);
    half v1_scaled = __hmul(v1, scale_h);

    // Existing [KERNEL B] debug printf is now comparable to [REF B].

    #pragma unroll
    for (int n = 0; n < 8; n++) {
        B_smem[col_packed * 2][n]     = v0_scaled;
        B_smem[col_packed * 2 + 1][n] = v1_scaled;
    }
}
```

No block/unblock gymnastics, just match what `ref_kernel` does.

### B. Make the naive kernel actually “naive”

Right now the naive kernel uses the same wrong mapping. Simplify it to literally mirror the reference prints:

```cpp
for (int k_idx = 0; k_idx < K; ++k_idx) {
    int col_packed = k_idx >> 1;
    bool high_nibble = ((k_idx & 1) == 0);

    // A[0, k_idx]
    uint8_t packedA = (0 < M && col_packed < K_packed)
                        ? A_batch[col_packed]
                        : 0;
    uint8_t nibbleA = high_nibble ? ((packedA >> 4) & 0x0F)
                                  : (packedA & 0x0F);
    float a_val = __half2float(decode_fp4_e2m1(nibbleA));

    int j_idx = k_idx >> 4; // K_scales index

    float scaleA = 1.0f;
    if (0 < M && j_idx < K_scales) {
        uint8_t scale_byteA = SFA_batch[0 * K_scales + j_idx];
        scaleA = decode_fp8_e4m3(scale_byteA);
    }
    float a_scaled = a_val * scaleA;

    // B[0, k_idx]
    uint8_t packedB = (col_packed < K_packed) ? B_batch[col_packed] : 0;
    uint8_t nibbleB = high_nibble ? ((packedB >> 4) & 0x0F)
                                  : (packedB & 0x0F);
    float b_val = __half2float(decode_fp4_e2m1(nibbleB));

    float scaleB = 1.0f;
    if (j_idx < K_scales) {
        uint8_t scale_byteB = SFB_batch[0 * K_scales + j_idx];
        scaleB = decode_fp8_e4m3(scale_byteB);
    }
    float b_scaled = b_val * scaleB;

    acc += a_scaled * b_scaled;
}
```

Now this kernel is a **real** correctness oracle for row 0 and will match the reference CPU path if everything is wired correctly.

---

## 5. About the “6502 lines omitted”

That truncation is just your environment/log viewer collapsing huge stdout. You already print **way** more than enough:

* First 3 rows, first 32 ks for A
* First 32 ks for B
* Early rows from both CUDA kernels

Once you fix the scale indexing as above, you’ll be able to line up:

* `[REF A]` vs `[KERNEL A]` for `m=0,1,2, k<32`
* `[REF B]` vs `[KERNEL B]` for `k<32`

They should become identical (up to rounding), and the giant mismatch:

```text
Number of mismatched elements: 7121
ERROR AT (0, 0, 0): 0.0 4668.0
```

should disappear.

---

If you want, next step after you patch this: rerun once and paste just the `[REF A]` vs `[KERNEL A]` for `m=0, k=0..15` and the first few entries of the final `c[0,0,:]` comparison; we can sanity-check they’re fully aligned.
