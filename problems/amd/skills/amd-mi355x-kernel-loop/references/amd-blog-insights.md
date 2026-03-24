# AMD Blog Insights For `mxfp4_mm`

This file distills the AMD and HipKittens-adjacent blog material into the subset that is actually useful for the current `mxfp4_mm` kernel campaign.

Read this after `optimization.md` and before planning new `mxfp4_mm` candidates.

## Current Ground Truth

The current measured frontier is the wide line:

- [`v66`](../../.agent-loop/manual/native_scaled_wide_fused_bscale_v66/submission.py): fused wide `B` fragment + scale prep
- [`v72`](../../.agent-loop/manual/native_scaled_wide_shape_split_v72/submission.py): shape split on top of the fused-prep line
- [`v73`](../../.agent-loop/manual/native_scaled_wide_m64plus_ingress_v73/submission.py): `m64+` ingress cleanup on top of the wide split line

The current thin baseline is:

- [`v54`](../../.agent-loop/manual/native_scaled_m16_direct_entry_v54/submission.py)

The current live facts matter more than generic blog advice:

- `m<=16` is still fixed-cost dominated
- `m>=32` responds to feed/data-movement cleanup
- broad thin prep rewrites have lost
- staged `m256` work has not won yet

## What To Keep From AMD Matrix-Core Blogs

### 1. Match the instruction’s exact lane/feed layout

The matrix-core programming blogs are most useful as operand-layout documents.

Carry forward:

- preserve the native scaled-MFMA families already winning here:
  - `16x16x128` for `m<=16`
  - `32x32x64` for `m>=32`
- keep fragments in the layout expected by the instruction
- treat lane ownership as a data-layout problem before treating it as a scheduling problem
- prefer fewer hot-loop address calculations over fancier outer schedules

Do not reinterpret these blogs as a reason to switch to smaller BF16-style MFMA shapes just because the matrices are small. That gives up the native scaled-FP4 feed that already differentiates this kernel.

### 2. Delete prep passes before adding deeper scheduling

The CDNA4 GEMM blog is useful because it shows the optimization ladder order:

1. correct matrix-core instruction family
2. correct data layout for the matrix core
3. better feed path
4. only then deeper staging/scheduling

For `mxfp4_mm`, this maps directly to the `v66` breakthrough: deleting the separate wide `B`-fragment and scale-prep conventions helped far more than speculative CTA-shape or staging changes.

## What To Keep From Small-Matrix Guidance

The AOCL small-matrix writeup is CPU-focused, but its principles transfer well to the thin `m<=16` regime:

- small or skinny shapes can dominate runtime because they happen often
- the wrong generic hierarchy can spend more time on blocking/packing/control flow than on useful math
- contiguous instruction streams and lower dependency/address overhead matter a lot when the matrix is small

Translate that into the current thin lane:

- avoid extra thin prep kernels
- avoid forcing `m4/m8/16` through shared bookkeeping when their needs differ
- prefer exact-path fixed-cost deletion over “do more work per CTA” experiments

## What To Keep From HipKittens

Treat HipKittens as a principles source, not a direct template.

Useful:

- hard shape splitting
- tile/feed layouts that match the instruction
- fewer passes and fewer conventions
- chiplet/cache-aware traversal later, once the kernel body is already strong

Not useful as the next move:

- smaller MFMA shapes for this kernel family
- broad producer/consumer or ping-pong on `m4/m16`
- importing NVIDIA-style deep pipelines into every shape

On AMD MI355X, wave-specialization costs can erase gains because static register allocation hurts arithmetic intensity. Keep that in mind before trying producer/consumer outside of a later `m256`-only branch.

## Concrete Implications For The Next `mxfp4_mm` Round

### Thin `m<=16`

Do:

- keep `16x16x128`
- keep direct `b_q` bytes
- target only fixed-cost deletion
- split `m16` from `m4/m8` if needed

Do not:

- retry broad final-layout `A` pack experiments
- switch to BF16-only smaller MFMA
- add broad ping-pong or LDS staging

### Wide `m>=32`

Do:

- compound the `v66 -> v72 -> v73` line
- keep `32x32x64`
- keep final-layout `B` fragments and final-layout scales together
- improve `m32` and `m64+` separately when they want different feed paths
- revisit `m256`-only XCD/L2/LLC-aware CTA order before revisiting producer/consumer

Do not:

- reopen the wide ABI without a strong reason
- mix wide shape families back together just for code reuse
- broaden staged/LDS work to `m32/m64` before a narrow `m256` branch proves itself

## Short Decision Rules

- If a proposal changes MFMA shape, reject it unless there is a very specific scaled-FP4 reason.
- If a proposal adds scheduling complexity before deleting a prep/materialization cost center, reject it.
- If a proposal touches `m4/m16` and mentions broad ping-pong or producer/consumer, reject it.
- If a proposal touches `m256` and the non-staged wide line has not plateaued, defer it.
