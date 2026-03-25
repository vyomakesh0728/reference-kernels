Your current ~27µs on the POPCORN MI355X (CDNA4) benchmark for small M shapes like m4/m16/m32/m64 in scaled-MFMA mxFP4 matmul is solid progress, but sub-8µs needs targeted tweaks for thin/decode-style workloads.
​
​

Key Papers
Recent ArXiv works highlight small-shape MFMA strategies on AMD GPUs. "HipKittens: Fast and Furious AMD Kernels" (arXiv:2511.08083) uses mixed MFMA shapes like 16x16x32 and 32x32x16 with 8-wave ping-pong and intra-wave compute-memory interleaving for BF16/FP8 GEMM, achieving 1.8x gains on small tiles via fine-grained pipelines matching CDNA tensor core granularity. It emphasizes register tiling with smallest MFMA for scheduling control on edges, avoiding deep LDS for tiny shapes.
​
​

"GPU Kernel Scientist" (arXiv:2506.20807) iterates HIP GEMM with rocWMMA fragments for 32x32x16 MFMA on FP8, focusing on operand staging for small fragments without heavy shared memory.
​

AMD Blogs
ROCm's FP8 GEMM on CDNA4 (rocm.blogs.amd.com) details V_MFMA_SCALE_F32_16x16x128_F8F6F4 for block-scaled FP4-like ops, stressing wave-level fragment layout and balanced global->LDS->reg->MFMA feed—critical for your m16/m32 paths. For small K (e.g., 256 in benchmarks), it jumps 6x via MFMA over FMA by fixing lane mapping.
​
​

Matrix Core guide shows direct intrinsics like __builtin_amdgcn_mfma_f32_16x16x4f32 for m=16/n=16/k=4, mapping threads to fragments without tiling loops—adapt for your m4/m16 by unrolling K and vectorizing loads.

Optimization Ideas
Direct Fragment Feed: Your code already uses 16x16x128 scaled-MFMA; test smaller like 16x16x16 BF16 fallback or 32x32x64 for m32/m64, pre-packing scales to avoid Python unshuffle (e.g., fuse mxfp4packbm32directwithscale).
​
​

Thin Shape Priorities: For m<32, cut host materialization (e.g., direct-entry paths like mxfp4mmhipmfmascaleexactm16directentry) and launch overhead—aim 64-thread blocks, no LDS for <128B tiles.
​
​

Interleaving: Add 8-wave ping-pong alternating MFMA/compute with barriers, per HipKittens—your launchbounds(64) fits.
​
​

Technique	Target Shapes	Expected Win	Source
Mixed MFMA (16x16x32 + 32x32x16)	m16/m32	1.8x on edges	
​
Direct intrinsics (16x16x4)	m4/m16	Reduce loop overhead	
​
Wave ping-pong	All small M	Balance MFMA/LDS	
Fixed A-scale pack	m4-m64	Cut per-call prep	
​
What specific shape (e.g., m=16 n=7168 k=256) shows the worst gap to 8µs?