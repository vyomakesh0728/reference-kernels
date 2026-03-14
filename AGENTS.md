# AGENTS.md -- Agent Instructions for reference-kernels

## Mission
Beat the aiter library baselines on all three AMD MI355X kernel problems
by writing faster Triton (or HIP) kernels submitted through popcorn-cli.

## Agent Workflow

### Reading a problem
1. Read `problems/amd_202602/<problem>/task.py` for input/output types
2. Read `problems/amd_202602/<problem>/reference.py` for correctness reference + generate_input
3. Read `problems/amd_202602/<problem>/task.yml` for test shapes and benchmark shapes
4. Read `problems/amd_202602/<problem>/README.md` (if exists) for optimization hints
5. Read `problems/amd/<problem-dir>/submission.py` for current best submission

### Writing a submission
- Output a single `submission.py` file
- Must define `def custom_kernel(data: input_t) -> output_t:`
- Must pass correctness: outputs checked against reference with rtol/atol tolerance
- Must not use cross-call caches, global state tricks, or benchmark cheating
- Preserve `#!POPCORN` header lines if present

### Evaluating
- No local MI355X GPU available
- All eval goes through `popcorn-cli submit` on remote cluster
- The agent loop (`problems/amd/agent_loop/`) automates this

## Problem-Specific Notes

### mxfp4-mm
- Input A is bf16, B is pre-quantized MXFP4 with shuffled layout
- Must quantize A on-the-fly with per-1x32 block scaling
- Key optimization: fuse quantization into the matmul kernel
- MXFP4 format: E2M1 values packed 2 per byte, E8M0 scales per 32-element block
- Tile sizes to explore: BLOCK_M in {16,32,64,128}, BLOCK_N in {128,256}, BLOCK_K in {64,128}
- Small M dimension (decode-style) -- optimize for thin matrices

### moe-mxfp4
- DeepSeek-R1 MoE: gate_up GEMM + SwiGLU + down GEMM per expert
- Weights pre-shuffled for (16,16) layout, MXFP4 quantized
- Key optimization: fuse routing + quant + GEMM + activation
- Expert parallelism and load balancing matter
- Shared expert can be fused or run separately
- Weight dimensions padded to 256-alignment

### mixed-mla
- Multi-head Latent Attention decode (not prefill)
- KV cache available in bf16, fp8, and mxfp4 formats
- fp8 path is current baseline (a8w8 via aiter)
- Key optimization: use mxfp4 KV with fused dequant for lower bandwidth
- Split-K with NUM_KV_SPLITS=32, exploit MQA pattern (1 KV head, 16 Q heads)
- Batch sizes range 4-256, KV seq lens 1024-8192

## Constraints
- Target GPU: AMD Instinct MI355X (CDNA4 architecture)
- Triton works on AMD via ROCm backend
- aiter library is available in the eval environment
- Python 3.x, PyTorch with ROCm support
- Timeout: 420-600s for benchmarks, 900-1200s for MLA ranked

## Optimization Priorities
1. Correctness first -- a fast wrong answer scores zero
2. Memory bandwidth -- most of these are memory-bound at small batch sizes
3. Quantization fusion -- avoid separate quant passes when possible
4. Tile size tuning -- MI355X has different optimal tiles than NVIDIA GPUs
5. Occupancy -- balance register pressure vs parallelism
