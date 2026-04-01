# AGENTS.md -- Agent Instructions for reference-kernels

## Mission
Beat the aiter library baselines on all three AMD MI355X kernel problems
by writing faster Triton (or HIP) kernels submitted through popcorn-cli.

## After context reset (compaction / new session)

1. Read **this file** (`AGENTS.md`), especially **Popcorn eval loop** and **Iteration log (HISTORY.md)** below.
2. Open **`HISTORY.md`** at repo root for the latest iteration notes and patterns not to repeat.
3. For mixed-mla team notes and MFMA direction, see **`CLAUDE.md`** (if present).
4. For low-level kernel optimization, see **`docs/mi355x_isa_reference.md`** (MFMA, memory ops, conversions, novel techniques).
5. **Codex CLI:** start at **`codex.md`** in the repo root (pointer to this file).

## Popcorn eval loop (do not skip)

Remote GPU only—no local MI355X.

| Mode | Use | Rate limit |
|------|-----|------------|
| **`test`** | Correctness first; default after code changes | Unlimited |
| **`benchmark`** | Official **8 shapes**, timing table; **geomean = (∏ mean_i)^(1/8)** in one unit (not printed—compute yourself) | Unlimited |
| **`leaderboard`** | **Published** aggregate / rank; same per-shape protocol, server applies `ranking_by` from `task.yml` | Typically **1/hour**—reserve for “ship it” |

**Rule:** **`test` → `benchmark` → `leaderboard`**. Do not iterate on `leaderboard` for tuning.

Example:

```bash
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test --no-tui problems/amd_202602/mixed-mla/submission.py
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode benchmark --no-tui problems/amd_202602/mixed-mla/submission.py
```

## Iteration log (`HISTORY.md`)

Maintain **`HISTORY.md`** at the **repository root** (see template there). After any **meaningful** code change or **Popcorn** run you might compare later, add a dated section—**newest first** under `## Log`.

**Required habits (so we don’t repeat mistakes across compactions):**

1. **Log failures and dead ends** — not only wins. *“Tried X → failed / reverted because Y”* is often the highest-value row.
2. **Evidence** — paste the **exact** `popcorn-cli …` line(s), **GitHub Actions / workflow URL** if the harness gives one, and **numbers**: test pass/fail, **mismatch_ratio** if any, benchmark **per-shape means** and/or **self-computed geomean**, leaderboard score if you ran it.
3. **Rule / spec tension** — if the code **conflicts** with a stated rule (e.g. `AGENTS.md` “no cross-call caches” vs buffer reuse in `submission.py`), call it out: *what we do, why, or what we’ll fix.*
4. **Next bet** — one line: *first experiment or file to try on the next session* so we don’t re-derive the roadmap from zero.

This file is **institutional memory** when chat context is gone.

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
