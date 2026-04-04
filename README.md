## Reference Kernels

This repo holds reference kernels for the KernelBot which hosts regular competitions on [discord.gg/gpumode](discord.gg/gpumode).

You can see what's going on [gpumode.com](https://www.gpumode.com/)

## Competition
1. [PMPP practice problems](https://github.com/gpu-mode/reference-kernels/tree/main/problems/pmpp_v2)
2. [AMD $100K kernel competition](problems/amd)
3. [BioML kernels](problems/bioml)
4. [AMD $100K distributed kernel competition](problems/amd_distributed)
5. [NVIDIA Blackwell NVFP4 competition](problems/nvidia)
6. [AMD $1.1M competition](problems/amd_202602)
7. [Helion IRL hackathon](problems/helion)
8. [Princeton course](problems/princeton)

## Making a Leaderboard Submission

Please take a look at `vectoradd_py` to see multiple examples of expected submisisons ranging from PyTorch code to Triton to inline CUDA.

## AI / agent workflow (AMD hackathon & kernel work)

- **`AGENTS.md`** — canonical instructions, **Popcorn `test` → `benchmark` → `leaderboard`** loop, and rules for **`HISTORY.md`** (evidence, failures, next bet).
- **`HISTORY.md`** — append **newest-first** iteration log so context compactions don’t erase patterns.
- **`codex.md`** — entry point for **Codex CLI** (points here).
- **`.cursor/rules/agents-history.mdc`** — Cursor always-on reminder to read the above.

---

## mixed-mla Status (2026-04-02)

**Current:** 66µs geomean | **Target:** 29µs (#1) | **Gap:** 2.3×

### Systematically Ruled Out (Dead Ends)

| Approach | Why It Failed |
|----------|---------------|
| **Pre-compiled .co files** | Submission only accepts single `.py` file |
| **HIP graphs** | Harness blocks `torch.cuda.Stream`, `Event`, `graph` |
| **torch.ops.*.default bypass** | aiter MLA not registered as `torch.ops` |
| **load_inline HIP** | JIT compilation >17 min, times out |
| **Write-through (ns=1)** | 3× **worse** - split-K parallelism matters |
| **Software mxfp4 dequant** | 30-68× slower |
| **Non-persistent mode** | Harness compatibility issues |

### Why Triton mxfp4 is the Best Remaining Path

1. **`tl.dot_scaled` confirmed working on gfx950** (workflow 23896940925)
2. **2× bandwidth savings** - mxfp4 KV (4 bits) vs fp8 KV (8 bits)
3. **Fused attention** - single Triton kernel, no Python dispatch per stage
4. **Avoids harness restrictions** - no streams/graphs needed
5. **Roofline math**: bs=256,kv=8k reads 1.2GB → 150µs at 8TB/s; mxfp4 halves this

### The Challenge

- `QK_HEAD_DIM=576` is **not a power of 2** - `tl.dot()` on AMD needs power-of-2 dims
- aiter handles this with 18 MFMA tiles (576/32=18)
- Options: (a) pad to 640/768, (b) loop over tiles, (c) use `tl.dot_scaled` software path

### Key Files

- `problems/amd_202602/mixed-mla/submission.py` — current best (aiter fp8, 66µs)
- `problems/amd_202602/mixed-mla/test_dotscaled_v3.py` — working `tl.dot_scaled` test
- `CLAUDE.md` — team notes and detailed profiling data
- `HISTORY.md` — full iteration log with evidence

---

## Contributing New Problems

To add a new problem, create a new folder in the `problems/glory` directory where you need to add the following files:
- `reference.py` - This is the PyTorch reference implementation of the problem.
- `task.yml` - This is the problem specification that will be used to generate test cases for different shapes
- `task.py` - Specifies the schema of the inputs and outputs for the problem

You can evaluate problems with your own Modal account (they give you a free $30) by borrowing this [neat script from @gau-nernst](https://github.com/gpu-mode/reference-kernels/pull/96#issue-3850136894)


