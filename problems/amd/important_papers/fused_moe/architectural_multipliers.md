# MoE Architectural Multipliers

These are the highest-value structural bets for this repo.

## Best Bets

1. Padding-free routed-expert packing plus fused reorder/linear flow.
   This is the clearest transfer from ScatterMoE and SonicMoE.
   It attacks sparse-per-expert padding waste and Python-side regroup overhead directly.

2. Persistent expert-tile pipeline across `stage1 -> SwiGLU -> stage2`.
   This is the durable FlashMoE and SonicMoE idea.
   Keep expert metadata resident, overlap IO with compute, and avoid relaunch boundaries.

3. Shared-expert split plus regime-specialized kernels.
   DeepSeek-style shared experts and fine-grained routed experts should not share one schedule.
   Use separate sparse and dense paths instead of one generic kernel.

## Repo-Specific Translation

- keep `topk_ids` and `topk_weights` fixed
- pack touched `(token, expert, weight)` entries once
- do not dequantize all experts up front
- do not run expert loops in Python on benchmark lanes
- specialize by live regime, not by one global block size

## Ideas To Reject

- token rounding that changes expert multiplicity or routing
- training-only load-balancing or auxiliary-loss tricks
- CUDA-only implementation details such as CUTLASS, cuBLASDx, or NVSHMEM as direct code templates
- copying the `mxfp4_mm` scaled-MFMA ladder before the MoE structure is fixed
