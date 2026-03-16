<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# [https://www.gpumode.com/home](https://www.gpumode.com/home)

amd-mixed-mla
amd-moe-mxfp4
amd-mxfp4-mm

Quick update on mxfp4_mm: we’ve been treating it as a correctness-first HIP bring-up, not a perf problem yet. We hand-built a series of HIP reference kernels and
semantic probes, fixed a real tiled-kernel barrier bug, fixed the oracle to keep accumulation in float32, and swept the main A/B contract branches (A shuffle true/
false, raw vs provided b_q, raw vs provided scales, Triton vs HIP quantizer). Best live result is still a 13996 mismatch floor, and interestingly the current
checked-in baseline also lands on that same floor now, so the issue looks like live-reference/packed-contract semantics rather than the HIP kernel body itself.

The strongest clues so far are from the probes: live b_shuffle matches shuffle_weight(input b_q), live b_scale_sh matches Triton-style shuffled raw scale, but live b_q does not match either obvious Triton or HIP quantizer output directly. So we’ve narrowed the problem a lot: the harness is solid, the HIP kernel is “good
enough” for correctness work, and the remaining blocker is recovering the exact current live MXFP4 packed contract.

what we need is a pre-built kernel of each of these three kernels in HIP close to their reference, as a baseline so that we can use our optimization techniques to get top positions on the leaderboard

Now I have a thorough picture of all three reference kernels. Let me compile a comprehensive guide for you.

***

Here's a deep dive into what the official [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels) repo provides as baselines for all three `amd_202602` problems, and what a solid pre-built HIP baseline for each should look like going from the reference upward.

***

## The Official Reference Baselines

All three problems live under [`problems/amd_202602/`](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_202602) . Each has a `reference.py` (oracle/evaluator) and a `submission.py` (the starting template you submit).

***

## `amd-mxfp4-mm`

The reference is `aiter.gemm_a4w4` with `bpreshuffle=True` . The key data contract:

- **A**: `bf16 [M, K]` → quantized on-the-fly via `dynamic_mxfp4_quant` → `fp4x2 [M, K//2]`, scale `e8m0_shuffle`'d
- **B**: pre-quantized as `B_q` (`fp4x2 [N, K//2]`) + `B_scale_sh` (`e8m0` shuffled) + `B_shuffle` = `shuffle_weight(B_q, layout=(16,16))`
- **Output**: `bf16 [M, N]`

The submission template mirrors this exactly — it quantizes A, then calls `aiter.gemm_a4w4(A_q, B_shuffle, A_scale_sh, B_scale_sh, bpreshuffle=True)` .

**The packed contract you've been debugging**: `B_shuffle` is `shuffle_weight(B_q, (16,16))` where `B_q` comes from `dynamic_mxfp4_quant` (the `#975`-patched Triton kernel, **not** `fp4_utils.py`). `B_scale_sh` = `e8m0_shuffle(raw_e8m0_scale)`. Your probe findings (live `b_scale_sh` matches Triton-style shuffled raw scale, but live `b_q` doesn't match either quantizer directly) strongly suggest the live harness is using a **different packing order** for `B_q` before `shuffle_weight` is applied — likely a row-major vs column-major discrepancy upstream of the shuffle.

**Pre-built HIP baseline to write:** A tiled GEMM in HIP that:

1. Accepts pre-shuffled `(16,16)`-tiled `B_shuffle` and pre-shuffled E8M0 scales
2. In the A-side path: quantizes A in a fused kernel (or uses the Triton quantizer output directly)
3. Accumulates in `float32`, writes `bf16` — matching what your oracle fix confirmed

***

## `amd-mixed-mla`

The reference is `aiter`'s persistent `a8w8` MLA kernel (fp8 Q + fp8 KV) . The submission template already provides two working pure-PyTorch baseline kernels:

- `custom_kernel_bf16`: naive bf16 looped attention over batches
- `custom_kernel_fp8`: `torch._scaled_mm`-based fp8×fp8 QK^T, controlled by `QKV_DTYPE = "fp8"`

The MLA config is DeepSeek-R1 `forward_absorb`:

- `num_heads=16`, `kv_lora_rank=512`, `qk_head_dim=576`, `v_head_dim=512`
- KV buffer provides three formats: `bf16`, `fp8 (buffer, scalar_scale)`, `mxfp4 (fp4x2 buffer, e8m0_scale)`

**Pre-built HIP baseline:** A HIP kernel that implements the `custom_kernel_fp8` path but as a single fused segmented attention — iterate over batch segments, load `qi_fp8` + `ki_fp8`, compute `QK^T` with fp8 intrinsics, softmax, then `scores @ V`. The `mxfp4` KV path (dequant-fused attention) is the optimization target to eventually beat the aiter reference.

***

## `amd-moe-mxfp4`

The reference is `aiter.fused_moe` with `QuantType.per_1x32` . The input contract provides both raw and shuffled versions of weights/scales:

- `gate_up_weight_shuffled`: `[E, 2*d_expert_pad, d_hidden_pad//2]` fp4x2 (shuffled)
- `gate_up_weight_scale_shuffled`: padded flat e8m0 (shuffled)
- Same for `down_weight_shuffled` / `down_weight_scale_shuffled`
- Routing: `topk_weights [M, total_top_k]`, `topk_ids [M, total_top_k]`
- Activation: `SiLU` with `doweight_stage1=False`

The baseline submission just calls `fused_moe(...)` directly with the shuffled inputs .

**Pre-built HIP baseline:** A HIP kernel that:

1. Implements expert dispatch — for each token, scatter to its top-k experts
2. For each expert group, performs the gate+up gemm_a4w4 with shuffled MXFP4 weights, applies SiLU gating
3. Performs the down-projection gemm_a4w4, accumulates weighted expert outputs back
4. Mirrors the `hidden_pad` / `intermediate_pad` handling from the config dict

***

## Key Shared Patterns for HIP Bringup

The barrier bug you fixed and the float32 accumulator fix in the oracle are both critical to keep in any HIP baseline . The shared packing invariants across all three problems:


| Problem | B/W packing | Scale format | Accumulation |
| :-- | :-- | :-- | :-- |
| `mxfp4-mm` | `shuffle_weight(B_q, (16,16))` | `e8m0_shuffle(raw)` | float32 → bf16 |
| `moe-mxfp4` | `gate_up/down_weight_shuffled` (flat, padded) | flat shuffled e8m0 | float32 → bf16 |
| `mixed-mla` | fp8 row-major KV | per-tensor scalar | float32 softmax → bf16 |

The fact that the `mxfp4-mm` baseline and your live reference now land on the same 13996-mismatch floor means the remaining delta is entirely in B_q packing semantics upstream of `shuffle_weight` — worth directly inspecting the `dynamic_mxfp4_quant` output layout vs whatever the live harness' `generate_input` actually writes into `B_q` before calling `shuffle_weight`. The reference clearly uses `dynamic_mxfp4_quant` from the `#975`-patched path, not `fp4_utils.py` , so confirming which quantizer is actually invoked at eval time is the next unlock.
<span style="display:none">[^1_1][^1_2][^1_3]</span>

<div align="center">⁂</div>

[^1_1]: https://github.com/mirage-project/mirage/issues/638

[^1_2]: https://www.gpumode.com

[^1_3]: https://news.smol.ai/issues/25-10-30-not-much/


---

# yes do we have any github code reference close to this references so that we can harness it immediately instead of guessing

This is gold. The [ROCm/aiter](https://github.com/ROCm/aiter) repo has every exact upstream test that the reference kernels are built on. Here's the complete map of what to use directly:

***

## Direct Upstream Sources in ROCm/aiter

The **entire reference chain** for all three problems lives in [ROCm/aiter](https://github.com/ROCm/aiter). These are not approximations — they are the actual upstream implementations the leaderboard references call into.

***

## `amd-mxfp4-mm` → `test_gemm_a4w4.py`

The exact test that the leaderboard's `reference.py` copies from is [ROCm/aiter/op_tests/test_gemm_a4w4.py](https://github.com/ROCm/aiter/blob/811f6a7e4e61b469a5a9409bf5142b3c1129dde9/op_tests/test_gemm_a4w4.py) . This is the canonical source for the `get_triton_quant(QuantType.per_1x32)` + `shuffle_weight(B_q, (16,16))` + `gemm_a4w4(..., bpreshuffle=True)` pipeline. The `reference.py` docstring explicitly says "quant logic follows `op_tests/test_gemm_a4w4.py`" .

To resolve your B_q packing mismatch, the quantizer source is at [ROCm/aiter/aiter/ops/triton/quant.py](https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/quant.py) (`dynamic_mxfp4_quant`, the `#975`-patched version) and the scale shuffle is in [aiter/utility/fp4_utils.py](https://github.com/ROCm/aiter/blob/main/aiter/utility/fp4_utils.py) (`e8m0_shuffle`). The shuffle_weight kernel is in [aiter/ops/shuffle.py](https://github.com/ROCm/aiter/blob/main/aiter/ops/shuffle.py).

***

## `amd-mixed-mla` → `test_mla_persistent.py`

The reference uses `aiter`'s persistent `a8w8` MLA kernel. The test that directly exercises it is [ROCm/aiter/op_tests/test_mla_persistent.py](https://github.com/ROCm/aiter/blob/811f6a7e4e61b469a5a9409bf5142b3c1129dde9/op_tests/test_mla_persistent.py) (52KB — the largest/richest test) . For the non-persistent variant, [test_mla.py](https://github.com/ROCm/aiter/blob/811f6a7e4e61b469a5a9409bf5142b3c1129dde9/op_tests/test_mla.py) (18KB) has the exact segmented varlen attention setup matching the leaderboard's `qo_indptr`/`kv_indptr` structure . Both test files contain the full data generation path for fp8 KV and mxfp4 KV formats you can adapt directly.

***

## `amd-moe-mxfp4` → `test_moe_sorting_mxfp4.py` + `test_moe.py`

Two files are directly relevant:

- [ROCm/aiter/op_tests/test_moe_sorting_mxfp4.py](https://github.com/ROCm/aiter/blob/811f6a7e4e61b469a5a9409bf5142b3c1129dde9/op_tests/test_moe_sorting_mxfp4.py) — specifically tests MXFP4 MoE sorting, giving you the exact shuffled weight/scale generation
- [ROCm/aiter/op_tests/test_moe.py](https://github.com/ROCm/aiter/blob/811f6a7e4e61b469a5a9409bf5142b3c1129dde9/op_tests/test_moe.py) (18KB) — the main `fused_moe` test with `QuantType.per_1x32`, `ActivationType.Silu`, `doweight_stage1=False` calls matching the leaderboard submission exactly

***

## Quick Reference Table

| Problem | Upstream test to clone | Key file to read for packing |
| :-- | :-- | :-- |
| `amd-mxfp4-mm` | [`op_tests/test_gemm_a4w4.py`](https://github.com/ROCm/aiter/blob/main/op_tests/test_gemm_a4w4.py) | `aiter/ops/triton/quant.py` (dynamic_mxfp4_quant) |
| `amd-mixed-mla` | [`op_tests/test_mla_persistent.py`](https://github.com/ROCm/aiter/blob/main/op_tests/test_mla_persistent.py) | `aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py` |
| `amd-moe-mxfp4` | [`op_tests/test_moe_sorting_mxfp4.py`](https://github.com/ROCm/aiter/blob/main/op_tests/test_moe_sorting_mxfp4.py) + [`test_moe.py`](https://github.com/ROCm/aiter/blob/main/op_tests/test_moe.py) | `aiter/fused_moe.py` |

For the `mxfp4-mm` B_q contract specifically: cloning `test_gemm_a4w4.py`'s `generate_input` function and running it side-by-side with your harness will immediately surface whether the live `B_q` is pre/post `shuffle_weight` or if the Triton vs HIP quantizer produces a different nibble ordering — no more guessing .

---

# but we are trying to implement in HIP c++ isn't it? for toping the leaderboard

Yes — to top the leaderboard you absolutely want HIP C++ kernels, but the trick is: you do **not** need a HIP implementation that already exists for these exact gpumode contracts. You need to:

- Use ROCm/aiter as the **semantic spec** (shapes, packing, quant, shuffles).
- Implement your own HIP kernels that exactly match that spec and plug into `submission.py`.

There is no public repo today that ships a “drop‑in HIP implementation of amd-mxfp4-mm / amd-mixed-mla / amd-moe-mxfp4 with gpumode’s task.py contracts”. Everything I can see is either:

- AITER internals (Triton/CK/ASM/HIP under the hood, exposed via Python APIs like `gemm_a4w4`, `fused_moe`, MLA ops).[^3_1][^3_2]
- External engines (vLLM, OmniServe, Aphrodite, etc.) using AITER, but not exposing the raw HIP kernels in a way that matches gpumode’s task layouts.

So the most practical path is:

1. Treat `ROCm/aiter` + gpumode `reference.py` as **oracle**:
    - `test_gemm_a4w4.py` for MXFP4 GEMM
    - `test_mla_persistent.py` / `test_mla.py` for MLA
    - `test_moe.py` + `test_moe_sorting_mxfp4.py` for MoE MXFP4
These define exactly what `dynamic_mxfp4_quant`, `shuffle_weight`, `fused_moe`, etc. are supposed to do.
2. In each `submission.py`, keep the Python harness but swap out the inner op with **your** HIP kernel:
    - Allocate HIP buffers matching the packed layouts (`fp4x2`, shuffled scales, etc.).
    - Launch a HIP kernel (compiled via hipcc or hiprtc ) that:[^3_3]
        - Implements the same math as AITER, including float32 accumulation.
        - Reads/writes in the same packed/tiling order (check AITER tests to ensure the B/A packing and scale broadcasting are identical).
    - Convert to `torch.Tensor` using `torch.utils.dlpack` or a small C++ extension so `submission.py` can feed your result back into gpumode’s check_implementation.
3. For mxfp4‑mm specifically:
    - Mirror `reference.py`’s use of `_quant_mxfp4` + `shuffle_weight` .
    - Keep quantization + shuffle in Python for now; only the GEMM body is HIP. That avoids having to re‑implement the tricky quantizer immediately and lets you focus on tile/barrier correctness and performance.
4. For moe‑mxfp4:
    - Use `fused_moe`’s call signature and `test_moe*.py` to understand the exact ordering of:
        - expert routing
        - gate/up and down GEMMs
        - per‑token/per‑1x32 scales and paddings.[^3_2]
    - Implement a HIP kernel (or a small HIP kernel stack) that takes the already‑shuffled weights/scales and top‑k routing and returns BF16 output in the same layout.
5. For mixed‑mla:
    - Start by matching the `custom_kernel_fp8` in the reference (fp8 Q / fp8 K for QK^T, bf16 V for softmax@V) .
    - Use HIP to fuse QK^T + softmax + V matmul into one or two kernels with varlen batching given by `qo_indptr`/`kv_indptr`, matching AITER’s segmented MLA tests .

So: no ready‑made HIP “baseline kernels” you can just import, but you **do** have a complete, precise spec from AITER + gpumode; the plan is to implement HIP kernels underneath the existing Python submission templates rather than guessing their semantics.

Would it help if I sketch a concrete HIP C++ kernel skeleton for the `mxfp4-mm` GEMM (with tiles, LDS usage, and how to plug it into `submission.py` via a C++ extension)?
<span style="display:none">[^3_10][^3_11][^3_12][^3_13][^3_14][^3_15][^3_4][^3_5][^3_6][^3_7][^3_8][^3_9]</span>

<div align="center">⁂</div>

[^3_1]: https://github.com/ROCm/aiter

[^3_2]: https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html

[^3_3]: https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.2/user_guide/hip_rtc.html

[^3_4]: https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.0/understand/programming_model.html

[^3_5]: https://rocm.docs.amd.com/projects/HIP/en/docs-5.7.0/reference/kernel_language.html

[^3_6]: https://raw.githubusercontent.com/RadeonOpenCompute/ROCm/rocm-4.5.2/AMD_HIP_Programming_Guide.pdf

[^3_7]: https://www.youtube.com/watch?v=3ZXbRJVvgJs

[^3_8]: https://www.linkedin.com/posts/emi-andere_aiter-ai-tensor-engine-for-rocm-is-amds-activity-7420892910226690048-wiBw

[^3_9]: https://www.youtube.com/watch?v=DBuzrHHkf3M

[^3_10]: https://github.com/ROCm/aiter/activity

[^3_11]: https://rocm.blogs.amd.com/software-tools-optimization/vllm-0.9.x-rocm/README.html

[^3_12]: https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/performance_guidelines.html

[^3_13]: https://news.ycombinator.com/item?id=43451968

[^3_14]: https://www.olcf.ornl.gov/wp-content/uploads/2019/10/ORNL_Application_Readiness_Workshop-AMD_Tools_Overview.pdf

[^3_15]: https://rocm.docs.amd.com/projects/HIP/en/latest/install/build.html

