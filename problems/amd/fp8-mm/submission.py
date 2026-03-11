#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
# AGENT_LOOP_META: {"attempt": 42, "generator": {"kind": "llm", "model": "(codex default)", "parallel_agents": 3, "provider": "codex_cli", "use_plan": true}, "gpu": "MI355X", "leaderboard": "amd-mxfp4-mm", "policy_profile": {"family": "kernel_explore", "focus": "preserve shuffled MXFP4 semantics before tuning", "name": "contract_repair", "trigger_signals": ["contract_repair", "runtime_repair", "submission_repair"]}, "problem": "mxfp4_mm", "variant": {"BLOCK_K": 128, "BLOCK_M": 64, "BLOCK_N": 256, "CONTRACT_NATIVE": true, "GROUP_M": 4, "NUM_STAGES": 3, "NUM_WARPS": 8, "family": "kernel_explore", "strategy": "contract_bf16_dequant_matmul", "variant_name": "kernel_contract_m64n256k128"}, "variant_index": 4}

import aiter
from aiter import QuantType, dtypes
import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def _bf16_matmul_scaffold(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, k, BLOCK_K):
        k_offsets = k_start + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_ptrs = b_ptr + offs_n[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < m) & (k_offsets[None, :] < k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < n) & (k_offsets[:, None] < k), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


def _unused_triton_bf16_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[0]
    c = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(m, 64) * triton.cdiv(n, 128),)
    _bf16_matmul_scaffold[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=32,
        GROUP_M=4,
        num_warps=8,
        num_stages=3,
    )
    return c


def custom_kernel(data: input_t) -> output_t:
    a, b, b_q, b_shuffle, b_scale_sh = data
    del b, b_q
    quant = aiter.get_triton_quant(QuantType.per_1x32)
    a_q, a_scale_sh = quant(a.contiguous(), shuffle=True)
    return aiter.gemm_a4w4(
        a_q,
        b_shuffle,
        a_scale_sh,
        b_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
