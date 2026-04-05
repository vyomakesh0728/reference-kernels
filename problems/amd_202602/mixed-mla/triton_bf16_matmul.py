#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Triton MLA decode with bf16 KV using tl.dot for MFMA.

Key insight: Must use tl.dot() to trigger MFMA instructions.
Single-pass online softmax with 2D matrix operations.

For QK: Q(1, 576) @ K(N, 576)^T -> need Q as (M, K) and K^T as (K, N)
Since q_seq=1, we tile across heads: process multiple heads per threadblock.

Architecture per threadblock:
- Grid: (bs, num_head_groups) where each group processes BLOCK_H heads
- QK: (BLOCK_H, 576) @ (576, BLOCK_N) via tiled K=64 dot products
- V:  (BLOCK_H, BLOCK_N) @ (BLOCK_N, 512) via tiled K dot products
"""

import torch
from task import input_t, output_t

try:
    import triton
    import triton.language as tl

    TRITON_OK = True
except ImportError:
    TRITON_OK = False

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)
LOG2E = 1.4426950408889634

_cache = {}

if TRITON_OK:

    @triton.jit
    def _mla_bf16_dot_kernel(
        Q,  # (bs, heads, 576) bf16
        KV,  # (total_kv, 576) bf16
        Out,  # (bs, heads, 512) bf16
        kv_indptr,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kd,
        stride_ob,
        stride_oh,
        stride_od,
        SM_SCALE_LOG2E: tl.constexpr,
        QK_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BLOCK_H: tl.constexpr,  # heads per block (e.g., 16)
        BLOCK_N: tl.constexpr,  # KV tokens per block (e.g., 16 or 32)
        BLOCK_D: tl.constexpr,  # dim tile for dot (e.g., 64)
    ):
        """MLA decode using tl.dot for MFMA."""
        bid = tl.program_id(0)  # batch
        hg = tl.program_id(1)  # head group

        kv_start = tl.load(kv_indptr + bid).to(tl.int32)
        kv_end = tl.load(kv_indptr + bid + 1).to(tl.int32)
        kv_len = kv_end - kv_start

        offs_h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_v = tl.arange(0, V_DIM)

        # Online softmax state per head
        m_i = tl.full((BLOCK_H,), value=-float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_H, V_DIM), dtype=tl.float32)

        for n_start in range(0, kv_len, BLOCK_N):
            n_end = tl.minimum(n_start + BLOCK_N, kv_len)
            n_size = n_end - n_start
            offs_n = tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_size

            # QK: (BLOCK_H, QK_DIM) @ (QK_DIM, BLOCK_N) -> (BLOCK_H, BLOCK_N)
            # Tile across K dimension
            qk = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)

            for d_start in range(0, QK_DIM, BLOCK_D):
                offs_d = d_start + tl.arange(0, BLOCK_D)
                mask_d = offs_d < QK_DIM

                # Load Q tile: (BLOCK_H, BLOCK_D)
                q_ptrs = (
                    Q
                    + bid * stride_qb
                    + offs_h[:, None] * stride_qh
                    + offs_d[None, :] * stride_qd
                )
                q_tile = tl.load(q_ptrs, mask=mask_d[None, :], other=0.0)

                # Load K tile: (BLOCK_N, BLOCK_D) then transpose
                k_ptrs = (
                    KV
                    + (kv_start + n_start + offs_n[:, None]) * stride_kt
                    + offs_d[None, :] * stride_kd
                )
                k_tile = tl.load(
                    k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0
                )

                # (BLOCK_H, BLOCK_D) @ (BLOCK_D, BLOCK_N) -> accumulate into qk
                qk += tl.dot(q_tile, tl.trans(k_tile))

            # Apply softmax scaling
            qk = qk * SM_SCALE_LOG2E
            # Mask invalid positions
            qk = tl.where(mask_n[None, :], qk, -float("inf"))

            # Online softmax update (per head)
            m_ij = tl.max(qk, axis=1)  # (BLOCK_H,)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new[:, None])
            p = tl.where(mask_n[None, :], p, 0.0)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            # V: (BLOCK_H, BLOCK_N) @ (BLOCK_N, V_DIM) -> accumulate
            # Tile across V dimension
            for v_start in range(0, V_DIM, BLOCK_D):
                v_offs = v_start + tl.arange(0, BLOCK_D)
                v_mask = v_offs < V_DIM

                # Load V tile: (BLOCK_N, BLOCK_D)
                v_ptrs = (
                    KV
                    + (kv_start + n_start + offs_n[:, None]) * stride_kt
                    + v_offs[None, :] * stride_kd
                )
                v_tile = tl.load(
                    v_ptrs, mask=mask_n[:, None] & v_mask[None, :], other=0.0
                )

                # Cast p to bf16 for dot
                p_bf16 = p.to(tl.bfloat16)

                # (BLOCK_H, BLOCK_N) @ (BLOCK_N, BLOCK_D) -> (BLOCK_H, BLOCK_D)
                pv = tl.dot(p_bf16, v_tile)

                # Accumulate into correct output slice
                acc_slice_ptrs = v_start + tl.arange(0, BLOCK_D)
                for j in range(BLOCK_D):
                    if v_start + j < V_DIM:
                        acc_col = acc[:, v_start + j]
                        pv_col = pv[:, j]
                        acc = tl.where(
                            (tl.arange(0, V_DIM) == v_start + j)[None, :],
                            (acc_col + pv_col)[:, None]
                            * tl.ones((1, V_DIM), dtype=tl.float32),
                            acc,
                        )

            m_i = m_new

        # Normalize
        acc = acc / l_i[:, None]

        # Store
        for h in range(BLOCK_H):
            out_ptrs = (
                Out
                + bid * stride_ob
                + (hg * BLOCK_H + h) * stride_oh
                + offs_v * stride_od
            )
            tl.store(out_ptrs, acc[h, :].to(tl.bfloat16))


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    kv_bf16 = kv_data["bf16"]
    q_reshaped = q.view(bs, NUM_HEADS, QK_HEAD_DIM)
    kv_flat = kv_bf16.view(kv_bf16.shape[0], kv_bf16.shape[-1])

    out_key = ("bf16_out", bs)
    if out_key not in _cache:
        _cache[out_key] = torch.empty(
            (bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device
        )
    out = _cache[out_key]

    BLOCK_H = NUM_HEADS  # All 16 heads in one block
    grid = (bs, NUM_HEADS // BLOCK_H)

    _mla_bf16_dot_kernel[grid](
        q_reshaped,
        kv_flat,
        out,
        kv_indptr,
        q_reshaped.stride(0),
        q_reshaped.stride(1),
        q_reshaped.stride(2),
        kv_flat.stride(0),
        kv_flat.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        SM_SCALE_LOG2E=SM_SCALE * LOG2E,
        QK_DIM=QK_HEAD_DIM,
        V_DIM=V_HEAD_DIM,
        BLOCK_H=BLOCK_H,
        BLOCK_N=16,
        BLOCK_D=64,
    )
    return out
