#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Triton MLA decode with bf16 KV using tl.dot for MFMA.

Key fix: Use tl.dot for QK computation (generates MFMA instructions)
instead of element-wise multiply + reduce.

For decode (q_seq=1):
- Q: (1, 576) per head -> broadcast to (BLOCK_N, 576) is wasteful
- Better: load Q once, tile K as (BLOCK_N, BLOCK_D), use tl.dot

The challenge: tl.dot requires 2D @ 2D. For decode with q_seq=1,
we need Q as (1, D) @ K^T as (D, BLOCK_N) = scores (1, BLOCK_N).

Split-K for large KV lengths.
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
    def _mla_bf16_kernel(
        Q,
        KV,
        Out,
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
        BLOCK_N: tl.constexpr,
    ):
        """MLA decode with bf16 KV. Uses tl.dot for QK^T."""
        bid = tl.program_id(0)
        hid = tl.program_id(1)

        kv_start = tl.load(kv_indptr + bid).to(tl.int32)
        kv_end = tl.load(kv_indptr + bid + 1).to(tl.int32)
        kv_len = kv_end - kv_start

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((V_DIM,), dtype=tl.float32)
        offs_v = tl.arange(0, V_DIM)

        # Load Q once: (QK_DIM,) - reused across all KV blocks
        # Pad to 1024 (next power of 2 for tl.arange)
        offs_qk = tl.arange(0, 1024)
        mask_qk = offs_qk < QK_DIM
        q_ptr = Q + bid * stride_qb + hid * stride_qh + offs_qk * stride_qd
        q_vec = tl.load(q_ptr, mask=mask_qk, other=0.0)

        q_row = tl.reshape(q_vec, (1, 1024))

        for n_start in range(0, kv_len, BLOCK_N):
            n_end = tl.minimum(n_start + BLOCK_N, kv_len)
            n_size = n_end - n_start
            offs_n = tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_size

            # Load K: (BLOCK_N, 1024) with padding
            k_ptrs = (
                KV
                + (kv_start + n_start + offs_n[:, None]) * stride_kt
                + offs_qk[None, :] * stride_kd
            )
            k_val = tl.load(
                k_ptrs,
                mask=mask_n[:, None] & mask_qk[None, :],
                other=0.0,
            )  # (BLOCK_N, 1024) bf16

            # QK^T via tl.dot: (1, 1024) @ (1024, BLOCK_N) = (1, BLOCK_N)
            k_T = tl.trans(k_val)  # (1024, BLOCK_N)
            qk_2d = tl.dot(q_row, k_T)  # (1, BLOCK_N) f32
            qk = tl.reshape(qk_2d, (BLOCK_N,))

            qk = tl.where(mask_n, qk, -float("inf"))
            qk = qk * SM_SCALE_LOG2E

            # Online softmax
            m_ij = tl.max(qk)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new)
            p = tl.where(mask_n, p, 0.0)

            l_i = l_i * alpha + tl.sum(p)
            acc = acc * alpha

            # V accumulation: p @ V
            # V: first 512 dims of KV
            v_ptrs = (
                KV
                + (kv_start + n_start + offs_n[:, None]) * stride_kt
                + offs_v[None, :] * stride_kd
            )
            v_val = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(p[:, None] * v_val, axis=0)

            m_i = m_new

        acc = acc / l_i
        out_ptrs = Out + bid * stride_ob + hid * stride_oh + offs_v * stride_od
        tl.store(out_ptrs, acc.to(tl.bfloat16))

    @triton.jit
    def _mla_bf16_splitk_stage1(
        Q,
        KV,
        SplitOut,
        SplitLse,
        kv_indptr,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kd,
        stride_sob,
        stride_soh,
        stride_sos,
        stride_sod,
        stride_slb,
        stride_slh,
        stride_sls,
        SM_SCALE_LOG2E: tl.constexpr,
        LOG2E_C: tl.constexpr,
        QK_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        """Split-K stage1 with tl.dot for QK."""
        bid = tl.program_id(0)
        hid = tl.program_id(1)
        sid = tl.program_id(2)

        kv_start = tl.load(kv_indptr + bid).to(tl.int32)
        kv_end = tl.load(kv_indptr + bid + 1).to(tl.int32)
        kv_len = kv_end - kv_start

        split_size = (kv_len + NUM_SPLITS - 1) // NUM_SPLITS
        s_start = sid * split_size
        s_end = tl.minimum(s_start + split_size, kv_len)

        offs_v = tl.arange(0, V_DIM)

        if s_start >= kv_len:
            out_ptrs = (
                SplitOut
                + bid * stride_sob
                + hid * stride_soh
                + sid * stride_sos
                + offs_v * stride_sod
            )
            tl.store(out_ptrs, tl.zeros((V_DIM,), dtype=tl.bfloat16))
            tl.store(
                SplitLse + bid * stride_slb + hid * stride_slh + sid * stride_sls,
                float("-inf"),
            )
            return

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((V_DIM,), dtype=tl.float32)

        # Load Q once
        offs_qk = tl.arange(0, 1024)
        mask_qk = offs_qk < QK_DIM
        q_ptr = Q + bid * stride_qb + hid * stride_qh + offs_qk * stride_qd
        q_vec = tl.load(q_ptr, mask=mask_qk, other=0.0)
        q_row = tl.reshape(q_vec, (1, 1024))

        for n_start in range(s_start, s_end, BLOCK_N):
            n_end = tl.minimum(n_start + BLOCK_N, s_end)
            n_size = n_end - n_start
            offs_n = tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_size

            # QK^T via tl.dot: (1, 1024) @ (1024, BLOCK_N) = (1, BLOCK_N)
            k_ptrs = (
                KV
                + (kv_start + n_start + offs_n[:, None]) * stride_kt
                + offs_qk[None, :] * stride_kd
            )
            k_val = tl.load(k_ptrs, mask=mask_n[:, None] & mask_qk[None, :], other=0.0)
            k_T = tl.trans(k_val)
            qk_2d = tl.dot(q_row, k_T)
            qk = tl.reshape(qk_2d, (BLOCK_N,))

            qk = tl.where(mask_n, qk, -float("inf"))
            qk = qk * SM_SCALE_LOG2E

            m_ij = tl.max(qk)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new)
            p = tl.where(mask_n, p, 0.0)

            l_i = l_i * alpha + tl.sum(p)
            acc = acc * alpha

            v_ptrs = (
                KV
                + (kv_start + n_start + offs_n[:, None]) * stride_kt
                + offs_v[None, :] * stride_kd
            )
            v_val = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(p[:, None] * v_val, axis=0)
            m_i = m_new

        out_ptrs = (
            SplitOut
            + bid * stride_sob
            + hid * stride_soh
            + sid * stride_sos
            + offs_v * stride_sod
        )
        tl.store(out_ptrs, (acc / l_i).to(tl.bfloat16))

        lse = m_i / LOG2E_C + tl.log(l_i)
        tl.store(SplitLse + bid * stride_slb + hid * stride_slh + sid * stride_sls, lse)

    @triton.jit
    def _mla_reduce_kernel(
        SplitOut,
        SplitLse,
        Out,
        stride_sob,
        stride_soh,
        stride_sos,
        stride_sod,
        stride_slb,
        stride_slh,
        stride_sls,
        stride_ob,
        stride_oh,
        stride_od,
        V_DIM: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        """Reduce partial results across splits."""
        bid = tl.program_id(0)
        hid = tl.program_id(1)
        offs_v = tl.arange(0, V_DIM)

        max_lse = -float("inf")
        for s in range(NUM_SPLITS):
            lse = tl.load(
                SplitLse + bid * stride_slb + hid * stride_slh + s * stride_sls
            )
            max_lse = tl.maximum(max_lse, lse)

        acc = tl.zeros((V_DIM,), dtype=tl.float32)
        sum_w = 0.0
        for s in range(NUM_SPLITS):
            lse = tl.load(
                SplitLse + bid * stride_slb + hid * stride_slh + s * stride_sls
            )
            w = tl.math.exp(lse - max_lse)
            sum_w += w
            partial = tl.load(
                SplitOut
                + bid * stride_sob
                + hid * stride_soh
                + s * stride_sos
                + offs_v * stride_sod
            ).to(tl.float32)
            acc += w * partial

        acc = acc / sum_w
        out_ptrs = Out + bid * stride_ob + hid * stride_oh + offs_v * stride_od
        tl.store(out_ptrs, acc.to(tl.bfloat16))


def _get_num_splits(bs, kvl):
    if kvl <= 1024:
        return 1
    if kvl <= 4096:
        return 4
    return 8


def _get_bufs(bs, num_splits, dev):
    key = ("splitk_bufs", bs, num_splits)
    if key not in _cache:
        _cache[key] = (
            torch.empty(
                (bs, NUM_HEADS, num_splits, V_HEAD_DIM),
                dtype=torch.bfloat16,
                device=dev,
            ),
            torch.empty((bs, NUM_HEADS, num_splits), dtype=torch.float32, device=dev),
        )
    return _cache[key]


def _triton_bf16_mla(q, kv, kv_indptr, out, bs, kvl):
    num_splits = _get_num_splits(bs, kvl)

    if num_splits == 1:
        grid = (bs, NUM_HEADS)
        _mla_bf16_kernel[grid](
            q,
            kv,
            out,
            kv_indptr,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            kv.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            SM_SCALE_LOG2E=SM_SCALE * LOG2E,
            QK_DIM=QK_HEAD_DIM,
            V_DIM=V_HEAD_DIM,
            BLOCK_N=64,
        )
    else:
        split_out, split_lse = _get_bufs(bs, num_splits, q.device)
        grid_s1 = (bs, NUM_HEADS, num_splits)
        _mla_bf16_splitk_stage1[grid_s1](
            q,
            kv,
            split_out,
            split_lse,
            kv_indptr,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            kv.stride(1),
            split_out.stride(0),
            split_out.stride(1),
            split_out.stride(2),
            split_out.stride(3),
            split_lse.stride(0),
            split_lse.stride(1),
            split_lse.stride(2),
            SM_SCALE_LOG2E=SM_SCALE * LOG2E,
            LOG2E_C=LOG2E,
            QK_DIM=QK_HEAD_DIM,
            V_DIM=V_HEAD_DIM,
            BLOCK_N=64,
            NUM_SPLITS=num_splits,
        )
        grid_r = (bs, NUM_HEADS)
        _mla_reduce_kernel[grid_r](
            split_out,
            split_lse,
            out,
            split_out.stride(0),
            split_out.stride(1),
            split_out.stride(2),
            split_out.stride(3),
            split_lse.stride(0),
            split_lse.stride(1),
            split_lse.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            V_DIM=V_HEAD_DIM,
            NUM_SPLITS=num_splits,
        )
    return out


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

    return _triton_bf16_mla(q_reshaped, kv_flat, kv_indptr, out, bs, kvl)
