#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Triton MLA decode with bf16 KV and split-K parallelism.

Key insight: bf16 KV eliminates all quant/dequant overhead.
Split-K gives parallelism across KV length for large shapes.

Architecture:
- Stage 1: Each split computes partial QK softmax + V accumulation
- Stage 2: Reduce across splits using log-sum-exp correction

For decode (q_seq=1), each threadblock handles one (batch, head, split).
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
    def _mla_stage1_kernel(
        Q,
        KV,
        SplitOut,  # (bs, heads, num_splits, V_DIM)
        SplitLse,  # (bs, heads, num_splits)
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
        BLOCK_D: tl.constexpr,
        NUM_D_BLOCKS: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        """Stage 1: Each program handles one (batch, head, split)."""
        bid = tl.program_id(0)
        hid = tl.program_id(1)
        sid = tl.program_id(2)  # split index

        kv_start = tl.load(kv_indptr + bid).to(tl.int32)
        kv_end = tl.load(kv_indptr + bid + 1).to(tl.int32)
        kv_len = kv_end - kv_start

        # Split range
        split_size = (kv_len + NUM_SPLITS - 1) // NUM_SPLITS
        split_start = sid * split_size
        split_end = tl.minimum(split_start + split_size, kv_len)

        if split_start >= kv_len:
            # This split has no work
            offs_v = tl.arange(0, V_DIM)
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
        offs_v = tl.arange(0, V_DIM)

        for n_start in range(split_start, split_end, BLOCK_N):
            n_end = tl.minimum(n_start + BLOCK_N, split_end)
            n_size = n_end - n_start
            offs_n = tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_size

            # QK dot product via tiling
            qk = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for d_block in tl.static_range(NUM_D_BLOCKS):
                d_start = d_block * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                mask_d = offs_d < QK_DIM

                q_ptr = Q + bid * stride_qb + hid * stride_qh + offs_d * stride_qd
                q_val = tl.load(q_ptr, mask=mask_d, other=0.0).to(tl.float32)

                k_ptrs = (
                    KV
                    + (kv_start + n_start + offs_n[:, None]) * stride_kt
                    + offs_d[None, :] * stride_kd
                )
                k_val = tl.load(
                    k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0
                ).to(tl.float32)

                qk += tl.sum(q_val[None, :] * k_val, axis=1)

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

            # V accumulation
            v_ptrs = (
                KV
                + (kv_start + n_start + offs_n[:, None]) * stride_kt
                + offs_v[None, :] * stride_kd
            )
            v_val = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(p[:, None] * v_val, axis=0)

            m_i = m_new

        # Store partial output and LSE
        out_ptrs = (
            SplitOut
            + bid * stride_sob
            + hid * stride_soh
            + sid * stride_sos
            + offs_v * stride_sod
        )
        tl.store(out_ptrs, (acc / l_i).to(tl.bfloat16))

        lse = m_i / LOG2E_C + tl.log(l_i)  # Convert from log2 to ln
        tl.store(SplitLse + bid * stride_slb + hid * stride_slh + sid * stride_sls, lse)

    @triton.jit
    def _mla_reduce_kernel(
        SplitOut,  # (bs, heads, num_splits, V_DIM) bf16
        SplitLse,  # (bs, heads, num_splits) f32
        Out,  # (bs, heads, V_DIM) bf16
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

        # Find global max LSE
        max_lse = -float("inf")
        for s in range(NUM_SPLITS):
            lse = tl.load(
                SplitLse + bid * stride_slb + hid * stride_slh + s * stride_sls
            )
            max_lse = tl.maximum(max_lse, lse)

        # Weighted sum
        acc = tl.zeros((V_DIM,), dtype=tl.float32)
        sum_w = 0.0
        for s in range(NUM_SPLITS):
            lse = tl.load(
                SplitLse + bid * stride_slb + hid * stride_slh + s * stride_sls
            )
            w = tl.math.exp(lse - max_lse)  # ln-space
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

    @triton.jit
    def _mla_single_pass_kernel(
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
        BLOCK_D: tl.constexpr,
        NUM_D_BLOCKS: tl.constexpr,
    ):
        """Single-pass kernel for small KV lengths."""
        bid = tl.program_id(0)
        hid = tl.program_id(1)

        kv_start = tl.load(kv_indptr + bid).to(tl.int32)
        kv_end = tl.load(kv_indptr + bid + 1).to(tl.int32)
        kv_len = kv_end - kv_start

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((V_DIM,), dtype=tl.float32)
        offs_v = tl.arange(0, V_DIM)

        for n_start in range(0, kv_len, BLOCK_N):
            n_end = tl.minimum(n_start + BLOCK_N, kv_len)
            n_size = n_end - n_start
            offs_n = tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_size

            qk = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for d_block in tl.static_range(NUM_D_BLOCKS):
                d_start = d_block * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                mask_d = offs_d < QK_DIM

                q_ptr = Q + bid * stride_qb + hid * stride_qh + offs_d * stride_qd
                q_val = tl.load(q_ptr, mask=mask_d, other=0.0).to(tl.float32)

                k_ptrs = (
                    KV
                    + (kv_start + n_start + offs_n[:, None]) * stride_kt
                    + offs_d[None, :] * stride_kd
                )
                k_val = tl.load(
                    k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0
                ).to(tl.float32)
                qk += tl.sum(q_val[None, :] * k_val, axis=1)

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

        acc = acc / l_i
        out_ptrs = Out + bid * stride_ob + hid * stride_oh + offs_v * stride_od
        tl.store(out_ptrs, acc.to(tl.bfloat16))


def _get_num_splits(bs, kvl):
    """Choose split count based on shape."""
    if kvl <= 1024:
        return 1  # Single pass
    if kvl <= 2048:
        return 2
    if kvl <= 4096:
        return 4
    return 8  # 8192


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
            torch.empty((bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=dev),
        )
    return _cache[key]


BLOCK_D = 64
NUM_D_BLOCKS = (QK_HEAD_DIM + BLOCK_D - 1) // BLOCK_D


def _triton_bf16_mla(q, kv, kv_indptr, out, bs, kvl):
    num_splits = _get_num_splits(bs, kvl)

    if num_splits == 1:
        grid = (bs, NUM_HEADS)
        _mla_single_pass_kernel[grid](
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
            BLOCK_D=BLOCK_D,
            NUM_D_BLOCKS=NUM_D_BLOCKS,
        )
    else:
        split_out, split_lse, _ = _get_bufs(bs, num_splits, q.device)
        grid_s1 = (bs, NUM_HEADS, num_splits)
        _mla_stage1_kernel[grid_s1](
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
            BLOCK_D=BLOCK_D,
            NUM_D_BLOCKS=NUM_D_BLOCKS,
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

    # Use bf16 KV - no quant/dequant needed
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
