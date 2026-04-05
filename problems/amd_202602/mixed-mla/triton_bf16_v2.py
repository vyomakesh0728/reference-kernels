#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Triton MLA decode with bf16 KV using tl.dot() for MFMA.

Each threadblock processes one (batch, head) pair.
Q is (1, 576) per head — reshape to (M, K) for tl.dot.
K is (BLOCK_N, 576) — use tl.dot(Q_tile, K_tile^T) for QK.
V is (BLOCK_N, 512) — use scalar p*v for V (avoid 2D acc indexing).

Strategy: Use tl.dot for QK (the expensive part), scalar for V.
QK is 576-dim dot product per token — this dominates.
V is just weighting by scalar attention weights.
"""

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
    def _mla_bf16_v2_kernel(
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
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_D_BLOCKS_QK: tl.constexpr,
        NUM_D_BLOCKS_V: tl.constexpr,
    ):
        """MLA decode: tl.dot for QK, vector p*V for output."""
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

            # QK via tiled tl.dot: Q(1, K_tile) @ K(K_tile, BLOCK_N)
            # Reshape Q(576) as tiles of (1, BLOCK_D)
            # K(BLOCK_N, 576) loaded as tiles of (BLOCK_N, BLOCK_D), transposed
            qk = tl.zeros((BLOCK_N,), dtype=tl.float32)

            for d_block in tl.static_range(NUM_D_BLOCKS_QK):
                d_start = d_block * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                mask_d = offs_d < QK_DIM

                # Q tile: (1, BLOCK_D) -> (BLOCK_D,)
                q_ptrs = Q + bid * stride_qb + hid * stride_qh + offs_d * stride_qd
                q_tile = tl.load(q_ptrs, mask=mask_d, other=0.0).to(tl.float32)

                # K tile: (BLOCK_N, BLOCK_D)
                k_ptrs = (
                    KV
                    + (kv_start + n_start + offs_n[:, None]) * stride_kt
                    + offs_d[None, :] * stride_kd
                )
                k_tile = tl.load(
                    k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0
                ).to(tl.float32)

                # Vector dot: q(BLOCK_D) broadcast * k(BLOCK_N, BLOCK_D) -> sum -> (BLOCK_N,)
                qk += tl.sum(q_tile[None, :] * k_tile, axis=1)

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

            # V accumulation: p(BLOCK_N) @ V(BLOCK_N, V_DIM)
            # Tile V across dim to keep loads manageable
            for v_block in tl.static_range(NUM_D_BLOCKS_V):
                v_start = v_block * BLOCK_D
                v_offs = v_start + tl.arange(0, BLOCK_D)
                v_mask = v_offs < V_DIM

                v_ptrs = (
                    KV
                    + (kv_start + n_start + offs_n[:, None]) * stride_kt
                    + v_offs[None, :] * stride_kd
                )
                v_tile = tl.load(
                    v_ptrs, mask=mask_n[:, None] & v_mask[None, :], other=0.0
                ).to(tl.float32)

                # p(BLOCK_N, 1) * V(BLOCK_N, BLOCK_D) -> sum over BLOCK_N -> (BLOCK_D,)
                pv = tl.sum(p[:, None] * v_tile, axis=0)  # (BLOCK_D,)

                # Store into acc at correct positions
                acc_ptrs = v_start + tl.arange(0, BLOCK_D)
                acc_mask = acc_ptrs < V_DIM
                # Use pointer arithmetic to write into acc
                # acc[v_start:v_start+BLOCK_D] += pv
                acc = tl.where(
                    (offs_v >= v_start) & (offs_v < v_start + BLOCK_D),
                    acc
                    + tl.where(
                        (offs_v >= v_start) & (offs_v < v_start + BLOCK_D),
                        # Extract the right element from pv
                        # pv is (BLOCK_D,) and we need pv[offs_v - v_start]
                        # But we can't index dynamically...
                        # Instead, broadcast pv to V_DIM and mask
                        0.0,  # placeholder
                        0.0,
                    ),
                    acc,
                )

            m_i = m_new

        acc = acc / l_i

        out_ptrs = Out + bid * stride_ob + hid * stride_oh + offs_v * stride_od
        tl.store(out_ptrs, acc.to(tl.bfloat16))
