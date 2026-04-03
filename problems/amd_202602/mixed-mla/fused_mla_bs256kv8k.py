#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Tier 2: Fused HIP kernel for bs=256,kv=8k. Single launch, no split-K materialization."""
import torch
from task import input_t, output_t
import torch.utils.cpp_extension

# Constants
NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS  # 16

# Fused HIP kernel source
FUSED_MLA_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

#define M_MFMA 16
#define K_MFMA 32
#define N_MFMA 16

// FP8 E4M3 to FP32 conversion
__device__ inline float fp8_to_f32(uint8_t x) {
    union { uint32_t u; float f; } u2f;
    uint32_t mant = x & 0x7;
    uint32_t exp = (x >> 3) & 0xF;
    uint32_t sign = (x >> 7) << 31;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        u2f.u = sign | ((exp - 1 + 127) << 23) | (mant << 20);
    } else if (exp == 15) {
        u2f.u = sign | 0x7F800000u | (mant << 20);
    } else {
        u2f.u = sign | ((exp - 1 + 127) << 23) | (mant << 20);
    }
    return u2f.f;
}

// BF16 to FP32
__device__ inline float bf16_to_f32(uint16_t x) {
    union { uint32_t u; float f; } u2f;
    u2f.u = ((uint32_t)x) << 16;
    return u2f.f;
}

// MFMA QK^T: 16x16x32_fp8_fp8
// D = A * B where A[M,K], B[K,N], D[M,N]
// A: bf16 Q [16 heads, 32 dims per head chunk]
// B: fp8 K [32 dims, 16 KV tokens]
// D: float32 scores [16 heads, 16 KV tokens]
__device__ void mfma_qk_f32_16x16x32(
    const uint16_t* __restrict__ A,  // [M, K] bf16 row-major
    const uint8_t* __restrict__ B,   // [K, N] fp8 row-major
    float* __restrict__ D,            // [M, N] output
    int lda, int ldb, int ldd
) {
    // Use inline asm for gfx950 MFMA
    // This stub uses scalar for now - will replace with MFMA
    for (int m = 0; m < M_MFMA; m++) {
        for (int n = 0; n < N_MFMA; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K_MFMA; k++) {
                float a = bf16_to_f32(A[m * lda + k]);
                float b = fp8_to_f32(B[k * ldb + n]);
                acc += a * b;
            }
            D[m * ldd + n] = acc;
        }
    }
}

// Fused MLA kernel for bs=256, kv=8192
// Each workgroup handles one query token (batch * seq=1)
__global__ void fused_mla_bs256kv8k(
    const void* __restrict__ q_ptr,      // [256, 16, 576] bf16
    const void* __restrict__ k_ptr,      // [8192, 1, 64, 8] fp8 (reshaped from [8192, 512])
    const void* __restrict__ v_ptr,      // [8192, 1, 64, 8] fp8
    const float* __restrict__ kv_scale,  // [1] scale for KV
    void* __restrict__ out_ptr,          // [256, 16, 512] bf16 output
    float sm_scale,
    int batch_size,
    int kv_len
) {
    // Each thread block handles one batch element (one query token)
    int bid = blockIdx.x;
    if (bid >= batch_size) return;

    const uint16_t* q = static_cast<const uint16_t*>(q_ptr);
    const uint8_t* k = static_cast<const uint8_t*>(k_ptr);
    const uint8_t* v = static_cast<const uint8_t*>(v_ptr);
    float* out = static_cast<float*>(out_ptr);  // accumulate in fp32
    float scale = kv_scale[0];

    // Shared memory for tiles
    __shared__ float s_qk[M_MFMA * N_MFMA];     // [16, 16] scores
    __shared__ float s_softmax[M_MFMA * N_MFMA]; // [16, 16] softmax
    __shared__ float s_v_accum[M_MFMA * V_HEAD_DIM]; // [16, 512] per-head output

    // Initialize output accumulation
    for (int i = threadIdx.x; i < M_MFMA * V_HEAD_DIM; i += blockDim.x) {
        s_v_accum[i] = 0.0f;
    }
    __syncthreads();

    // Each thread handles one head
    int head = threadIdx.x;
    if (head >= M_MFMA) return;

    // Q pointer for this batch + head: [bid, head, :]
    const uint16_t* q_h = q + (bid * NUM_HEADS + head) * QK_HEAD_DIM;

    // Process K in tiles of N_MFMA=16 KV tokens
    const int kv_tiles = (kv_len + N_MFMA - 1) / N_MFMA;

    for (int kv_tile = 0; kv_tile < kv_tiles; kv_tile++) {
        int kv_start = kv_tile * N_MFMA;
        int kv_end = min(kv_start + N_MFMA, kv_len);
        int tile_len = kv_end - kv_start;

        // Load K tile: [tile_len, 64, 8] fp8 -> [32 dims, 16 tokens] for MFMA
        // K is stored as [kv_len, num_kv_heads, 64, 8] where 64*8=512
        // We need to load 32 dims at a time for MFMA
        __shared__ uint8_t s_k[K_MFMA * N_MFMA];  // [32, 16]
        __shared__ float s_v[N_MFMA * V_HEAD_DIM]; // [16, 512] V values

        // Load K tile (32 dims x 16 tokens max)
        for (int i = threadIdx.x; i < tile_len * K_MFMA; i += blockDim.x) {
            int token = kv_start + (i / K_MFMA);
            int dim = i % K_MFMA;
            if (token < kv_end && dim < QK_HEAD_DIM) {
                // k[token, 0, dim//8, dim%8]
                int k_idx = token * 64 * 8 + (dim / 8) * 8 + (dim % 8);
                s_k[dim * N_MFMA + (token - kv_start)] = k[k_idx];
            }
        }

        // Load V tile: [16 tokens, 512 dims]
        for (int i = threadIdx.x; i < tile_len * V_HEAD_DIM; i += blockDim.x) {
            int token = kv_start + (i / V_HEAD_DIM);
            int vdim = i % V_HEAD_DIM;
            if (token < kv_end) {
                // v[token, 0, vdim//8, vdim%8]
                int v_idx = token * 64 * 8 + (vdim / 8) * 8 + (vdim % 8);
                s_v[(token - kv_start) * V_HEAD_DIM + vdim] = fp8_to_f32(v[v_idx]) * scale;
            }
        }
        __syncthreads();

        // QK^T via MFMA: [16 heads, 16 tokens]
        // Q has 576 dims, need 18 MFMA calls (576/32 = 18)
        float qk_local[N_MFMA] = {0};

        for (int qk_chunk = 0; qk_chunk < (QK_HEAD_DIM + K_MFMA - 1) / K_MFMA; qk_chunk++) {
            int k_start = qk_chunk * K_MFMA;
            int k_end = min(k_start + K_MFMA, QK_HEAD_DIM);
            int k_len = k_end - k_start;

            // Load Q chunk: [16 heads, 32 dims] -> actually just 1 head's Q chunk
            __shared__ uint16_t s_q[K_MFMA];  // [32] dims for this head
            for (int i = threadIdx.x; i < k_len; i++) {
                s_q[i] = q_h[k_start + i];
            }
            __syncthreads();

            if (threadIdx.x < M_MFMA && threadIdx.x < tile_len) {
                // Scalar QK^T for this head, this token
                for (int k = 0; k < k_len; k++) {
                    float q_val = bf16_to_f32(s_q[k]);
                    float k_val = fp8_to_f32(s_k[k * N_MFMA + threadIdx.x]) * scale;
                    qk_local[threadIdx.x] += q_val * k_val;
                }
            }
            __syncthreads();
        }

        // Scale and softmax within this tile
        float max_val = -INFINITY;
        for (int i = 0; i < tile_len; i++) {
            qk_local[i] *= sm_scale;
            if (qk_local[i] > max_val) max_val = qk_local[i];
        }

        // Exp and sum
        float sum = 0.0f;
        for (int i = 0; i < tile_len; i++) {
            qk_local[i] = expf(qk_local[i] - max_val);
            sum += qk_local[i];
        }

        // Accumulate into output: softmax * V
        // V is [16 tokens, 512 dims], output is [1 head, 512 dims]
        for (int vdim = threadIdx.x; vdim < V_HEAD_DIM; vdim += blockDim.x) {
            float v_acc = 0.0f;
            for (int i = 0; i < tile_len; i++) {
                v_acc += (qk_local[i] / sum) * s_v[i * V_HEAD_DIM + vdim];
            }
            s_v_accum[head * V_HEAD_DIM + vdim] += v_acc;
        }
        __syncthreads();
    }

    // Write output
    if (threadIdx.x < M_MFMA) {
        for (int vdim = 0; vdim < V_HEAD_DIM; vdim++) {
            out[bid * NUM_HEADS * V_HEAD_DIM + head * V_HEAD_DIM + vdim] =
                s_v_accum[head * V_HEAD_DIM + vdim];
        }
    }
}
"""

# Compile the kernel
def _compile_kernel():
    try:
        module = torch.utils.cpp_extension.load_inline(
            name='fused_mla_bs256kv8k',
            cpp_sources=FUSED_MLA_SRC,
            extra_cuda_cflags=['-O3', '--offload-arch=gfx950'],
            with_cuda=True,
        )
        return module
    except Exception as e:
        print(f"Failed to compile: {e}")
        return None

_module = None
_out_buf = None

def custom_kernel(data: input_t) -> output_t:
    global _module, _out_buf

    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    # Only use fused kernel for bs=256, kv=8192
    if bs == 256 and kvl == 8192:
        if _module is None:
            _module = _compile_kernel()
            if _module is None:
                # Fall back to aiter
                return _fallback_kernel(data)
            if _out_buf is None:
                _out_buf = torch.empty((bs, NUM_HEADS, V_HEAD_DIM),
                                       dtype=torch.bfloat16, device=q.device)

        kv_fp8, kv_scale = kv_data["fp8"]
        kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])

        # Reshape Q: [total_q, 16, 576]
        q_view = q.view(-1, NUM_HEADS, QK_HEAD_DIM)

        # For simplicity, extract K and V from the packed format
        # kv_4d is [8192, 1, 1, 512] in fp8
        # We need to reshape to [8192, 64, 8] for our kernel
        k_reshaped = kv_4d[:, 0, 0, :].contiguous()  # [8192, 512]
        v_reshaped = k_reshaped  # Same for MLA

        # Launch kernel
        threads = 256
        blocks = bs
        _module.fused_mla_bs256kv8k(
            q_view, k_reshaped, v_reshaped, kv_scale,
            _out_buf, SM_SCALE, bs, kvl,
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
        )

        return _out_buf

    # Fallback to aiter for other shapes
    return _fallback_kernel(data)

def _fallback_kernel(data: input_t) -> output_t:
    """Original aiter path for non-target shapes."""
    from aiter import dtypes as aiter_dtypes
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
    from aiter.mla import mla_decode_fwd
    from aiter.ops.quant import dynamic_per_tensor_quant

    FP8_DTYPE = aiter_dtypes.fp8
    _cache = {}

    def _get_config(bs, kvl):
        if kvl <= 1024:
            if bs <= 32: return (8, False, 2, True)
            if bs <= 64: return (4, False, 2, True)
            return (4, False, 2, True)
        else:
            if bs <= 4:  return (32, False, 2, True)
            if bs <= 32: return (8, True, 1, False)
            if bs <= 64: return (8, True, 1, False)
            return (16, True, 1, False)

    def _get_or_build(bs, kvl, qd, kvd, qo, kvi, ns, dev, ps, fm):
        key = (bs, kvl, ns, qd, ps, fm)
        if key in _cache: return _cache[key]
        tkv = bs * kvl
        kl = (kvi[1:] - kvi[:-1]).to(torch.int32)
        ki = torch.arange(tkv, dtype=torch.int32, device=dev)
        info = get_mla_metadata_info_v1(bs, 1, NUM_HEADS, qd, kvd,
                                        is_sparse=False, fast_mode=fm,
                                        num_kv_splits=ns, intra_batch_mode=True)
        w = [torch.empty(s, dtype=t, device=dev) for s, t in info]
        wm, wi, ws, ri, rf, rp = w
        get_mla_metadata_v1(qo, kvi, kl, NUM_HEADS//NUM_KV_HEADS, NUM_KV_HEADS,
                            True, wm, ws, wi, ri, rf, rp, page_size=ps,
                            kv_granularity=max(ps,16), max_seqlen_qo=1,
                            uni_seqlen_qo=1, fast_mode=fm, max_split_per_batch=ns,
                            intra_batch_mode=True, dtype_q=qd, dtype_kv=kvd)
        e = {"meta": {"work_meta_data":wm,"work_indptr":wi,"work_info_set":ws,
                      "reduce_indptr":ri,"reduce_final_map":rf,"reduce_partial_map":rp},
              "kl":kl, "ki":ki,
              "out":torch.empty((bs,NUM_HEADS,V_HEAD_DIM),dtype=torch.bfloat16,device=dev)}
        _cache[key] = e
        return e

    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"]); kvl = int(config["kv_seq_len"])
    ns, use_a8w8, ps, fm = _get_config(bs, kvl)
    kv_fp8, kv_scale = kv_data["fp8"]
    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])

    if use_a8w8:
        bkey = ("dq", q.numel())
        if bkey not in _cache:
            _cache[bkey] = (torch.empty_like(q, dtype=FP8_DTYPE),
                            torch.empty(1, dtype=torch.float32, device=q.device))
        qi, qs = _cache[bkey]
        dynamic_per_tensor_quant(qi, q, qs)
        qv = qi.view(-1, NUM_HEADS, QK_HEAD_DIM)
    else:
        qv = q.view(-1, NUM_HEADS, QK_HEAD_DIM); qs = None

    c = _get_or_build(bs, kvl, qv.dtype, kv_fp8.dtype, qo_indptr, kv_indptr,
                      ns, q.device, ps, fm)

    mla_decode_fwd(qv, kv_4d, c["out"], qo_indptr, kv_indptr, c["ki"], c["kl"],
                   1, page_size=ps, nhead_kv=NUM_KV_HEADS, sm_scale=SM_SCALE,
                   logit_cap=0.0, num_kv_splits=ns, q_scale=qs, kv_scale=kv_scale,
                   intra_batch_mode=True, **c["meta"])
    return c["out"]
