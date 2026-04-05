#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""HIP MLA decode with split-K, bf16 KV, vectorized loads.

Architecture:
- Grid: (bs * num_splits, 16 heads)
- Block: 64 threads (1 wavefront)
- Each block: 1 (batch, head, split) chunk
- QK: vectorized dot product with warp reduce
- V: parallel accumulation (8 dims per thread)
- Reduce: log-sum-exp across splits

Hybrid: HIP for small/medium shapes, aiter fp8 for large (where BW matters).
"""

import os

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ.setdefault("CXX", "clang++")

import hashlib
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

_cache = {}
_MODULE = None

CPP_WRAPPER = """
void mla_decode_splitk(
    torch::Tensor q, torch::Tensor kv, torch::Tensor split_out,
    torch::Tensor split_lse, torch::Tensor kv_indptr,
    int batch_size, int num_splits, float sm_scale
);
void mla_reduce(
    torch::Tensor split_out, torch::Tensor split_lse,
    torch::Tensor out, int batch_size, int num_splits
);
"""

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>

constexpr int QK_DIM = 576;
constexpr int V_DIM = 512;
constexpr int N_HEADS = 16;
constexpr int WARP_SIZE = 64;
constexpr int DIMS_PER_THREAD_QK = (QK_DIM + WARP_SIZE - 1) / WARP_SIZE;  // 9
constexpr int DIMS_PER_THREAD_V = (V_DIM + WARP_SIZE - 1) / WARP_SIZE;    // 8

__launch_bounds__(64)
__global__ void mla_decode_splitk_kernel(
    const __hip_bfloat16* __restrict__ q,
    const __hip_bfloat16* __restrict__ kv,
    float* __restrict__ split_out,
    float* __restrict__ split_lse,
    const int* __restrict__ kv_indptr,
    int num_splits,
    float sm_scale
) {
    const int linear_bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int bid = linear_bid / num_splits;
    const int sid = linear_bid % num_splits;
    const int tid = threadIdx.x;

    const int kv_start = kv_indptr[bid];
    const int kv_end = kv_indptr[bid + 1];
    const int kv_len = kv_end - kv_start;

    const int split_size = (kv_len + num_splits - 1) / num_splits;
    const int s_start = sid * split_size;
    const int s_end = min(s_start + split_size, kv_len);

    const int split_out_offset = ((bid * N_HEADS + hid) * num_splits + sid) * V_DIM;
    const int split_lse_offset = (bid * N_HEADS + hid) * num_splits + sid;

    if (s_start >= kv_len) {
        for (int i = 0; i < DIMS_PER_THREAD_V; i++) {
            int d = tid + i * WARP_SIZE;
            if (d < V_DIM) split_out[split_out_offset + d] = 0.0f;
        }
        if (tid == 0) split_lse[split_lse_offset] = -1e30f;
        return;
    }

    // Load Q into registers
    const __hip_bfloat16* q_ptr = q + (bid * N_HEADS + hid) * QK_DIM;
    float q_reg[DIMS_PER_THREAD_QK];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD_QK; i++) {
        int d = tid + i * WARP_SIZE;
        q_reg[i] = (d < QK_DIM) ? static_cast<float>(q_ptr[d]) : 0.0f;
    }

    // Online softmax + V accumulation
    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float v_acc[DIMS_PER_THREAD_V];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD_V; i++) v_acc[i] = 0.0f;

    for (int t = s_start; t < s_end; t++) {
        const __hip_bfloat16* kv_ptr = kv + (kv_start + t) * QK_DIM;

        // QK dot product - coalesced loads at stride 64
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD_QK; i++) {
            int d = tid + i * WARP_SIZE;
            if (d < QK_DIM) {
                partial += q_reg[i] * static_cast<float>(kv_ptr[d]);
            }
        }

        // Warp reduce
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) {
            partial += __shfl_down(partial, offset, WARP_SIZE);
        }
        float qk = __shfl(partial, 0, WARP_SIZE) * sm_scale;

        // Online softmax update
        float m_new = fmaxf(m_prev, qk);
        float exp_diff = expf(m_prev - m_new);
        float exp_qk = expf(qk - m_new);
        float l_new = l_prev * exp_diff + exp_qk;

        // V accumulation with rescaling
        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD_V; i++) {
            int d = tid + i * WARP_SIZE;
            if (d < V_DIM) {
                float v_val = static_cast<float>(kv_ptr[d]);
                v_acc[i] = v_acc[i] * exp_diff + exp_qk * v_val;
            }
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Store partial output (unnormalized) and LSE
    float inv_l = 1.0f / l_prev;
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD_V; i++) {
        int d = tid + i * WARP_SIZE;
        if (d < V_DIM) {
            split_out[split_out_offset + d] = v_acc[i] * inv_l;
        }
    }

    if (tid == 0) {
        split_lse[split_lse_offset] = m_prev + logf(l_prev);
    }
}

__launch_bounds__(64)
__global__ void mla_reduce_kernel(
    const float* __restrict__ split_out,
    const float* __restrict__ split_lse,
    __hip_bfloat16* __restrict__ out,
    int num_splits
) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int tid = threadIdx.x;

    const int lse_base = (bid * N_HEADS + hid) * num_splits;
    const int out_base = (bid * N_HEADS + hid) * V_DIM;

    // Find max LSE
    float max_lse = -1e30f;
    for (int s = 0; s < num_splits; s++) {
        float lse = split_lse[lse_base + s];
        max_lse = fmaxf(max_lse, lse);
    }

    // Weighted sum across splits
    constexpr int DIMS_PER_THREAD = (V_DIM + WARP_SIZE - 1) / WARP_SIZE;
    float acc[DIMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; i++) acc[i] = 0.0f;
    float sum_w = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        float lse = split_lse[lse_base + s];
        float w = expf(lse - max_lse);
        sum_w += w;

        const float* partial = split_out + ((bid * N_HEADS + hid) * num_splits + s) * V_DIM;
        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD; i++) {
            int d = tid + i * WARP_SIZE;
            if (d < V_DIM) {
                acc[i] += w * partial[d];
            }
        }
    }

    float inv_sum = 1.0f / sum_w;
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; i++) {
        int d = tid + i * WARP_SIZE;
        if (d < V_DIM) {
            out[out_base + d] = static_cast<__hip_bfloat16>(acc[i] * inv_sum);
        }
    }
}

void mla_decode_splitk(
    torch::Tensor q, torch::Tensor kv, torch::Tensor split_out,
    torch::Tensor split_lse, torch::Tensor kv_indptr,
    int batch_size, int num_splits, float sm_scale
) {
    dim3 block(64);
    dim3 grid(batch_size * num_splits, N_HEADS);
    hipLaunchKernelGGL(
        mla_decode_splitk_kernel,
        grid, block, 0, 0,
        reinterpret_cast<const __hip_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __hip_bfloat16*>(kv.data_ptr<at::BFloat16>()),
        split_out.data_ptr<float>(),
        split_lse.data_ptr<float>(),
        kv_indptr.data_ptr<int>(),
        num_splits, sm_scale
    );
}

void mla_reduce(
    torch::Tensor split_out, torch::Tensor split_lse,
    torch::Tensor out, int batch_size, int num_splits
) {
    dim3 block(64);
    dim3 grid(batch_size, N_HEADS);
    hipLaunchKernelGGL(
        mla_reduce_kernel,
        grid, block, 0, 0,
        split_out.data_ptr<float>(),
        split_lse.data_ptr<float>(),
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()),
        num_splits
    );
}
"""

EXPORT_FUNCTIONS = ["mla_decode_splitk", "mla_reduce"]


def _module():
    global _MODULE
    if _MODULE is None:
        build_root = Path(tempfile.gettempdir()) / "mla_hip_splitk"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode()).hexdigest()[:12]
        _MODULE = load_inline(
            name=f"mla_splitk_{digest}",
            cpp_sources=[CPP_WRAPPER],
            cuda_sources=[HIP_SRC],
            functions=EXPORT_FUNCTIONS,
            extra_cuda_cflags=["--offload-arch=gfx950", "-std=c++20", "-O3"],
            build_directory=str(build_root),
            verbose=False,
        )
    return _MODULE


def _get_num_splits(bs, kvl):
    if kvl <= 1024:
        return 1
    if kvl <= 4096:
        return 4
    return 8


def _get_bufs(bs, num_splits, dev):
    key = ("hip_splitk", bs, num_splits)
    if key not in _cache:
        _cache[key] = (
            torch.empty(
                (bs, NUM_HEADS, num_splits, V_HEAD_DIM), dtype=torch.float32, device=dev
            ),
            torch.empty((bs, NUM_HEADS, num_splits), dtype=torch.float32, device=dev),
            torch.empty((bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=dev),
        )
    return _cache[key]


# aiter fallback for large shapes
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8


def _get_or_build_aiter(bs, kvl, qd, kvd, qo, kvi, ns, dev, ps, fm):
    key = ("aiter", bs, kvl, ns, qd, ps, fm)
    if key in _cache:
        return _cache[key]
    tkv = bs * kvl
    kl = (kvi[1:] - kvi[:-1]).to(torch.int32)
    ki = torch.arange(tkv, dtype=torch.int32, device=dev)
    info = get_mla_metadata_info_v1(
        bs,
        1,
        NUM_HEADS,
        qd,
        kvd,
        is_sparse=False,
        fast_mode=fm,
        num_kv_splits=ns,
        intra_batch_mode=True,
    )
    w = [torch.empty(s, dtype=t, device=dev) for s, t in info]
    wm, wi, ws, ri, rf, rp = w
    get_mla_metadata_v1(
        qo,
        kvi,
        kl,
        NUM_HEADS // NUM_KV_HEADS,
        NUM_KV_HEADS,
        True,
        wm,
        ws,
        wi,
        ri,
        rf,
        rp,
        page_size=ps,
        kv_granularity=max(ps, 16),
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=fm,
        max_split_per_batch=ns,
        intra_batch_mode=True,
        dtype_q=qd,
        dtype_kv=kvd,
    )
    e = {
        "meta": {
            "work_meta_data": wm,
            "work_indptr": wi,
            "work_info_set": ws,
            "reduce_indptr": ri,
            "reduce_final_map": rf,
            "reduce_partial_map": rp,
        },
        "kl": kl,
        "ki": ki,
        "out": torch.empty(
            (bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=dev
        ),
    }
    _cache[key] = e
    return e


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    # Use HIP for all shapes
    kv_bf16 = kv_data["bf16"]
    q_reshaped = q.view(bs, NUM_HEADS, QK_HEAD_DIM)
    kv_flat = kv_bf16.view(kv_bf16.shape[0], kv_bf16.shape[-1])

    num_splits = _get_num_splits(bs, kvl)
    split_out, split_lse, out = _get_bufs(bs, num_splits, q.device)

    mod = _module()
    mod.mla_decode_splitk(
        q_reshaped,
        kv_flat,
        split_out,
        split_lse,
        kv_indptr,
        bs,
        num_splits,
        SM_SCALE,
    )

    if num_splits > 1:
        mod.mla_reduce(split_out, split_lse, out, bs, num_splits)
    else:
        # Single split - just convert from f32 to bf16
        out.copy_(split_out.squeeze(2).to(torch.bfloat16))

    return out
