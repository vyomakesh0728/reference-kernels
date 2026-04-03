#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
RESEARCH: Pre-compiled .co file loading via hipModuleLoad

===============================================================================
FINDINGS SUMMARY
===============================================================================

1. SUBMISSION FORMAT CONSTRAINT
   - Popcorn only accepts a single submission.py file
   - No mechanism to ship additional binary .co files
   - Task.yml shows: files: [{"name": "submission.py", "source": "@SUBMISSION@"}]

2. AITER'S .co FILES LOCATION
   - On eval machine: /home/runner/aiter/hsa/gfx950/mla/*.co
   - Kernel naming pattern: mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co
   - Contains pre-compiled assembly for gfx950

3. HIP PYTHON MODULE APPROACH
   - hip-python package provides: from hip import hip, hiprtc
   - hipModuleLoad(filename) - loads .co from file path
   - hipModuleLoadData(bytes) - loads from memory
   - hipModuleGetFunction(module, b"kernel_name") - gets kernel handle
   - hipModuleLaunchKernel(...) - launches kernel

4. KERNEL SIGNATURE (from asm_mla_decode_fwd.cpp)
   Arguments:
   - q: void* [num_seqs, num_heads, head_size]
   - kv_buffer: void* [num_page, page_size, num_kv_heads, head_size]
   - qo_indptr: void* [batch_size+1]
   - kv_indptr: void* [batch_size+1]
   - kv_page_indices: void* [num_page_used]
   - kv_last_page_lens: void* [batch_size]
   - max_seqlen_q: int
   - softmax_scale: float
   - logits: void* [batch_size, num_kv_splits, num_heads, v_head_dim]
   - attn_lse: void* [batch_size, num_kv_splits, num_heads, 1]
   - output: void*
   - num_seqs, num_heads, num_kv_heads: int
   - strides: int (various)
   - stream: hipStream_t

5. CHALLENGES
   - Can't ship .co files with submission
   - Need to load aiter's existing .co files from /home/runner/aiter/...
   - Need to match exact kernel function name inside .co
   - Argument packing must match aiter's ABI
   - hip-python might not be installed in eval environment

6. ESTIMATED OVERHEAD REDUCTION
   Current Python dispatch: ~270us (CPU time)
   With hipModuleLoad direct: ~10-20us (one hipLaunchKernel call)
   Potential savings: 250us per inference

7. DEAD END WARNING
   - Even if we can load .co files, we still need metadata setup
   - aiter's metadata functions are Python/C++ hybrid
   - The real overhead is spread across multiple small operations

===============================================================================
"""

import torch
import sys
import os
import ctypes
from task import input_t, output_t

# Try to import hip-python
try:
    from hip import hip, hiprtc

    HIP_PYTHON_AVAILABLE = True
except ImportError:
    HIP_PYTHON_AVAILABLE = False
    print("WARNING: hip-python not available, falling back to aiter", file=sys.stderr)

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)


# Helper for hip-python error checking
def hip_check(call_result):
    if not HIP_PYTHON_AVAILABLE:
        return None
    err = call_result[0]
    result = call_result[1:] if len(call_result) > 1 else None
    if len(call_result) == 2:
        result = call_result[1]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(f"HIP error: {err}")
    return result


def find_aiter_co_files():
    """Find aiter's precompiled .co files on the eval machine."""
    possible_paths = [
        "/home/runner/aiter/hsa/gfx950/mla",
        "/opt/rocm/aiter/hsa/gfx950/mla",
        os.path.expanduser("~/aiter/hsa/gfx950/mla"),
    ]

    for path in possible_paths:
        if os.path.isdir(path):
            return path
    return None


def get_kernel_path(bs, kvl, use_fp8_q=True, page_size=1):
    """Get the appropriate .co kernel file for the given config."""
    base_path = find_aiter_co_files()
    if not base_path:
        return None

    # Pattern: mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co for fp8
    # Pattern: mla_a16w8_qh16_... for bf16 Q

    if use_fp8_q:
        kernel_name = "mla_a8w8_qh16_qseqlen1_gqaratio16"
    else:
        kernel_name = "mla_a16w8_qh16_m16x4_n16x1_coex0_mask1"

    if page_size > 1:
        kernel_name += f"_ps{page_size}"
    else:
        kernel_name += "_ps"

    kernel_path = os.path.join(base_path, f"{kernel_name}.co")
    if os.path.exists(kernel_path):
        return kernel_path
    return None


def load_co_module(kernel_path):
    """Load a .co file as a HIP module."""
    if not HIP_PYTHON_AVAILABLE:
        return None, None

    # Read the .co file
    with open(kernel_path, "rb") as f:
        code = f.read()

    # Load as module
    module = hip_check(hip.hipModuleLoadData(bytearray(code)))

    # Get the kernel function (need to know exact name)
    # The function name is typically the same as the file name without .co
    func_name = os.path.basename(kernel_path).replace(".co", "")

    try:
        kernel = hip_check(hip.hipModuleGetFunction(module, func_name.encode()))
        return module, kernel
    except:
        # Try common function name patterns
        for suffix in ["", "_fwd", "_kernel"]:
            try:
                kernel = hip_check(
                    hip.hipModuleGetFunction(module, (func_name + suffix).encode())
                )
                return module, kernel
            except:
                pass

    return module, None


def launch_mla_kernel_direct(
    kernel,
    q,
    kv,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_lens,
    max_seqlen_q,
    sm_scale,
    logits,
    attn_lse,
    output,
    num_seqs,
    num_heads,
    num_kv_heads,
    stream=None,
):
    """Launch MLA kernel directly via hipModuleLaunchKernel."""
    if not HIP_PYTHON_AVAILABLE or kernel is None:
        return False

    # Build argument list
    # Warning: This is a best-guess at the argument layout
    # The actual layout depends on aiter's ABI
    args = (
        q.data_ptr(),  # void* q
        kv.data_ptr(),  # void* kv_buffer
        qo_indptr.data_ptr(),  # void* qo_indptr
        kv_indptr.data_ptr(),  # void* kv_indptr
        kv_indices.data_ptr(),  # void* kv_page_indices
        kv_lens.data_ptr(),  # void* kv_last_page_lens
        ctypes.c_int(max_seqlen_q),
        ctypes.c_float(sm_scale),
        logits.data_ptr(),
        attn_lse.data_ptr(),
        output.data_ptr(),
        ctypes.c_int(num_seqs),
        ctypes.c_int(num_heads),
        ctypes.c_int(num_kv_heads),
        # Strides...
        ctypes.c_int(q.stride(0)),
        ctypes.c_int(kv.stride(0) if len(kv.shape) >= 1 else 1),
        ctypes.c_int(attn_lse.stride(0)),
        ctypes.c_int(attn_lse.stride(1) if len(attn_lse.shape) > 1 else 1),
        ctypes.c_int(attn_lse.stride(2) if len(attn_lse.shape) > 2 else 1),
        ctypes.c_int(output.stride(0)),
        ctypes.c_int(output.stride(1) if len(output.shape) > 1 else 1),
    )

    # Grid and block dimensions (from aiter analysis)
    # Persistent kernels use CU-count as grid size
    grid = hip.dim3(256, 1, 1)  # 256 CUs on MI355X
    block = hip.dim3(256, 1, 1)  # 4 waves per CU

    # Launch!
    hip_check(
        hip.hipModuleLaunchKernel(
            kernel,
            *grid,
            *block,
            sharedMemBytes=32768,  # 32KB LDS
            stream=stream,
            kernelParams=None,
            extra=args,
        )
    )
    return True


# ============================================================================
# FALLBACK: Use standard aiter (this is what we use when hip-python isn't available)
# ============================================================================

from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd
from aiter.ops.quant import dynamic_per_tensor_quant

FP8_DTYPE = aiter_dtypes.fp8
_cache = {}


def _get_config(bs, kvl):
    if kvl <= 1024:
        if bs <= 32:
            return (8, False, 2, True)
        if bs <= 64:
            return (4, False, 2, True)
        return (4, False, 2, True)
    return (32, True, 1, False)


def _get_or_build(bs, kvl, qd, kvd, qo, kvi, ns, dev, ps, fm):
    key = (bs, kvl, ns, qd, ps, fm)
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

    # Probe available methods
    if not _cache.get("_probed"):
        _cache["_probed"] = True
        print(f"hip-python available: {HIP_PYTHON_AVAILABLE}", file=sys.stderr)
        co_path = find_aiter_co_files()
        print(f"aiter .co path: {co_path}", file=sys.stderr)
        if co_path:
            import glob

            co_files = glob.glob(os.path.join(co_path, "*.co"))[:5]
            print(f"Sample .co files: {co_files}", file=sys.stderr)

    # Standard aiter path
    ns, use_a8w8, ps, fm = _get_config(bs, kvl)
    kv_fp8, kv_scale = kv_data["fp8"]
    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])

    if use_a8w8:
        bkey = ("dq", q.numel())
        if bkey not in _cache:
            _cache[bkey] = (
                torch.empty_like(q, dtype=FP8_DTYPE),
                torch.empty(1, dtype=torch.float32, device=q.device),
            )
        qi, qs = _cache[bkey]
        dynamic_per_tensor_quant(qi, q, qs)
        qv = qi.view(-1, NUM_HEADS, QK_HEAD_DIM)
    else:
        qv = q.view(-1, NUM_HEADS, QK_HEAD_DIM)
        qs = None

    c = _get_or_build(
        bs, kvl, qv.dtype, kv_fp8.dtype, qo_indptr, kv_indptr, ns, q.device, ps, fm
    )
    mla_decode_fwd(
        qv,
        kv_4d,
        c["out"],
        qo_indptr,
        kv_indptr,
        c["ki"],
        c["kl"],
        1,
        page_size=ps,
        nhead_kv=NUM_KV_HEADS,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=ns,
        q_scale=qs,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **c["meta"],
    )
    return c["out"]
