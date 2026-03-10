#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
# AGENT_LOOP_META: {"attempt": 6, "gpu": "MI355X", "leaderboard": "amd-mxfp4-mm", "problem": "mxfp4_mm", "variant": {"family": "anchor", "strategy": "contract_anchor", "variant_name": "aiter_contract_anchor"}, "variant_index": 0}
import aiter
from aiter import QuantType, dtypes
from task import input_t, output_t


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
