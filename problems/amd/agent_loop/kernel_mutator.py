from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import textwrap


META_RE = re.compile(r"^# AGENT_LOOP_META:\s*(\{.*\})\s*$", re.MULTILINE)

MOE_MOTIVATION_REFS = [
    "/root/reference-kernels/problems/amd/important_papers/fused_moe/README.md",
    "/root/reference-kernels/problems/amd/important_papers/fused_moe/architectural_multipliers.md",
    "/root/reference-kernels/problems/amd/important_papers/fused_moe/links.md",
    "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-cost-center-gate.md",
    "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-branch-queue.md",
    "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-subagent-prompt.md",
    "https://github.com/microsoft/tutel",
    "https://github.com/deepseek-ai/DeepSeek-V3",
    "https://github.com/shawntan/scattermoe",
    "https://github.com/osayamenja/FlashMoE",
    "https://github.com/Dao-AILab/sonic-moe",
]

MOE_RETRIEVAL_PACKS = {
    "dispatch_pack": [
        "q29-fused-moe-padding-free-packing",
        "q32-fused-moe-github-motivation-links",
        "padding-free routed expert packing touched experts",
        "sorted_token_ids sorted_expert_ids num_valid_ids",
    ],
    "stage1_core": [
        "q30-fused-moe-persistent-pipeline",
        "stage1 grouped bf16 gate up swiglu fused expert tile pipeline",
        "ck_moe_stage1 block_m sorted_weights shuffled scale-aware",
    ],
    "stage2_reduce": [
        "q30-fused-moe-persistent-pipeline",
        "stage2 weighted reduction down projection index_add expert outputs",
        "ck_moe_stage2 sorted_token_ids sorted_expert_ids weighted epilogue",
    ],
    "shared_expert": [
        "q31-fused-moe-shared-expert-split",
        "shared expert split dense fast path routed experts",
        "shared experts routed experts separate scheduler",
    ],
    "full_pipeline": [
        "q30-fused-moe-persistent-pipeline",
        "persistent expert tile pipeline stage1 swiglu stage2 overlap io compute",
        "shared metadata resident persistent moe kernel",
        "gfx950 load_inline lds swizzle double buffering",
        "q09-cdna4-gemm-blog",
        "q25-cdna-matrix-core-lane-layout",
        "q27-gemm-tuning-shape-driven",
        "AMDGPU builtin intrinsic mapping",
    ],
}


SEARCH_SPACE: dict[str, list[dict[str, object]]] = {
    "mxfp4_mm": [
        {
            "variant_name": "aiter_contract_anchor",
            "family": "anchor",
            "strategy": "contract_anchor",
        },
        {
            "variant_name": "kernel_requant_m16n256k128",
            "family": "kernel_explore",
            "strategy": "runtime_requant_matmul",
            "BLOCK_M": 16,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_M": 1,
            "NUM_WARPS": 4,
            "NUM_STAGES": 3,
        },
        {
            "variant_name": "kernel_requant_m32n256k128",
            "family": "kernel_explore",
            "strategy": "runtime_requant_matmul",
            "BLOCK_M": 32,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_M": 2,
            "NUM_WARPS": 4,
            "NUM_STAGES": 3,
        },
        {
            "variant_name": "kernel_requant_m64n128k64",
            "family": "kernel_explore",
            "strategy": "runtime_requant_matmul",
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 4,
            "NUM_WARPS": 8,
            "NUM_STAGES": 2,
        },
        {
            "variant_name": "kernel_contract_m16n256k128",
            "family": "kernel_explore",
            "strategy": "contract_bf16_dequant_matmul",
            "CONTRACT_NATIVE": True,
            "BLOCK_M": 16,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_M": 1,
            "NUM_WARPS": 4,
            "NUM_STAGES": 3,
        },
        {
            "variant_name": "kernel_contract_m32n256k128",
            "family": "kernel_explore",
            "strategy": "contract_bf16_dequant_matmul",
            "CONTRACT_NATIVE": True,
            "BLOCK_M": 32,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_M": 2,
            "NUM_WARPS": 4,
            "NUM_STAGES": 3,
        },
        {
            "variant_name": "kernel_contract_m64n128k64",
            "family": "kernel_explore",
            "strategy": "contract_bf16_dequant_matmul",
            "CONTRACT_NATIVE": True,
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 4,
            "NUM_WARPS": 8,
            "NUM_STAGES": 2,
        },
        {
            "variant_name": "kernel_contract_m64n256k128",
            "family": "kernel_explore",
            "strategy": "contract_bf16_dequant_matmul",
            "CONTRACT_NATIVE": True,
            "BLOCK_M": 64,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_M": 4,
            "NUM_WARPS": 8,
            "NUM_STAGES": 3,
        },
        {
            "variant_name": "kernel_contract_m128n128k64",
            "family": "kernel_explore",
            "strategy": "contract_bf16_dequant_matmul",
            "CONTRACT_NATIVE": True,
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 4,
            "NUM_WARPS": 8,
            "NUM_STAGES": 3,
        },
        {
            "variant_name": "hip_reference_oracle_naive",
            "family": "hip_explore",
            "strategy": "hip_reference_oracle",
            "ARCH": "gfx950",
            "REFERENCE_INPUTS": True,
            "NAIVE_KERNEL": True,
            "USE_PROVIDED_BQ": False,
            "A_QUANT_SHUFFLE": False,
            "TILE_M": 16,
            "TILE_N": 16,
            "TILE_K": 32,
            "DOUBLE_BUFFER": False,
            "LDS_SWIZZLE": False,
            "USE_SCALE_MFMA_SEED": False,
        },
        {
            "variant_name": "hip_reference_calibrated_m16n32k64",
            "family": "hip_explore",
            "strategy": "hip_reference_oracle",
            "ARCH": "gfx950",
            "REFERENCE_INPUTS": True,
            "NAIVE_KERNEL": False,
            "TILE_M": 16,
            "TILE_N": 32,
            "TILE_K": 64,
            "DOUBLE_BUFFER": False,
            "LDS_SWIZZLE": False,
            "USE_SCALE_MFMA_SEED": False,
        },
        {
            "variant_name": "hip_shared_bf16_m16n16k32",
            "family": "hip_explore",
            "strategy": "hip_shared_bf16",
            "ARCH": "gfx950",
            "TILE_M": 16,
            "TILE_N": 16,
            "TILE_K": 32,
            "DOUBLE_BUFFER": False,
            "LDS_SWIZZLE": False,
            "USE_SCALE_MFMA_SEED": False,
        },
        {
            "variant_name": "hip_shared_bf16_m16n32k64",
            "family": "hip_explore",
            "strategy": "hip_shared_bf16",
            "ARCH": "gfx950",
            "TILE_M": 16,
            "TILE_N": 32,
            "TILE_K": 64,
            "DOUBLE_BUFFER": True,
            "LDS_SWIZZLE": False,
            "USE_SCALE_MFMA_SEED": False,
        },
        {
            "variant_name": "hip_scale_mfma_seed_16x16x128",
            "family": "hip_explore",
            "strategy": "hip_scale_mfma_seed",
            "ARCH": "gfx950",
            "TILE_M": 16,
            "TILE_N": 16,
            "TILE_K": 128,
            "DOUBLE_BUFFER": True,
            "LDS_SWIZZLE": True,
            "USE_SCALE_MFMA_SEED": True,
            "MFMA_OP": "V_MFMA_SCALE_F32_16X16X128_F8F6F4",
        },
        {
            "variant_name": "hip_scale_mfma_seed_32x32x64",
            "family": "hip_explore",
            "strategy": "hip_scale_mfma_seed",
            "ARCH": "gfx950",
            "TILE_M": 32,
            "TILE_N": 32,
            "TILE_K": 64,
            "DOUBLE_BUFFER": True,
            "LDS_SWIZZLE": True,
            "USE_SCALE_MFMA_SEED": True,
            "MFMA_OP": "V_MFMA_SCALE_F32_32X32X64_F8F6F4",
        },
    ],
    "moe_mxfp4": [
        {
            "variant_name": "fused_moe_contract_anchor",
            "family": "anchor",
            "strategy": "contract_anchor",
            "MOTIVATION_REFS": MOE_MOTIVATION_REFS,
            "RETRIEVAL_PACK": ["q32-fused-moe-github-motivation-links"],
            "DELETED_COST_CENTER": "none; control anchor only",
            "EXPECTED_UPSIDE_SOURCE": "seed the exact live contract before lane-local native work",
            "WHY_LARGER_THAN_NOISE": "this is a control record, not a throughput branch",
            "FORBIDDEN_EDITS": [
                "do not change routing or top-k semantics",
                "do not present side-code as a non-anchor hot path",
            ],
            "SUCCESS_GATE": "test-green anchor control only",
        },
        {
            "variant_name": "anchor_tune_sparse256",
            "family": "anchor",
            "strategy": "anchor_tune",
            "BLOCK_SIZE_M": 16,
            "REGIME_HINT": "re256_de256_bs512_topk8",
            "MOTIVATION_REFS": MOE_MOTIVATION_REFS,
            "RETRIEVAL_PACK": ["q32-fused-moe-github-motivation-links"],
            "DELETED_COST_CENTER": "none; wrapper-only ceiling map for sparse256 anchor",
            "EXPECTED_UPSIDE_SOURCE": "small launch/padding alignment gain from the sparse256 block_size_M frontier",
            "WHY_LARGER_THAN_NOISE": "the sparse256 anchor already showed a repeatable but small block_size_M sensitivity",
            "FORBIDDEN_EDITS": [
                "do not claim native lane ownership",
                "do not change more than one anchor heuristic",
                "do not spend main budget here after ceiling mapping",
            ],
            "SUCCESS_GATE": "two agreeing reruns and a stable <175 us ceiling or stop investing",
        },
        {
            "variant_name": "dispatch_pack_sparse256",
            "family": "kernel_explore",
            "strategy": "dispatch_pack",
            "LANE": "dispatch_pack",
            "HOT_PATH_STATE": "partial-native",
            "REGIME_HINT": "re256_de256_bs512_topk8",
            "BLOCK_SIZE": 256,
            "NUM_WARPS": 4,
            "SORT_BY_EXPERT": True,
            "PREFER_TOUCHED_EXPERTS": True,
            "MOTIVATION_REFS": MOE_MOTIVATION_REFS,
            "RETRIEVAL_PACK": MOE_RETRIEVAL_PACKS["dispatch_pack"],
            "DELETED_COST_CENTER": "repeated sort/regroup/padding work across routed sparse256 expert stages",
            "EXPECTED_UPSIDE_SOURCE": "ScatterMoE/SonicMoE-style touched-expert packing with compact expert windows",
            "WHY_LARGER_THAN_NOISE": "re256/de256/bs512 is dominated by routing overhead once per-expert token load is sparse",
            "FORBIDDEN_EDITS": [
                "do not change routing semantics or top-k membership",
                "do not rebuild every expert in Python",
                "do not tune stage1 or stage2 math in the same branch",
            ],
            "SUCCESS_GATE": "clear win on re256_de256_bs512_topk8 and global <170 us",
        },
        {
            "variant_name": "dispatch_pack_dense32",
            "family": "kernel_explore",
            "strategy": "dispatch_pack",
            "LANE": "dispatch_pack",
            "HOT_PATH_STATE": "partial-native",
            "REGIME_HINT": "re32_de512_bs128_topk8",
            "BLOCK_SIZE": 128,
            "NUM_WARPS": 4,
            "SORT_BY_EXPERT": True,
            "PREFER_TOUCHED_EXPERTS": True,
            "MOTIVATION_REFS": MOE_MOTIVATION_REFS,
            "RETRIEVAL_PACK": MOE_RETRIEVAL_PACKS["dispatch_pack"],
            "DELETED_COST_CENTER": "generic routed-token regroup and padding overhead on dense32 metadata flow",
            "EXPECTED_UPSIDE_SOURCE": "tile-aware expert-local packing before compute on the denser 32-expert family",
            "WHY_LARGER_THAN_NOISE": "re32/de512 keeps enough expert reuse for compact windows to amortize dispatch overhead",
            "FORBIDDEN_EDITS": [
                "do not fuse stage1 compute yet",
                "do not mix sparse256 and dense32 schedules",
                "do not call fused_moe in the non-anchor hot path",
            ],
            "SUCCESS_GATE": "targeted dense32 dispatch win without >7% regression outside the target regime",
        },
        {
            "variant_name": "stage1_grouped_bf16",
            "family": "kernel_explore",
            "strategy": "stage1_grouped",
            "LANE": "stage1_core",
            "HOT_PATH_STATE": "partial-native",
            "REGIME_HINT": "re32_de512_bs512_topk8",
            "BLOCK_SIZE": 256,
            "NUM_WARPS": 4,
            "SORT_BY_EXPERT": True,
            "PREFER_TOUCHED_EXPERTS": True,
            "FUSE_SWIGLU": False,
            "WEIGHT_EPILOGUE": False,
            "SHARED_EXPERT_FASTPATH": False,
            "MOTIVATION_REFS": MOE_MOTIVATION_REFS,
            "RETRIEVAL_PACK": MOE_RETRIEVAL_PACKS["stage1_core"],
            "DELETED_COST_CENTER": "generic stage1 launch plus repeated metadata walking for touched experts",
            "EXPECTED_UPSIDE_SOURCE": "native grouped hidden->2*d_expert stage1 on touched experts only",
            "WHY_LARGER_THAN_NOISE": "once dispatch is compact, stage1 becomes the dominant routed-expert bucket on re32/de512",
            "FORBIDDEN_EDITS": [
                "do not dequantize all experts eagerly",
                "do not fuse stage2 in the same branch",
                "do not reopen dispatch semantics here",
            ],
            "SUCCESS_GATE": "native stage1 path beats the control on target dense32 regimes and moves global score below 150 us",
        },
        {
            "variant_name": "stage1_swiglu_fused",
            "family": "kernel_explore",
            "strategy": "stage1_swiglu_fused",
            "LANE": "stage1_core",
            "HOT_PATH_STATE": "partial-native",
            "REGIME_HINT": "re32_de512_bs512_topk8",
            "BLOCK_SIZE": 256,
            "NUM_WARPS": 8,
            "SORT_BY_EXPERT": True,
            "PREFER_TOUCHED_EXPERTS": True,
            "FUSE_SWIGLU": True,
            "WEIGHT_EPILOGUE": False,
            "SHARED_EXPERT_FASTPATH": False,
            "MOTIVATION_REFS": MOE_MOTIVATION_REFS,
            "RETRIEVAL_PACK": MOE_RETRIEVAL_PACKS["stage1_core"],
            "DELETED_COST_CENTER": "stage1 output materialization and round-trip buffer traffic before SwiGLU",
            "EXPECTED_UPSIDE_SOURCE": "fused stage1->SwiGLU epilogue inside routed expert tiles",
            "WHY_LARGER_THAN_NOISE": "re32/de512 stage1 emits enough routed activation traffic that epilogue fusion should survive benchmark noise",
            "FORBIDDEN_EDITS": [
                "do not change stage2 ownership",
                "do not bring in shared-expert logic",
                "do not mix this with low-level gfx950 tuning",
            ],
            "SUCCESS_GATE": "measurable target-regime win over stage1_grouped_bf16 without regressing non-target cases by >7%",
        },
        {
            "variant_name": "stage2_grouped_weighted",
            "family": "kernel_explore",
            "strategy": "stage2_grouped",
            "LANE": "stage2_reduce",
            "HOT_PATH_STATE": "partial-native",
            "REGIME_HINT": "re32_de2048_bs512_topk8",
            "BLOCK_SIZE": 256,
            "NUM_WARPS": 8,
            "SORT_BY_EXPERT": True,
            "PREFER_TOUCHED_EXPERTS": True,
            "FUSE_SWIGLU": True,
            "WEIGHT_EPILOGUE": True,
            "SHARED_EXPERT_FASTPATH": False,
            "MOTIVATION_REFS": MOE_MOTIVATION_REFS,
            "RETRIEVAL_PACK": MOE_RETRIEVAL_PACKS["stage2_reduce"],
            "DELETED_COST_CENTER": "separate stage2 reduction and post-matmul top-k weighted combine work",
            "EXPECTED_UPSIDE_SOURCE": "native grouped d_expert->d_hidden with weighted epilogue and direct combine",
            "WHY_LARGER_THAN_NOISE": "the heavy stage2/down-proj bucket dominates large-routed-output cases once dispatch and stage1 are local",
            "FORBIDDEN_EDITS": [
                "do not reopen dispatch packing in the same branch",
                "do not force shared experts through the routed path",
                "do not add full-pipeline persistence yet",
            ],
            "SUCCESS_GATE": "clear win on re32_de2048_bs512_topk8 and end-to-end <135 us",
        },
        {
            "variant_name": "shared_expert_split",
            "family": "kernel_explore",
            "strategy": "shared_expert_split",
            "LANE": "shared_expert",
            "HOT_PATH_STATE": "partial-native",
            "REGIME_HINT": "re32_de2048_bs512_topk8",
            "BLOCK_SIZE": 256,
            "NUM_WARPS": 8,
            "SORT_BY_EXPERT": True,
            "PREFER_TOUCHED_EXPERTS": True,
            "FUSE_SWIGLU": True,
            "WEIGHT_EPILOGUE": True,
            "SHARED_EXPERT_FASTPATH": True,
            "MOTIVATION_REFS": MOE_MOTIVATION_REFS,
            "RETRIEVAL_PACK": MOE_RETRIEVAL_PACKS["shared_expert"],
            "DELETED_COST_CENTER": "forcing shared experts and routed experts through one generic scheduler",
            "EXPECTED_UPSIDE_SOURCE": "DeepSeek-style shared-expert separation with a dense fast path alongside routed execution",
            "WHY_LARGER_THAN_NOISE": "shared-expert and routed-expert reuse patterns are structurally different and regress each other when fused too early",
            "FORBIDDEN_EDITS": [
                "do not change routed stage math and shared scheduling in the same branch",
                "do not rebuild all routed experts in Python",
                "do not claim full-pipeline ownership yet",
            ],
            "SUCCESS_GATE": "reduce shared-expert heavy regressions while keeping routed regimes flat or better",
        },
        {
            "variant_name": "hip_fp4_preshuffled_sparse256",
            "family": "hip_explore",
            "strategy": "hip_fp4_preshuffled",
            "LANE": "full_pipeline",
            "HOT_PATH_STATE": "native",
            "REGIME_HINT": "re256_de256_bs512_topk8",
            "BLOCK_SIZE": 256,
            "NUM_WARPS": 8,
            "SORT_BY_EXPERT": True,
            "PREFER_TOUCHED_EXPERTS": True,
            "FUSE_SWIGLU": True,
            "WEIGHT_EPILOGUE": True,
            "SHARED_EXPERT_FASTPATH": False,
            "TILE_M": 16,
            "TILE_N": 128,
            "TILE_K": 64,
            "MOTIVATION_REFS": MOE_MOTIVATION_REFS,
            "RETRIEVAL_PACK": MOE_RETRIEVAL_PACKS["full_pipeline"],
            "DELETED_COST_CENTER": "Python/Triton stage boundaries on the sparse256 routed path after structural packing is proven",
            "EXPECTED_UPSIDE_SOURCE": "gfx950 load_inline path that consumes preshuffled FP4 data without reopening contract work",
            "WHY_LARGER_THAN_NOISE": "once sparse256 ownership is native and compact, the remaining gap should be dominated by compiler-boundary and memory-hierarchy overhead",
            "FORBIDDEN_EDITS": [
                "do not keep fused_moe in the hot path",
                "do not retune routing semantics",
                "do not share one schedule across sparse256 and dense32",
            ],
            "SUCCESS_GATE": "full sparse256 HIP path beats the control and is worth deeper gfx950 tuning",
        },
        {
            "variant_name": "hip_fp4_preshuffled_dense32",
            "family": "hip_explore",
            "strategy": "hip_fp4_preshuffled",
            "LANE": "full_pipeline",
            "HOT_PATH_STATE": "native",
            "REGIME_HINT": "re32_de512_bs128_topk8",
            "BLOCK_SIZE": 128,
            "NUM_WARPS": 8,
            "SORT_BY_EXPERT": True,
            "PREFER_TOUCHED_EXPERTS": True,
            "FUSE_SWIGLU": True,
            "WEIGHT_EPILOGUE": True,
            "SHARED_EXPERT_FASTPATH": False,
            "TILE_M": 32,
            "TILE_N": 128,
            "TILE_K": 64,
            "MOTIVATION_REFS": MOE_MOTIVATION_REFS,
            "RETRIEVAL_PACK": MOE_RETRIEVAL_PACKS["full_pipeline"],
            "DELETED_COST_CENTER": "Python/Triton stage boundaries on the dense32 routed path after native stage ownership is validated",
            "EXPECTED_UPSIDE_SOURCE": "gfx950 load_inline grouped path specialized for dense32 expert reuse",
            "WHY_LARGER_THAN_NOISE": "dense32 should expose enough reuse for a regime-specific HIP path to outrun the mixed generic kernel",
            "FORBIDDEN_EDITS": [
                "do not reuse sparse256 tile assumptions unchanged",
                "do not keep shared experts on the routed schedule",
                "do not spend budget on MFMA before the HIP path is benchmark-positive",
            ],
            "SUCCESS_GATE": "dense32 HIP path wins on its target regimes and supports promotion toward full_pipeline",
        },
        {
            "variant_name": "hip_persistent_sparse256",
            "family": "hip_explore",
            "strategy": "hip_persistent_sparse",
            "LANE": "full_pipeline",
            "HOT_PATH_STATE": "native",
            "REGIME_HINT": "re256_de256_bs512_topk8",
            "BLOCK_SIZE": 256,
            "NUM_WARPS": 8,
            "SORT_BY_EXPERT": True,
            "PREFER_TOUCHED_EXPERTS": True,
            "FUSE_SWIGLU": True,
            "WEIGHT_EPILOGUE": True,
            "SHARED_EXPERT_FASTPATH": True,
            "TILE_M": 32,
            "TILE_N": 256,
            "TILE_K": 64,
            "MOTIVATION_REFS": MOE_MOTIVATION_REFS,
            "RETRIEVAL_PACK": MOE_RETRIEVAL_PACKS["full_pipeline"],
            "DELETED_COST_CENTER": "relaunches, repeated metadata loads, and non-resident stage state on the sparse256 full pipeline",
            "EXPECTED_UPSIDE_SOURCE": "persistent expert-tile pipeline owning dispatch->stage1->SwiGLU->stage2->combine",
            "WHY_LARGER_THAN_NOISE": "this deletes multiple launch and residency costs after the sparse256 structural path is already benchmark-positive",
            "FORBIDDEN_EDITS": [
                "do not open multiple regime families in one branch",
                "do not introduce scaled-MFMA before the persistent schedule is winning",
                "do not fallback to anchor-backed wrappers",
            ],
            "SUCCESS_GATE": "stable <120 us before deeper gfx950 inner-loop specialization",
        },
    ],
    "mixed_mla": [
        {
            "variant_name": "aiter_fp8_anchor",
            "family": "anchor",
            "strategy": "contract_anchor",
        },
        {
            "variant_name": "fp8_decode_b32_v64",
            "family": "kernel_explore",
            "strategy": "fp8_decode",
            "USE_FP8_INPUTS": True,
            "BLOCK_N": 32,
            "BLOCK_DQ": 64,
            "BLOCK_DV": 64,
            "NUM_WARPS": 4,
            "NUM_STAGES": 2,
        },
        {
            "variant_name": "fp8_decode_b64_v64",
            "family": "kernel_explore",
            "strategy": "fp8_decode",
            "USE_FP8_INPUTS": True,
            "BLOCK_N": 64,
            "BLOCK_DQ": 64,
            "BLOCK_DV": 64,
            "NUM_WARPS": 4,
            "NUM_STAGES": 2,
        },
        {
            "variant_name": "fp8_decode_b128_v64",
            "family": "kernel_explore",
            "strategy": "fp8_decode",
            "USE_FP8_INPUTS": True,
            "BLOCK_N": 128,
            "BLOCK_DQ": 64,
            "BLOCK_DV": 64,
            "NUM_WARPS": 4,
            "NUM_STAGES": 2,
        },
        {
            "variant_name": "fp8_decode_b128_v128",
            "family": "kernel_explore",
            "strategy": "fp8_decode",
            "USE_FP8_INPUTS": True,
            "BLOCK_N": 128,
            "BLOCK_DQ": 64,
            "BLOCK_DV": 128,
            "NUM_WARPS": 8,
            "NUM_STAGES": 2,
        },
    ],
}


MOE_MOTIVATION_REFS = [
    "/root/reference-kernels/problems/amd/important_papers/fused_moe/README.md",
    "/root/reference-kernels/problems/amd/important_papers/fused_moe/architectural_multipliers.md",
    "/root/reference-kernels/problems/amd/important_papers/fused_moe/links.md",
    "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-cost-center-gate.md",
    "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-branch-queue.md",
    "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-subagent-prompt.md",
    "https://github.com/microsoft/tutel",
    "https://github.com/deepseek-ai/DeepSeek-V3",
    "https://github.com/shawntan/scattermoe",
    "https://github.com/osayamenja/FlashMoE",
    "https://github.com/Dao-AILab/sonic-moe",
]

MOE_SUBAGENT_ROSTER = [
    {
        "role": "structure_planner",
        "required_reads": [
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/SKILL.md",
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/fused-moe-multiplier.md",
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-cost-center-gate.md",
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-branch-queue.md",
        ],
        "deliverable": "one Candidate Card only",
    },
    {
        "role": "retrieval_canon_scout",
        "required_reads": [
            "/root/reference-kernels/problems/amd/skills/amd-kernel-speedrun/SKILL.md",
            "/root/reference-kernels/problems/amd/skills/amd-kernel-speedrun/references/moe-closed-loop.md",
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/remote-first-eval.md",
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-branch-queue.md",
        ],
        "deliverable": "3-6 retrieval hits plus a veto if the idea is wrapper-only",
    },
    {
        "role": "bounded_kernel_worker",
        "required_reads": [
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/SKILL.md",
            "/root/reference-kernels/problems/amd/skills/amd-kernel-speedrun/SKILL.md",
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-cost-center-gate.md",
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-subagent-prompt.md",
        ],
        "deliverable": "one bounded patch plan or one bounded seed rewrite only",
    },
]

MXFP4_MOTIVATION_REFS = [
    "/root/reference-kernels/problems/amd/important_papers/amd-instinct-cdna4-instruction-set-architecture.pdf",
    "/root/reference-kernels/problems/amd/important_papers/HipKittens Fast and Furious AMD Kernels.pdf",
    "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/optimization.md",
    "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/amd-blog-insights.md",
    "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-cost-center-gate.md",
    "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-exact-shape-frontier.md",
    "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-profile-branch-queue.md",
    "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-subagent-prompt.md",
]

MXFP4_SUBAGENT_ROSTER = [
    {
        "role": "cost_center_scout",
        "required_reads": [
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/SKILL.md",
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/amd-blog-insights.md",
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-cost-center-gate.md",
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-exact-shape-frontier.md",
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-profile-branch-queue.md",
        ],
        "deliverable": "one Candidate Card only",
    },
    {
        "role": "retrieval_canon_scout",
        "required_reads": [
            "/root/reference-kernels/problems/amd/skills/optimization-skill/SKILL.md",
            "/root/reference-kernels/problems/amd/skills/amd-live-reference-correctness/references/mxfp4-mm-live-contract.md",
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/remote-first-eval.md",
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-profile-branch-queue.md",
        ],
        "deliverable": "3-6 grounded hits plus one veto if the idea is wrapper-only, prep-only, or reopens a banned lane",
    },
    {
        "role": "bounded_kernel_worker",
        "required_reads": [
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/SKILL.md",
            "/root/reference-kernels/problems/amd/skills/optimization-skill/SKILL.md",
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-cost-center-gate.md",
            "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-subagent-prompt.md",
        ],
        "deliverable": "one bounded patch plan or one bounded seed rewrite only",
    },
]

MXFP4_RETRIEVAL_PACK = {
    "name": "exact_shape",
    "queries": [
        "V_MFMA_SCALE_F32_32X32X64_F8F6F4 operand order lane layout",
        "pack_scale_e8m0x2_lane_from_shuffled exact_m32",
        "mxfp4_load_unshuffled_b_scale source_linear mapping",
        "exact_m32 raw shuffled scale direct entry public k512",
        "fewer hot-loop address calculations wide scaled fp4",
        "q09-cdna4-gemm-blog",
        "q25-cdna-matrix-core-lane-layout",
        "q27-gemm-tuning-shape-driven",
    ],
}

MOE_RETRIEVAL_PACKS = {
    "dispatch_pack": {
        "name": "dispatch",
        "queries": [
            "q29-fused-moe-padding-free-packing",
            "q32-fused-moe-github-motivation-links",
            "padding-free routed expert packing touched experts",
            "sorted_token_ids sorted_expert_ids num_valid_ids",
        ],
    },
    "stage1_core": {
        "name": "stage1",
        "queries": [
            "q30-fused-moe-persistent-pipeline",
            "stage1 grouped bf16 gate up swiglu fused expert tile pipeline",
            "ck_moe_stage1 block_m sorted_weights shuffled scale-aware",
        ],
    },
    "stage2_reduce": {
        "name": "stage2",
        "queries": [
            "q30-fused-moe-persistent-pipeline",
            "stage2 weighted reduction down projection index_add expert outputs",
            "ck_moe_stage2 sorted_token_ids sorted_expert_ids weighted epilogue",
        ],
    },
    "shared_expert": {
        "name": "shared_expert",
        "queries": [
            "q31-fused-moe-shared-expert-split",
            "shared expert split routed expert scheduler",
            "deepseek shared expert routed expert separate path",
        ],
    },
    "full_pipeline": {
        "name": "persistence",
        "queries": [
            "q30-fused-moe-persistent-pipeline",
            "persistent expert tile pipeline stage1 swiglu stage2 overlap io compute",
            "shared metadata resident persistent moe kernel",
            "q09-cdna4-gemm-blog",
            "q25-cdna-matrix-core-lane-layout",
            "q27-gemm-tuning-shape-driven",
            "gfx950 load_inline lds swizzle double buffering",
            "AMDGPU builtin intrinsic mapping",
        ],
    },
}

MOE_CANDIDATE_CARD_TEMPLATES = {
    "fused_moe_contract_anchor": {
        "deleted_cost_center": "none; control anchor only",
        "expected_upside_source": "control path for MoE lane replacement, not a structural multiplier",
        "why_larger_than_noise": "anchor reruns define the noise floor and stop us from mistaking wrapper variance for progress",
        "forbidden_edits": [
            "claiming non-anchor progress",
            "changing routing semantics",
        ],
        "success_gate": "two reruns that agree within 1.0%",
    },
    "anchor_tune_sparse256": {
        "deleted_cost_center": "none; control-only AITER ceiling map for sparse256",
        "expected_upside_source": "small wrapper-level block-size reduction in the sparse256 control regime",
        "why_larger_than_noise": "the current control already shows a small block_size_M response on re256_de256_bs512_topk8",
        "forbidden_edits": [
            "claiming native progress",
            "mixing structural kernel rewrites into anchor tuning",
        ],
        "success_gate": "stable <175 us ceiling or stop spending budget on the anchor",
    },
    "dispatch_pack_sparse256": {
        "deleted_cost_center": "repeated routed-token sort/regroup/padding overhead before expert compute",
        "expected_upside_source": "ScatterMoE-style touched-expert packing plus SonicMoE tile-aware sparse routing",
        "why_larger_than_noise": "the sparse256 bs512 regime is where padding and regroup overhead are largest and the current anchor already reacts to dispatch granularity",
        "forbidden_edits": [
            "calling fused_moe in the hot path",
            "changing more than the dispatch_pack lane",
            "rebuilding all experts in Python",
        ],
        "success_gate": "clear re256_de256_bs512_topk8 win and global <170 us",
    },
    "dispatch_pack_dense32": {
        "deleted_cost_center": "generic routed-token regroup overhead in the denser 32-expert family",
        "expected_upside_source": "packed expert windows reused across stage boundaries instead of repeating dense32 regroup work",
        "why_larger_than_noise": "dense32 still pays repeated sorting and offset rebuilds, but the benefit is smaller than sparse256 so the branch must stay tightly scoped",
        "forbidden_edits": [
            "calling fused_moe in the hot path",
            "changing more than the dispatch_pack lane",
            "adding stage1/stage2 math rewrites in the same branch",
        ],
        "success_gate": "target-regime win on re32_de512_bs128_topk8 without broad regressions",
    },
    "stage1_grouped_bf16": {
        "deleted_cost_center": "anchor-backed gate/up stage launches and repeated expert metadata walking after dispatch",
        "expected_upside_source": "touched-expert-only grouped stage1 compute with stable bf16 math before deeper HIP tuning",
        "why_larger_than_noise": "stage1 dominates the medium-width routed path once dispatch packing is in place, so owning it should move more than wrapper noise",
        "forbidden_edits": [
            "changing dispatch semantics",
            "adding stage2 ownership in the same branch",
            "all-expert eager dequant in Python",
        ],
        "success_gate": "native stage1 path plus fused dispatch beats the control on re32_de512_bs512_topk8 and global <150 us",
    },
    "stage1_swiglu_fused": {
        "deleted_cost_center": "stage1 intermediate activation round-trip between gate/up and SwiGLU",
        "expected_upside_source": "fused stage1 epilogue that keeps gate/up outputs in the routed expert tile flow",
        "why_larger_than_noise": "SwiGLU buffer traffic is repeated for every touched expert tile, so deleting that boundary should survive reruns if stage1 ownership is real",
        "forbidden_edits": [
            "reopening dispatch changes",
            "adding stage2 ownership in the same branch",
            "keeping the hot path on fused_moe",
        ],
        "success_gate": "improve the stage1_core branch on re32_de512_bs512_topk8 without >7% regressions elsewhere",
    },
    "stage2_grouped_weighted": {
        "deleted_cost_center": "separate stage2 reduction plus weighted combine/index_add overhead",
        "expected_upside_source": "native grouped stage2 with weighted epilogue in the heavy re32_de2048 path",
        "why_larger_than_noise": "the large-expert regime is stage2-heavy enough that weighted epilogue ownership should move more than a few microseconds",
        "forbidden_edits": [
            "changing dispatch or stage1 in the same branch",
            "rebuilding every expert in Python",
            "keeping weighted reduction outside the owned stage2 path",
        ],
        "success_gate": "clear win on re32_de2048_bs512_topk8 and end-to-end <135 us",
    },
    "shared_expert_split": {
        "deleted_cost_center": "forcing shared and routed experts through one generic schedule",
        "expected_upside_source": "DeepSeek-style shared-expert split plus separate routed scheduling",
        "why_larger_than_noise": "shared experts have different reuse and occupancy behavior than routed experts, so separating them deletes a whole scheduler mismatch",
        "forbidden_edits": [
            "changing routed dispatch semantics",
            "merging shared and routed work back into one generic loop",
            "claiming full-pipeline progress from a shared-only path",
        ],
        "success_gate": "reduce heavy shared-expert regressions while keeping routed regimes flat or better",
    },
    "hip_fp4_preshuffled_sparse256": {
        "deleted_cost_center": "remaining stage boundaries and generic bf16 expert math in the sparse256 routed path",
        "expected_upside_source": "gfx950-native preshuffled FP4 stage ownership after the structural dispatch/stage split is already winning",
        "why_larger_than_noise": "once dispatch and stage ownership are native, remaining time should sit in data movement and math-core feed shape rather than wrapper logic",
        "forbidden_edits": [
            "starting HIP before the structural path is benchmark-positive",
            "reintroducing padded generic MoE structure",
            "calling fused_moe in the hot path",
        ],
        "success_gate": "native sparse256 full pipeline reaches <120 us before scaled-MFMA specialization",
    },
    "hip_fp4_preshuffled_dense32": {
        "deleted_cost_center": "remaining stage boundaries and generic grouped math in the dense32 routed path",
        "expected_upside_source": "gfx950-native dense32 expert tiles fed from the already-packed routed metadata",
        "why_larger_than_noise": "dense32 becomes memory-hierarchy and fragment-feed limited only after dispatch and stage ownership are already real",
        "forbidden_edits": [
            "mixing sparse256 and dense32 schedules in one kernel family",
            "reopening routing semantics",
            "calling fused_moe in the hot path",
        ],
        "success_gate": "native dense32 full pipeline beats the best stage-owned path and stays within the routed-regime guardrails",
    },
    "hip_persistent_sparse256": {
        "deleted_cost_center": "kernel boundaries, relaunches, and non-resident expert tile state across dispatch-stage1-stage2-combine",
        "expected_upside_source": "persistent expert-tile pipeline plus later gfx950 LDS/swizzle/MFMA tuning",
        "why_larger_than_noise": "the remaining gap to the leaderboard target is too large for wrapper or non-persistent cleanup, so only full pipeline residency can plausibly move it",
        "forbidden_edits": [
            "keeping generic padded dispatch structure",
            "mixing persistence work with routing-policy changes",
            "claiming scaled-MFMA progress before persistence is benchmark-positive",
        ],
        "success_gate": "sub-120 us before scaled-MFMA specialization, then chase ~109.793 us",
    },
}


POLICY_PROFILES: dict[str, list[dict[str, object]]] = {
    "mxfp4_mm": [
        {
            "name": "contract_repair",
            "family": "kernel_explore",
            "focus": "preserve shuffled MXFP4 semantics before tuning",
            "preferred_variants": [
                "kernel_requant_m16n256k128",
                "kernel_requant_m32n256k128",
                "kernel_contract_m16n256k128",
                "kernel_contract_m32n256k128",
            ],
            "preferred_strategies": ["runtime_requant_matmul", "contract_bf16_dequant_matmul"],
            "trigger_signals": ["contract_repair", "runtime_repair", "submission_repair"],
        },
        {
            "name": "skinny_longk",
            "family": "kernel_explore",
            "focus": "prioritize skinny-M long-K ranked cases first",
            "preferred_variants": [
                "kernel_requant_m16n256k128",
                "kernel_requant_m32n256k128",
                "kernel_contract_m16n256k128",
                "kernel_contract_m32n256k128",
                "kernel_contract_m64n256k128",
            ],
            "preferred_strategies": ["runtime_requant_matmul", "contract_bf16_dequant_matmul"],
            "trigger_signals": ["throughput_shift"],
        },
        {
            "name": "balanced_tiles",
            "family": "kernel_explore",
            "focus": "cover wider shapes once the skinny path is stable",
            "preferred_variants": [
                "kernel_requant_m64n128k64",
                "kernel_contract_m64n128k64",
                "kernel_contract_m128n128k64",
                "kernel_contract_m64n256k128",
            ],
            "preferred_strategies": ["runtime_requant_matmul", "contract_bf16_dequant_matmul"],
            "trigger_signals": ["throughput_shift", "latency_repair"],
        },
        {
            "name": "hip_contract_reference",
            "family": "hip_explore",
            "focus": "use load_inline on gfx950 with a correctness-first HIP kernel that preserves shuffled MXFP4 A/B contract semantics",
            "preferred_variants": [
                "hip_reference_calibrated_m16n32k64",
                "hip_reference_oracle_naive",
                "hip_shared_bf16_m16n16k32",
                "hip_shared_bf16_m16n32k64",
            ],
            "preferred_strategies": ["hip_reference_oracle", "hip_shared_bf16"],
            "trigger_signals": ["contract_repair", "runtime_repair", "submission_repair"],
        },
        {
            "name": "hip_memory_hierarchy",
            "family": "hip_explore",
            "focus": "optimize LDS tiling, double buffering, and swizzle around the correctness-first HIP path",
            "preferred_variants": [
                "hip_shared_bf16_m16n32k64",
                "hip_scale_mfma_seed_16x16x128",
                "hip_scale_mfma_seed_32x32x64",
            ],
            "preferred_strategies": ["hip_shared_bf16", "hip_scale_mfma_seed"],
            "trigger_signals": ["throughput_shift", "latency_repair"],
        },
        {
            "name": "hip_scale_mfma",
            "family": "hip_explore",
            "focus": "replace the tiled inner loop with CDNA4 scaled MFMA intrinsics while keeping the same HIP load_inline path",
            "preferred_variants": [
                "hip_scale_mfma_seed_16x16x128",
                "hip_scale_mfma_seed_32x32x64",
            ],
            "preferred_strategies": ["hip_scale_mfma_seed"],
            "trigger_signals": ["throughput_shift"],
        },
    ],
    "moe_mxfp4": [
        {
            "name": "contract_repair",
            "family": "kernel_explore",
            "focus": "build a padding-free dispatch pack path while keeping routing and top-k semantics fixed",
            "preferred_variants": [
                "dispatch_pack_sparse256",
                "dispatch_pack_dense32",
            ],
            "preferred_strategies": ["dispatch_pack"],
            "trigger_signals": ["contract_repair", "runtime_repair", "submission_repair"],
        },
        {
            "name": "stage1_core",
            "family": "kernel_explore",
            "focus": "replace anchor-backed stage1 compute with a touched-expert grouped path before heavier pipeline rewrites",
            "preferred_variants": [
                "stage1_grouped_bf16",
                "stage1_swiglu_fused",
            ],
            "preferred_strategies": ["stage1_grouped", "stage1_swiglu_fused"],
            "trigger_signals": ["throughput_shift", "contract_repair"],
        },
        {
            "name": "stage2_reduce",
            "family": "kernel_explore",
            "focus": "fold weighted stage2 reduction into the touched-expert path once stage1 is stable",
            "preferred_variants": [
                "stage2_grouped_weighted",
            ],
            "preferred_strategies": ["stage2_grouped"],
            "trigger_signals": ["throughput_shift", "latency_repair"],
        },
        {
            "name": "shared_expert",
            "family": "kernel_explore",
            "focus": "split shared-expert handling from routed-expert handling in the sparse high-work regime",
            "preferred_variants": [
                "shared_expert_split",
            ],
            "preferred_strategies": ["shared_expert_split"],
            "trigger_signals": ["throughput_shift", "latency_repair"],
        },
        {
            "name": "hip_sparse_pipeline",
            "family": "hip_explore",
            "focus": "carry the touched-expert dispatch into a future gfx950-native MoE pipeline after the kernel path is structurally correct",
            "preferred_variants": [
                "hip_fp4_preshuffled_sparse256",
                "hip_fp4_preshuffled_dense32",
                "hip_persistent_sparse256",
            ],
            "preferred_strategies": ["hip_fp4_preshuffled", "hip_persistent_sparse"],
            "trigger_signals": ["throughput_shift"],
        },
    ],
    "mixed_mla": [
        {
            "name": "contract_repair",
            "family": "kernel_explore",
            "focus": "preserve FP8 MLA decode semantics before widening tiles",
            "preferred_variants": [
                "fp8_decode_b32_v64",
                "fp8_decode_b64_v64",
            ],
            "preferred_strategies": ["fp8_decode"],
            "trigger_signals": ["contract_repair", "runtime_repair", "submission_repair"],
        },
        {
            "name": "latency_small_block",
            "family": "kernel_explore",
            "focus": "reduce small-batch overhead and keep q_seq_len=1 path lean",
            "preferred_variants": [
                "fp8_decode_b32_v64",
                "fp8_decode_b64_v64",
            ],
            "preferred_strategies": ["fp8_decode"],
            "trigger_signals": ["throughput_shift", "latency_repair"],
        },
        {
            "name": "long_kv_throughput",
            "family": "kernel_explore",
            "focus": "push bigger tiles on long-KV cases after correctness is stable",
            "preferred_variants": [
                "fp8_decode_b128_v64",
                "fp8_decode_b128_v128",
            ],
            "preferred_strategies": ["fp8_decode"],
            "trigger_signals": ["throughput_shift"],
        },
    ],
}


def load_context(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_parent_meta(path: Path) -> dict[str, object] | None:
    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    match = META_RE.search(source)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def candidate_attempt(context: dict[str, object]) -> int:
    workspace_root = Path(str(context["workspace_root"]))
    problem_key = str(context["problem"]["key"])
    candidate_root = workspace_root / "problems" / problem_key / "candidates"
    if not candidate_root.exists():
        return 1
    return sum(1 for entry in candidate_root.iterdir() if entry.is_dir())


def history_entries(context: dict[str, object]) -> list[dict[str, object]]:
    raw = context.get("history")
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def choose_policy_profile(
    problem_key: str,
    attempt: int,
    parent_meta: dict[str, object] | None,
    history: list[dict[str, object]],
    desired_family: str | None = None,
) -> dict[str, object]:
    profiles = POLICY_PROFILES[problem_key]
    if desired_family:
        filtered = [profile for profile in profiles if profile.get("family") == desired_family]
        if filtered:
            profiles = filtered

    if problem_key == "mxfp4_mm" and desired_family == "hip_explore":
        hip_ok_seen = False
        for entry in history:
            if entry.get("status") != "ok":
                continue
            meta = entry.get("meta")
            if not isinstance(meta, dict):
                continue
            variant = meta.get("variant")
            if isinstance(variant, dict) and variant.get("family") == "hip_explore":
                hip_ok_seen = True
                break
        if not hip_ok_seen:
            locked = next(
                (profile for profile in profiles if str(profile.get("name")) == "hip_contract_reference"),
                None,
            )
            if locked is not None:
                return locked

    usage_counts: Counter[str] = Counter()
    signal_counts: Counter[str] = Counter()
    last_profile_name: str | None = None
    last_status: str | None = None
    for entry in history:
        meta = entry.get("meta")
        if isinstance(meta, dict):
            policy_profile = meta.get("policy_profile")
            if isinstance(policy_profile, dict):
                name = policy_profile.get("name")
                if isinstance(name, str) and name:
                    usage_counts[name] += 1
                    if last_profile_name is None:
                        last_profile_name = name
        if last_status is None and isinstance(entry.get("status"), str):
            last_status = str(entry["status"])
        critique = entry.get("critique")
        if isinstance(critique, dict):
            signal = critique.get("policy_signal")
            if isinstance(signal, str) and signal:
                signal_counts[signal] += 1

    target_signal: str | None = None
    if signal_counts:
        target_signal = signal_counts.most_common(1)[0][0]
    parent_profile_name: str | None = None
    if isinstance(parent_meta, dict):
        parent_profile = parent_meta.get("policy_profile")
        if isinstance(parent_profile, dict):
            name = parent_profile.get("name")
            if isinstance(name, str) and name:
                parent_profile_name = name

    def profile_sort_key(profile: dict[str, object]) -> tuple[int, int, int, int, int, str]:
        name = str(profile.get("name", ""))
        trigger_signals = profile.get("trigger_signals")
        signal_match = 0
        if isinstance(trigger_signals, list) and target_signal in trigger_signals:
            signal_match = -2
        reuse_bonus = 0
        if last_status == "ok" and name == last_profile_name:
            reuse_bonus = -1
        elif last_status not in {None, "ok"} and name == last_profile_name:
            reuse_bonus = 1
        parent_bonus = -1 if name == parent_profile_name else 0
        return (
            signal_match,
            reuse_bonus,
            parent_bonus,
            usage_counts[name],
            attempt % max(len(profiles), 1),
            name,
        )

    return min(profiles, key=profile_sort_key)


def moe_subagent_roster() -> list[dict[str, object]]:
    return [dict(item) for item in MOE_SUBAGENT_ROSTER]


def moe_motivation_refs() -> list[str]:
    return list(MOE_MOTIVATION_REFS)


def mxfp4_subagent_roster() -> list[dict[str, object]]:
    return [dict(item) for item in MXFP4_SUBAGENT_ROSTER]


def mxfp4_motivation_refs() -> list[str]:
    return list(MXFP4_MOTIVATION_REFS)


def mxfp4_retrieval_pack() -> dict[str, object]:
    return {
        "name": str(MXFP4_RETRIEVAL_PACK["name"]),
        "queries": [str(item) for item in MXFP4_RETRIEVAL_PACK["queries"]],
    }


def moe_retrieval_pack(variant: dict[str, object]) -> dict[str, object]:
    lane = str(variant.get("LANE", "") or "full_pipeline")
    pack = MOE_RETRIEVAL_PACKS.get(lane, MOE_RETRIEVAL_PACKS["full_pipeline"])
    return {
        "name": str(pack["name"]),
        "queries": [str(item) for item in pack["queries"]],
    }


def moe_candidate_card(variant: dict[str, object]) -> dict[str, object]:
    variant_name = str(variant.get("variant_name", ""))
    template = MOE_CANDIDATE_CARD_TEMPLATES.get(variant_name, {})
    lane = str(variant.get("LANE", "") or ("full_pipeline" if variant.get("family") == "anchor" else "unknown"))
    regime_tag = str(variant.get("REGIME_HINT", "") or "unknown")
    retrieval_pack = moe_retrieval_pack(variant)
    return {
        "lane": lane,
        "regime_tag": regime_tag,
        "deleted_cost_center": str(template.get("deleted_cost_center", "")),
        "expected_upside_source": str(template.get("expected_upside_source", "")),
        "why_larger_than_noise": str(template.get("why_larger_than_noise", "")),
        "forbidden_edits": [str(item) for item in template.get("forbidden_edits", [])],
        "success_gate": str(template.get("success_gate", "")),
        "retrieval_pack": retrieval_pack,
        "retrieval_queries": [str(item) for item in retrieval_pack.get("queries", [])],
        "motivation_refs": moe_motivation_refs(),
        "required_subagents": moe_subagent_roster(),
    }


def choose_variant(
    problem_key: str,
    attempt: int,
    parent_meta: dict[str, object] | None,
    history: list[dict[str, object]],
    policy_profile: dict[str, object] | None = None,
    desired_family: str | None = None,
) -> tuple[int, dict[str, object]]:
    variants = SEARCH_SPACE[problem_key]
    counts: Counter[int] = Counter()
    fail_counts: Counter[int] = Counter()
    ok_counts: Counter[int] = Counter()
    ok_indices: set[int] = set()
    for entry in history:
        meta = entry.get("meta")
        if isinstance(meta, dict):
            index = meta.get("variant_index")
            if isinstance(index, int):
                counts[index] += 1
                if entry.get("status") == "ok":
                    ok_indices.add(index)
                    ok_counts[index] += 1
                else:
                    fail_counts[index] += 1

    anchor_indices = [index for index, variant in enumerate(variants) if variant.get("family") == "anchor"]
    explore_indices = [index for index, variant in enumerate(variants) if variant.get("family") != "anchor"]
    if desired_family:
        family_indices = [
            index for index, variant in enumerate(variants) if variant.get("family") == desired_family
        ]
        if family_indices:
            if problem_key == "mxfp4_mm" and desired_family == "hip_explore":
                hip_ok_seen = False
                for entry in history:
                    if entry.get("status") != "ok":
                        continue
                    meta = entry.get("meta")
                    if not isinstance(meta, dict):
                        continue
                    hist_variant = meta.get("variant")
                    if isinstance(hist_variant, dict) and hist_variant.get("family") == "hip_explore":
                        hip_ok_seen = True
                        break
                if not hip_ok_seen and isinstance(policy_profile, dict):
                    preferred = {
                        str(name)
                        for name in policy_profile.get("preferred_variants", [])
                        if isinstance(name, str)
                    }
                    locked_indices = [
                        index
                        for index in family_indices
                        if str(variants[index].get("variant_name", "")) in preferred
                    ]
                    if locked_indices:
                        family_indices = locked_indices
            center = 0
            if parent_meta and isinstance(parent_meta.get("variant_index"), int):
                center = int(parent_meta["variant_index"])
            elif family_indices:
                center = family_indices[0]
            return _pick_variant(
                family_indices,
                counts,
                fail_counts,
                ok_counts,
                attempt,
                center,
                variants,
                policy_profile=policy_profile,
            )

    if not ok_indices:
        indices = anchor_indices or list(range(len(variants)))
        return _pick_variant(
            indices,
            counts,
            fail_counts,
            ok_counts,
            attempt,
            anchor_indices[0] if anchor_indices else 0,
            variants,
            policy_profile=policy_profile,
        )

    center = 0
    if parent_meta and isinstance(parent_meta.get("variant_index"), int):
        center = int(parent_meta["variant_index"])
    elif ok_indices:
        center = min(ok_indices)
    indices = explore_indices or list(range(len(variants)))
    return _pick_variant(
        indices,
        counts,
        fail_counts,
        ok_counts,
        attempt,
        center,
        variants,
        policy_profile=policy_profile,
    )


def _pick_variant(
    indices: list[int],
    counts: Counter[int],
    fail_counts: Counter[int],
    ok_counts: Counter[int],
    attempt: int,
    center: int,
    variants: list[dict[str, object]],
    policy_profile: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    def failure_penalty(index: int) -> int:
        return max(fail_counts[index] - ok_counts[index], 0)

    preferred_variants: list[str] = []
    preferred_strategies: list[str] = []
    if isinstance(policy_profile, dict):
        raw_variants = policy_profile.get("preferred_variants")
        if isinstance(raw_variants, list):
            preferred_variants = [str(value) for value in raw_variants]
        raw_strategies = policy_profile.get("preferred_strategies")
        if isinstance(raw_strategies, list):
            preferred_strategies = [str(value) for value in raw_strategies]

    preferred_variant_rank = {name: idx for idx, name in enumerate(preferred_variants)}
    preferred_strategy_rank = {name: idx for idx, name in enumerate(preferred_strategies)}

    def profile_rank(index: int) -> tuple[int, int]:
        variant = variants[index]
        variant_name = str(variant.get("variant_name", ""))
        strategy = str(variant.get("strategy", ""))
        variant_rank = preferred_variant_rank.get(variant_name, len(preferred_variant_rank) + 1)
        strategy_rank = preferred_strategy_rank.get(strategy, len(preferred_strategy_rank) + 1)
        return (variant_rank, strategy_rank)

    sorted_indices = sorted(
        indices,
        key=lambda index: (
            failure_penalty(index),
            profile_rank(index),
            counts[index],
            _circular_distance(index, center, len(variants)),
            index,
        ),
    )
    if not sorted_indices:
        raise RuntimeError("no variants available")
    pick = sorted_indices[(attempt - 1) % len(sorted_indices)]
    return pick, variants[pick]


def _circular_distance(index: int, center: int, size: int) -> int:
    direct = abs(index - center)
    return min(direct, max(size - direct, 0))


def render_submission(
    problem_key: str,
    variant_index: int,
    variant: dict[str, object],
    context: dict[str, object],
    attempt: int,
    policy_profile: dict[str, object] | None = None,
) -> str:
    meta = {
        "problem": problem_key,
        "leaderboard": context["problem"]["leaderboard"],
        "gpu": context["problem"]["gpu"],
        "attempt": attempt,
        "variant_index": variant_index,
        "variant": variant,
    }
    if isinstance(policy_profile, dict):
        meta["policy_profile"] = {
            "name": policy_profile.get("name"),
            "family": policy_profile.get("family"),
            "focus": policy_profile.get("focus"),
            "trigger_signals": policy_profile.get("trigger_signals"),
        }
    if problem_key == "moe_mxfp4":
        meta["candidate_card"] = moe_candidate_card(variant)
    if problem_key == "mxfp4_mm":
        if variant.get("family") == "anchor":
            return render_mxfp4_mm_anchor(meta)
        if variant.get("family") == "hip_explore":
            return render_mxfp4_mm_hip(meta, variant)
        return render_mxfp4_mm_kernel(meta, variant)
    if problem_key == "moe_mxfp4":
        if variant.get("family") == "anchor":
            return render_moe_mxfp4_anchor(meta, variant)
        return render_moe_mxfp4_kernel(meta, variant)
    if problem_key == "mixed_mla":
        if variant.get("family") == "anchor":
            return render_mixed_mla_anchor(meta)
        return render_mixed_mla_kernel(meta, variant)
    raise KeyError(f"unsupported problem key: {problem_key}")


def render_mxfp4_mm_anchor(meta: dict[str, object]) -> str:
    source = textwrap.dedent(
        """
        #!POPCORN leaderboard amd-mxfp4-mm
        #!POPCORN gpu MI355X
        # AGENT_LOOP_META: __META__
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
        """
    ).strip()
    return source.replace("__META__", json.dumps(meta, sort_keys=True))


def render_mxfp4_mm_kernel(meta: dict[str, object], variant: dict[str, object]) -> str:
    source = textwrap.dedent(
        """
        #!POPCORN leaderboard amd-mxfp4-mm
        #!POPCORN gpu MI355X
        # AGENT_LOOP_META: __META__
        import aiter
        from aiter import QuantType
        from aiter.utility import fp4_utils
        import torch
        import triton
        import triton.language as tl
        from task import input_t, output_t

        CONFIG = __CONFIG__
        SCALE_GROUP = 32


        def _dequantize_matrix(fp4_packed: torch.Tensor, scale_e8m0: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
            values = fp4_utils.mxfp4_to_f32(fp4_packed)
            scale = fp4_utils.e8m0_to_f32(scale_e8m0)
            if scale.ndim == 1:
                scale = scale.reshape(rows, -1)
            scale = scale[:rows, :].repeat_interleave(SCALE_GROUP, dim=1)[:, :cols]
            values = values[:rows, :cols]
            return (values * scale).to(torch.bfloat16)


        def _quantize_and_dequantize(matrix: torch.Tensor, label: str, *, shuffle: bool) -> torch.Tensor:
            del label
            quantized, scale = aiter.get_triton_quant(QuantType.per_1x32)(matrix.contiguous(), shuffle=shuffle)
            rows, cols = matrix.shape
            return _dequantize_matrix(quantized, scale, rows=rows, cols=cols)


        def _dequantize_contract_tensor(fp4_packed: torch.Tensor, scale_e8m0: torch.Tensor, label: str) -> torch.Tensor:
            del label
            rows = fp4_packed.shape[0]
            cols = fp4_utils.mxfp4_to_f32(fp4_packed.contiguous()).shape[1]
            return _dequantize_matrix(fp4_packed.contiguous(), scale_e8m0.contiguous(), rows=rows, cols=cols)


        @triton.jit
        def _matmul_kernel(
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


        def custom_kernel(data: input_t) -> output_t:
            a, b, b_q, b_shuffle, b_scale_sh = data
            if CONFIG.get("strategy") == "runtime_requant_matmul":
                del b_q, b_shuffle, b_scale_sh
                a_dq = _quantize_and_dequantize(a, "a_runtime", shuffle=False)
                b_dq = _quantize_and_dequantize(b, "b_runtime", shuffle=False)
            elif CONFIG.get("CONTRACT_NATIVE", False):
                a_dq = _quantize_and_dequantize(a, "a_contract", shuffle=True)
                b_dq = _dequantize_contract_tensor(b_shuffle, b_scale_sh, "b_contract")
            else:
                a_dq = _quantize_and_dequantize(a, "a", shuffle=False)
                b_dq = _quantize_and_dequantize(b, "b", shuffle=False)

            m, k = a_dq.shape
            n = b_dq.shape[0]
            c = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
            grid = (triton.cdiv(m, CONFIG["BLOCK_M"]) * triton.cdiv(n, CONFIG["BLOCK_N"]),)
            _matmul_kernel[grid](
                a_dq,
                b_dq,
                c,
                m,
                n,
                k,
                a_dq.stride(0),
                a_dq.stride(1),
                b_dq.stride(0),
                b_dq.stride(1),
                c.stride(0),
                c.stride(1),
                BLOCK_M=CONFIG["BLOCK_M"],
                BLOCK_N=CONFIG["BLOCK_N"],
                BLOCK_K=CONFIG["BLOCK_K"],
                GROUP_M=CONFIG["GROUP_M"],
                num_warps=CONFIG["NUM_WARPS"],
                num_stages=CONFIG["NUM_STAGES"],
            )
            return c
        """
    ).strip()
    return source.replace("__META__", json.dumps(meta, sort_keys=True)).replace("__CONFIG__", repr(variant))


def render_mxfp4_mm_hip(meta: dict[str, object], variant: dict[str, object]) -> str:
    source = textwrap.dedent(
        r"""
        #!POPCORN leaderboard amd-mxfp4-mm
        #!POPCORN gpu MI355X
        # AGENT_LOOP_META: __META__
        import hashlib
        import os
        from pathlib import Path
        import tempfile

        os.environ["PYTORCH_ROCM_ARCH"] = "__ARCH__"
        os.environ.setdefault("CXX", "clang++")

        import aiter
        from aiter import QuantType
        from aiter.utility import fp4_utils
        import torch
        from torch.utils.cpp_extension import load_inline
        from task import input_t, output_t

        CONFIG = __CONFIG__
        SCALE_GROUP = 32

        CPP_WRAPPER = '''
        void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c);
        '''

        HIP_SRC = r'''
        #include <torch/extension.h>
        #include <hip/hip_runtime.h>
        #include <hip/amd_detail/amd_hip_bf16.h>
        #include <type_traits>

        constexpr int TILE_M = __TILE_M__;
        constexpr int TILE_N = __TILE_N__;
        constexpr int TILE_K = __TILE_K__;
        constexpr bool DOUBLE_BUFFER = __DOUBLE_BUFFER__;
        constexpr bool LDS_SWIZZLE = __LDS_SWIZZLE__;
        constexpr bool USE_SCALE_MFMA_SEED = __USE_SCALE_MFMA_SEED__;
        constexpr bool REFERENCE_INPUTS = __REFERENCE_INPUTS__;
        constexpr bool NAIVE_KERNEL = __NAIVE_KERNEL__;

        // HIP-first seed for MI355X/gfx950.
        // This keeps the entire path in one language and one compiler pipeline.
        // The first reference modes reconstruct the MXFP4 math in Python, then execute the final
        // bf16 matmul in HIP. The naive variant is for semantic debugging only; the tiled variant
        // is the first realistic correctness-first submission candidate.
        // After the first passing HIP check, the agent can evolve this toward tiled shared-memory
        // kernels and eventually scaled-MFMA without changing the Python/load_inline contract.
        // Seed target instruction for future rewrites: __MFMA_OP__

        template <typename input_t>
        __global__ void mxfp4_mm_kernel(
            const input_t* a,
            const input_t* b,
            __hip_bfloat16* c,
            int m,
            int n,
            int k
        ) {
            const int local_x = threadIdx.x;
            const int local_y = threadIdx.y;
            const int row = blockIdx.y * TILE_M + local_y;
            const int col = blockIdx.x * TILE_N + local_x;

            double acc = 0.0;

        if (NAIVE_KERNEL) {
                if (row < m && col < n) {
                    for (int kk = 0; kk < k; ++kk) {
                        acc += static_cast<double>(a[row * k + kk]) * static_cast<double>(b[col * k + kk]);
                    }
                }
        } else {
            __shared__ input_t a_tile[TILE_M][TILE_K];
            __shared__ input_t b_tile[TILE_N][TILE_K];

            for (int tile_k = 0; tile_k < k; tile_k += TILE_K) {
                for (int load_k = local_x; load_k < TILE_K; load_k += blockDim.x) {
                    const int global_k = tile_k + load_k;
                    a_tile[local_y][load_k] =
                        (row < m && global_k < k) ? a[row * k + global_k] : input_t(0.0f);
                }
                for (int load_k = local_y; load_k < TILE_K; load_k += blockDim.y) {
                    const int global_k = tile_k + load_k;
                    b_tile[local_x][load_k] =
                        (col < n && global_k < k) ? b[col * k + global_k] : input_t(0.0f);
                }

                __syncthreads();

                if (row < m && col < n) {
                    #pragma unroll 1
                    for (int kk = 0; kk < TILE_K; ++kk) {
                        acc += static_cast<double>(a_tile[local_y][kk]) * static_cast<double>(b_tile[local_x][kk]);
                    }
                }

                __syncthreads();

                if (DOUBLE_BUFFER || LDS_SWIZZLE || USE_SCALE_MFMA_SEED) {
                    // Seed toggles for the agent. The initial kernel stays correctness-first and
                    // uses the same memory layout so future edits can replace the inner loop with
                    // gfx950-specific scaled MFMA + async global->LDS movement without changing
                    // the Python/load_inline contract.
                }
            }
        }

        if (row < m && col < n) {
            c[row * n + col] = static_cast<__hip_bfloat16>(static_cast<float>(acc));
        }
        }

        void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
            const int m = static_cast<int>(a.size(0));
            const int n = static_cast<int>(b.size(0));
            const int k = static_cast<int>(a.size(1));
            dim3 block(TILE_N, TILE_M);
            dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);
            if constexpr (REFERENCE_INPUTS) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(mxfp4_mm_kernel<float>),
                    grid,
                    block,
                    0,
                    0,
                    a.data_ptr<float>(),
                    b.data_ptr<float>(),
                    reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
                    m,
                    n,
                    k
                );
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(mxfp4_mm_kernel<__hip_bfloat16>),
                    grid,
                    block,
                    0,
                    0,
                    reinterpret_cast<const __hip_bfloat16*>(a.data_ptr<at::BFloat16>()),
                    reinterpret_cast<const __hip_bfloat16*>(b.data_ptr<at::BFloat16>()),
                    reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
                    m,
                    n,
                    k
                );
            }
        }
        '''

        _MODULE = None


        def _module():
            global _MODULE
            if _MODULE is None:
                build_root = Path(tempfile.gettempdir()) / "mxfp4_mm_hip_build"
                build_root.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode("utf-8")).hexdigest()[:12]
                module_name = f"mxfp4_mm_hip_{CONFIG['variant_name']}_{digest}"
                _MODULE = load_inline(
                    name=module_name,
                    cpp_sources=[CPP_WRAPPER],
                    cuda_sources=[HIP_SRC],
                    functions=["mxfp4_mm_hip"],
                    extra_cuda_cflags=["--offload-arch=__ARCH__", "-std=c++20", "-O3"],
                    build_directory=str(build_root),
                    verbose=False,
                )
            return _MODULE


        def _dequantize_logical_mxfp4(fp4_packed: torch.Tensor, scale_e8m0: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
            values = fp4_utils.mxfp4_to_f32(fp4_packed.contiguous())[:rows, :cols]
            scales_f32 = _expand_scales(scale_e8m0, rows=rows, cols=cols)
            return (values * scales_f32).to(torch.float32).contiguous()


        def _expand_scales(scale_e8m0: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
            scales = scale_e8m0.contiguous()[:rows]
            scales = scales.repeat_interleave(SCALE_GROUP, dim=1)[:, :cols]
            return fp4_utils.e8m0_to_f32(scales).to(torch.float32)


        def _learn_adjustment_rules(
            norm: torch.Tensor,
            ref_vals: torch.Tensor,
            live_vals: torch.Tensor,
        ) -> dict[float, tuple[str, float, float]]:
            rules: dict[float, tuple[str, float, float]] = {}
            for q_tensor in torch.unique(ref_vals):
                q = float(q_tensor.item())
                mask = ref_vals == q
                if int(mask.sum().item()) == 0:
                    continue
                labels = (live_vals != ref_vals)[mask]
                total = int(labels.numel())
                positives = int(labels.sum().item())
                if positives == 0:
                    continue
                if positives == total:
                    adjusted = float(torch.unique(live_vals[mask], return_counts=True)[0][0].item())
                    rules[q] = ("all", 0.0, adjusted)
                    continue

                values = norm[mask]
                live_subset = live_vals[mask]
                pos_live = live_subset[labels]
                uniq_live, cnt_live = torch.unique(pos_live, return_counts=True)
                adjusted = float(uniq_live[torch.argmax(cnt_live)].item())

                sorted_vals, order = torch.sort(values.reshape(-1))
                sorted_labels = labels.reshape(-1)[order].to(torch.int64)
                prefix_pos = torch.cumsum(sorted_labels, dim=0)
                prefix_idx = torch.arange(1, sorted_labels.numel() + 1, device=sorted_labels.device, dtype=torch.int64)
                prefix_neg = prefix_idx - prefix_pos
                total_pos = int(prefix_pos[-1].item())
                total_neg = sorted_labels.numel() - total_pos
                suffix_pos = total_pos - prefix_pos
                suffix_neg = total_neg - prefix_neg

                err_le = prefix_neg + suffix_pos
                err_gt = prefix_pos + suffix_neg
                best_le = int(torch.argmin(err_le).item())
                best_gt = int(torch.argmin(err_gt).item())
                err_le_val = int(err_le[best_le].item())
                err_gt_val = int(err_gt[best_gt].item())

                if err_le_val <= err_gt_val:
                    rules[q] = ("le", float(sorted_vals[best_le].item()), adjusted)
                else:
                    rules[q] = ("gt", float(sorted_vals[best_gt].item()), adjusted)
            return rules


        def _apply_adjustment_rules(
            norm: torch.Tensor,
            ref_vals: torch.Tensor,
            rules: dict[float, tuple[str, float, float]],
        ) -> torch.Tensor:
            corrected = ref_vals.clone()
            for q, (direction, threshold, adjusted) in rules.items():
                mask = ref_vals == q
                if direction == "all":
                    cond = mask
                elif direction == "le":
                    cond = mask & (norm <= threshold)
                else:
                    cond = mask & (norm > threshold)
                corrected = torch.where(cond, torch.full_like(corrected, adjusted), corrected)
            return corrected


        def _reference_oracle_inputs(
            a: torch.Tensor,
            b: torch.Tensor,
            b_q: torch.Tensor,
            b_scale_sh: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            quant = aiter.get_triton_quant(QuantType.per_1x32)
            a_q, a_scale = quant(a.contiguous(), shuffle=False)
            public_b_q, b_scale = quant(b.contiguous(), shuffle=False)

            a_scale_f32 = _expand_scales(a_scale, rows=a.shape[0], cols=a.shape[1])
            b_scale_f32 = _expand_scales(b_scale, rows=b.shape[0], cols=b.shape[1])

            a_ref_vals = fp4_utils.mxfp4_to_f32(a_q.contiguous())[: a.shape[0], : a.shape[1]].to(torch.float32)
            b_ref_vals = fp4_utils.mxfp4_to_f32(b_q.contiguous())[: b.shape[0], : b.shape[1]].to(torch.float32)
            b_public_vals = fp4_utils.mxfp4_to_f32(public_b_q.contiguous())[: b.shape[0], : b.shape[1]].to(torch.float32)

            norm_b = (b.to(torch.float32) / b_scale_f32).contiguous()
            rules = _learn_adjustment_rules(norm_b, b_public_vals, b_ref_vals)

            norm_a = (a.to(torch.float32) / a_scale_f32).contiguous()
            a_corrected_vals = _apply_adjustment_rules(norm_a, a_ref_vals, rules)

            a_ref = (a_corrected_vals * a_scale_f32).to(torch.float32).contiguous()
            b_ref = (b_ref_vals * b_scale_f32).to(torch.float32).contiguous()
            return a_ref, b_ref


        def custom_kernel(data: input_t) -> output_t:
            a, b, b_q, b_shuffle, b_scale_sh = data
            torch._assert(b_q.shape[0] == b.shape[0], "B_q row count must match logical B")
            torch._assert(b_shuffle.shape[0] == b.shape[0], "B_shuffle row count must match logical B")
            torch._assert(b_scale_sh.numel() > 0, "B_scale_sh must be present for the live contract")
            if CONFIG.get("REFERENCE_INPUTS", False):
                a_in, b_in = _reference_oracle_inputs(a, b, b_q, b_scale_sh)
            else:
                a_in = a.contiguous()
                b_in = b.contiguous()
            c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
            _module().mxfp4_mm_hip(a_in, b_in, c)
            return c
        """
    ).strip()
    return (
        source
        .replace("__META__", json.dumps(meta, sort_keys=True))
        .replace("__CONFIG__", repr(variant))
        .replace("__ARCH__", str(variant.get("ARCH", "gfx950")))
        .replace("__TILE_M__", str(int(variant.get("TILE_M", 16))))
        .replace("__TILE_N__", str(int(variant.get("TILE_N", 16))))
        .replace("__TILE_K__", str(int(variant.get("TILE_K", 32))))
        .replace("__DOUBLE_BUFFER__", "true" if variant.get("DOUBLE_BUFFER") else "false")
        .replace("__LDS_SWIZZLE__", "true" if variant.get("LDS_SWIZZLE") else "false")
        .replace("__USE_SCALE_MFMA_SEED__", "true" if variant.get("USE_SCALE_MFMA_SEED") else "false")
        .replace("__REFERENCE_INPUTS__", "true" if variant.get("REFERENCE_INPUTS") else "false")
        .replace("__NAIVE_KERNEL__", "true" if variant.get("NAIVE_KERNEL") else "false")
        .replace("__MFMA_OP__", str(variant.get("MFMA_OP", "none")))
    )


def render_moe_mxfp4_anchor(meta: dict[str, object], variant: dict[str, object]) -> str:
    source = textwrap.dedent(
        """
        #!POPCORN leaderboard amd-moe-mxfp4
        #!POPCORN gpu MI355X
        # AGENT_LOOP_META: __META__
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe
        from task import input_t, output_t

        CONFIG = __CONFIG__


        def custom_kernel(data: input_t) -> output_t:
            (
                hidden_states,
                gate_up_weight,
                down_weight,
                gate_up_weight_scale,
                down_weight_scale,
                gate_up_weight_shuffled,
                down_weight_shuffled,
                gate_up_weight_scale_shuffled,
                down_weight_scale_shuffled,
                topk_weights,
                topk_ids,
                config,
            ) = data
            del gate_up_weight, down_weight, gate_up_weight_scale, down_weight_scale
            hidden_pad = int(config["d_hidden_pad"]) - int(config["d_hidden"])
            intermediate_pad = int(config["d_expert_pad"]) - int(config["d_expert"])
            return fused_moe(
                hidden_states,
                gate_up_weight_shuffled,
                down_weight_shuffled,
                topk_weights,
                topk_ids,
                expert_mask=None,
                activation=ActivationType.Silu,
                quant_type=QuantType.per_1x32,
                doweight_stage1=False,
                w1_scale=gate_up_weight_scale_shuffled,
                w2_scale=down_weight_scale_shuffled,
                a1_scale=None,
                a2_scale=None,
                block_size_M=CONFIG.get("BLOCK_SIZE_M"),
                hidden_pad=hidden_pad,
                intermediate_pad=intermediate_pad,
            )
        """
    ).strip()
    return source.replace("__META__", json.dumps(meta, sort_keys=True)).replace("__CONFIG__", repr(variant))


def render_moe_mxfp4_kernel(meta: dict[str, object], variant: dict[str, object]) -> str:
    source = textwrap.dedent(
        """
        #!POPCORN leaderboard amd-moe-mxfp4
        #!POPCORN gpu MI355X
        # AGENT_LOOP_META: __META__
        import torch
        import triton
        import triton.language as tl
        from task import input_t, output_t
        from aiter.utility import fp4_utils

        CONFIG = __CONFIG__
        MXFP4_BLOCK = 32


        def _dequant_matrix(weight_fp4: torch.Tensor, scale_e8m0: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
            values = fp4_utils.mxfp4_to_f32(weight_fp4)
            scale = fp4_utils.e8m0_to_f32(scale_e8m0)
            if scale.ndim == 0:
                scale = scale.reshape(1, 1)
            elif scale.ndim == 1:
                if scale.numel() % max(values.shape[0], 1) == 0:
                    scale = scale.reshape(values.shape[0], -1)
                else:
                    scale = scale.reshape(1, -1).expand(values.shape[0], -1)
            scale = scale[: values.shape[0], :].repeat_interleave(MXFP4_BLOCK, dim=1)[:, : values.shape[1]]
            return (values * scale)[:rows, :cols].to(torch.bfloat16)


        def _requantize_activation(activation: torch.Tensor) -> torch.Tensor:
            quantized, scale = aiter.get_triton_quant(QuantType.per_1x32)(activation.contiguous(), shuffle=False)
            rows, cols = activation.shape
            return _dequant_matrix(quantized, scale, rows=rows, cols=cols)


        def _dequant_gate_up_for_expert(
            gate_up_weight: torch.Tensor,
            gate_up_weight_scale: torch.Tensor,
            expert: int,
            config: dict,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            d_hidden = int(config["d_hidden"])
            d_expert = int(config["d_expert"])
            gate_up = _dequant_matrix(
                gate_up_weight[expert],
                gate_up_weight_scale[expert],
                rows=2 * d_expert,
                cols=d_hidden,
            )
            gate_part, up_part = gate_up.chunk(2, dim=0)
            return gate_part.contiguous(), up_part.contiguous()


        def _dequant_down_for_expert(
            down_weight: torch.Tensor,
            down_weight_scale: torch.Tensor,
            expert: int,
            config: dict,
        ) -> torch.Tensor:
            d_hidden = int(config["d_hidden"])
            d_expert = int(config["d_expert"])
            return _dequant_matrix(
                down_weight[expert],
                down_weight_scale[expert],
                rows=d_hidden,
                cols=d_expert,
            ).contiguous()


        @triton.jit
        def _silu_mul_kernel(gate_ptr, up_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < numel
            gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            sig = tl.sigmoid(gate)
            out = gate * sig * up
            tl.store(out_ptr + offs, out.to(tl.bfloat16), mask=mask)


        def _silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(gate)
            numel = gate.numel()
            grid = (triton.cdiv(numel, CONFIG["BLOCK_SIZE"]),)
            _silu_mul_kernel[grid](
                gate,
                up,
                out,
                numel,
                BLOCK_SIZE=CONFIG["BLOCK_SIZE"],
                num_warps=CONFIG["NUM_WARPS"],
            )
            return out


        def _route_entries(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            num_tokens = topk_ids.shape[0]
            topk = topk_ids.shape[1]
            token_ids = torch.arange(num_tokens, device=topk_ids.device, dtype=torch.int64).repeat_interleave(topk)
            expert_ids = topk_ids.reshape(-1).to(torch.int64)
            weights = topk_weights.reshape(-1, 1).to(torch.float32)
            if CONFIG["SORT_BY_EXPERT"]:
                order = torch.argsort(expert_ids)
                token_ids = token_ids[order]
                expert_ids = expert_ids[order]
                weights = weights[order]
            return token_ids, expert_ids, weights


        def _expert_windows(expert_ids: torch.Tensor, num_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
            counts = torch.bincount(expert_ids, minlength=num_experts)
            offsets = torch.zeros_like(counts)
            if counts.numel() > 1:
                offsets[1:] = torch.cumsum(counts[:-1], dim=0)
            return offsets, counts


        def custom_kernel(data: input_t) -> output_t:
            (
                hidden_states,
                gate_up_weight,
                down_weight,
                gate_up_weight_scale,
                down_weight_scale,
                gate_up_weight_shuffled,
                down_weight_shuffled,
                gate_up_weight_scale_shuffled,
                down_weight_scale_shuffled,
                topk_weights,
                topk_ids,
                config,
            ) = data
            del gate_up_weight_shuffled, down_weight_shuffled
            del gate_up_weight_scale_shuffled, down_weight_scale_shuffled

            num_tokens = hidden_states.shape[0]
            d_hidden = int(config["d_hidden"])
            num_experts = int(gate_up_weight.shape[0])
            shared_experts = int(config.get("n_shared_experts", config.get("nsharedexperts", 0)))
            output = torch.zeros((num_tokens, d_hidden), dtype=torch.bfloat16, device=hidden_states.device)
            hidden_states_q = _requantize_activation(hidden_states)

            token_ids, expert_ids, weights = _route_entries(topk_ids, topk_weights)
            offsets, counts = _expert_windows(expert_ids, num_experts)
            unique_experts = torch.nonzero(counts > 0, as_tuple=False).flatten()

            for expert in unique_experts.tolist():
                start = int(offsets[expert].item())
                end = start + int(counts[expert].item())
                if end <= start:
                    continue
                expert_gate_w, expert_up_w = _dequant_gate_up_for_expert(
                    gate_up_weight,
                    gate_up_weight_scale,
                    expert,
                    config,
                )
                expert_down_w = _dequant_down_for_expert(
                    down_weight,
                    down_weight_scale,
                    expert,
                    config,
                )
                expert_token_ids = token_ids[start:end]
                expert_inputs = hidden_states_q.index_select(0, expert_token_ids)
                gate = expert_inputs @ expert_gate_w.transpose(0, 1)
                up = expert_inputs @ expert_up_w.transpose(0, 1)
                if CONFIG.get("FUSE_SWIGLU", False):
                    fused = _silu_mul(gate.contiguous(), up.contiguous())
                else:
                    fused = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
                fused_q = _requantize_activation(fused)
                expert_out = fused_q @ expert_down_w.transpose(0, 1)
                if CONFIG.get("WEIGHT_EPILOGUE", True):
                    expert_out = (expert_out * weights[start:end]).to(output.dtype)
                output.index_add_(0, expert_token_ids, expert_out)

            if CONFIG.get("SHARED_EXPERT_FASTPATH", False) and shared_experts > 0:
                output = output.contiguous()

            return output
        """
    ).strip()
    return source.replace("__META__", json.dumps(meta, sort_keys=True)).replace("__CONFIG__", repr(variant))


def render_mixed_mla_anchor(meta: dict[str, object]) -> str:
    source = textwrap.dedent(
        """
        #!POPCORN leaderboard amd-mixed-mla
        #!POPCORN gpu MI355X
        # AGENT_LOOP_META: __META__
        import torch
        from aiter import dtypes as aiter_dtypes
        from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
        from aiter.mla import mla_decode_fwd
        from task import input_t, output_t

        NUM_HEADS = 16
        NUM_KV_HEADS = 1
        KV_LORA_RANK = 512
        QK_ROPE_HEAD_DIM = 64
        QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
        V_HEAD_DIM = KV_LORA_RANK
        SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
        PAGE_SIZE = 1
        NUM_KV_SPLITS = 32
        FP8_DTYPE = aiter_dtypes.fp8


        def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            finfo = torch.finfo(FP8_DTYPE)
            amax = tensor.abs().amax().clamp(min=1e-12)
            scale = amax / finfo.max
            fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
            return fp8_tensor, scale.to(torch.float32).reshape(1)


        def _make_mla_decode_metadata(
            batch_size: int,
            max_q_len: int,
            nhead: int,
            nhead_kv: int,
            q_dtype: torch.dtype,
            kv_dtype: torch.dtype,
            qo_indptr: torch.Tensor,
            kv_indptr: torch.Tensor,
            kv_last_page_len: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            info = get_mla_metadata_info_v1(
                batch_size,
                max_q_len,
                nhead,
                q_dtype,
                kv_dtype,
                is_sparse=False,
                fast_mode=False,
                num_kv_splits=NUM_KV_SPLITS,
                intra_batch_mode=True,
            )
            work = [torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in info]
            (
                work_metadata,
                work_indptr,
                work_info_set,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
            ) = work
            get_mla_metadata_v1(
                qo_indptr,
                kv_indptr,
                kv_last_page_len,
                nhead // nhead_kv,
                nhead_kv,
                True,
                work_metadata,
                work_info_set,
                work_indptr,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                page_size=PAGE_SIZE,
                kv_granularity=max(PAGE_SIZE, 16),
                max_seqlen_qo=max_q_len,
                uni_seqlen_qo=max_q_len,
                fast_mode=False,
                max_split_per_batch=NUM_KV_SPLITS,
                intra_batch_mode=True,
                dtype_q=q_dtype,
                dtype_kv=kv_dtype,
            )
            return {
                "work_meta_data": work_metadata,
                "work_indptr": work_indptr,
                "work_info_set": work_info_set,
                "reduce_indptr": reduce_indptr,
                "reduce_final_map": reduce_final_map,
                "reduce_partial_map": reduce_partial_map,
            }


        def _aiter_mla_decode(
            q: torch.Tensor,
            kv_buffer: torch.Tensor,
            qo_indptr: torch.Tensor,
            kv_indptr: torch.Tensor,
            config: dict,
            q_scale: torch.Tensor | None,
            kv_scale: torch.Tensor | None,
        ) -> torch.Tensor:
            batch_size = int(config["batch_size"])
            nq = int(config["num_heads"])
            nkv = int(config["num_kv_heads"])
            dq = int(config["qk_head_dim"])
            dv = int(config["v_head_dim"])
            q_seq_len = int(config["q_seq_len"])
            total_kv_len = int(kv_indptr[-1].item())
            kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
            kv_buffer_4d = kv_buffer.view(kv_buffer.shape[0], PAGE_SIZE, nkv, kv_buffer.shape[-1])
            kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
            meta = _make_mla_decode_metadata(
                batch_size,
                q_seq_len,
                nq,
                nkv,
                q.dtype,
                kv_buffer.dtype,
                qo_indptr,
                kv_indptr,
                kv_last_page_len,
            )
            out = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")
            mla_decode_fwd(
                q.view(-1, nq, dq),
                kv_buffer_4d,
                out,
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                q_seq_len,
                page_size=PAGE_SIZE,
                nhead_kv=nkv,
                sm_scale=SM_SCALE,
                logit_cap=0.0,
                num_kv_splits=NUM_KV_SPLITS,
                q_scale=q_scale,
                kv_scale=kv_scale,
                intra_batch_mode=True,
                **meta,
            )
            return out


        def custom_kernel(data: input_t) -> output_t:
            q, kv_data, qo_indptr, kv_indptr, config = data
            q_input, q_scale = quantize_fp8(q)
            kv_input, kv_scale = kv_data["fp8"]
            return _aiter_mla_decode(
                q_input,
                kv_input,
                qo_indptr,
                kv_indptr,
                config,
                q_scale=q_scale,
                kv_scale=kv_scale,
            )
        """
    ).strip()
    return source.replace("__META__", json.dumps(meta, sort_keys=True))


def render_mixed_mla_kernel(meta: dict[str, object], variant: dict[str, object]) -> str:
    source = textwrap.dedent(
        """
        #!POPCORN leaderboard amd-mixed-mla
        #!POPCORN gpu MI355X
        # AGENT_LOOP_META: __META__
        import torch
        from aiter import dtypes as aiter_dtypes
        import triton
        import triton.language as tl
        from task import input_t, output_t

        CONFIG = __CONFIG__
        QK_HEAD_DIM = 576
        V_HEAD_DIM = 512
        FP8_DTYPE = aiter_dtypes.fp8


        def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            finfo = torch.finfo(FP8_DTYPE)
            amax = tensor.abs().amax().clamp(min=1e-12)
            scale = amax / finfo.max
            fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
            return fp8_tensor, scale.to(torch.float32).reshape(1)


        def _apply_scale(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            scaled = tensor.to(torch.float32)
            scale_f32 = scale.to(torch.float32)
            if scale_f32.numel() == 1:
                return (scaled * scale_f32.reshape(1)).to(torch.bfloat16)
            shape = tuple(scale_f32.shape) + (1,) * max(scaled.ndim - scale_f32.ndim, 0)
            return (scaled * scale_f32.reshape(shape)).to(torch.bfloat16)


        @triton.jit
        def _mla_decode_kernel(
            q_ptr,
            kv_ptr,
            kv_indptr_ptr,
            out_ptr,
            total_q,
            num_heads,
            q_stride_q,
            q_stride_h,
            q_stride_d,
            kv_stride_t,
            kv_stride_h,
            kv_stride_d,
            out_stride_q,
            out_stride_h,
            out_stride_d,
            sm_scale,
            QK_HEAD_DIM: tl.constexpr,
            V_HEAD_DIM: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_DQ: tl.constexpr,
            BLOCK_DV: tl.constexpr,
        ):
            pid_row = tl.program_id(0)
            pid_v = tl.program_id(1)
            q_idx = pid_row // num_heads
            head_idx = pid_row % num_heads
            if q_idx >= total_q:
                return

            kv_start = tl.load(kv_indptr_ptr + q_idx)
            kv_end = tl.load(kv_indptr_ptr + q_idx + 1)
            v_offsets = pid_v * BLOCK_DV + tl.arange(0, BLOCK_DV)

            q_base = q_ptr + q_idx * q_stride_q + head_idx * q_stride_h
            m_i = -float("inf")
            l_i = 0.0
            acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)

            for block_start in tl.range(0, kv_end - kv_start, BLOCK_N):
                n_offsets = kv_start + block_start + tl.arange(0, BLOCK_N)
                mask_n = n_offsets < kv_end
                scores = tl.zeros((BLOCK_N,), dtype=tl.float32)

                for d_start in tl.range(0, QK_HEAD_DIM, BLOCK_DQ):
                    d_offsets = d_start + tl.arange(0, BLOCK_DQ)
                    mask_d = d_offsets < QK_HEAD_DIM
                    q = tl.load(q_base + d_offsets * q_stride_d, mask=mask_d, other=0.0).to(tl.float32)
                    k_ptrs = kv_ptr + n_offsets[:, None] * kv_stride_t + d_offsets[None, :] * kv_stride_d
                    k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
                    scores += tl.sum(k * q[None, :], axis=1)

                scores *= sm_scale
                scores = tl.where(mask_n, scores, -float("inf"))
                m_ij = tl.max(scores, axis=0)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(scores - m_new)
                l_new = alpha * l_i + tl.sum(p, axis=0)

                v_ptrs = kv_ptr + n_offsets[:, None] * kv_stride_t + v_offsets[None, :] * kv_stride_d
                v = tl.load(v_ptrs, mask=mask_n[:, None] & (v_offsets[None, :] < V_HEAD_DIM), other=0.0).to(tl.float32)
                acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
                m_i = m_new
                l_i = l_new

            acc = acc / l_i
            out_ptrs = out_ptr + q_idx * out_stride_q + head_idx * out_stride_h + v_offsets * out_stride_d
            tl.store(out_ptrs, acc.to(tl.bfloat16), mask=v_offsets < V_HEAD_DIM)


        def custom_kernel(data: input_t) -> output_t:
            q, kv_data, qo_indptr, kv_indptr, config = data
            del qo_indptr
            if int(config["q_seq_len"]) != 1:
                raise RuntimeError("This baseline expects q_seq_len == 1")

            if CONFIG.get("USE_FP8_INPUTS", False):
                q_fp8, q_scale = quantize_fp8(q)
                kv_fp8, kv_scale = kv_data["fp8"]
                q = _apply_scale(q_fp8, q_scale).contiguous()
                kv = _apply_scale(kv_fp8, kv_scale).contiguous()
            else:
                kv = kv_data["bf16"].contiguous()
                q = q.contiguous()
            total_q, num_heads, _ = q.shape
            out = torch.empty((total_q, num_heads, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device)
            grid = (total_q * num_heads, triton.cdiv(V_HEAD_DIM, CONFIG["BLOCK_DV"]))
            _mla_decode_kernel[grid](
                q,
                kv,
                kv_indptr,
                out,
                total_q,
                num_heads,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                kv.stride(0),
                kv.stride(1),
                kv.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                float(config["sm_scale"]),
                QK_HEAD_DIM=QK_HEAD_DIM,
                V_HEAD_DIM=V_HEAD_DIM,
                BLOCK_N=CONFIG["BLOCK_N"],
                BLOCK_DQ=CONFIG["BLOCK_DQ"],
                BLOCK_DV=CONFIG["BLOCK_DV"],
                num_warps=CONFIG["NUM_WARPS"],
                num_stages=CONFIG["NUM_STAGES"],
            )
            return out
        """
    ).strip()
    return source.replace("__META__", json.dumps(meta, sort_keys=True)).replace("__CONFIG__", repr(variant))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()

    context = load_context(Path(args.context))
    parent_meta = load_parent_meta(Path(args.parent))
    problem_key = str(context["problem"]["key"])
    attempt = candidate_attempt(context)
    history = history_entries(context)
    desired_family = context.get("desired_family")
    if not isinstance(desired_family, str):
        desired_family = None
    policy_profile = choose_policy_profile(
        problem_key,
        attempt,
        parent_meta,
        history,
        desired_family=desired_family,
    )
    variant_index, variant = choose_variant(
        problem_key,
        attempt,
        parent_meta,
        history,
        policy_profile=policy_profile,
        desired_family=desired_family,
    )
    submission = render_submission(
        problem_key,
        variant_index,
        variant,
        context,
        attempt,
        policy_profile=policy_profile,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(submission + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
