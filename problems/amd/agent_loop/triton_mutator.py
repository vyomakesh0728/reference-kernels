from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import textwrap


META_RE = re.compile(r"^# AGENT_LOOP_META:\s*(\{.*\})\s*$", re.MULTILINE)


SEARCH_SPACE: dict[str, list[dict[str, object]]] = {
    "mxfp4_mm": [
        {
            "variant_name": "aiter_contract_anchor",
            "family": "anchor",
            "strategy": "contract_anchor",
        },
        {
            "variant_name": "triton_requant_m16n256k128",
            "family": "triton_explore",
            "strategy": "runtime_requant_matmul",
            "BLOCK_M": 16,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_M": 1,
            "NUM_WARPS": 4,
            "NUM_STAGES": 3,
        },
        {
            "variant_name": "triton_requant_m32n256k128",
            "family": "triton_explore",
            "strategy": "runtime_requant_matmul",
            "BLOCK_M": 32,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_M": 2,
            "NUM_WARPS": 4,
            "NUM_STAGES": 3,
        },
        {
            "variant_name": "triton_requant_m64n128k64",
            "family": "triton_explore",
            "strategy": "runtime_requant_matmul",
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 4,
            "NUM_WARPS": 8,
            "NUM_STAGES": 2,
        },
        {
            "variant_name": "triton_contract_m16n256k128",
            "family": "triton_explore",
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
            "variant_name": "triton_contract_m32n256k128",
            "family": "triton_explore",
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
            "variant_name": "triton_contract_m64n128k64",
            "family": "triton_explore",
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
            "variant_name": "triton_contract_m64n256k128",
            "family": "triton_explore",
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
            "variant_name": "triton_contract_m128n128k64",
            "family": "triton_explore",
            "strategy": "contract_bf16_dequant_matmul",
            "CONTRACT_NATIVE": True,
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 4,
            "NUM_WARPS": 8,
            "NUM_STAGES": 3,
        },
    ],
    "moe_mxfp4": [
        {
            "variant_name": "fused_moe_contract_anchor",
            "family": "anchor",
            "strategy": "contract_anchor",
        },
        {
            "variant_name": "routing_swiglu_128",
            "family": "triton_explore",
            "strategy": "routing_prototype",
            "BLOCK_SIZE": 128,
            "NUM_WARPS": 4,
            "SORT_BY_EXPERT": True,
        },
        {
            "variant_name": "routing_swiglu_256",
            "family": "triton_explore",
            "strategy": "routing_prototype",
            "BLOCK_SIZE": 256,
            "NUM_WARPS": 4,
            "SORT_BY_EXPERT": True,
        },
        {
            "variant_name": "routing_swiglu_256_unsorted",
            "family": "triton_explore",
            "strategy": "routing_prototype",
            "BLOCK_SIZE": 256,
            "NUM_WARPS": 4,
            "SORT_BY_EXPERT": False,
        },
        {
            "variant_name": "routing_swiglu_512",
            "family": "triton_explore",
            "strategy": "routing_prototype",
            "BLOCK_SIZE": 512,
            "NUM_WARPS": 8,
            "SORT_BY_EXPERT": True,
        },
        {
            "variant_name": "routing_swiglu_1024",
            "family": "triton_explore",
            "strategy": "routing_prototype",
            "BLOCK_SIZE": 1024,
            "NUM_WARPS": 8,
            "SORT_BY_EXPERT": True,
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
            "family": "triton_explore",
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
            "family": "triton_explore",
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
            "family": "triton_explore",
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
            "family": "triton_explore",
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


POLICY_PROFILES: dict[str, list[dict[str, object]]] = {
    "mxfp4_mm": [
        {
            "name": "contract_repair",
            "family": "triton_explore",
            "focus": "preserve shuffled MXFP4 semantics before tuning",
            "preferred_variants": [
                "triton_requant_m16n256k128",
                "triton_requant_m32n256k128",
                "triton_contract_m16n256k128",
                "triton_contract_m32n256k128",
            ],
            "preferred_strategies": ["runtime_requant_matmul", "contract_bf16_dequant_matmul"],
            "trigger_signals": ["contract_repair", "runtime_repair", "submission_repair"],
        },
        {
            "name": "skinny_longk",
            "family": "triton_explore",
            "focus": "prioritize skinny-M long-K ranked cases first",
            "preferred_variants": [
                "triton_requant_m16n256k128",
                "triton_requant_m32n256k128",
                "triton_contract_m16n256k128",
                "triton_contract_m32n256k128",
                "triton_contract_m64n256k128",
            ],
            "preferred_strategies": ["runtime_requant_matmul", "contract_bf16_dequant_matmul"],
            "trigger_signals": ["throughput_shift"],
        },
        {
            "name": "balanced_tiles",
            "family": "triton_explore",
            "focus": "cover wider shapes once the skinny path is stable",
            "preferred_variants": [
                "triton_requant_m64n128k64",
                "triton_contract_m64n128k64",
                "triton_contract_m128n128k64",
                "triton_contract_m64n256k128",
            ],
            "preferred_strategies": ["runtime_requant_matmul", "contract_bf16_dequant_matmul"],
            "trigger_signals": ["throughput_shift", "latency_repair"],
        },
    ],
    "moe_mxfp4": [
        {
            "name": "contract_repair",
            "family": "triton_explore",
            "focus": "stabilize routing and shuffled-weight semantics",
            "preferred_variants": [
                "routing_swiglu_128",
                "routing_swiglu_256",
            ],
            "preferred_strategies": ["routing_prototype"],
            "trigger_signals": ["contract_repair", "runtime_repair", "submission_repair"],
        },
        {
            "name": "routing_balance",
            "family": "triton_explore",
            "focus": "test safer grouped-routing schedules first",
            "preferred_variants": [
                "routing_swiglu_256",
                "routing_swiglu_256_unsorted",
                "routing_swiglu_512",
            ],
            "preferred_strategies": ["routing_prototype"],
            "trigger_signals": ["throughput_shift"],
        },
        {
            "name": "routing_throughput",
            "family": "triton_explore",
            "focus": "push larger routing/fusion blocks after correctness is stable",
            "preferred_variants": [
                "routing_swiglu_512",
                "routing_swiglu_1024",
            ],
            "preferred_strategies": ["routing_prototype"],
            "trigger_signals": ["throughput_shift", "latency_repair"],
        },
    ],
    "mixed_mla": [
        {
            "name": "contract_repair",
            "family": "triton_explore",
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
            "family": "triton_explore",
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
            "family": "triton_explore",
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
    if problem_key == "mxfp4_mm":
        if variant.get("family") == "anchor":
            return render_mxfp4_mm_anchor(meta)
        return render_mxfp4_mm_triton(meta, variant)
    if problem_key == "moe_mxfp4":
        if variant.get("family") == "anchor":
            return render_moe_mxfp4_anchor(meta)
        return render_moe_mxfp4_triton(meta, variant)
    if problem_key == "mixed_mla":
        if variant.get("family") == "anchor":
            return render_mixed_mla_anchor(meta)
        return render_mixed_mla_triton(meta, variant)
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


def render_mxfp4_mm_triton(meta: dict[str, object], variant: dict[str, object]) -> str:
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


def render_moe_mxfp4_anchor(meta: dict[str, object]) -> str:
    source = textwrap.dedent(
        """
        #!POPCORN leaderboard amd-moe-mxfp4
        #!POPCORN gpu MI355X
        # AGENT_LOOP_META: __META__
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe
        from task import input_t, output_t


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
                hidden_pad=hidden_pad,
                intermediate_pad=intermediate_pad,
            )
        """
    ).strip()
    return source.replace("__META__", json.dumps(meta, sort_keys=True))


def render_moe_mxfp4_triton(meta: dict[str, object], variant: dict[str, object]) -> str:
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


        def _load_weights(
            gate_up_weight: torch.Tensor,
            down_weight: torch.Tensor,
            gate_up_weight_scale: torch.Tensor,
            down_weight_scale: torch.Tensor,
            config: dict,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            experts = gate_up_weight.shape[0]
            d_hidden = int(config["d_hidden"])
            d_expert = int(config["d_expert"])

            gate_w = []
            up_w = []
            down_w = []
            for expert in range(experts):
                gate_up = _dequant_matrix(
                    gate_up_weight[expert],
                    gate_up_weight_scale[expert],
                    rows=2 * d_expert,
                    cols=d_hidden,
                )
                gate_part, up_part = gate_up.chunk(2, dim=0)
                gate_w.append(gate_part.contiguous())
                up_w.append(up_part.contiguous())
                down_part = _dequant_matrix(
                    down_weight[expert],
                    down_weight_scale[expert],
                    rows=d_hidden,
                    cols=d_expert,
                )
                down_w.append(down_part.contiguous())

            return (
                torch.stack(gate_w),
                torch.stack(up_w),
                torch.stack(down_w),
            )


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

            gate_w, up_w, down_w = _load_weights(
                gate_up_weight,
                down_weight,
                gate_up_weight_scale,
                down_weight_scale,
                config,
            )

            num_tokens = hidden_states.shape[0]
            total_top_k = topk_ids.shape[1]
            output = torch.zeros((num_tokens, int(config["d_hidden"])), dtype=torch.bfloat16, device=hidden_states.device)

            token_ids = torch.arange(num_tokens, device=hidden_states.device, dtype=torch.int64).repeat_interleave(total_top_k)
            expert_ids = topk_ids.reshape(-1).to(torch.int64)
            weights = topk_weights.reshape(-1, 1).to(torch.bfloat16)

            order = torch.argsort(expert_ids) if CONFIG["SORT_BY_EXPERT"] else torch.arange(expert_ids.numel(), device=expert_ids.device)
            token_ids = token_ids[order]
            expert_ids = expert_ids[order]
            weights = weights[order]
            counts = torch.bincount(expert_ids, minlength=gate_w.shape[0])

            start = 0
            for expert, count in enumerate(counts.tolist()):
                if count == 0:
                    continue
                end = start + count
                expert_token_ids = token_ids[start:end]
                expert_inputs = hidden_states.index_select(0, expert_token_ids)
                gate = expert_inputs @ gate_w[expert].transpose(0, 1)
                up = expert_inputs @ up_w[expert].transpose(0, 1)
                fused = _silu_mul(gate.contiguous(), up.contiguous())
                expert_out = fused @ down_w[expert].transpose(0, 1)
                expert_out = expert_out * weights[start:end]
                output.index_add_(0, expert_token_ids, expert_out)
                start = end

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


def render_mixed_mla_triton(meta: dict[str, object], variant: dict[str, object]) -> str:
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
