from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shutil

from .config import AppConfig
from .harness import HarnessSummary, KernelHarness


VARIANT_NAME_RE = re.compile(r'"variant_name":\s*"([^"]+)"')
HIP_SRC_RE = re.compile(r'HIP_SRC = r""".*?^"""', re.DOTALL | re.MULTILINE)
CPP_WRAPPER_RE = re.compile(r'CPP_WRAPPER = """.*?^"""', re.DOTALL | re.MULTILINE)


@dataclass(frozen=True)
class HandrolledMove:
    name: str
    description: str
    family: str = "generic"


class HandrolledOptimizer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.harness = KernelHarness(config)

    def run_campaign(
        self,
        *,
        problem_key: str,
        rounds: int,
        stages: list[str],
        sleep_seconds: float = 0.0,
        leaderboard_on_improve: bool = False,
    ) -> list[dict[str, object]]:
        if problem_key != "mxfp4_mm":
            raise RuntimeError("hand-rolled optimizer currently supports only mxfp4_mm")

        state = self._load_state(problem_key)
        if state is None:
            state = self._initialize_state(problem_key=problem_key, stages=stages)
        else:
            self._ensure_failure_memory(problem_key, state)

        moves = self._moves_for_problem(problem_key)
        records: list[dict[str, object]] = []

        for _ in range(rounds):
            move = self._select_next_move(moves, state)
            state["round"] += 1

            round_dir = self._round_dir(problem_key, int(state["round"]), move.name)
            round_dir.mkdir(parents=True, exist_ok=False)
            candidate_path = round_dir / "submission.py"
            plan_path = round_dir / "plan.json"

            parent_path = Path(str(state["working_path"]))
            parent_text = parent_path.read_text(encoding="utf-8")
            candidate_text = self._apply_move(problem_key, move, parent_text)
            candidate_path.write_text(candidate_text, encoding="utf-8")
            plan_path.write_text(
                json.dumps(
                    {
                        "problem": problem_key,
                        "round": state["round"],
                        "move": move.name,
                        "description": move.description,
                        "parent_path": str(parent_path),
                        "candidate_path": str(candidate_path),
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            run_dir = self.harness.create_run(
                problem_key,
                source_path=candidate_path,
                stages=stages,
                family="handrolled_hip",
                label=f"handrolled-{move.name}",
            )
            summary = self.harness.resume_run(run_dir)
            benchmark_objective = self._stage_objective(summary, "benchmark")
            self._update_failure_memory(problem_key, state, move, summary)
            keep = self._should_keep(summary, benchmark_objective, float(state["best_objective_ns"]))

            if keep:
                shutil.copy2(candidate_path, parent_path)
                state["working_path"] = str(parent_path)
                state["best_objective_ns"] = benchmark_objective
                state["best_run_dir"] = str(run_dir)
                state["best_move"] = move.name
                state["best_round"] = int(state["round"])
                if leaderboard_on_improve:
                    self._run_optional_leaderboard(problem_key, parent_path, move.name)

            record = {
                "round": int(state["round"]),
                "move": move.name,
                "family": move.family,
                "description": move.description,
                "candidate_path": str(candidate_path),
                "run_dir": str(run_dir),
                "test_status": self._stage_status(summary, "test"),
                "benchmark_status": self._stage_status(summary, "benchmark"),
                "benchmark_objective_ns": benchmark_objective,
                "decision": "keep" if keep else "revert",
                "best_objective_ns": float(state["best_objective_ns"]),
                "blocked_families": list(state.get("blocked_families", [])),
                "created_at": datetime.now(UTC).isoformat(),
            }
            self._append_journal(problem_key, record)
            self._write_state(problem_key, state)
            records.append(record)

            if sleep_seconds > 0:
                import time

                time.sleep(sleep_seconds)

        return records

    def _initialize_state(self, *, problem_key: str, stages: list[str]) -> dict[str, object]:
        problem = self.config.require_problem(problem_key)
        working_dir = self._working_dir(problem_key)
        working_dir.mkdir(parents=True, exist_ok=True)

        baseline_run = self.harness.create_run(
            problem_key,
            source_path=problem.submission_path,
            stages=stages,
            family="handrolled_hip",
            label="handrolled-baseline",
        )
        summary = self.harness.resume_run(baseline_run)
        baseline_objective = self._stage_objective(summary, "benchmark")
        if baseline_objective is None:
            raise RuntimeError("baseline benchmark did not complete successfully for hand-rolled loop")

        state = {
            "problem": problem_key,
            "created_at": datetime.now(UTC).isoformat(),
            "round": 0,
            "move_cursor": 0,
            "working_path": str(problem.submission_path),
            "best_objective_ns": baseline_objective,
            "best_run_dir": str(baseline_run),
            "best_move": "baseline",
            "best_round": 0,
            "family_failure_counts": {},
            "blocked_families": [],
        }
        self._write_state(problem_key, state)
        self._append_journal(
            problem_key,
            {
                "round": 0,
                "move": "baseline",
                "family": "baseline",
                "description": "baseline benchmark for hand-rolled campaign",
                "candidate_path": str(problem.submission_path),
                "run_dir": str(baseline_run),
                "test_status": self._stage_status(summary, "test"),
                "benchmark_status": self._stage_status(summary, "benchmark"),
                "benchmark_objective_ns": baseline_objective,
                "decision": "keep",
                "best_objective_ns": baseline_objective,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        return state

    def _moves_for_problem(self, problem_key: str) -> list[HandrolledMove]:
        if problem_key != "mxfp4_mm":
            return []
        return [
            HandrolledMove(
                name="deaiter_exact_m16_scaled_mfma",
                description="Replace the exact m=16 AITER branch with the native FP4 scaled-MFMA 16x16x128 path while preserving the corrected live contract reconstruction.",
                family="deaiter_m16",
            ),
            HandrolledMove(
                name="deaiter_exact_m64_scaled_split16",
                description="Replace the exact m=64 AITER branch with four native FP4 scaled-MFMA 16-row slices so the hot regime stays on a custom low-precision path instead of asm dispatch.",
                family="deaiter_m64",
            ),
            HandrolledMove(
                name="deaiter_exact_m256_native_hip",
                description="Replace the large-shape torch.mm fallback for exact m=256 with the current native HIP tiled path to start removing non-native engines from the trunk.",
                family="deaiter_m256",
            ),
            HandrolledMove(
                name="deaiter_native_regime_dispatch_scaled_v1",
                description="Remove the AITER exact-m16/exact-m64 branches and the large torch.mm fallback together, routing the current benchmark regimes through native scaled-MFMA or native HIP kernels only.",
                family="deaiter_combo",
            ),
        ]

    def _apply_move(self, problem_key: str, move: HandrolledMove, source_text: str) -> str:
        if problem_key != "mxfp4_mm":
            raise RuntimeError(f"unsupported problem for hand-rolled move application: {problem_key}")

        updated = source_text
        updated = self._rename_variant(updated, move.name)

        if move.name == "lds_pad_banks":
            return self._apply_lds_padding(updated)

        if move.name == "deaiter_exact_m16_scaled_mfma":
            return self._apply_deaiter_exact_m16_scaled_mfma(updated)

        if move.name == "deaiter_exact_m64_scaled_split16":
            return self._apply_deaiter_exact_m64_scaled_split16(updated)

        if move.name == "deaiter_exact_m256_native_hip":
            return self._apply_deaiter_exact_m256_native_hip(updated)

        if move.name == "deaiter_native_regime_dispatch_scaled_v1":
            return self._apply_deaiter_native_regime_dispatch_scaled_v1(updated)

        if move.name == "b_tile_transpose_safe":
            return self._apply_b_tile_transpose(updated)

        if move.name == "double_buffer_prefetch":
            return self._apply_double_buffer_prefetch(updated)

        if move.name == "double_buffer_tiles":
            return self._apply_double_buffer(updated)

        raise RuntimeError(f"unknown hand-rolled move: {move.name}")

    def _apply_shape_dispatch(self, source_text: str, *, medium_threshold: int) -> str:
        if "def _select_kernel_regime(" in source_text:
            source_text = re.sub(
                r"if m <= \d+:\n        return \"tiny_m\"\n    if m <= \d+:\n        return \"medium_m\"\n",
                f"if m <= 16:\n        return \"tiny_m\"\n    if m <= {medium_threshold}:\n        return \"medium_m\"\n",
                source_text,
                count=1,
            )
            return source_text

        helper = f'''
def _select_kernel_regime(m: int, k: int) -> str:
    if m <= 16:
        return "tiny_m"
    if m <= {medium_threshold}:
        return "medium_m"
    return "fallback"


'''
        source_text = source_text.replace("\n\ndef custom_kernel(data: input_t) -> output_t:\n", f"\n{helper}def custom_kernel(data: input_t) -> output_t:\n", 1)
        dispatch_block = f'''    regime = _select_kernel_regime(a_in.shape[0], a_in.shape[1])
    if regime == "medium_m":
        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
        _module().mxfp4_mm_hip(a_in, b_in, c)
        return c
    if regime == "fallback":
        return torch.mm(a_in, b_in.t()).to(torch.bfloat16)
'''
        source_text = source_text.replace(
            '    c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)\n    _module().mxfp4_mm_hip(a_in, b_in, c)\n    return c\n',
            dispatch_block + '    c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)\n    _module().mxfp4_mm_hip(a_in, b_in, c)\n    return c\n',
            1,
        )
        return source_text

    def _replace_exact_aiter_guard(self, source_text: str, guard: str) -> str:
        updated, count = re.subn(
            r"if a\.shape\[0\] in \(16, 64\):",
            guard,
            source_text,
            count=1,
        )
        if count != 1:
            raise RuntimeError("failed to locate exact-m16/m64 AITER guard")
        return updated

    def _inject_after_contract_asserts(self, source_text: str, block: str) -> str:
        anchor = '    torch._assert(b_scale_sh.numel() > 0, "B_scale_sh must be present for the live contract")\n'
        if anchor not in source_text:
            raise RuntimeError("failed to locate contract assert anchor for custom_kernel injection")
        return source_text.replace(anchor, anchor + block, 1)

    def _apply_deaiter_exact_m16_scaled_mfma(self, source_text: str) -> str:
        updated = self._apply_mfma_scale_exact_m16(source_text)
        updated = self._replace_exact_aiter_guard(updated, "if a.shape[0] == 64:")
        return updated

    def _apply_deaiter_exact_m64_scaled_split16(self, source_text: str) -> str:
        updated = self._apply_mfma_scale_exact_m16(source_text)
        if "import json\n" not in updated:
            updated = updated.replace("import hashlib\n", "import hashlib\nimport json\n", 1)
        updated = self._replace_exact_aiter_guard(updated, "if a.shape[0] == 16:")
        helper_anchor = "\ndef custom_kernel(data: input_t) -> output_t:\n"
        if "def _log_m64_numeric_diagnostics(" not in updated:
            helper = '''

def _log_m64_numeric_diagnostics(
    *,
    a_shape: tuple[int, int],
    b_rows: int,
    candidate: torch.Tensor,
    anchor: torch.Tensor,
) -> None:
    seen = globals().setdefault("_M64_DIAG_SEEN", set())
    shape_key = (int(a_shape[0]), int(b_rows), int(a_shape[1]))
    if shape_key in seen:
        return
    seen.add(shape_key)

    candidate_f32 = candidate.to(torch.float32)
    anchor_f32 = anchor.to(torch.float32)
    abs_diff = (candidate_f32 - anchor_f32).abs()
    rel_diff = abs_diff / anchor_f32.abs().clamp_min(1e-6)

    def bucket_counts(tensor: torch.Tensor, thresholds: list[float]) -> dict[str, int]:
        out: dict[str, int] = {}
        prev = 0
        total = tensor.numel()
        for threshold in thresholds:
            count = int((tensor <= threshold).sum().item())
            out[f"<= {threshold:g}"] = count - prev
            prev = count
        out[f"> {thresholds[-1]:g}"] = total - prev
        return out

    tol_counts = {}
    for atol, rtol in ((1e-3, 1e-3), (1e-3, 1e-2), (1e-3, 2e-2), (5e-3, 5e-2)):
        ok = abs_diff <= (atol + rtol * anchor_f32.abs())
        tol_counts[f"atol={atol:g},rtol={rtol:g}"] = int(ok.sum().item())

    payload = {
        "m64_scaled_diag": {
            "shape": {"m": shape_key[0], "n": shape_key[1], "k": shape_key[2]},
            "mae": float(abs_diff.mean().item()),
            "max_abs": float(abs_diff.max().item()),
            "mean_rel": float(rel_diff.mean().item()),
            "max_rel": float(rel_diff.max().item()),
            "abs_hist": bucket_counts(abs_diff, [1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 4.0, 8.0]),
            "rel_hist": bucket_counts(rel_diff, [1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]),
            "tol_counts": tol_counts,
            "numel": int(abs_diff.numel()),
        }
    }
    print(json.dumps(payload, sort_keys=True))
'''
            updated = updated.replace(helper_anchor, helper + helper_anchor, 1)
        block = """    if a.shape[0] == 64 and (a.shape[1] % 128) == 0 and (b.shape[0] % 16) == 0:\n        b_packed, b_scale = _get_b_contract_mfma_fp4(b, b_q, b_scale_sh)\n        c = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)\n        inflight = globals().setdefault(\"_MFMA_SCALE_INFLIGHT\", [])\n        chunks = []\n        for offset in (0, 16, 32, 48):\n            a_slice = a.narrow(0, offset, 16).contiguous()\n            a_chunk, a_scale_chunk = _get_a_contract_mfma_fp4(a_slice, b, b_q, b_scale_sh)\n            c_chunk = c.narrow(0, offset, 16)\n            chunks.append((a_chunk, a_scale_chunk, c_chunk))\n        inflight.append((chunks, b_packed, b_scale))\n        if len(inflight) > 64:\n            del inflight[:-64]\n        for a_chunk, a_scale_chunk, c_chunk in chunks:\n            _module().mxfp4_mm_hip_mfma_scale_exact_m16(a_chunk, b_packed, a_scale_chunk, b_scale, c_chunk)\n        a_q_sh, a_scale_sh = _get_corrected_a_preshuffle(a, b, b_q, b_scale_sh)\n        anchor = aiter.gemm_a4w4(\n            a_q_sh.contiguous(),\n            b_shuffle.contiguous(),\n            a_scale_sh.contiguous(),\n            b_scale_sh.contiguous(),\n            dtype=dtypes.bf16,\n            bpreshuffle=True,\n        )\n        _log_m64_numeric_diagnostics(\n            a_shape=(int(a.shape[0]), int(a.shape[1])),\n            b_rows=int(b.shape[0]),\n            candidate=c,\n            anchor=anchor,\n        )\n        return anchor\n"""
        return self._inject_after_contract_asserts(updated, block)

    def _apply_deaiter_exact_m256_native_hip(self, source_text: str) -> str:
        block = """    if a.shape[0] == 256:\n        a_in, b_in = _reference_oracle_inputs(a, b, b_q, b_scale_sh)\n        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)\n        _module().mxfp4_mm_hip(a_in, b_in, c)\n        return c\n"""
        return self._inject_after_contract_asserts(source_text, block)

    def _apply_deaiter_native_regime_dispatch_scaled_v1(self, source_text: str) -> str:
        updated = self._apply_mfma_scale_exact_m16(source_text)
        updated = self._replace_exact_aiter_guard(updated, "if False:")
        block = """    if a.shape[0] == 16 and (a.shape[1] % 128) == 0 and (b.shape[0] % 16) == 0:\n        a_packed, a_scale = _get_a_contract_mfma_fp4(a, b, b_q, b_scale_sh)\n        b_packed, b_scale = _get_b_contract_mfma_fp4(b, b_q, b_scale_sh)\n        c = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)\n        inflight = globals().setdefault(\"_MFMA_SCALE_INFLIGHT\", [])\n        inflight.append((a_packed, a_scale, b_packed, b_scale))\n        if len(inflight) > 64:\n            del inflight[:-64]\n        _module().mxfp4_mm_hip_mfma_scale_exact_m16(a_packed, b_packed, a_scale, b_scale, c)\n        return c\n    if a.shape[0] == 64 and (a.shape[1] % 128) == 0 and (b.shape[0] % 16) == 0:\n        a_packed, a_scale = _get_a_contract_mfma_fp4(a, b, b_q, b_scale_sh)\n        b_packed, b_scale = _get_b_contract_mfma_fp4(b, b_q, b_scale_sh)\n        c = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)\n        inflight = globals().setdefault(\"_MFMA_SCALE_INFLIGHT\", [])\n        chunks = []\n        for offset in (0, 16, 32, 48):\n            a_chunk = a_packed[offset:offset + 16].contiguous()\n            a_scale_chunk = a_scale[offset:offset + 16].contiguous()\n            c_chunk = c.narrow(0, offset, 16)\n            chunks.append((a_chunk, a_scale_chunk, c_chunk))\n        inflight.append((chunks, b_packed, b_scale))\n        if len(inflight) > 64:\n            del inflight[:-64]\n        for a_chunk, a_scale_chunk, c_chunk in chunks:\n            _module().mxfp4_mm_hip_mfma_scale_exact_m16(a_chunk, b_packed, a_scale_chunk, b_scale, c_chunk)\n        return c\n    if a.shape[0] == 256:\n        a_in, b_in = _reference_oracle_inputs(a, b, b_q, b_scale_sh)\n        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)\n        _module().mxfp4_mm_hip(a_in, b_in, c)\n        return c\n"""
        return self._inject_after_contract_asserts(updated, block)

    def _apply_medium_variant(self, source_text: str, *, tile_n: int, tile_k: int, variant_suffix: str) -> str:
        if f"mxfp4_mm_kernel_{variant_suffix}" in source_text:
            source_text = re.sub(
                r"constexpr int TILE_N_" + variant_suffix.upper() + r" = \d+;",
                f"constexpr int TILE_N_{variant_suffix.upper()} = {tile_n};",
                source_text,
            )
            source_text = re.sub(
                r"constexpr int TILE_K_" + variant_suffix.upper() + r" = \d+;",
                f"constexpr int TILE_K_{variant_suffix.upper()} = {tile_k};",
                source_text,
            )
            return source_text

        anchor = 'void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c) {\n'
        kernel = f'''
constexpr int TILE_N_{variant_suffix.upper()} = {tile_n};
constexpr int TILE_K_{variant_suffix.upper()} = {tile_k};

__global__ void mxfp4_mm_kernel_{variant_suffix}(
    const float* a,
    const float* b,
    __hip_bfloat16* c,
    int m,
    int n,
    int k
) {{
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int row = blockIdx.y * TILE_M + local_y;
    const int col = blockIdx.x * TILE_N_{variant_suffix.upper()} + local_x;

    double acc = 0.0;
    __shared__ float a_tile[TILE_M][TILE_K_{variant_suffix.upper()}];
    __shared__ float b_tile[TILE_N_{variant_suffix.upper()}][TILE_K_{variant_suffix.upper()}];

    for (int tile_k = 0; tile_k < k; tile_k += TILE_K_{variant_suffix.upper()}) {{
        if (local_x < TILE_K_{variant_suffix.upper()} / 4) {{
            const int k_vec = local_x * 4;
            const int global_k = tile_k + k_vec;
            if (row < m && global_k + 3 < k) {{
                const float4 vec = *reinterpret_cast<const float4*>(a + row * k + global_k);
                a_tile[local_y][k_vec + 0] = vec.x;
                a_tile[local_y][k_vec + 1] = vec.y;
                a_tile[local_y][k_vec + 2] = vec.z;
                a_tile[local_y][k_vec + 3] = vec.w;
            }} else {{
                #pragma unroll
                for (int lane = 0; lane < 4; ++lane) {{
                    const int kk = global_k + lane;
                    a_tile[local_y][k_vec + lane] = (row < m && kk < k) ? a[row * k + kk] : 0.0f;
                }}
            }}
        }}

        {{
            const int k_vec = local_y * 4;
            const int global_k = tile_k + k_vec;
            if (col < n && global_k + 3 < k) {{
                const float4 vec = *reinterpret_cast<const float4*>(b + col * k + global_k);
                b_tile[local_x][k_vec + 0] = vec.x;
                b_tile[local_x][k_vec + 1] = vec.y;
                b_tile[local_x][k_vec + 2] = vec.z;
                b_tile[local_x][k_vec + 3] = vec.w;
            }} else {{
                #pragma unroll
                for (int lane = 0; lane < 4; ++lane) {{
                    const int kk = global_k + lane;
                    b_tile[local_x][k_vec + lane] = (col < n && kk < k) ? b[col * k + kk] : 0.0f;
                }}
            }}
        }}

        __syncthreads();

        if (row < m && col < n) {{
            #pragma unroll 4
            for (int kk = 0; kk < TILE_K_{variant_suffix.upper()}; ++kk) {{
                acc += static_cast<double>(a_tile[local_y][kk]) * static_cast<double>(b_tile[local_x][kk]);
            }}
        }}

        __syncthreads();
    }}

    if (row < m && col < n) {{
        c[row * n + col] = static_cast<__hip_bfloat16>(static_cast<float>(acc));
    }}
}}

void mxfp4_mm_hip_{variant_suffix}(torch::Tensor a, torch::Tensor b, torch::Tensor c) {{
    const int m = static_cast<int>(a.size(0));
    const int n = static_cast<int>(b.size(0));
    const int k = static_cast<int>(a.size(1));
    dim3 block(TILE_N_{variant_suffix.upper()}, TILE_M);
    dim3 grid((n + TILE_N_{variant_suffix.upper()} - 1) / TILE_N_{variant_suffix.upper()}, (m + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_{variant_suffix},
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
}}

'''
        source_text = source_text.replace(anchor, kernel + anchor, 1)
        source_text = source_text.replace(
            'void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c);',
            f'void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c);\nvoid mxfp4_mm_hip_{variant_suffix}(torch::Tensor a, torch::Tensor b, torch::Tensor c);',
            1,
        )
        source_text = source_text.replace(
            '    if regime == "medium_m":\n        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)\n        _module().mxfp4_mm_hip(a_in, b_in, c)\n        return c\n',
            f'    if regime == "medium_m":\n        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)\n        _module().mxfp4_mm_hip_{variant_suffix}(a_in, b_in, c)\n        return c\n',
            1,
        )
        return source_text

    def _apply_tiny_variant(self, source_text: str, *, tile_n: int, tile_k: int, variant_suffix: str) -> str:
        if f"mxfp4_mm_kernel_{variant_suffix}" in source_text:
            source_text = re.sub(
                r"constexpr int TILE_N_" + variant_suffix.upper() + r" = \d+;",
                f"constexpr int TILE_N_{variant_suffix.upper()} = {tile_n};",
                source_text,
            )
            source_text = re.sub(
                r"constexpr int TILE_K_" + variant_suffix.upper() + r" = \d+;",
                f"constexpr int TILE_K_{variant_suffix.upper()} = {tile_k};",
                source_text,
            )
            return source_text

        anchor = 'void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c) {\n'
        kernel = f'''
constexpr int TILE_N_{variant_suffix.upper()} = {tile_n};
constexpr int TILE_K_{variant_suffix.upper()} = {tile_k};

__global__ void mxfp4_mm_kernel_{variant_suffix}(
    const float* a,
    const float* b,
    __hip_bfloat16* c,
    int m,
    int n,
    int k
) {{
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int row = blockIdx.y * TILE_M + local_y;
    const int col = blockIdx.x * TILE_N_{variant_suffix.upper()} + local_x;

    double acc = 0.0;
    __shared__ float a_tile[TILE_M][TILE_K_{variant_suffix.upper()}];
    __shared__ float b_tile[TILE_N_{variant_suffix.upper()}][TILE_K_{variant_suffix.upper()}];

    for (int tile_k = 0; tile_k < k; tile_k += TILE_K_{variant_suffix.upper()}) {{
        if (local_x < TILE_K_{variant_suffix.upper()} / 4) {{
            const int k_vec = local_x * 4;
            const int global_k = tile_k + k_vec;
            if (row < m && global_k + 3 < k) {{
                const float4 vec = *reinterpret_cast<const float4*>(a + row * k + global_k);
                a_tile[local_y][k_vec + 0] = vec.x;
                a_tile[local_y][k_vec + 1] = vec.y;
                a_tile[local_y][k_vec + 2] = vec.z;
                a_tile[local_y][k_vec + 3] = vec.w;
            }} else {{
                #pragma unroll
                for (int lane = 0; lane < 4; ++lane) {{
                    const int kk = global_k + lane;
                    a_tile[local_y][k_vec + lane] = (row < m && kk < k) ? a[row * k + kk] : 0.0f;
                }}
            }}
        }}

        {{
            const int k_vec = local_y * 4;
            const int global_k = tile_k + k_vec;
            if (col < n && global_k + 3 < k) {{
                const float4 vec = *reinterpret_cast<const float4*>(b + col * k + global_k);
                b_tile[local_x][k_vec + 0] = vec.x;
                b_tile[local_x][k_vec + 1] = vec.y;
                b_tile[local_x][k_vec + 2] = vec.z;
                b_tile[local_x][k_vec + 3] = vec.w;
            }} else {{
                #pragma unroll
                for (int lane = 0; lane < 4; ++lane) {{
                    const int kk = global_k + lane;
                    b_tile[local_x][k_vec + lane] = (col < n && kk < k) ? b[col * k + kk] : 0.0f;
                }}
            }}
        }}

        __syncthreads();

        if (row < m && col < n) {{
            #pragma unroll 4
            for (int kk = 0; kk < TILE_K_{variant_suffix.upper()}; ++kk) {{
                acc += static_cast<double>(a_tile[local_y][kk]) * static_cast<double>(b_tile[local_x][kk]);
            }}
        }}

        __syncthreads();
    }}

    if (row < m && col < n) {{
        c[row * n + col] = static_cast<__hip_bfloat16>(static_cast<float>(acc));
    }}
}}

void mxfp4_mm_hip_{variant_suffix}(torch::Tensor a, torch::Tensor b, torch::Tensor c) {{
    const int m = static_cast<int>(a.size(0));
    const int n = static_cast<int>(b.size(0));
    const int k = static_cast<int>(a.size(1));
    dim3 block(TILE_N_{variant_suffix.upper()}, TILE_M);
    dim3 grid((n + TILE_N_{variant_suffix.upper()} - 1) / TILE_N_{variant_suffix.upper()}, (m + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_{variant_suffix},
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
}}

'''
        source_text = source_text.replace(anchor, kernel + anchor, 1)
        source_text = source_text.replace(
            'void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c);',
            f'void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c);\nvoid mxfp4_mm_hip_{variant_suffix}(torch::Tensor a, torch::Tensor b, torch::Tensor c);',
            1,
        )
        source_text = source_text.replace(
            '    c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)\n    _module().mxfp4_mm_hip(a_in, b_in, c)\n    return c\n',
            f'    c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)\n    _module().mxfp4_mm_hip_{variant_suffix}(a_in, b_in, c)\n    return c\n',
            1,
        )
        return source_text

    def _apply_lds_padding(self, source_text: str) -> str:
        updated = source_text.replace(
            "__shared__ float a_tile[TILE_M][TILE_K];",
            "__shared__ float a_tile[TILE_M][TILE_K + 1];",
        )
        updated = updated.replace(
            "__shared__ float b_tile[TILE_N][TILE_K];",
            "__shared__ float b_tile[TILE_N][TILE_K + 1];",
        )
        return updated

    def _apply_mfma_medium_bridge(
        self,
        source_text: str,
        *,
        max_medium_m: int,
        exact_m: int | None = None,
        variant: str = "bf16x32",
    ) -> str:
        updated = source_text
        updated = updated.replace(
            'void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c);',
            'void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c);\nvoid mxfp4_mm_hip_mfma_medium(torch::Tensor a, torch::Tensor b, torch::Tensor c);',
            1,
        )
        updated = updated.replace(
            '            functions=["mxfp4_mm_hip"],',
            '            functions=["mxfp4_mm_hip", "mxfp4_mm_hip_mfma_medium"],',
            1,
        )

        updated = updated.replace(
            "_B_CACHE: dict[tuple[object, ...], tuple[dict[float, tuple[str, float, float]], torch.Tensor]] = {}\n",
            "_B_CACHE: dict[tuple[object, ...], tuple[dict[float, tuple[str, float, float]], torch.Tensor]] = {}\n_B_BF16_CACHE: dict[tuple[object, ...], torch.Tensor] = {}\n",
            1,
        )

        helper_anchor = "\ndef _reference_oracle_inputs(\n"
        if "def _get_b_contract_bf16(" not in updated:
            helper = '''

def _get_b_contract_bf16(
    b: torch.Tensor,
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> torch.Tensor:
    key = _b_contract_cache_key(b, b_q, b_scale_sh)
    cached = _B_BF16_CACHE.get(key)
    if cached is not None:
        return cached

    _, b_ref = _get_b_contract(b, b_q, b_scale_sh)
    b_ref_bf16 = b_ref.to(torch.bfloat16).contiguous()
    if len(_B_BF16_CACHE) >= 4:
        _B_BF16_CACHE.clear()
    _B_BF16_CACHE[key] = b_ref_bf16
    return b_ref_bf16
'''
            updated = updated.replace(helper_anchor, helper + helper_anchor, 1)

        kernel_anchor = "void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c) {\n"
        if "__global__ void mxfp4_mm_kernel_mfma_medium(" not in updated:
            if variant == "bf16x16_1k":
                kernel = r'''

using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
using bit16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
typedef bit16x4 _B16x4;
typedef struct _B16x8
{
    _B16x4 xy[2];
} _B16x8;

__device__ __forceinline__ floatx4 gcn_mfma16x16x32_bf16_instr(
    const _B16x8& inpA,
    const _B16x8& inpB,
    const floatx4& inpC
) {
    bit16x8 tmpA = __builtin_shufflevector(inpA.xy[0], inpA.xy[1], 0, 1, 2, 3, 4, 5, 6, 7);
    bit16x8 tmpB = __builtin_shufflevector(inpB.xy[0], inpB.xy[1], 0, 1, 2, 3, 4, 5, 6, 7);
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(tmpA, tmpB, inpC, 0, 0, 0);
}

__device__ __forceinline__ floatx4 gcn_mfma16x16x16bf16_1k_instr(
    const _B16x4& inpA,
    const _B16x4& inpB,
    const floatx4& inpC
) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(inpA, inpB, inpC, 0, 0, 0);
}

template <typename input_t>
__global__ void mxfp4_mm_kernel_bf16_scalar(
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

    float acc = 0.0f;
    __shared__ input_t a_tile[TILE_M][TILE_K + 1];
    __shared__ input_t b_tile[TILE_N][TILE_K + 1];

    for (int tile_k = 0; tile_k < k; tile_k += TILE_K) {
        if (local_x < TILE_K / 4) {
            const int k_vec = local_x * 4;
            #pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                const int kk = tile_k + k_vec + lane;
                a_tile[local_y][k_vec + lane] = (row < m && kk < k) ? a[row * k + kk] : input_t(0.0f);
            }
        }

        {
            const int k_vec = local_y * 4;
            #pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                const int kk = tile_k + k_vec + lane;
                b_tile[local_x][k_vec + lane] = (col < n && kk < k) ? b[col * k + kk] : input_t(0.0f);
            }
        }

        __syncthreads();

        if (row < m && col < n) {
            #pragma unroll 4
            for (int kk = 0; kk < TILE_K; ++kk) {
                acc += static_cast<float>(a_tile[local_y][kk]) * static_cast<float>(b_tile[local_x][kk]);
            }
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = static_cast<__hip_bfloat16>(acc);
    }
}

__global__ void mxfp4_mm_kernel_mfma_medium(
    const __hip_bfloat16* a,
    const __hip_bfloat16* b,
    __hip_bfloat16* c,
    int m,
    int n,
    int k
) {
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;

    const int lane = threadIdx.x;
    const int tile_row = blockIdx.y * MFMA_M;
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane_col = lane & 15;
    const int lane_group = lane >> 4;

    const auto* a_bits = reinterpret_cast<uint16_t const*>(a);
    const auto* b_bits = reinterpret_cast<uint16_t const*>(b);
    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        _B16x4 a_reg{};
        _B16x4 b_reg{};
        #pragma unroll
        for (int lane_i = 0; lane_i < 4; ++lane_i) {
            const int kk = tile_k + lane_group * 4 + lane_i;
            const int a_row = tile_row + lane_col;
            const int b_col = tile_col + lane_col;
            a_reg[lane_i] = (a_row < m && kk < k) ? a_bits[a_row * k + kk] : uint16_t{0};
            b_reg[lane_i] = (b_col < n && kk < k) ? b_bits[b_col * k + kk] : uint16_t{0};
        }
        acc = gcn_mfma16x16x16bf16_1k_instr(a_reg, b_reg, acc);
    }

    const int out_col = tile_col + lane_col;
    const int out_row_base = tile_row + lane_group * 4;
    #pragma unroll
    for (int row_i = 0; row_i < 4; ++row_i) {
        const int out_row = out_row_base + row_i;
        if (out_row < m && out_col < n) {
            c[out_row * n + out_col] = static_cast<__hip_bfloat16>(acc[row_i]);
        }
    }
}

void mxfp4_mm_hip_mfma_medium(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const int m = static_cast<int>(a.size(0));
    const int n = static_cast<int>(b.size(0));
    const int k = static_cast<int>(a.size(1));

    if ((m % 16 == 0) && (n % 16 == 0) && (k % 32 == 0) && m <= 128) {
        dim3 block(64);
        dim3 grid((n + 16 - 1) / 16, (m + 16 - 1) / 16);
        hipLaunchKernelGGL(
            mxfp4_mm_kernel_mfma_medium,
            grid,
            block,
            0,
            0,
            reinterpret_cast<__hip_bfloat16 const*>(a.data_ptr<at::BFloat16>()),
            reinterpret_cast<__hip_bfloat16 const*>(b.data_ptr<at::BFloat16>()),
            reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
            m,
            n,
            k
        );
        return;
    }

    dim3 block(TILE_N, TILE_M);
    dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_bf16_scalar<__hip_bfloat16>,
        grid,
        block,
        0,
        0,
        reinterpret_cast<__hip_bfloat16 const*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<__hip_bfloat16 const*>(b.data_ptr<at::BFloat16>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        m,
        n,
        k
    );
}

'''
            else:
                kernel = r'''

using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
using bit16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
typedef bit16x4 _B16x4;
typedef struct _B16x8
{
    _B16x4 xy[2];
} _B16x8;

__device__ __forceinline__ floatx4 gcn_mfma16x16x32_bf16_instr(
    const _B16x8& inpA,
    const _B16x8& inpB,
    const floatx4& inpC
) {
    bit16x8 tmpA = __builtin_shufflevector(inpA.xy[0], inpA.xy[1], 0, 1, 2, 3, 4, 5, 6, 7);
    bit16x8 tmpB = __builtin_shufflevector(inpB.xy[0], inpB.xy[1], 0, 1, 2, 3, 4, 5, 6, 7);
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(tmpA, tmpB, inpC, 0, 0, 0);
}

template <typename input_t>
__global__ void mxfp4_mm_kernel_bf16_scalar(
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

    float acc = 0.0f;
    __shared__ input_t a_tile[TILE_M][TILE_K + 1];
    __shared__ input_t b_tile[TILE_N][TILE_K + 1];

    for (int tile_k = 0; tile_k < k; tile_k += TILE_K) {
        if (local_x < TILE_K / 4) {
            const int k_vec = local_x * 4;
            #pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                const int kk = tile_k + k_vec + lane;
                a_tile[local_y][k_vec + lane] = (row < m && kk < k) ? a[row * k + kk] : input_t(0.0f);
            }
        }

        {
            const int k_vec = local_y * 4;
            #pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                const int kk = tile_k + k_vec + lane;
                b_tile[local_x][k_vec + lane] = (col < n && kk < k) ? b[col * k + kk] : input_t(0.0f);
            }
        }

        __syncthreads();

        if (row < m && col < n) {
            #pragma unroll 4
            for (int kk = 0; kk < TILE_K; ++kk) {
                acc += static_cast<float>(a_tile[local_y][kk]) * static_cast<float>(b_tile[local_x][kk]);
            }
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = static_cast<__hip_bfloat16>(acc);
    }
}

__global__ void mxfp4_mm_kernel_mfma_medium(
    const __hip_bfloat16* a,
    const __hip_bfloat16* b,
    __hip_bfloat16* c,
    int m,
    int n,
    int k
) {
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;

    const int lane = threadIdx.x;
    const int tile_row = blockIdx.y * MFMA_M;
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane_col = lane & 15;
    const int lane_group = lane >> 4;

    const auto* a_bits = reinterpret_cast<uint16_t const*>(a);
    const auto* b_bits = reinterpret_cast<uint16_t const*>(b);
    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        _B16x8 a_reg{};
        _B16x8 b_reg{};
        #pragma unroll
        for (int pack = 0; pack < 2; ++pack) {
            #pragma unroll
            for (int lane_i = 0; lane_i < 4; ++lane_i) {
                const int kk = tile_k + lane_group * 8 + pack * 4 + lane_i;
                const int a_row = tile_row + lane_col;
                const int b_col = tile_col + lane_col;
                a_reg.xy[pack][lane_i] = (a_row < m && kk < k) ? a_bits[a_row * k + kk] : uint16_t{0};
                b_reg.xy[pack][lane_i] = (b_col < n && kk < k) ? b_bits[b_col * k + kk] : uint16_t{0};
            }
        }
        acc = gcn_mfma16x16x32_bf16_instr(a_reg, b_reg, acc);
    }

    const int out_col = tile_col + lane_col;
    const int out_row_base = tile_row + lane_group * 4;
    #pragma unroll
    for (int row_i = 0; row_i < 4; ++row_i) {
        const int out_row = out_row_base + row_i;
        if (out_row < m && out_col < n) {
            c[out_row * n + out_col] = static_cast<__hip_bfloat16>(acc[row_i]);
        }
    }
}

void mxfp4_mm_hip_mfma_medium(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const int m = static_cast<int>(a.size(0));
    const int n = static_cast<int>(b.size(0));
    const int k = static_cast<int>(a.size(1));

    if ((m % 16 == 0) && (n % 16 == 0) && (k % 32 == 0) && m <= 128) {
        dim3 block(64);
        dim3 grid((n + 16 - 1) / 16, (m + 16 - 1) / 16);
        hipLaunchKernelGGL(
            mxfp4_mm_kernel_mfma_medium,
            grid,
            block,
            0,
            0,
            reinterpret_cast<__hip_bfloat16 const*>(a.data_ptr<at::BFloat16>()),
            reinterpret_cast<__hip_bfloat16 const*>(b.data_ptr<at::BFloat16>()),
            reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
            m,
            n,
            k
        );
        return;
    }

    dim3 block(TILE_N, TILE_M);
    dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_bf16_scalar<__hip_bfloat16>,
        grid,
        block,
        0,
        0,
        reinterpret_cast<__hip_bfloat16 const*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<__hip_bfloat16 const*>(b.data_ptr<at::BFloat16>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        m,
        n,
        k
    );
}

'''
            updated = updated.replace(kernel_anchor, kernel + kernel_anchor, 1)

        m_check = f"a_in.shape[0] == {exact_m}" if exact_m is not None else f"a_in.shape[0] <= {max_medium_m}"
        medium_block_old = (
            '    if regime == "medium_m":\n'
            '        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)\n'
            '        _module().mxfp4_mm_hip(a_in, b_in, c)\n'
            '        return c\n'
        )
        medium_block_new = f'''    use_mfma_medium = (
        regime == "medium_m"
        and {m_check}
        and (a_in.shape[0] % 16) == 0
        and (a_in.shape[1] % 16) == 0
        and (b_in.shape[0] % 16) == 0
    )
    if use_mfma_medium:
        a_mfma = a_in.to(torch.bfloat16).contiguous()
        b_mfma = _get_b_contract_bf16(b, b_q, b_scale_sh)
        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
        _module().mxfp4_mm_hip_mfma_medium(a_mfma, b_mfma, c)
        return c
    if regime == "medium_m":
        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
        _module().mxfp4_mm_hip(a_in, b_in, c)
        return c
'''
        if medium_block_old in updated:
            updated = updated.replace(medium_block_old, medium_block_new, 1)
        else:
            updated = re.sub(
                r"    use_mfma_medium = \(\n(?:.*\n)+?    if regime == \"medium_m\":\n        c = torch.empty\(\(a_in.shape\[0\], b_in.shape\[0\]\), dtype=torch.bfloat16, device=a_in.device\)\n        _module\(\)\.mxfp4_mm_hip\(a_in, b_in, c\)\n        return c\n",
                medium_block_new,
                updated,
                count=1,
            )
        return updated

    def _apply_mfma_launch_bounds(self, source_text: str) -> str:
        updated = source_text
        updated = updated.replace(
            "__global__ void mxfp4_mm_kernel_mfma_medium(",
            "__launch_bounds__(64)\n__global__ void mxfp4_mm_kernel_mfma_medium(",
            1,
        )
        return updated

    def _apply_mfma_scale_exact_m16(self, source_text: str) -> str:
        updated = source_text
        updated = updated.replace(
            "void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c);\nvoid mxfp4_mm_hip_mfma_medium(torch::Tensor a, torch::Tensor b, torch::Tensor c);",
            "void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c);\nvoid mxfp4_mm_hip_mfma_medium(torch::Tensor a, torch::Tensor b, torch::Tensor c);\nvoid mxfp4_mm_hip_mfma_scale_exact_m16(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c);",
            1,
        )
        updated = updated.replace(
            '            functions=["mxfp4_mm_hip", "mxfp4_mm_hip_mfma_medium"],',
            '            functions=["mxfp4_mm_hip", "mxfp4_mm_hip_mfma_medium", "mxfp4_mm_hip_mfma_scale_exact_m16"],',
            1,
        )
        updated = updated.replace(
            '_B_BF16_CACHE: dict[tuple[object, ...], torch.Tensor] = {}\n',
            '_B_BF16_CACHE: dict[tuple[object, ...], torch.Tensor] = {}\n_B_MFMA_FP4_CACHE: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}\n',
            1,
        )

        helper_anchor = "\ndef _reference_oracle_inputs(\n"
        if "def _get_b_contract_mfma_fp4(" not in updated:
            helper = '''

def _get_b_contract_mfma_fp4(
    b: torch.Tensor,
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = _b_contract_cache_key(b, b_q, b_scale_sh)
    cached = _B_MFMA_FP4_CACHE.get(key)
    if cached is not None:
        return cached

    quant = _quant()
    _, b_scale = quant(b.contiguous(), shuffle=False)
    b_ref_vals = fp4_utils.mxfp4_to_f32(b_q.contiguous())[: b.shape[0], : b.shape[1]].to(torch.float32)
    b_packed = fp4_utils.f32_to_mxfp4(b_ref_vals.t().contiguous()).contiguous().view(torch.uint8)
    b_scale_u8 = b_scale.contiguous().view(torch.uint8)
    if len(_B_MFMA_FP4_CACHE) >= 4:
        _B_MFMA_FP4_CACHE.clear()
    _B_MFMA_FP4_CACHE[key] = (b_packed, b_scale_u8)
    return b_packed, b_scale_u8


def _get_a_contract_mfma_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    quant = _quant()
    a_q, a_scale = quant(a.contiguous(), shuffle=False)
    a_scale_f32 = _expand_scales(a_scale, rows=a.shape[0], cols=a.shape[1])
    a_ref_vals = fp4_utils.mxfp4_to_f32(a_q.contiguous())[: a.shape[0], : a.shape[1]].to(torch.float32)
    rules, _ = _get_b_contract(b, b_q, b_scale_sh)
    norm_a = (a.to(torch.float32) / a_scale_f32).contiguous()
    a_corrected_vals = _apply_adjustment_rules(norm_a, a_ref_vals, rules).contiguous()
    a_packed = fp4_utils.f32_to_mxfp4(a_corrected_vals).contiguous().view(torch.uint8)
    return a_packed, a_scale.contiguous().view(torch.uint8)
'''
            updated = updated.replace(helper_anchor, helper + helper_anchor, 1)

        kernel_anchor = "void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c) {\n"
        if "mxfp4_mm_kernel_mfma_scale_exact_m16" not in updated:
            kernel = r'''

using i32x8_t = int __attribute__((ext_vector_type(8)));

__device__ __forceinline__ unsigned char fp4_extract(unsigned char packed, int idx) {
    return (idx == 0) ? (packed & 0xFu) : (packed >> 4);
}

__device__ __forceinline__ unsigned char fp4_pack(unsigned char lo, unsigned char hi) {
    return (lo & 0xFu) | ((hi & 0xFu) << 4);
}

__device__ __forceinline__ int pack_scale_e8m0x4(const uint8_t* scale_ptr) {
    return static_cast<int>(scale_ptr[0])
        | (static_cast<int>(scale_ptr[1]) << 8)
        | (static_cast<int>(scale_ptr[2]) << 16)
        | (static_cast<int>(scale_ptr[3]) << 24);
}

__global__ void mxfp4_mm_kernel_mfma_scale_exact_m16(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_packed,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale,
    __hip_bfloat16* __restrict__ c,
    int m,
    int n,
    int k,
    int a_scale_stride,
    int b_scale_stride
) {
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 128;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_row = blockIdx.y * MFMA_M;
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane16 = lane & 15;
    const int group4 = lane >> 4;
    const int a_bytes_per_row = k / 2;
    const int b_bytes_per_row = n / 2;

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            a_buf.v[i] = 0;
            b_buf.v[i] = 0;
        }

        const unsigned char* ldg_a = a_packed + (tile_row + lane16) * a_bytes_per_row + tile_k / 2 + group4 * 16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            a_buf.b[i] = ldg_a[i];
        }

        const unsigned char* ldg_b = b_packed + (tile_k + group4 * 32) * b_bytes_per_row + tile_col / 2 + lane16 / 2;
        const int b_nibble = lane16 & 1;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const unsigned char byte0 = ldg_b[b_bytes_per_row * (2 * i)];
            const unsigned char byte1 = ldg_b[b_bytes_per_row * (2 * i + 1)];
            b_buf.b[i] = fp4_pack(fp4_extract(byte0, b_nibble), fp4_extract(byte1, b_nibble));
        }

        const int scale_block = tile_k / 32;
        const int scale_a = pack_scale_e8m0x4(a_scale + (tile_row + lane16) * a_scale_stride + scale_block);
        const int scale_b = pack_scale_e8m0x4(b_scale + (tile_col + lane16) * b_scale_stride + scale_block);
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_buf.v, b_buf.v, acc, 4, 4, 0, scale_a, 0, scale_b);
    }

    const int out_col = tile_col + lane16;
    const int out_row_base = tile_row + group4 * 4;
    #pragma unroll
    for (int row_i = 0; row_i < 4; ++row_i) {
        const int out_row = out_row_base + row_i;
        if (out_row < m && out_col < n) {
            c[out_row * n + out_col] = static_cast<__hip_bfloat16>(acc[row_i]);
        }
    }
}

void mxfp4_mm_hip_mfma_scale_exact_m16(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c) {
    const int m = static_cast<int>(a_scale.size(0));
    const int n = static_cast<int>(b_scale.size(0));
    const int k = static_cast<int>(a_packed.size(1) * 2);

    dim3 block(64);
    dim3 grid((n + 16 - 1) / 16, (m + 16 - 1) / 16);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m16,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_packed.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        m,
        n,
        k,
        static_cast<int>(a_scale.size(1)),
        static_cast<int>(b_scale.size(1))
    );
}

'''
            updated = updated.replace(kernel_anchor, kernel + kernel_anchor, 1)

        old_block = '''    use_mfma_medium = (
        regime == "medium_m"
        and a_in.shape[0] == 16
        and (a_in.shape[0] % 16) == 0
        and (a_in.shape[1] % 16) == 0
        and (b_in.shape[0] % 16) == 0
    )
    if use_mfma_medium:
        a_mfma = a_in.to(torch.bfloat16).contiguous()
        b_mfma = _get_b_contract_bf16(b, b_q, b_scale_sh)
        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
        _module().mxfp4_mm_hip_mfma_medium(a_mfma, b_mfma, c)
        return c
'''
        new_block = '''    use_mfma_scale = (
        regime == "medium_m"
        and a_in.shape[0] == 16
        and (a_in.shape[1] % 128) == 0
        and (b_in.shape[0] % 16) == 0
    )
    if use_mfma_scale:
        a_packed, a_scale = _get_a_contract_mfma_fp4(a, b, b_q, b_scale_sh)
        b_packed, b_scale = _get_b_contract_mfma_fp4(b, b_q, b_scale_sh)
        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
        inflight = globals().setdefault("_MFMA_SCALE_INFLIGHT", [])
        inflight.append((a_packed, a_scale, b_packed, b_scale))
        if len(inflight) > 16:
            del inflight[:-16]
        _module().mxfp4_mm_hip_mfma_scale_exact_m16(a_packed, b_packed, a_scale, b_scale, c)
        return c
    use_mfma_medium = (
        regime == "medium_m"
        and a_in.shape[0] == 16
        and (a_in.shape[0] % 16) == 0
        and (a_in.shape[1] % 16) == 0
        and (b_in.shape[0] % 16) == 0
    )
    if use_mfma_medium:
        a_mfma = a_in.to(torch.bfloat16).contiguous()
        b_mfma = _get_b_contract_bf16(b, b_q, b_scale_sh)
        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
        _module().mxfp4_mm_hip_mfma_medium(a_mfma, b_mfma, c)
        return c
'''
        if old_block in updated:
            updated = updated.replace(old_block, new_block, 1)
        return updated

    def _apply_mfma_scale_launch_bounds(self, source_text: str, *, kernel_name: str = "mxfp4_mm_kernel_mfma_scale_exact_m16") -> str:
        return source_text.replace(
            f"__global__ void {kernel_name}(",
            f"__launch_bounds__(64)\n__global__ void {kernel_name}(",
            1,
        )

    def _apply_mfma_scale_big_inflight(self, source_text: str, *, limit: int = 256) -> str:
        return source_text.replace(
            "        if len(inflight) > 16:\n            del inflight[:-16]\n",
            f"        if len(inflight) > {limit}:\n            del inflight[:-{limit}]\n",
        )

    def _apply_mfma_scale_sync_debug(self, source_text: str, *, kernel_name: str) -> str:
        return source_text.replace(
            f"        _module().{kernel_name}(a_packed, b_packed, a_scale, b_scale, c)\n        return c\n",
            f"        _module().{kernel_name}(a_packed, b_packed, a_scale, b_scale, c)\n        torch.cuda.synchronize()\n        return c\n",
            1,
        )

    def _apply_mfma_scale_exact_m32_split_m16(self, source_text: str) -> str:
        updated = self._apply_mfma_scale_exact_m16(source_text)
        anchor = """    use_mfma_scale = (
        regime == "medium_m"
        and a_in.shape[0] == 16
        and (a_in.shape[1] % 128) == 0
        and (b_in.shape[0] % 16) == 0
    )
"""
        insert = """    use_mfma_scale_split_m32 = (
        regime == "medium_m"
        and a_in.shape[0] == 32
        and (a_in.shape[1] % 128) == 0
        and (b_in.shape[0] % 16) == 0
    )
    if use_mfma_scale_split_m32:
        a_packed, a_scale = _get_a_contract_mfma_fp4(a, b, b_q, b_scale_sh)
        b_packed, b_scale = _get_b_contract_mfma_fp4(b, b_q, b_scale_sh)
        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
        inflight = globals().setdefault("_MFMA_SCALE_INFLIGHT", [])
        a_top = a_packed[:16].contiguous()
        a_bottom = a_packed[16:32].contiguous()
        a_scale_top = a_scale[:16].contiguous()
        a_scale_bottom = a_scale[16:32].contiguous()
        c_top = c.narrow(0, 0, 16)
        c_bottom = c.narrow(0, 16, 16)
        inflight.append((a_top, a_scale_top, a_bottom, a_scale_bottom, b_packed, b_scale, c_top, c_bottom))
        if len(inflight) > 64:
            del inflight[:-64]
        _module().mxfp4_mm_hip_mfma_scale_exact_m16(a_top, b_packed, a_scale_top, b_scale, c_top)
        _module().mxfp4_mm_hip_mfma_scale_exact_m16(a_bottom, b_packed, a_scale_bottom, b_scale, c_bottom)
        return c

"""
        if anchor not in updated:
            raise RuntimeError("failed to locate exact-m16 native scaled block anchor for split-m32 insertion")
        return updated.replace(anchor, insert + anchor, 1)

    def _apply_mfma_scale_exact_m32(self, source_text: str) -> str:
        updated = source_text
        updated = updated.replace(
            "void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c);\nvoid mxfp4_mm_hip_mfma_medium(torch::Tensor a, torch::Tensor b, torch::Tensor c);",
            "void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c);\nvoid mxfp4_mm_hip_mfma_medium(torch::Tensor a, torch::Tensor b, torch::Tensor c);\nvoid mxfp4_mm_hip_mfma_scale_exact_m32(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c);",
            1,
        )
        updated = updated.replace(
            '            functions=["mxfp4_mm_hip", "mxfp4_mm_hip_mfma_medium"],',
            '            functions=["mxfp4_mm_hip", "mxfp4_mm_hip_mfma_medium", "mxfp4_mm_hip_mfma_scale_exact_m32"],',
            1,
        )
        updated = updated.replace(
            '_B_BF16_CACHE: dict[tuple[object, ...], torch.Tensor] = {}\n',
            '_B_BF16_CACHE: dict[tuple[object, ...], torch.Tensor] = {}\n_B_MFMA_FP4_CACHE: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}\n',
            1,
        )

        helper_anchor = "\ndef _reference_oracle_inputs(\n"
        if "def _get_b_contract_mfma_fp4(" not in updated:
            helper = '''

def _get_b_contract_mfma_fp4(
    b: torch.Tensor,
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = _b_contract_cache_key(b, b_q, b_scale_sh)
    cached = _B_MFMA_FP4_CACHE.get(key)
    if cached is not None:
        return cached

    quant = _quant()
    _, b_scale = quant(b.contiguous(), shuffle=False)
    b_ref_vals = fp4_utils.mxfp4_to_f32(b_q.contiguous())[: b.shape[0], : b.shape[1]].to(torch.float32)
    b_packed = fp4_utils.f32_to_mxfp4(b_ref_vals.t().contiguous()).contiguous().view(torch.uint8)
    b_scale_u8 = b_scale.contiguous().view(torch.uint8)
    if len(_B_MFMA_FP4_CACHE) >= 4:
        _B_MFMA_FP4_CACHE.clear()
    _B_MFMA_FP4_CACHE[key] = (b_packed, b_scale_u8)
    return b_packed, b_scale_u8


def _get_a_contract_mfma_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    quant = _quant()
    a_q, a_scale = quant(a.contiguous(), shuffle=False)
    a_scale_f32 = _expand_scales(a_scale, rows=a.shape[0], cols=a.shape[1])
    a_ref_vals = fp4_utils.mxfp4_to_f32(a_q.contiguous())[: a.shape[0], : a.shape[1]].to(torch.float32)
    rules, _ = _get_b_contract(b, b_q, b_scale_sh)
    norm_a = (a.to(torch.float32) / a_scale_f32).contiguous()
    a_corrected_vals = _apply_adjustment_rules(norm_a, a_ref_vals, rules).contiguous()
    a_packed = fp4_utils.f32_to_mxfp4(a_corrected_vals).contiguous().view(torch.uint8)
    return a_packed, a_scale.contiguous().view(torch.uint8)
'''
            updated = updated.replace(helper_anchor, helper + helper_anchor, 1)

        kernel_anchor = "void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c) {\n"
        if "mxfp4_mm_kernel_mfma_scale_exact_m32" not in updated:
            kernel = r'''

using i32x8_t = int __attribute__((ext_vector_type(8)));
using floatx16 = float __attribute__((ext_vector_type(16)));

__device__ __forceinline__ unsigned char fp4_extract(unsigned char packed, int idx) {
    return (idx == 0) ? (packed & 0xFu) : (packed >> 4);
}

__device__ __forceinline__ unsigned char fp4_pack(unsigned char lo, unsigned char hi) {
    return (lo & 0xFu) | ((hi & 0xFu) << 4);
}

__device__ __forceinline__ int pack_scale_e8m0x2(const uint8_t* scale_ptr) {
    return static_cast<int>(scale_ptr[0])
        | (static_cast<int>(scale_ptr[1]) << 8)
        | (127 << 16)
        | (127 << 24);
}

__global__ void mxfp4_mm_kernel_mfma_scale_exact_m32(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_packed,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale,
    __hip_bfloat16* __restrict__ c,
    int m,
    int n,
    int k,
    int a_scale_stride,
    int b_scale_stride
) {
    constexpr int MFMA_M = 32;
    constexpr int MFMA_N = 32;
    constexpr int MFMA_K = 64;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_row = blockIdx.y * MFMA_M;
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane32 = lane & 31;
    const int group = lane >> 5;
    const int a_bytes_per_row = k / 2;
    const int b_bytes_per_row = n / 2;

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    floatx16 acc{};

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            a_buf.v[i] = 0;
            b_buf.v[i] = 0;
        }

        const unsigned char* ldg_a = a_packed + (tile_row + lane32) * a_bytes_per_row + tile_k / 2 + group * 16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            a_buf.b[i] = ldg_a[i];
        }

        const unsigned char* ldg_b = b_packed + (tile_k + group * 32) * b_bytes_per_row + tile_col / 2 + lane32 / 2;
        const int b_nibble = lane32 & 1;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const unsigned char byte0 = ldg_b[b_bytes_per_row * (2 * i)];
            const unsigned char byte1 = ldg_b[b_bytes_per_row * (2 * i + 1)];
            b_buf.b[i] = fp4_pack(fp4_extract(byte0, b_nibble), fp4_extract(byte1, b_nibble));
        }

        const int scale_block = tile_k / 32;
        const int scale_a = pack_scale_e8m0x2(a_scale + (tile_row + lane32) * a_scale_stride + scale_block);
        const int scale_b = pack_scale_e8m0x2(b_scale + (tile_col + lane32) * b_scale_stride + scale_block);
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a_buf.v, b_buf.v, acc, 4, 4, 0, scale_a, 0, scale_b);
    }

    const int out_col = tile_col + lane32;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row_base = tile_row + group * 4 + i * 8;
        if (row_base + 0 < m && out_col < n) c[(row_base + 0) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 0]);
        if (row_base + 1 < m && out_col < n) c[(row_base + 1) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 1]);
        if (row_base + 2 < m && out_col < n) c[(row_base + 2) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 2]);
        if (row_base + 3 < m && out_col < n) c[(row_base + 3) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 3]);
    }
}

void mxfp4_mm_hip_mfma_scale_exact_m32(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c) {
    const int m = static_cast<int>(a_scale.size(0));
    const int n = static_cast<int>(b_scale.size(0));
    const int k = static_cast<int>(a_packed.size(1) * 2);

    dim3 block(64);
    dim3 grid((n + 32 - 1) / 32, (m + 32 - 1) / 32);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m32,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_packed.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        m,
        n,
        k,
        static_cast<int>(a_scale.size(1)),
        static_cast<int>(b_scale.size(1))
    );
}

'''
            updated = updated.replace(kernel_anchor, kernel + kernel_anchor, 1)

        old_block = '''    use_mfma_medium = (
        regime == "medium_m"
        and a_in.shape[0] == 16
        and (a_in.shape[0] % 16) == 0
        and (a_in.shape[1] % 16) == 0
        and (b_in.shape[0] % 16) == 0
    )
    if use_mfma_medium:
        a_mfma = a_in.to(torch.bfloat16).contiguous()
        b_mfma = _get_b_contract_bf16(b, b_q, b_scale_sh)
        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
        _module().mxfp4_mm_hip_mfma_medium(a_mfma, b_mfma, c)
        return c
'''
        new_block = '''    use_mfma_scale = (
        regime == "medium_m"
        and a_in.shape[0] == 32
        and (a_in.shape[1] % 64) == 0
        and (b_in.shape[0] % 32) == 0
    )
    if use_mfma_scale:
        a_packed, a_scale = _get_a_contract_mfma_fp4(a, b, b_q, b_scale_sh)
        b_packed, b_scale = _get_b_contract_mfma_fp4(b, b_q, b_scale_sh)
        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
        inflight = globals().setdefault("_MFMA_SCALE_INFLIGHT", [])
        inflight.append((a_packed, a_scale, b_packed, b_scale))
        if len(inflight) > 16:
            del inflight[:-16]
        _module().mxfp4_mm_hip_mfma_scale_exact_m32(a_packed, b_packed, a_scale, b_scale, c)
        return c
    use_mfma_medium = (
        regime == "medium_m"
        and a_in.shape[0] == 16
        and (a_in.shape[0] % 16) == 0
        and (a_in.shape[1] % 16) == 0
        and (b_in.shape[0] % 16) == 0
    )
    if use_mfma_medium:
        a_mfma = a_in.to(torch.bfloat16).contiguous()
        b_mfma = _get_b_contract_bf16(b, b_q, b_scale_sh)
        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
        _module().mxfp4_mm_hip_mfma_medium(a_mfma, b_mfma, c)
        return c
'''
        if old_block in updated:
            updated = updated.replace(old_block, new_block, 1)
        else:
            updated = re.sub(
                r"    use_mfma_scale = \(\n(?:.*\n)+?    if use_mfma_medium:\n        a_mfma = a_in.to\(torch.bfloat16\)\.contiguous\(\)\n        b_mfma = _get_b_contract_bf16\(b, b_q, b_scale_sh\)\n        c = torch.empty\(\(a_in.shape\[0\], b_in.shape\[0\]\), dtype=torch.bfloat16, device=a_in.device\)\n        _module\(\)\.mxfp4_mm_hip_mfma_medium\(a_mfma, b_mfma, c\)\n        return c\n",
                new_block,
                updated,
                count=1,
            )
        return updated

    def _apply_b_tile_transpose(self, source_text: str) -> str:
        updated = source_text.replace(
            "__shared__ float b_tile[TILE_N][TILE_K];",
            "__shared__ float b_tile[TILE_K][TILE_N];",
        )
        updated = updated.replace(
            "                b_tile[local_x][k_vec + 0] = vec.x;\n                b_tile[local_x][k_vec + 1] = vec.y;\n                b_tile[local_x][k_vec + 2] = vec.z;\n                b_tile[local_x][k_vec + 3] = vec.w;",
            "                b_tile[k_vec + 0][local_x] = vec.x;\n                b_tile[k_vec + 1][local_x] = vec.y;\n                b_tile[k_vec + 2][local_x] = vec.z;\n                b_tile[k_vec + 3][local_x] = vec.w;",
        )
        updated = updated.replace(
            "                    b_tile[local_x][k_vec + lane] = (col < n && kk < k) ? b[col * k + kk] : 0.0f;",
            "                    b_tile[k_vec + lane][local_x] = (col < n && kk < k) ? b[col * k + kk] : 0.0f;",
        )
        updated = updated.replace(
            "                acc += static_cast<double>(a_tile[local_y][kk]) * static_cast<double>(b_tile[local_x][kk]);",
            "                acc += static_cast<double>(a_tile[local_y][kk]) * static_cast<double>(b_tile[kk][local_x]);",
        )
        return updated

    def _apply_double_buffer(self, source_text: str) -> str:
        updated = source_text.replace(
            "__shared__ float a_tile[TILE_M][TILE_K];",
            "__shared__ float a_tile[2][TILE_M][TILE_K + 1];",
        )
        updated = updated.replace(
            "__shared__ float b_tile[TILE_N][TILE_K];",
            "__shared__ float b_tile[2][TILE_N][TILE_K + 1];",
        )
        updated = updated.replace(
            "    for (int tile_k = 0; tile_k < k; tile_k += TILE_K) {\n",
            "    int buf = 0;\n    for (int tile_k = 0; tile_k < k; tile_k += TILE_K) {\n",
        )
        updated = updated.replace(
            "                a_tile[local_y][k_vec + 0] = vec.x;\n                a_tile[local_y][k_vec + 1] = vec.y;\n                a_tile[local_y][k_vec + 2] = vec.z;\n                a_tile[local_y][k_vec + 3] = vec.w;",
            "                a_tile[buf][local_y][k_vec + 0] = vec.x;\n                a_tile[buf][local_y][k_vec + 1] = vec.y;\n                a_tile[buf][local_y][k_vec + 2] = vec.z;\n                a_tile[buf][local_y][k_vec + 3] = vec.w;",
        )
        updated = updated.replace(
            "                    a_tile[local_y][k_vec + lane] = (row < m && kk < k) ? a[row * k + kk] : 0.0f;",
            "                    a_tile[buf][local_y][k_vec + lane] = (row < m && kk < k) ? a[row * k + kk] : 0.0f;",
        )
        updated = updated.replace(
            "                b_tile[local_x][k_vec + 0] = vec.x;\n                b_tile[local_x][k_vec + 1] = vec.y;\n                b_tile[local_x][k_vec + 2] = vec.z;\n                b_tile[local_x][k_vec + 3] = vec.w;",
            "                b_tile[buf][local_x][k_vec + 0] = vec.x;\n                b_tile[buf][local_x][k_vec + 1] = vec.y;\n                b_tile[buf][local_x][k_vec + 2] = vec.z;\n                b_tile[buf][local_x][k_vec + 3] = vec.w;",
        )
        updated = updated.replace(
            "                    b_tile[local_x][k_vec + lane] = (col < n && kk < k) ? b[col * k + kk] : 0.0f;",
            "                    b_tile[buf][local_x][k_vec + lane] = (col < n && kk < k) ? b[col * k + kk] : 0.0f;",
        )
        updated = updated.replace(
            "                acc += static_cast<double>(a_tile[local_y][kk]) * static_cast<double>(b_tile[local_x][kk]);",
            "                acc += static_cast<double>(a_tile[buf][local_y][kk]) * static_cast<double>(b_tile[buf][local_x][kk]);",
        )
        updated = updated.replace(
            "        __syncthreads();\n    }\n",
            "        __syncthreads();\n        buf ^= 1;\n    }\n",
            1,
        )
        return updated

    def _apply_double_buffer_prefetch(self, source_text: str) -> str:
        updated = source_text
        updated = updated.replace(
            "    for (int tile_k = 0; tile_k < k; tile_k += TILE_K) {\n",
            "    const int tile_count = (k + TILE_K - 1) / TILE_K;\n    for (int tile_index = 0; tile_index < tile_count; ++tile_index) {\n        const int tile_k = tile_index * TILE_K;\n",
            1,
        )
        updated = updated.replace(
            "        __syncthreads();\n    }\n",
            "        __syncthreads();\n        const int next_tile_k = tile_k + TILE_K;\n        (void)next_tile_k;\n    }\n",
            1,
        )
        return updated

    def _rename_variant(self, source_text: str, variant_name: str) -> str:
        source_text = re.sub(
            r'("variant_name":\s*")[^"]+(")',
            rf'\g<1>{variant_name}\2',
            source_text,
            count=1,
        )
        source_text = re.sub(
            r'("name":\s*")[^"]+(")',
            rf'\g<1>{variant_name}\2',
            source_text,
            count=1,
        )
        return source_text

    def _replace_block(self, target: str, donor: str, pattern: re.Pattern[str]) -> str:
        donor_match = pattern.search(donor)
        target_match = pattern.search(target)
        if donor_match is None or target_match is None:
            raise RuntimeError(f"failed to locate transplant block for pattern {pattern.pattern}")
        return target[: target_match.start()] + donor_match.group(0) + target[target_match.end() :]

    def _stage_status(self, summary: HarnessSummary, stage_name: str) -> str | None:
        for stage in summary.stages:
            if stage.get("name") == stage_name:
                status = stage.get("status")
                return str(status) if status is not None else None
        return None

    def _stage_objective(self, summary: HarnessSummary, stage_name: str) -> float | None:
        for stage in summary.stages:
            if stage.get("name") == stage_name:
                objective = stage.get("objective")
                return float(objective) if isinstance(objective, (int, float)) else None
        return None

    def _should_keep(
        self,
        summary: HarnessSummary,
        benchmark_objective: float | None,
        current_best_ns: float,
    ) -> bool:
        if self._stage_status(summary, "test") != "ok":
            return False
        if self._stage_status(summary, "benchmark") != "ok":
            return False
        if benchmark_objective is None:
            return False
        return benchmark_objective < (current_best_ns - self.config.workspace.improvement_epsilon)

    def _run_optional_leaderboard(self, problem_key: str, source_path: Path, move_name: str) -> None:
        run_dir = self.harness.create_run(
            problem_key,
            source_path=source_path,
            stages=["leaderboard"],
            family="handrolled_hip",
            label=f"handrolled-leaderboard-{move_name}",
        )
        self.harness.resume_run(run_dir)

    def _state_path(self, problem_key: str) -> Path:
        return self._working_dir(problem_key) / "state.json"

    def _journal_path(self, problem_key: str) -> Path:
        return self._working_dir(problem_key) / "journal.jsonl"

    def _working_dir(self, problem_key: str) -> Path:
        path = self.config.workspace.root / "handrolled" / problem_key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _round_dir(self, problem_key: str, round_index: int, move_name: str) -> Path:
        return self._working_dir(problem_key) / f"round-{round_index:04d}-{move_name}"

    def _load_state(self, problem_key: str) -> dict[str, object] | None:
        path = self._state_path(problem_key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_state(self, problem_key: str, state: dict[str, object]) -> None:
        self._state_path(problem_key).write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _append_journal(self, problem_key: str, record: dict[str, object]) -> None:
        with self._journal_path(problem_key).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _family_for_move_name(self, move_name: str) -> str:
        if move_name.startswith("mfma_scale_exact_m32_split_m16"):
            return "native_scale_m32_split16"
        if move_name.startswith("mfma_scale_exact_m32"):
            return "native_scale_m32_direct"
        if move_name.startswith("mfma_scale_exact_m16"):
            return "native_scale_m16"
        return move_name

    def _ensure_failure_memory(self, problem_key: str, state: dict[str, object]) -> None:
        if "family_failure_counts" in state and "blocked_families" in state:
            return
        counts: dict[str, int] = {}
        journal_path = self._journal_path(problem_key)
        if journal_path.exists():
            for line in journal_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                family = str(record.get("family") or self._family_for_move_name(str(record.get("move", ""))))
                if self._record_has_benchmark_only_failure(record):
                    counts[family] = counts.get(family, 0) + 1
        state["family_failure_counts"] = counts
        state["blocked_families"] = sorted(
            family for family, count in counts.items() if count >= 2 and family != "baseline"
        )

    def _select_next_move(self, moves: list[HandrolledMove], state: dict[str, object]) -> HandrolledMove:
        blocked = set(str(item) for item in state.get("blocked_families", []))
        cursor = int(state["move_cursor"])
        for offset in range(len(moves)):
            move = moves[(cursor + offset) % len(moves)]
            if move.family not in blocked:
                state["move_cursor"] = cursor + offset + 1
                return move
        move = moves[cursor % len(moves)]
        state["move_cursor"] = cursor + 1
        return move

    def _update_failure_memory(
        self,
        problem_key: str,
        state: dict[str, object],
        move: HandrolledMove,
        summary: HarnessSummary,
    ) -> None:
        self._ensure_failure_memory(problem_key, state)
        record = {
            "test_status": self._stage_status(summary, "test"),
            "benchmark_status": self._stage_status(summary, "benchmark"),
        }
        if not self._record_has_benchmark_only_failure(record):
            return
        counts = dict(state.get("family_failure_counts", {}))
        counts[move.family] = int(counts.get(move.family, 0)) + 1
        state["family_failure_counts"] = counts
        blocked = set(str(item) for item in state.get("blocked_families", []))
        if counts[move.family] >= 2 and move.family != "baseline":
            blocked.add(move.family)
        state["blocked_families"] = sorted(blocked)

    def _record_has_benchmark_only_failure(self, record: dict[str, object]) -> bool:
        if record.get("test_status") != "ok":
            return False
        benchmark_status = record.get("benchmark_status")
        return benchmark_status in {"runtime_error", "check_fail", "submit_error"}
