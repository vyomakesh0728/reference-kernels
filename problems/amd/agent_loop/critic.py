from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from .evaluator import EvaluationResult


@dataclass(frozen=True)
class EvaluationCritique:
    problem_key: str
    candidate_id: str
    status: str
    summary: str
    failure_kind: str | None
    failure_signature: str | None
    benchmark_case_count: int
    benchmark_pass_count: int
    benchmark_fail_count: int
    workflow_url: str | None
    policy_signal: str | None
    policy_rationale: str | None
    next_actions: list[str]


def build_critique(
    problem_key: str,
    candidate_id: str,
    result: EvaluationResult,
) -> EvaluationCritique:
    metrics = result.metrics
    benchmark_case_count = _as_int(metrics.get("benchmark_case_count"))
    benchmark_pass_count = _as_int(metrics.get("benchmark_pass_count"))
    benchmark_fail_count = _as_int(metrics.get("benchmark_fail_count"))
    workflow_url = _as_str(metrics.get("workflow_url"))

    failure_kind = _as_str(metrics.get("failure_kind"))
    failure_signature = _as_str(metrics.get("failure_signature"))
    policy_signal: str | None = None
    policy_rationale: str | None = None

    summary = f"{result.status} with {benchmark_pass_count}/{benchmark_case_count} benchmark cases passing"
    next_actions: list[str] = []

    if result.status == "ok":
        objective = metrics.get("geom_mean_ns")
        if isinstance(objective, (int, float)):
            summary = f"all checks passed; geometric mean {float(objective) / 1_000.0:.1f} us"
        wall_time_seconds = metrics.get("wall_time_seconds")
        if isinstance(wall_time_seconds, (int, float)):
            summary = f"{summary}; wall clock {float(wall_time_seconds):.1f}s"
        next_actions = [
            "Mutate around the best-performing schedule instead of changing semantics.",
            "Keep the live MI355X contract unchanged unless a correctness issue appears.",
        ]
        policy_signal = "throughput_shift"
        policy_rationale = "The candidate passed, so the next step is schedule/search refinement."
        if metrics.get("leaderboard_over_reference") is True:
            next_actions.append(
                "This ranked run beat correctness but exceeded the current ranked reference budget."
            )
            policy_signal = "latency_repair"
            policy_rationale = "The candidate is correct but too slow at ranked wall clock."
    elif result.status == "timeout":
        timeout_seconds = metrics.get("timeout_seconds")
        if isinstance(timeout_seconds, (int, float)):
            summary = f"submission exceeded {float(timeout_seconds):.1f}s client timeout"
        else:
            summary = "submission timed out"
        failure_kind = "timeout"
        next_actions = [
            "Treat this candidate as stalled or too slow for the current ranked submission budget.",
            "Prefer the last known-good anchor before retrying more aggressive variants.",
        ]
        policy_signal = "latency_repair"
        policy_rationale = "The candidate exceeded the ranked submission budget."
    elif result.status == "runtime_error":
        runtime_error = _as_str(metrics.get("runtime_error")) or "runtime error during evaluation"
        summary = runtime_error
        if "shape '" in runtime_error:
            failure_kind = failure_kind or "shape_contract_error"
            next_actions = [
                "Re-check the live task tuple/shape contract before mutating schedule parameters.",
                "Prefer a contract-faithful anchor variant before exploring aggressive rewrites.",
            ]
            policy_signal = "contract_repair"
            policy_rationale = "The failure looks like a live shape or API contract mismatch."
        else:
            next_actions = [
                "Minimize the kernel to the smallest contract-faithful path and re-run.",
                "Avoid introducing optional fast-path assumptions until the baseline passes.",
            ]
            policy_signal = "runtime_repair"
            policy_rationale = "The kernel failed during execution before we could trust any performance result."
    elif result.status == "check_fail":
        mismatch_count = _as_int(metrics.get("mismatch_count"))
        if mismatch_count:
            summary = f"correctness mismatch ({mismatch_count} mismatched elements)"
        failure_kind = failure_kind or "correctness_mismatch"
        next_actions = [
            "Preserve the live input/layout contract exactly before tuning tile sizes.",
            "Remove or harden caches that can survive across benchmark cases.",
        ]
        policy_signal = "contract_repair"
        policy_rationale = "The kernel produced incorrect outputs, so correctness must dominate the next mutation."
    else:
        summary = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "submission failed before evaluation"
        next_actions = [
            "Check Popcorn submission stderr and result artifacts for infrastructure or packaging issues.",
        ]
        policy_signal = "submission_repair"
        policy_rationale = "The candidate failed before we received a valid evaluation result."

    if problem_key == "mxfp4_mm" and failure_kind == "correctness_mismatch":
        next_actions.insert(
            0,
            "Use the shuffled MXFP4 contract path first: quantize A with shuffle=True and consume B_shuffle/B_scale_sh.",
        )
    if problem_key == "moe_mxfp4" and result.status != "ok":
        next_actions.insert(
            0,
            "Anchor on fused_moe with the provided shuffled weights/scales before exploring Triton packing variants.",
        )
    if problem_key == "mixed_mla" and result.status != "ok":
        next_actions.insert(
            0,
            "Anchor on the fp8 MLA decode path before exploring bf16 Triton decode variants.",
        )

    return EvaluationCritique(
        problem_key=problem_key,
        candidate_id=candidate_id,
        status=result.status,
        summary=summary,
        failure_kind=failure_kind,
        failure_signature=failure_signature,
        benchmark_case_count=benchmark_case_count,
        benchmark_pass_count=benchmark_pass_count,
        benchmark_fail_count=benchmark_fail_count,
        workflow_url=workflow_url,
        policy_signal=policy_signal,
        policy_rationale=policy_rationale,
        next_actions=next_actions,
    )


def write_critique(path: Path, critique: EvaluationCritique) -> None:
    path.write_text(json.dumps(asdict(critique), indent=2, sort_keys=True), encoding="utf-8")


def load_critique(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
