from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from pathlib import Path
import json
import os
import re
import statistics
import subprocess
import time

from .config import ProblemConfig, WorkspaceConfig


LINE_RE = re.compile(r"^\s*([^:]+):\s*(.*?)\s*$")
MEAN_RE = re.compile(r"^benchmark\.(\d+)\.mean$")
WORKFLOW_RE = re.compile(r"Workflow \[(\d+)\]\(<(https://[^>]+)>\)")
BENCHMARK_CASE_RE = re.compile(
    r"^(?P<failed>❌\s+)?k:\s*(?P<k>\d+);\s*m:\s*(?P<m>\d+);\s*n:\s*(?P<n>\d+);\s*seed:\s*(?P<seed>\d+)(?P<suffix>.*)$"
)
MEASURE_RE = re.compile(r"⏱\s*(?P<mean>[0-9.]+)\s*±\s*(?P<std>[0-9.]+)\s*(?P<unit>ns|µs|us|ms|s)")
MISMATCH_RE = re.compile(r"Number of mismatched elements:\s*(\d+)")
RUNTIME_ERROR_RE = re.compile(r"RuntimeError:\s*(.+)")
SECTION_RE = re.compile(r"^##\s+(Benchmarks|Ranked Benchmark):\s*$")


@dataclass(frozen=True)
class EvaluationResult:
    command: list[str]
    return_code: int
    status: str
    objective: float | None
    metrics: dict[str, object]
    stdout: str
    stderr: str
    result_text: str


def parse_result_text(text: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for line in text.splitlines():
        match = LINE_RE.match(line)
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        parsed[key] = _coerce_value(value)

    means: list[float] = []
    for key, value in parsed.items():
        if MEAN_RE.match(key) and isinstance(value, (int, float)):
            means.append(float(value))

    if means:
        parsed["geom_mean_ns"] = _geometric_mean(means)
        parsed["mean_of_means_ns"] = statistics.fmean(means)
        parsed["benchmark_case_count"] = len(means)

    parsed.update(_parse_kernelbot_markdown(text))

    return parsed


def run_popcorn_submission(
    workspace: WorkspaceConfig,
    problem: ProblemConfig,
    submission_path: Path,
    artifacts_dir: Path,
    mode: str | None = None,
) -> EvaluationResult:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifacts_dir / "result.txt"
    stdout_path = artifacts_dir / "stdout.txt"
    stderr_path = artifacts_dir / "stderr.txt"

    command = [
        workspace.popcorn_cli,
        "submit",
        "--no-tui",
        "--gpu",
        problem.gpu,
        "--leaderboard",
        problem.leaderboard,
        "--mode",
        mode or problem.mode,
        "--output",
        str(result_path),
        str(submission_path),
    ]
    env = os.environ.copy()
    if workspace.api_url:
        env["POPCORN_API_URL"] = workspace.api_url

    selected_mode = mode or problem.mode
    timeout_seconds = (
        workspace.leaderboard_timeout_seconds if selected_mode == "leaderboard" else None
    )
    started_at = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(submission_path.parent),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        wall_time = time.monotonic() - started_at
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        result_text = result_path.read_text(encoding="utf-8") if result_path.exists() else stdout
        metrics = parse_result_text(result_text) if result_text else {}
        metrics["failure_kind"] = "timeout"
        metrics["failure_signature"] = f"timeout>{timeout_seconds}s"
        metrics["timeout_seconds"] = timeout_seconds
        metrics["wall_time_seconds"] = wall_time
        metrics["check"] = "fail"
        metadata_path = artifacts_dir / "parsed_metrics.json"
        metadata_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return EvaluationResult(
            command=command,
            return_code=124,
            status="timeout",
            objective=None,
            metrics=metrics,
            stdout=stdout,
            stderr=stderr,
            result_text=result_text,
        )

    wall_time = time.monotonic() - started_at

    stdout = completed.stdout
    stderr = completed.stderr
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    if result_path.exists():
        result_text = result_path.read_text(encoding="utf-8")
    else:
        result_text = stdout

    metrics = parse_result_text(result_text)
    metrics["wall_time_seconds"] = wall_time
    if selected_mode == "leaderboard" and workspace.leaderboard_reference_seconds is not None:
        metrics["leaderboard_reference_seconds"] = workspace.leaderboard_reference_seconds
        metrics["leaderboard_over_reference"] = wall_time > workspace.leaderboard_reference_seconds
    objective = _extract_objective(metrics, problem.objective)

    status = "submit_error"
    if metrics.get("failure_kind") == "timeout":
        status = "timeout"
    elif completed.returncode == 0 and metrics.get("check") == "pass":
        status = "ok"
    elif completed.returncode == 0 and metrics.get("failure_kind") == "runtime_error":
        status = "runtime_error"
    elif completed.returncode == 0:
        status = "check_fail"

    metadata_path = artifacts_dir / "parsed_metrics.json"
    metadata_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return EvaluationResult(
        command=command,
        return_code=completed.returncode,
        status=status,
        objective=objective,
        metrics=metrics,
        stdout=stdout,
        stderr=stderr,
        result_text=result_text,
    )


def _coerce_value(value: str) -> object:
    if value in {"pass", "fail"}:
        return value
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def _extract_objective(metrics: dict[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _geometric_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute geometric mean of an empty list")
    if any(v <= 0 for v in values):
        raise ValueError(f"geometric mean requires positive values, got {values}")
    return exp(sum(log(v) for v in values) / len(values))


def _parse_kernelbot_markdown(text: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    lines = text.splitlines()

    workflow_match = WORKFLOW_RE.search(text)
    if workflow_match:
        parsed["workflow_id"] = int(workflow_match.group(1))
        parsed["workflow_url"] = workflow_match.group(2)

    if "❌ Running benchmarks failed" in text:
        parsed["failure_kind"] = "runtime_error"
        parsed["check"] = "fail"
    elif "❌ Benchmarking failed" in text or "failed testing:" in text:
        parsed["failure_kind"] = "correctness_mismatch"
        parsed["check"] = "fail"
    elif "✅" in text and "failed" not in text.lower():
        parsed["check"] = "pass"

    mismatch_match = MISMATCH_RE.search(text)
    if mismatch_match:
        parsed["mismatch_count"] = int(mismatch_match.group(1))
        parsed.setdefault("failure_kind", "correctness_mismatch")
        parsed["failure_signature"] = f"mismatch:{mismatch_match.group(1)}"

    runtime_match = RUNTIME_ERROR_RE.search(text)
    if runtime_match:
        runtime_error = runtime_match.group(1).strip()
        parsed["runtime_error"] = runtime_error
        parsed["failure_signature"] = runtime_error
        parsed.setdefault("failure_kind", "runtime_error")

    benchmark_sections = _parse_benchmark_sections(lines)
    benchmark_cases = benchmark_sections.get("benchmark", [])
    ranked_cases = benchmark_sections.get("ranked_benchmark", [])

    if benchmark_cases:
        parsed["benchmarks"] = benchmark_cases
        parsed.update(_summarize_section("benchmark", benchmark_cases))

    if ranked_cases:
        parsed["ranked_benchmarks"] = ranked_cases
        parsed.update(_summarize_section("ranked_benchmark", ranked_cases))

    if "ranked_benchmark_geom_mean_ns" in parsed:
        parsed["geom_mean_ns"] = parsed["ranked_benchmark_geom_mean_ns"]
        parsed["mean_of_means_ns"] = parsed["ranked_benchmark_mean_of_means_ns"]
        parsed["partial_geom_mean_ns"] = parsed["ranked_benchmark_partial_geom_mean_ns"]
        parsed["partial_mean_of_means_ns"] = parsed["ranked_benchmark_partial_mean_of_means_ns"]
        parsed["objective_section"] = "ranked_benchmark"
        if parsed.get("ranked_benchmark_fail_count", 0) == 0:
            parsed["check"] = "pass"
    elif "benchmark_geom_mean_ns" in parsed:
        parsed["geom_mean_ns"] = parsed["benchmark_geom_mean_ns"]
        parsed["mean_of_means_ns"] = parsed["benchmark_mean_of_means_ns"]
        parsed["partial_geom_mean_ns"] = parsed["benchmark_partial_geom_mean_ns"]
        parsed["partial_mean_of_means_ns"] = parsed["benchmark_partial_mean_of_means_ns"]
        parsed["objective_section"] = "benchmark"
        if parsed.get("benchmark_fail_count", 0) == 0:
            parsed["check"] = "pass"

    if parsed.get("ranked_benchmark_fail_count", 0) > 0 or parsed.get("benchmark_fail_count", 0) > 0:
        parsed["check"] = "fail"

    return parsed


def _parse_benchmark_sections(lines: list[str]) -> dict[str, list[dict[str, object]]]:
    sections: dict[str, list[dict[str, object]]] = {
        "benchmark": [],
        "ranked_benchmark": [],
    }
    current_section: str | None = None
    in_code = False
    current_case: dict[str, object] | None = None

    def flush() -> None:
        nonlocal current_case
        if current_section and current_case is not None:
            sections[current_section].append(current_case)
            current_case = None

    for raw_line in lines:
        stripped = raw_line.strip()
        section_match = SECTION_RE.match(stripped)
        if section_match:
            flush()
            current_section = (
                "ranked_benchmark"
                if section_match.group(1) == "Ranked Benchmark"
                else "benchmark"
            )
            in_code = False
            continue

        if current_section is None:
            continue

        if stripped == "```":
            if in_code:
                flush()
            in_code = not in_code
            continue

        if not in_code:
            continue

        if not stripped:
            continue

        if "failed testing" in stripped and current_case is not None:
            current_case["status"] = "fail"
            continue

        measure_match = MEASURE_RE.search(stripped)
        if measure_match and current_case is not None:
            mean = float(measure_match.group("mean"))
            current_case["mean_ns"] = _to_ns(mean, measure_match.group("unit"))
            current_case["status"] = "pass"
            continue

        case_data = _parse_case_line(stripped)
        if case_data is not None:
            flush()
            current_case = case_data

    flush()
    return sections


def _parse_case_line(line: str) -> dict[str, object] | None:
    if line.startswith("⚡") or line.startswith(">"):
        return None

    legacy_match = BENCHMARK_CASE_RE.match(line)
    if legacy_match:
        return {
            "k": int(legacy_match.group("k")),
            "m": int(legacy_match.group("m")),
            "n": int(legacy_match.group("n")),
            "seed": int(legacy_match.group("seed")),
            "status": "fail"
            if legacy_match.group("failed") or "failed testing" in legacy_match.group("suffix")
            else "pending",
        }

    normalized = line
    initial_status = "pending"
    if normalized.startswith("❌ "):
        normalized = normalized[2:].strip()
        initial_status = "fail"
    elif normalized.startswith("✅ "):
        normalized = normalized[2:].strip()

    if ":" not in normalized:
        return None

    fields: dict[str, object] = {}
    for chunk in normalized.split(";"):
        piece = chunk.strip()
        if not piece or ":" not in piece:
            continue
        key, value = piece.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        fields[key] = _coerce_value(value)

    if not fields:
        return None

    fields["status"] = initial_status
    return fields


def _summarize_section(prefix: str, cases: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        f"{prefix}_case_count": len(cases),
    }
    pass_means = [
        float(case["mean_ns"])
        for case in cases
        if isinstance(case.get("mean_ns"), (int, float))
    ]
    pass_count = sum(1 for case in cases if case.get("status") == "pass")
    fail_count = sum(1 for case in cases if case.get("status") == "fail")
    summary[f"{prefix}_pass_count"] = pass_count
    summary[f"{prefix}_fail_count"] = fail_count
    if pass_means:
        summary[f"{prefix}_partial_geom_mean_ns"] = _geometric_mean(pass_means)
        summary[f"{prefix}_partial_mean_of_means_ns"] = statistics.fmean(pass_means)
        if fail_count == 0:
            summary[f"{prefix}_geom_mean_ns"] = _geometric_mean(pass_means)
            summary[f"{prefix}_mean_of_means_ns"] = statistics.fmean(pass_means)
    return summary


def _to_ns(value: float, unit: str) -> float:
    if unit == "ns":
        return value
    if unit in {"µs", "us"}:
        return value * 1_000.0
    if unit == "ms":
        return value * 1_000_000.0
    if unit == "s":
        return value * 1_000_000_000.0
    raise ValueError(f"unsupported time unit: {unit}")
