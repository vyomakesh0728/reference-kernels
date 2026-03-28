#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _python_for_repo(repo: Path) -> str:
    venv_python = repo / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _run_status(repo: Path, config: str) -> dict[str, Any]:
    proc = subprocess.run(
        [
            _python_for_repo(repo),
            "-m",
            "agent_loop",
            "--config",
            config,
            "mxfp4-closed-loop",
            "status",
        ],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(proc.stdout)


def _run_report(repo: Path, config: str) -> dict[str, Any]:
    proc = subprocess.run(
        [
            _python_for_repo(repo),
            "-m",
            "agent_loop",
            "--config",
            config,
            "mxfp4-closed-loop",
            "status",
            "--report",
        ],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(proc.stdout)


def _shared_budget_exhausted(status: dict[str, Any]) -> bool:
    budget = status.get("budget", {})
    return int(budget.get("shared_test_bucket_usable", 0)) <= 0


def _all_hourly_budget_exhausted(status: dict[str, Any]) -> bool:
    budget = status.get("budget", {})
    return (
        int(budget.get("shared_test_bucket_remaining_before_reserve", 0)) <= 0
        and int(budget.get("leaderboard_remaining", 0)) <= 0
    )


def _watch_meta_path(repo: Path, variant: str, stage: str) -> Path:
    return repo / ".agent-loop" / "quota_watch" / f"{variant}-{stage}.json"


def _watch_is_live(repo: Path, variant: str, stage: str) -> bool:
    return _watch_meta_path(repo, variant, stage).exists()


def _start_watch(repo: Path, config: str, entry: dict[str, Any], stage: str) -> None:
    log_dir = repo / ".agent-loop" / "quota_watch"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{entry['variant']}-{stage}.autoloop.log"
    cmd = [
        _python_for_repo(repo),
        str(repo / "skills" / "amd-mi355x-kernel-loop" / "scripts" / "quota_watch_resume.py"),
        "--repo",
        str(repo),
        "--problem",
        "mxfp4_mm",
        "--variant",
        str(entry["variant"]),
        "--source",
        str((repo / str(entry["source"])).resolve()),
        "--lane",
        str(entry["lane"]),
        "--stage",
        stage,
        "--config",
        config,
    ]
    for key, arg_name in (
        ("hypothesis", "--hypothesis"),
        ("expected_gain", "--expected-gain"),
        ("next_patch", "--next-patch"),
    ):
        value = str(entry.get(key, "")).strip()
        if value:
            cmd.extend([arg_name, value])

    with log_path.open("ab") as log:
        subprocess.Popen(
            cmd,
            cwd=repo,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )


def _best_other_geomean(records: list[dict[str, Any]], variant: str) -> float | None:
    vals = [
        float(rec["benchmark_geomean"])
        for rec in records
        if rec.get("variant") != variant and rec.get("benchmark_status") == "ok" and rec.get("benchmark_geomean") is not None
    ]
    return min(vals) if vals else None


def _maybe_submit_or_watch(
    repo: Path,
    config: str,
    entry: dict[str, Any],
    stage: str,
) -> str:
    cmd = [
        _python_for_repo(repo),
        "-m",
        "agent_loop",
        "--config",
        config,
        "mxfp4-closed-loop",
        "submit",
        "--variant",
        str(entry["variant"]),
        "--source",
        str(entry["source"]),
        "--lane",
        str(entry["lane"]),
        "--stage",
        stage,
    ]
    proc = subprocess.run(cmd, cwd=repo, text=True, capture_output=True)
    if proc.returncode == 0:
        return "submitted"
    stderr = (proc.stderr or "") + (proc.stdout or "")
    if "quota is exhausted" in stderr.lower():
        if not _watch_is_live(repo, str(entry["variant"]), stage):
            _start_watch(repo, config, entry, stage)
            return "watch_started"
        return "watch_exists"
    return f"submit_failed: {stderr.strip()}"


def _next_stage_action(records: list[dict[str, Any]], entry: dict[str, Any]) -> tuple[str | None, str]:
    by_variant = {rec.get("variant"): rec for rec in records}
    rec = by_variant.get(entry["variant"])
    if rec is None:
        return ("test", "new variant")

    if rec.get("test_status") in {"pending", "requested", "running"}:
        return (None, "test already pending")
    if rec.get("test_status") not in {"ok"}:
        return ("test", "test not green")

    if rec.get("benchmark_status") in {"pending", "requested", "running"}:
        return (None, "benchmark already pending")
    if rec.get("benchmark_status") != "ok":
        return ("benchmark", "benchmark not green yet")

    if entry.get("profile_on_win_pct") is None:
        return (None, "profile disabled")
    if rec.get("profile_rocprof_status") in {"pending", "requested", "running", "ok"}:
        return (None, "profile already handled")
    current = rec.get("benchmark_geomean")
    if current is None:
        return (None, "no benchmark geomean")
    other_best = _best_other_geomean(records, str(entry["variant"]))
    if other_best is None:
        return (None, "no comparison baseline")
    improvement_pct = (other_best / float(current) - 1.0) * 100.0
    if improvement_pct >= float(entry["profile_on_win_pct"]):
        return ("profile_rocprof", f"profile triggered by {improvement_pct:.2f}% win")
    return (None, f"win {improvement_pct:.2f}% below profile gate")


def _stage_priority(entry: dict[str, Any]) -> list[str]:
    raw = entry.get("stage_priority")
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw]
    return ["test", "benchmark"]


def _record_by_variant(records: list[dict[str, Any]], variant: str) -> dict[str, Any] | None:
    for rec in records:
        if rec.get("variant") == variant:
            return rec
    return None


def _stage_needed(rec: dict[str, Any] | None, stage: str) -> bool:
    if rec is None:
        return stage == "test"
    if stage == "test":
        return rec.get("test_status") not in {"ok", "requested", "running"}
    if stage == "benchmark":
        return rec.get("benchmark_status") not in {"ok", "requested", "running"}
    if stage == "leaderboard":
        return rec.get("leaderboard_status") not in {"ok", "requested", "running"}
    return False


def _can_attempt_stage(rec: dict[str, Any] | None, stage: str, allow_benchmark_without_test: bool) -> bool:
    if stage == "test":
        return True
    if rec is None:
        return False
    if stage == "benchmark":
        return allow_benchmark_without_test or rec.get("test_status") == "ok"
    if stage == "leaderboard":
        return rec.get("benchmark_status") == "ok"
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep the mxfp4 closed-loop queue moving without babysitting.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", default="agent_loop.toml")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--stop-when-shared-exhausted", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    queue_path = Path(args.queue).resolve()

    while True:
        queue = json.loads(queue_path.read_text())
        status = _run_status(repo, args.config)
        report = _run_report(repo, args.config)
        records = report["records"]
        if args.stop_when_shared_exhausted and _all_hourly_budget_exhausted(status):
            print(json.dumps({
                "queue": str(queue_path),
                "status": "stopped_all_hourly_budget_exhausted",
                "budget": status.get("budget", {}),
            }, indent=2, sort_keys=True))
            return 0

        events: list[dict[str, str]] = []
        for entry in queue.get("entries", []):
            rec = _record_by_variant(records, str(entry["variant"]))
            chosen_stage = None
            chosen_reason = ""
            allow_benchmark_without_test = bool(entry.get("allow_benchmark_without_test", False))
            for stage in _stage_priority(entry):
                if not _stage_needed(rec, stage):
                    continue
                if not _can_attempt_stage(rec, stage, allow_benchmark_without_test):
                    chosen_reason = f"{stage} blocked by prior-stage gate"
                    continue
                chosen_stage = stage
                chosen_reason = f"next priority stage {stage}"
                break

            if chosen_stage is None:
                stage, reason = _next_stage_action(records, entry)
                if stage is None:
                    events.append({"variant": entry["variant"], "status": reason})
                    continue
                if stage == "profile_rocprof":
                    events.append({"variant": entry["variant"], "status": "profile gate hit; submit manually"})
                    continue
                chosen_stage = stage
                chosen_reason = reason

            result = _maybe_submit_or_watch(repo, args.config, entry, chosen_stage)
            events.append({"variant": entry["variant"], "status": f"{chosen_stage}: {result}", "reason": chosen_reason})

        print(json.dumps({"queue": str(queue_path), "events": events}, indent=2, sort_keys=True))
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
