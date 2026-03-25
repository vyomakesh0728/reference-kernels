#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


WINDOW = timedelta(hours=1)
PROBLEM_SPECS = {
    "mxfp4_mm": {
        "closed_loop": "mxfp4-closed-loop",
        "ledger": ".agent-loop/closed_loop/mxfp4_mm/experiment_ledger.jsonl",
    },
    "moe_mxfp4": {
        "closed_loop": "moe-closed-loop",
        "ledger": ".agent-loop/closed_loop/moe_mxfp4/experiment_ledger.jsonl",
    },
}


def _parse_ts(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if len(text) >= 5 and (text[-5] in {"+", "-"}) and text[-3] != ":":
        text = text[:-2] + ":" + text[-2:]
    return datetime.fromisoformat(text)


def _format_ts(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _python_for_repo(repo: Path) -> str:
    venv_python = repo / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _run_status(repo: Path, config: str, problem: str) -> dict[str, Any]:
    spec = PROBLEM_SPECS[problem]
    proc = subprocess.run(
        [
            _python_for_repo(repo),
            "-m",
            "agent_loop",
            "--config",
            config,
            str(spec["closed_loop"]),
            "status",
        ],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(proc.stdout)


def _load_ledger(repo: Path, problem: str) -> list[dict[str, Any]]:
    path = repo / str(PROBLEM_SPECS[problem]["ledger"])
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _history_times(
    ledger: list[dict[str, Any]],
    stages: set[str],
    now: datetime,
) -> list[datetime]:
    out: list[datetime] = []
    cutoff = now - WINDOW
    for rec in ledger:
        for item in rec.get("remote_history", []):
            if item.get("stage") not in stages:
                continue
            raw = item.get("requested_at")
            if not raw:
                continue
            ts = _parse_ts(raw)
            if ts >= cutoff:
                out.append(ts)
    out.sort()
    return out


def _eligible_from_window(timestamps: list[datetime], limit: int, now: datetime) -> datetime:
    if limit <= 0:
        return now + WINDOW
    if len(timestamps) < limit:
        return now
    idx = len(timestamps) - limit
    return timestamps[idx] + WINDOW


def _latest_stage_state(ledger: list[dict[str, Any]], variant: str, stage: str) -> dict[str, Any] | None:
    for rec in reversed(ledger):
        if rec.get("variant") != variant:
            continue
        for item in reversed(rec.get("remote_history", [])):
            if item.get("stage") == stage:
                return item
        return None
    return None


@dataclass
class Eligibility:
    when: datetime
    reason: str


def _compute_eligibility(
    status: dict[str, Any],
    ledger: list[dict[str, Any]],
    stage: str,
    now: datetime,
) -> Eligibility:
    budget = status["budget"]
    if stage == "leaderboard":
        leaderboard_times = _history_times(ledger, {"leaderboard"}, now)
        stage_ready = _eligible_from_window(
            leaderboard_times,
            int(budget["leaderboard_limit_per_hour"]),
            now,
        )
        policy_ready_after_raw = budget.get("leaderboard_policy_ready_after")
        if policy_ready_after_raw:
            policy_ready_after = _parse_ts(str(policy_ready_after_raw))
            ready_at = max(policy_ready_after, stage_ready)
        else:
            ready_at = stage_ready
        return Eligibility(when=ready_at, reason="leaderboard gate")

    shared_times = _history_times(ledger, {"test", "benchmark"}, now)
    shared_ready = _eligible_from_window(
        shared_times,
        int(budget["shared_test_bucket_usable"]),
        now,
    )

    stage_limit_key = f"{stage}_stage_limit_per_hour"
    stage_times = _history_times(ledger, {stage}, now)
    stage_ready = _eligible_from_window(
        stage_times,
        int(budget[stage_limit_key]),
        now,
    )
    ready_at = max(shared_ready, stage_ready)
    return Eligibility(when=ready_at, reason=f"{stage}+shared gate")


def _write_watch_files(
    repo: Path,
    problem: str,
    variant: str,
    stage: str,
    pid: int,
    when: datetime,
    submit_cmd: list[str],
) -> tuple[Path, Path]:
    out_dir = repo / ".agent-loop/quota_watch"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{variant}-{stage}"
    pid_path = out_dir / f"{stem}.pid"
    meta_path = out_dir / f"{stem}.json"
    pid_path.write_text(f"{pid}\n")
    meta_path.write_text(
        json.dumps(
            {
                "variant": variant,
                "problem": problem,
                "stage": stage,
                "pid": pid,
                "resume_at_utc": _format_ts(when),
                "submit_cmd": submit_cmd,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return pid_path, meta_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for closed-loop quota reset, then auto-submit one stage.")
    parser.add_argument("--repo", required=True, help="Repo root, e.g. /Users/v/reference-kernels/problems/amd")
    parser.add_argument("--problem", default="mxfp4_mm", choices=tuple(PROBLEM_SPECS))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--hot-path-state", default="")
    parser.add_argument("--regime-tag", default="")
    parser.add_argument("--replaced-stage", action="append", default=[])
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--expected-gain", default="")
    parser.add_argument("--next-patch", default="")
    parser.add_argument("--deleted-cost-center", default="")
    parser.add_argument("--expected-upside-source", default="")
    parser.add_argument("--why-larger-than-noise", default="")
    parser.add_argument("--forbidden-edit", action="append", default=[])
    parser.add_argument("--success-gate", default="")
    parser.add_argument("--motivation-ref", action="append", default=[])
    parser.add_argument("--retrieval-query", action="append", default=[])
    parser.add_argument("--stage", required=True, choices=("test", "benchmark", "leaderboard"))
    parser.add_argument("--config", default="agent_loop.toml")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    spec = PROBLEM_SPECS[args.problem]
    now = datetime.now(timezone.utc)
    status = _run_status(repo, args.config, args.problem)
    ledger = _load_ledger(repo, args.problem)

    latest = _latest_stage_state(ledger, args.variant, args.stage)
    if latest and latest.get("status") in {"requested", "running"} and not latest.get("finished_at"):
        print(
            json.dumps(
                {
                    "status": "already_pending",
                    "stage": args.stage,
                    "variant": args.variant,
                    "requested_at": latest.get("requested_at"),
                    "run_dir": latest.get("run_dir"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    eligibility = _compute_eligibility(status, ledger, args.stage, now)
    submit_cmd = [
        _python_for_repo(repo),
        "-m",
        "agent_loop",
        "--config",
        args.config,
        str(spec["closed_loop"]),
        "submit",
        "--variant",
        args.variant,
        "--source",
        str(Path(args.source).resolve()),
        "--lane",
        args.lane,
        "--stage",
        args.stage,
    ]
    if args.problem == "moe_mxfp4":
        if not args.hot_path_state or not args.regime_tag:
            raise SystemExit("--hot-path-state and --regime-tag are required for moe_mxfp4")
        submit_cmd.extend(
            [
                "--hot-path-state",
                args.hot_path_state,
                "--regime-tag",
                args.regime_tag,
            ]
        )
        for item in args.replaced_stage:
            submit_cmd.extend(["--replaced-stage", item])
        if args.hypothesis:
            submit_cmd.extend(["--hypothesis", args.hypothesis])
        if args.expected_gain:
            submit_cmd.extend(["--expected-gain", args.expected_gain])
        if args.next_patch:
            submit_cmd.extend(["--next-patch", args.next_patch])
        if args.deleted_cost_center:
            submit_cmd.extend(["--deleted-cost-center", args.deleted_cost_center])
        if args.expected_upside_source:
            submit_cmd.extend(["--expected-upside-source", args.expected_upside_source])
        if args.why_larger_than_noise:
            submit_cmd.extend(["--why-larger-than-noise", args.why_larger_than_noise])
        for item in args.forbidden_edit:
            submit_cmd.extend(["--forbidden-edit", item])
        if args.success_gate:
            submit_cmd.extend(["--success-gate", args.success_gate])
        for item in args.motivation_ref:
            submit_cmd.extend(["--motivation-ref", item])
        for item in args.retrieval_query:
            submit_cmd.extend(["--retrieval-query", item])
    pid_path, meta_path = _write_watch_files(
        repo,
        args.problem,
        args.variant,
        args.stage,
        os.getpid(),
        eligibility.when,
        submit_cmd,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "reason": eligibility.reason,
                    "resume_at_utc": _format_ts(eligibility.when),
                    "pid_file": str(pid_path),
                    "meta_file": str(meta_path),
                    "submit_cmd": submit_cmd,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    delay = max(0.0, (eligibility.when - now).total_seconds())
    if delay > 0:
        time.sleep(delay)

    while True:
        status = _run_status(repo, args.config, args.problem)
        ledger = _load_ledger(repo, args.problem)
        latest = _latest_stage_state(ledger, args.variant, args.stage)
        if latest and latest.get("status") in {"requested", "running"} and not latest.get("finished_at"):
            print(
                json.dumps(
                    {
                        "status": "already_pending",
                        "stage": args.stage,
                        "variant": args.variant,
                        "requested_at": latest.get("requested_at"),
                        "run_dir": latest.get("run_dir"),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        eligibility = _compute_eligibility(status, ledger, args.stage, datetime.now(timezone.utc))
        if eligibility.when <= datetime.now(timezone.utc):
            break
        time.sleep(args.poll_seconds)

    proc = subprocess.run(submit_cmd, cwd=repo, text=True, capture_output=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
