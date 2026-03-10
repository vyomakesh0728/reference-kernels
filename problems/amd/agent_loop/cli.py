from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .config import load_config
from .runner import ClosedLoopRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent_loop")
    parser.add_argument(
        "--config",
        default="agent_loop.toml",
        help="Path to the TOML config file",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    healthcheck = sub.add_parser("healthcheck", help="Verify local setup")
    healthcheck.add_argument("--problem")

    baseline = sub.add_parser("baseline", help="Register and submit the current repo baseline")
    baseline.add_argument("--problem", required=True)
    baseline.add_argument("--mode")

    loop = sub.add_parser("loop", help="Run N mutation/evaluation iterations")
    loop.add_argument("--problem", required=True)
    loop.add_argument("--iterations", type=int, default=1)
    loop.add_argument("--hypothesis", default="")
    loop.add_argument("--mutator-command")

    swarm = sub.add_parser(
        "swarm",
        help="Run round-robin closed-loop iterations across multiple problems",
    )
    swarm.add_argument("--rounds", type=int, default=1)
    swarm.add_argument("--problems", nargs="*")
    swarm.add_argument("--hypothesis-prefix", default="swarm")

    status = sub.add_parser("status", help="Show problem state")
    status.add_argument("--problem")

    promote = sub.add_parser("promote", help="Promote a stored candidate into the repo")
    promote.add_argument("--problem", required=True)
    promote.add_argument("--candidate", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "healthcheck":
        return _run_healthcheck(config, args.problem)

    runner = ClosedLoopRunner(config)
    try:
        if args.command == "baseline":
            summary = runner.run_baseline(args.problem, mode=args.mode)
            print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
            return 0

        if args.command == "loop":
            for iteration in range(args.iterations):
                summary = runner.run_iteration(
                    args.problem,
                    hypothesis=args.hypothesis or f"iteration {iteration + 1}",
                    mutator_command=args.mutator_command,
                )
                print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
            return 0

        if args.command == "swarm":
            selected = args.problems or sorted(config.problems)
            for round_index in range(args.rounds):
                for problem_key in selected:
                    summary = runner.run_iteration(
                        problem_key,
                        hypothesis=f"{args.hypothesis_prefix} round {round_index + 1} {problem_key}",
                    )
                    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
            return 0

        if args.command == "status":
            if args.problem:
                snapshot = runner.problem_snapshot(args.problem)
            else:
                snapshot = {
                    key: runner.problem_snapshot(key) for key in sorted(config.problems)
                }
            print(json.dumps(snapshot, indent=2, sort_keys=True))
            return 0

        if args.command == "promote":
            runner.promote_candidate(args.problem, args.candidate)
            print(
                json.dumps(
                    {"problem": args.problem, "candidate": args.candidate, "promoted": True},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
    finally:
        runner.close()

    parser.error(f"unsupported command: {args.command}")
    return 2


def _run_healthcheck(config, problem_key: str | None) -> int:
    report = {
        "config_path": str(config.config_path),
        "workspace_root": str(config.workspace.root),
        "popcorn_cli": config.workspace.popcorn_cli,
        "api_url": config.workspace.api_url or "(inherit popcorn-cli default)",
        "leaderboard_reference_seconds": config.workspace.leaderboard_reference_seconds,
        "leaderboard_timeout_seconds": config.workspace.leaderboard_timeout_seconds,
        "problems": sorted(config.problems),
    }
    report["popcorn_cli_path"] = shutil.which(config.workspace.popcorn_cli)
    report["popcorn_cli_version"] = _capture([config.workspace.popcorn_cli, "--version"])
    report["popcorn_auth_present"] = Path.home().joinpath(".popcorn.yaml").exists()
    if problem_key:
        problem = config.require_problem(problem_key)
        report["problem"] = {
            "key": problem.key,
            "submission_path": str(problem.submission_path),
            "submission_exists": problem.submission_path.exists(),
            "leaderboard": problem.leaderboard,
            "gpu": problem.gpu,
            "mode": problem.mode,
            "mutator_command": problem.mutator_command,
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _capture(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return completed.stdout.strip() or completed.stderr.strip()
    return completed.stderr.strip() or completed.stdout.strip() or f"rc={completed.returncode}"


if __name__ == "__main__":
    sys.exit(main())
