from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .config import load_config
from .handroll import HandrolledOptimizer
from .harness import KernelHarness
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
    loop.add_argument("--mode")
    loop.add_argument("--family")

    swarm = sub.add_parser(
        "swarm",
        help="Run round-robin closed-loop iterations across multiple problems",
    )
    swarm.add_argument("--rounds", type=int, default=1)
    swarm.add_argument("--problems", nargs="*")
    swarm.add_argument("--hypothesis-prefix", default="swarm")
    swarm.add_argument("--mode")
    swarm.add_argument("--family")
    swarm.add_argument("--bootstrap-baseline", action="store_true")

    campaign = sub.add_parser(
        "campaign",
        help="Run a long-lived round-robin mutation campaign with plateau stop criteria",
    )
    campaign.add_argument("--rounds", type=int, default=100)
    campaign.add_argument("--problems", nargs="*")
    campaign.add_argument("--hypothesis-prefix", default="campaign")
    campaign.add_argument("--mode", default="leaderboard")
    campaign.add_argument("--family")
    campaign.add_argument("--sleep-seconds", type=float, default=0.0)
    campaign.add_argument("--max-consecutive-non-improve", "--stall-limit", dest="max_consecutive_non_improve", type=int, default=8)
    campaign.add_argument("--max-check-fails", type=int, default=3)
    campaign.add_argument("--continue-on-error", action="store_true")
    campaign.add_argument("--bootstrap-baseline", action="store_true")

    campaign_parallel = sub.add_parser(
        "campaign-parallel",
        help="Launch one detached long-lived campaign process per problem",
    )
    campaign_parallel.add_argument("--rounds", type=int, default=1000)
    campaign_parallel.add_argument("--problems", nargs="*")
    campaign_parallel.add_argument("--hypothesis-prefix", default="campaign-parallel")
    campaign_parallel.add_argument("--mode", default="leaderboard")
    campaign_parallel.add_argument("--family")
    campaign_parallel.add_argument("--sleep-seconds", type=float, default=1.0)
    campaign_parallel.add_argument("--max-consecutive-non-improve", "--stall-limit", dest="max_consecutive_non_improve", type=int, default=20)
    campaign_parallel.add_argument("--max-check-fails", type=int, default=3)
    campaign_parallel.add_argument("--continue-on-error", action="store_true")
    campaign_parallel.add_argument("--bootstrap-baseline", action="store_true")
    campaign_parallel.add_argument("--stagger-seconds", type=float, default=2.0)

    status = sub.add_parser("status", help="Show problem state")
    status.add_argument("--problem")

    cleanup = sub.add_parser(
        "cleanup",
        help="Prune failed and stale mutation candidates while keeping compact summaries",
    )
    cleanup.add_argument("--problem")
    cleanup.add_argument("--stale-pending-hours", type=float, default=6.0)

    reset_problem = sub.add_parser(
        "reset-problem",
        help="Delete one problem's local workspace, candidates, pruned summaries, and DB state",
    )
    reset_problem.add_argument("--problem", required=True)

    promote = sub.add_parser("promote", help="Promote a stored candidate into the repo")
    promote.add_argument("--problem", required=True)
    promote.add_argument("--candidate", required=True)

    harness_run = sub.add_parser(
        "harness-run",
        help="Run a KernelBench-style staged harness over one submission source",
    )
    harness_run.add_argument("--problem", required=True)
    harness_run.add_argument("--source")
    harness_run.add_argument("--family")
    harness_run.add_argument("--label", default="")
    harness_run.add_argument("--stages", default="test,benchmark,leaderboard")
    harness_run.add_argument("--continue-after-fail", action="store_true")

    harness_resume = sub.add_parser(
        "harness-resume",
        help="Resume a previously created harness run directory",
    )
    harness_resume.add_argument("--problem", required=True)
    harness_resume.add_argument("--run-dir")
    harness_resume.add_argument("--continue-after-fail", action="store_true")

    harness_summary = sub.add_parser(
        "harness-summary",
        help="Summarize a harness run directory or the latest run for a problem",
    )
    harness_summary.add_argument("--problem", required=True)
    harness_summary.add_argument("--run-dir")

    handroll = sub.add_parser(
        "handroll-campaign",
        help="Run a hand-rolled keep/revert optimization loop from the current tracked working seed",
    )
    handroll.add_argument("--problem", required=True)
    handroll.add_argument("--rounds", type=int, default=1)
    handroll.add_argument("--sleep-seconds", type=float, default=0.0)
    handroll.add_argument("--stages", default="test,benchmark")
    handroll.add_argument("--leaderboard-on-improve", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command in {"harness-run", "harness-resume", "harness-summary"}:
        return _run_harness_command(config, args)

    if args.command == "handroll-campaign":
        optimizer = HandrolledOptimizer(config)
        records = optimizer.run_campaign(
            problem_key=args.problem,
            rounds=args.rounds,
            stages=[item.strip() for item in args.stages.split(",") if item.strip()],
            sleep_seconds=args.sleep_seconds,
            leaderboard_on_improve=bool(args.leaderboard_on_improve),
        )
        for record in records:
            print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    if args.command == "healthcheck":
        return _run_healthcheck(config, args.problem)

    if args.command == "campaign-parallel":
        return _launch_parallel_campaigns(config, args)

    runner = ClosedLoopRunner(config)
    try:
        if args.command == "baseline":
            summary = runner.run_baseline(args.problem, mode=args.mode)
            print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
            return 0

        if args.command == "loop":
            for iteration in range(args.iterations):
                desired_family = _problem_family(config, args.problem, args.family)
                selected_mode = runner.recommended_mode(
                    args.problem,
                    requested_mode=args.mode or config.require_problem(args.problem).mode,
                    desired_family=desired_family,
                )
                summary = runner.run_iteration(
                    args.problem,
                    hypothesis=args.hypothesis or f"iteration {iteration + 1}",
                    mutator_command=args.mutator_command,
                    mode=selected_mode,
                    desired_family=desired_family,
                )
                print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
            return 0

        if args.command == "swarm":
            selected = args.problems or sorted(config.problems)
            if args.bootstrap_baseline:
                for problem_key in selected:
                    summary = runner.bootstrap_problem(problem_key, mode=args.mode)
                    print(json.dumps({"stage": "baseline", **summary.__dict__}, indent=2, sort_keys=True))
            for round_index in range(args.rounds):
                for problem_key in selected:
                    desired_family = _problem_family(config, problem_key, args.family)
                    selected_mode = runner.recommended_mode(
                        problem_key,
                        requested_mode=args.mode or config.require_problem(problem_key).mode,
                        desired_family=desired_family,
                    )
                    summary = runner.run_iteration(
                        problem_key,
                        hypothesis=f"{args.hypothesis_prefix} round {round_index + 1} {problem_key}",
                        mode=selected_mode,
                        desired_family=desired_family,
                    )
                    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
            return 0

        if args.command == "campaign":
            selected = args.problems or sorted(config.problems)
            plateau_counts = {problem_key: 0 for problem_key in selected}
            check_fail_counts = {problem_key: 0 for problem_key in selected}
            if args.bootstrap_baseline:
                for problem_key in selected:
                    summary = runner.bootstrap_problem(problem_key, mode=args.mode)
                    print(json.dumps({"stage": "baseline", **summary.__dict__}, indent=2, sort_keys=True))
            for round_index in range(args.rounds):
                active = False
                for problem_key in selected:
                    if plateau_counts[problem_key] >= args.max_consecutive_non_improve:
                        continue
                    if check_fail_counts[problem_key] >= args.max_check_fails:
                        continue
                    active = True
                    desired_family = _problem_family(config, problem_key, args.family)
                    selected_mode = runner.recommended_mode(
                        problem_key,
                        requested_mode=args.mode or config.require_problem(problem_key).mode,
                        desired_family=desired_family,
                    )
                    try:
                        summary = runner.run_iteration(
                            problem_key,
                            hypothesis=f"{args.hypothesis_prefix} round {round_index + 1} {problem_key}",
                            mode=selected_mode,
                            desired_family=desired_family,
                        )
                    except Exception as exc:
                        if not args.continue_on_error:
                            raise
                        plateau_counts[problem_key] += 1
                        print(
                            json.dumps(
                                {
                                    "problem": problem_key,
                                    "round": round_index + 1,
                                    "status": "runner_error",
                                    "error": str(exc),
                                    "plateau_count": plateau_counts[problem_key],
                                    "check_fail_count": check_fail_counts[problem_key],
                                },
                                indent=2,
                                sort_keys=True,
                            )
                        )
                    else:
                        if summary.status == "check_fail":
                            check_fail_counts[problem_key] += 1
                        else:
                            check_fail_counts[problem_key] = 0
                        plateau_counts[problem_key] = 0 if summary.improved else plateau_counts[problem_key] + 1
                        payload = dict(summary.__dict__)
                        payload["round"] = round_index + 1
                        payload["plateau_count"] = plateau_counts[problem_key]
                        payload["check_fail_count"] = check_fail_counts[problem_key]
                        print(json.dumps(payload, indent=2, sort_keys=True))
                        if check_fail_counts[problem_key] >= args.max_check_fails:
                            print(
                                json.dumps(
                                    {
                                        "status": "campaign_halted",
                                        "reason": "max_check_fails_reached",
                                        "problem": problem_key,
                                        "round": round_index + 1,
                                        "max_check_fails": args.max_check_fails,
                                        "last_candidate_id": summary.candidate_id,
                                    },
                                    indent=2,
                                    sort_keys=True,
                                )
                            )
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                if not active:
                    print(
                        json.dumps(
                            {
                                "status": "campaign_complete",
                                "reason": "all problems reached plateau threshold or check-fail threshold",
                                "max_consecutive_non_improve": args.max_consecutive_non_improve,
                                "max_check_fails": args.max_check_fails,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                    )
                    break
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

        if args.command == "cleanup":
            summary = runner.compact_failed_candidates(
                args.problem,
                stale_pending_hours=args.stale_pending_hours,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        if args.command == "reset-problem":
            summary = runner.reset_problem_workspace(args.problem)
            print(json.dumps(summary, indent=2, sort_keys=True))
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


def _problem_family(config, problem_key: str, requested_family: str | None) -> str | None:
    if requested_family:
        return requested_family
    return config.require_problem(problem_key).default_family


def _run_harness_command(config, args) -> int:
    harness = KernelHarness(config)
    if args.command == "harness-run":
        problem = config.require_problem(args.problem)
        source_path = Path(args.source) if args.source else problem.submission_path
        stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
        run_dir = harness.create_run(
            args.problem,
            source_path=source_path,
            stages=stages,
            family=args.family,
            label=args.label,
        )
        summary = harness.resume_run(run_dir, continue_after_fail=args.continue_after_fail)
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "harness-resume":
        run_dir = Path(args.run_dir) if args.run_dir else harness.latest_run_dir(args.problem)
        if run_dir is None:
            raise SystemExit(f"no harness run found for {args.problem}")
        summary = harness.resume_run(run_dir, continue_after_fail=args.continue_after_fail)
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "harness-summary":
        run_dir = Path(args.run_dir) if args.run_dir else harness.latest_run_dir(args.problem)
        if run_dir is None:
            raise SystemExit(f"no harness run found for {args.problem}")
        summary = harness.summary(run_dir)
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0

    raise SystemExit(f"unknown harness command: {args.command}")


def _run_healthcheck(config, problem_key: str | None) -> int:
    report = {
        "config_path": str(config.config_path),
        "workspace_root": str(config.workspace.root),
        "popcorn_cli": config.workspace.popcorn_cli,
        "api_url": config.workspace.api_url or "(inherit popcorn-cli default)",
        "leaderboard_reference_seconds": config.workspace.leaderboard_reference_seconds,
        "leaderboard_timeout_seconds": config.workspace.leaderboard_timeout_seconds,
        "llm": {
            "enabled": config.llm.enabled,
            "provider": config.llm.provider,
            "model": config.llm.model,
            "api_url": config.llm.api_url,
            "api_key_env_var": config.llm.api_key_env_var,
            "api_key_present": bool(os.environ.get(config.llm.api_key_env_var)),
            "anthropic_api_url": config.llm.anthropic_api_url,
            "anthropic_api_key_env_var": config.llm.anthropic_api_key_env_var,
            "anthropic_api_key_present": bool(os.environ.get(config.llm.anthropic_api_key_env_var)),
            "anthropic_version": config.llm.anthropic_version,
            "openrouter_api_url": config.llm.openrouter_api_url,
            "openrouter_api_key_env_var": config.llm.openrouter_api_key_env_var,
            "openrouter_api_key_present": bool(os.environ.get(config.llm.openrouter_api_key_env_var)),
            "openrouter_http_referer": config.llm.openrouter_http_referer,
            "openrouter_title": config.llm.openrouter_title,
            "reasoning_effort": config.llm.reasoning_effort,
            "max_output_tokens": config.llm.max_output_tokens,
            "fallback_to_seed": config.llm.fallback_to_seed,
            "codex_cli": config.llm.codex_cli,
            "codex_model": config.llm.codex_model,
            "codex_path": shutil.which(config.llm.codex_cli),
            "codex_profile": config.llm.codex_profile,
            "codex_sandbox": config.llm.codex_sandbox,
            "codex_use_plan": config.llm.codex_use_plan,
            "codex_parallel_agents": config.llm.codex_parallel_agents,
        },
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


def _launch_parallel_campaigns(config, args) -> int:
    selected = args.problems or sorted(config.problems)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logs_dir = config.workspace.root / "logs" / f"parallel-{timestamp}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    launched: list[dict[str, object]] = []
    for index, problem_key in enumerate(selected):
        log_path = logs_dir / f"{problem_key}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "agent_loop",
            "--config",
            str(config.config_path),
            "campaign",
            "--rounds",
            str(args.rounds),
            "--problems",
            problem_key,
            "--hypothesis-prefix",
            f"{args.hypothesis_prefix}-{problem_key}",
            "--mode",
            args.mode,
            "--stall-limit",
            str(args.max_consecutive_non_improve),
            "--max-check-fails",
            str(args.max_check_fails),
            "--sleep-seconds",
            str(args.sleep_seconds),
        ]
        family = _problem_family(config, problem_key, args.family)
        if family:
            command.extend(["--family", family])
        if args.continue_on_error:
            command.append("--continue-on-error")
        if args.bootstrap_baseline:
            command.append("--bootstrap-baseline")

        process = subprocess.Popen(
            command,
            cwd=str(config.repo_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        log_handle.close()
        launched.append(
            {
                "problem": problem_key,
                "pid": process.pid,
                "log_path": str(log_path),
                "command": command,
            }
        )
        if index < len(selected) - 1 and args.stagger_seconds > 0:
            time.sleep(args.stagger_seconds)

    manifest = {
        "status": "launched",
        "timestamp": timestamp,
        "logs_dir": str(logs_dir),
        "workers": launched,
    }
    manifest_path = logs_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
