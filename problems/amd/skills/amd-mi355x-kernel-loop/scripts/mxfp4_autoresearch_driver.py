#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


VARIANT_RE = re.compile(r"\bv(\d+)\b")


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


def _next_variant_name(status: dict[str, Any]) -> str:
    max_num = 0
    for key in status.get("latest_variants", {}).keys():
        match = VARIANT_RE.search(str(key))
        if match:
            max_num = max(max_num, int(match.group(1)))
    return f"v{max_num + 1}"


def _budget_exhausted(status: dict[str, Any]) -> bool:
    budget = status.get("budget", {})
    return (
        int(budget.get("shared_test_bucket_usable", 0)) <= 0
        and int(budget.get("leaderboard_remaining", 0)) <= 0
    )


def _build_prompt(repo: Path, variant: str, source_path: Path, status: dict[str, Any]) -> str:
    best_variant = status.get("current_best_variant", "unknown")
    best_geomean = status.get("current_best_geomean_us")
    return f"""
You are running inside the repo-local mxfp4 autoresearch loop for /Users/v/reference-kernels/problems/amd.

Work on mxfp4_mm only.

Read and follow these local files first:
- {repo / 'skills/amd-mi355x-kernel-loop/SKILL.md'}
- {repo / 'skills/amd-mi355x-kernel-loop/references/mxfp4-subagent-prompt.md'}
- {repo / 'skills/amd-mi355x-kernel-loop/references/mxfp4-exact-shape-frontier.md'}
- {repo / 'skills/amd-mi355x-kernel-loop/references/mxfp4-profile-branch-queue.md'}

Current best measured trunk is {best_variant} at {best_geomean:.4f} us.

Research doctrine:
- use exactly three sub-agents/scouts
- every scout must use gpt-5.4 with reasoning_effort=xhigh
- think in an Atom-of-Thoughts style
- one exact-shape lane only
- one deleted cost center only
- if the lane is any A-pack reopen, require reuse/duplication proof before code
- local exact-shape A-pack feeder deletion is already closed by v121, v122, and v125
- benchmark immediately after a green test
- profile only after a >=5% geomean win over the current best measured trunk

Your task for this round:
1. Launch one mxfp4 research round using the local canon and the three-scout xhigh workflow.
2. Decide the narrowest legal next lane that could plausibly cut whole-call microseconds.
3. If the lane is not viable on paper, stop at a paper veto and update the local frontier/queue notes if needed.
4. If viable, create or update exactly one candidate branch at:
   {source_path}
5. Register it in the closed-loop ledger as variant {variant}.
6. Run preflight.
7. If preflight is acceptable, spend remote test.
8. If test is green, spend benchmark immediately.
9. Update the frontier/queue canon with the new result.

Ground rules:
- do not work on any problem other than mxfp4_mm
- do not reopen local A-pack feeder rewrites unless you can prove a new duplication law
- do not touch more than one exact shape in this round
- do not leave the repo in a half-implemented state

Return strict JSON matching the provided schema only.
""".strip()


def _schema_path(tmpdir: Path) -> Path:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["implemented", "paper_veto", "blocked", "no_change"],
            },
            "variant": {"type": "string"},
            "source_path": {"type": ["string", "null"]},
            "lane": {"type": ["string", "null"]},
            "test_status": {"type": ["string", "null"]},
            "benchmark_status": {"type": ["string", "null"]},
            "benchmark_geomean_us": {"type": ["number", "null"]},
            "summary": {"type": "string"},
            "next_focus": {"type": "string"},
            "changed_files": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "decision",
            "variant",
            "source_path",
            "lane",
            "test_status",
            "benchmark_status",
            "benchmark_geomean_us",
            "summary",
            "next_focus",
            "changed_files",
        ],
    }
    path = tmpdir / "round_schema.json"
    path.write_text(json.dumps(schema, indent=2))
    return path


def _blocked_result(variant: str, source_path: Path, reason: str) -> dict[str, Any]:
    return {
        "decision": "blocked",
        "variant": variant,
        "source_path": str(source_path),
        "lane": None,
        "test_status": None,
        "benchmark_status": None,
        "benchmark_geomean_us": None,
        "summary": reason,
        "next_focus": "Repair the autoresearch driver or manually resume the next round.",
        "changed_files": [],
    }


def _artifacts_dir(repo: Path) -> Path:
    path = repo / ".agent-loop" / "closed_loop" / "mxfp4_mm" / "autoresearch_driver"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_journal(repo: Path, record: dict[str, Any]) -> None:
    journal_path = _artifacts_dir(repo) / "rounds.jsonl"
    with journal_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def _coerce_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_codex_round(
    repo: Path,
    prompt: str,
    schema_path: Path,
    output_path: Path,
    model: str,
    *,
    variant: str,
    source_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    cmd = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--disable",
        "codex_hooks",
        "--enable",
        "multi_agent",
        "-C",
        str(repo),
        "-m",
        model,
        "-c",
        'model_reasoning_effort="xhigh"',
        "-c",
        'approval_policy="never"',
        "-c",
        'sandbox_mode="danger-full-access"',
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        artifacts_dir = _artifacts_dir(repo)
        (artifacts_dir / f"{variant}.stdout.txt").write_text(_coerce_text(exc.stdout))
        (artifacts_dir / f"{variant}.stderr.txt").write_text(_coerce_text(exc.stderr))
        return _blocked_result(
            variant,
            source_path,
            f"codex exec timed out after {timeout_seconds:.0f}s; see {artifacts_dir / f'{variant}.stdout.txt'} and {artifacts_dir / f'{variant}.stderr.txt'}",
        )
    artifacts_dir = _artifacts_dir(repo)
    (artifacts_dir / f"{variant}.stdout.txt").write_text(proc.stdout)
    (artifacts_dir / f"{variant}.stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        return _blocked_result(
            variant,
            source_path,
            f"codex exec failed with code {proc.returncode}; see {artifacts_dir / f'{variant}.stderr.txt'}",
        )
    if not output_path.exists():
        return _blocked_result(
            variant,
            source_path,
            f"codex exec produced no round result; see {artifacts_dir / f'{variant}.stdout.txt'} and {artifacts_dir / f'{variant}.stderr.txt'}",
        )
    raw = output_path.read_text().strip()
    if not raw:
        return _blocked_result(
            variant,
            source_path,
            f"codex exec wrote an empty round result; see {artifacts_dir / f'{variant}.stdout.txt'} and {artifacts_dir / f'{variant}.stderr.txt'}",
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        return _blocked_result(
            variant,
            source_path,
            f"codex exec returned invalid JSON ({exc}); see {artifacts_dir / f'{variant}.stdout.txt'} and {artifacts_dir / f'{variant}.stderr.txt'}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous Codex-driven mxfp4 research loop.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", default="agent_loop.toml")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--round-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--stop-when-budget-exhausted", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    rounds: list[dict[str, Any]] = []

    for _ in range(args.max_rounds):
        status = _run_status(repo, args.config)
        if args.stop_when_budget_exhausted and _budget_exhausted(status):
            break

        variant = _next_variant_name(status)
        source_dir = repo / ".agent-loop" / "manual" / f"autoresearch_{variant}"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_path = source_dir / "submission.py"

        with tempfile.TemporaryDirectory(prefix="mxfp4_autoresearch_") as td:
            tmpdir = Path(td)
            schema_path = _schema_path(tmpdir)
            output_path = tmpdir / "round_result.json"
            prompt = _build_prompt(repo, variant, source_path, status)
            result = _run_codex_round(
                repo,
                prompt,
                schema_path,
                output_path,
                args.model,
                variant=variant,
                source_path=source_path,
                timeout_seconds=args.round_timeout_seconds,
            )
            rounds.append(result)
            _append_journal(
                repo,
                {
                    "variant": variant,
                    "timestamp": time.time(),
                    "result": result,
                },
            )
            print(json.dumps({"round": len(rounds), "variant": variant, "decision": result.get("decision")}), file=sys.stderr)

        if result.get("decision") in {"paper_veto", "blocked"}:
            break

        time.sleep(args.poll_seconds)

    print(json.dumps({"rounds": rounds}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
