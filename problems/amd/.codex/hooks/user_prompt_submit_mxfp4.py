#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
KEYWORDS = ("mxfp4", "fp8-mm", "mi355x", "a-pack", "benchmark", "leaderboard", "kernel")


def _load_payload() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def _status() -> dict | None:
    try:
        out = subprocess.check_output(
            ["python3", "-m", "agent_loop", "mxfp4-closed-loop", "status"],
            cwd=REPO,
            text=True,
        )
        return json.loads(out)
    except Exception:
        return None


def main() -> int:
    payload = _load_payload()
    cwd = Path(payload.get("cwd") or REPO)
    prompt = str(payload.get("prompt") or "").lower()
    if REPO not in [cwd, *cwd.parents] or not any(k in prompt for k in KEYWORDS):
        print(json.dumps({"continue": True}))
        return 0

    status = _status()
    if not status:
        print(json.dumps({"continue": True}))
        return 0

    best_variant = status.get("current_best_variant", "unknown")
    best_geomean = status.get("current_best_geomean_us")
    context = [
        f"Current best measured mxfp4_mm branch is {best_variant} at {best_geomean:.4f} us."
        if isinstance(best_geomean, (int, float))
        else f"Current best measured mxfp4_mm branch is {best_variant}.",
        "Use skills/amd-mi355x-kernel-loop/SKILL.md and skills/amd-mi355x-kernel-loop/references/mxfp4-subagent-prompt.md explicitly before proposing the next kernel lane.",
        "Run the mxfp4 round as three gpt-5.4 xhigh scouts on one exact-shape lane, then collapse them into one Candidate Card before coding.",
        "Use Atom-of-Thoughts style decomposition: split the bottleneck into small law changes, paper-veto weak branches, then code only one narrow legal hypothesis.",
        "Search priority: whole-call cost-center deletion first, not local A-pack feeder rewrites.",
        "Current A-pack law: only reopen if reuse beats duplication with a legal mechanism; local exact-shape re-quant is closed.",
        "A-pack focus now means duplication-law research, not another local feeder swap.",
        "Remote-first discipline: preflight, test, benchmark, then profile only on >=5% geomean win.",
    ]

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": " ".join(context),
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
