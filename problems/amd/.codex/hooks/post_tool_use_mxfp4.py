#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


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


def _latest_actionable_message(status: dict) -> str | None:
    latest = status.get("latest_variants", {})
    actionable = []
    for variant, rec in latest.items():
        if not isinstance(rec, dict):
            continue
        if rec.get("test_status") == "ok" and rec.get("benchmark_status") == "pending":
            actionable.append(f"{variant} is test-green and benchmark-pending; benchmark it immediately.")
        if rec.get("benchmark_status") == "ok" and rec.get("benchmark_geomean") is not None:
            try:
                actionable.append(
                    f"{variant} benchmark is complete at {float(rec['benchmark_geomean']):.4f} us; use that result to decide the next lane before opening sibling branches."
                )
            except Exception:
                pass
    return actionable[0] if actionable else None


def main() -> int:
    payload = _load_payload()
    cwd = Path(payload.get("cwd") or REPO)
    if REPO not in [cwd, *cwd.parents]:
        print("{}")
        return 0

    tool_input = payload.get("tool_input") or {}
    command = str(tool_input.get("command") or "")
    if "agent_loop" not in command and "mxfp4-closed-loop" not in command:
        print("{}")
        return 0

    status = _status()
    if not status:
        print("{}")
        return 0

    message = _latest_actionable_message(status)
    if not message:
        print("{}")
        return 0

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": message,
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
