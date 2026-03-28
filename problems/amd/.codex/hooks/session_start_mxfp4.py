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


def main() -> int:
    payload = _load_payload()
    cwd = Path(payload.get("cwd") or REPO)
    if REPO not in [cwd, *cwd.parents]:
        print(json.dumps({"continue": True}))
        return 0

    status = _status()
    if not status:
        print(json.dumps({"continue": True}))
        return 0

    best_variant = status.get("current_best_variant", "unknown")
    best_geomean = status.get("current_best_geomean_us")
    latest = status.get("latest_variants", {})
    v125 = latest.get("v125")
    extra = [
        f"mxfp4_mm frontier: best measured trunk is {best_variant} at {best_geomean:.4f} us."
        if isinstance(best_geomean, (int, float))
        else f"mxfp4_mm frontier: best measured trunk is {best_variant}.",
        "Use the local skill canon explicitly: skills/amd-mi355x-kernel-loop/SKILL.md and skills/amd-mi355x-kernel-loop/references/mxfp4-subagent-prompt.md.",
        "When opening a new mxfp4 research round, use exactly three scouts and keep them on one exact-shape lane.",
        "All mxfp4 scouts must use gpt-5.4 with reasoning_effort=xhigh.",
        "Think in an Atom-of-Thoughts style: decompose the active cost center into small candidate laws, reject weak branches on paper, then code only the narrowest legal candidate.",
        "Autonomy rules: after a green remote test, benchmark immediately in the same turn.",
        "Profile only after a >=5% geomean win over the current best measured trunk.",
        "A-pack local deletion is closed: v121, v122, and v125 all proved that deleting temp bytes without breaking duplication law overpays internally.",
        "Current strategic question is not how to remove A-pack bytes locally, but what execution law changes total A-pack quant duplication.",
    ]
    if isinstance(v125, dict) and v125.get("benchmark_status") == "ok":
        extra.append(
            f"Latest A-pack register-only result: v125 benchmarked at {float(v125['benchmark_geomean']):.4f} us and is a hard negative; do not reopen local register-only A feeder rewrites."
        )

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": " ".join(extra),
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
