#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _collect_runs(root: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    if not root.exists():
        return runs
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stages = manifest.get("stages", [])
        if not isinstance(stages, list) or not stages:
            continue
        stage = stages[-1] if isinstance(stages[-1], dict) else {}
        runs.append(
            {
                "label": manifest.get("label", manifest_path.parent.name),
                "run_dir": str(manifest_path.parent),
                "source_path": manifest.get("source_path", ""),
                "status": stage.get("status", "unknown"),
                "mismatch_count": stage.get("mismatch_count"),
                "failure_signature": stage.get("failure_signature", ""),
                "workflow_url": stage.get("workflow_url", ""),
                "finished_at": stage.get("finished_at", ""),
            }
        )
    return runs


def _sort_key(row: dict[str, object]) -> tuple[int, str]:
    mismatch = row.get("mismatch_count")
    if isinstance(mismatch, int):
        return (mismatch, str(row.get("label", "")))
    return (10**18, str(row.get("label", "")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/Users/v/reference-kernels/problems/amd")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    harness_root = repo / ".agent-loop" / "harness_runs" / "mxfp4_mm"
    runs = _collect_runs(harness_root)
    runs.sort(key=_sort_key)

    if not runs:
        print("No harness runs found for mxfp4_mm.")
        return 0

    print("mxfp4_mm harness runs (best mismatch first)")
    for row in runs:
        mismatch = row["mismatch_count"]
        mismatch_text = str(mismatch) if mismatch is not None else "-"
        print(
            "\t".join(
                [
                    mismatch_text,
                    str(row["status"]),
                    str(row["label"]),
                    str(row["failure_signature"]),
                    str(row["workflow_url"]),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
