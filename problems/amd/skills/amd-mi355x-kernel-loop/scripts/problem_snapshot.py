#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROBLEM_ALIASES = {
    "mxfp4_mm": "mxfp4_mm",
    "mxfp4-mm": "mxfp4_mm",
    "mxfp4": "mxfp4_mm",
    "fp8-mm": "mxfp4_mm",
    "amd-mxfp4-mm": "mxfp4_mm",
    "moe": "moe_mxfp4",
    "moe_mxfp4": "moe_mxfp4",
    "moe-mxfp4": "moe_mxfp4",
    "amd-moe-mxfp4": "moe_mxfp4",
    "mla": "mixed_mla",
    "mixed_mla": "mixed_mla",
    "mixed-mla": "mixed_mla",
    "mla-decode": "mixed_mla",
    "amd-mixed-mla": "mixed_mla",
    "identity": "identity",
    "amd-identity": "identity",
}

PROBLEM_LAYOUT = {
    "mxfp4_mm": {
        "dir": "fp8-mm",
        "label": "mxfp4_mm",
        "current_base": ".agent-loop/manual/native_scaled_m32_bfrag_direct_v45/submission.py",
        "pure_anchor": ".agent-loop/manual/native_scaled_pure_compiled_bscale_v44/submission.py",
    },
    "moe_mxfp4": {
        "dir": "moe",
        "label": "moe_mxfp4",
    },
    "mixed_mla": {
        "dir": "mla-decode",
        "label": "mixed_mla",
    },
    "identity": {
        "dir": "identity",
        "label": "identity",
    },
}


def resolve_problem(raw: str) -> str:
    key = raw.strip().lower()
    if key not in PROBLEM_ALIASES:
        raise SystemExit(f"unknown problem: {raw}")
    return PROBLEM_ALIASES[key]


def rel(path: Path, repo: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except Exception:
        return str(path.resolve())


def latest_team_summaries(repo: Path) -> list[str]:
    summaries = sorted((repo / "team_results").glob("*/*/summary.md"))
    return [str(path.resolve()) for path in summaries[-6:]]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def best_mxfp4_record(repo: Path) -> dict[str, Any] | None:
    ledger = repo / ".agent-loop/closed_loop/mxfp4_mm/experiment_ledger.jsonl"
    rows = load_jsonl(ledger)
    if not rows:
        return None
    latest_by_variant: dict[str, dict[str, Any]] = {}
    for row in rows:
        variant = str(row.get("variant", ""))
        if variant:
            latest_by_variant[variant] = row
    candidates = [
        row
        for row in latest_by_variant.values()
        if row.get("benchmark_status") == "ok" and row.get("benchmark_geomean") is not None
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda row: float(row["benchmark_geomean"]))
    return {
        "variant": best.get("variant"),
        "benchmark_geomean_us": best.get("benchmark_geomean"),
        "leaderboard_status": best.get("leaderboard_status"),
        "source_path": best.get("source_path"),
        "per_shape_times": best.get("per_shape_times", {}),
    }


def build_snapshot(repo: Path, problem: str) -> dict[str, Any]:
    layout = PROBLEM_LAYOUT[problem]
    problem_dir = repo / layout["dir"]
    start_files = {
        "task_py": str((problem_dir / "task.py").resolve()),
        "reference_py": str((problem_dir / "reference.py").resolve()),
        "task_yml": str((problem_dir / "task.yml").resolve()),
        "submission_py": str((problem_dir / "submission.py").resolve()),
    }
    if (problem_dir / "README.md").exists():
        start_files["readme_md"] = str((problem_dir / "README.md").resolve())
    if (problem_dir / "template-hip.py").exists():
        start_files["template_hip_py"] = str((problem_dir / "template-hip.py").resolve())
    if (problem_dir / "template.py").exists():
        start_files["template_py"] = str((problem_dir / "template.py").resolve())

    repo_resources = {
        "agent_loop_readme": str((repo / "agent_loop/README.md").resolve()),
        "retrieval_readme": str((repo / "amd_kernel_rag/README.md").resolve()),
        "dataset_mining_readme": str((repo / "dataset_mining/kernelbot_data/README.md").resolve()),
        "team_results_readme": str((repo / "team_results/README.md").resolve()),
        "session_summary": str((repo / "session-summary.md").resolve()),
        "repo_correctness_skill": str((repo / "skills/amd-live-reference-correctness/SKILL.md").resolve()),
        "repo_optimization_skill": str((repo / "skills/optimization-skill/SKILL.md").resolve()),
    }

    snapshot: dict[str, Any] = {
        "repo_root": str(repo.resolve()),
        "problem": problem,
        "problem_dir": str(problem_dir.resolve()),
        "start_files": start_files,
        "repo_resources": repo_resources,
        "retrieval": {
            "index_exists": (repo / ".agent-loop/retrieval/amd-kernel-rag").exists(),
            "index_dir": str((repo / ".agent-loop/retrieval/amd-kernel-rag").resolve()),
            "readme": str((repo / "amd_kernel_rag/README.md").resolve()),
        },
        "dataset_mining": {
            "path": str((repo / "dataset_mining/kernelbot_data").resolve()),
            "readme": str((repo / "dataset_mining/kernelbot_data/README.md").resolve()),
            "note": "older AMD competition snapshot; use for priors, not live mxfp4 truth",
        },
        "team_summaries": latest_team_summaries(repo),
    }

    if problem == "mxfp4_mm":
        snapshot["mxfp4_current_base"] = str((repo / layout["current_base"]).resolve())
        snapshot["mxfp4_pure_anchor"] = str((repo / layout["pure_anchor"]).resolve())
        snapshot["mxfp4_best_record"] = best_mxfp4_record(repo)
        snapshot["mxfp4_manual_lineage_root"] = str((repo / ".agent-loop/manual").resolve())
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit a compact AMD kernel repo snapshot for one problem.")
    parser.add_argument("--repo", default="/Users/v/reference-kernels/problems/amd")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    problem = resolve_problem(args.problem)
    payload = build_snapshot(repo, problem)
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
