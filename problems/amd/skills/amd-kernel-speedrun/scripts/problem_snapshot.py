#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_problem(repo: Path, problem: str, history_limit: int) -> dict[str, object]:
    problem_dir = repo / ".agent-loop" / "problems" / problem
    knowledge = load_json(problem_dir / "knowledge.json")
    history = knowledge.get("history", [])

    repeated_summaries: dict[str, int] = {}
    for entry in history[:history_limit]:
        critique = entry.get("critique") if isinstance(entry.get("critique"), dict) else {}
        summary = critique.get("summary")
        if isinstance(summary, str) and summary:
            repeated_summaries[summary] = repeated_summaries.get(summary, 0) + 1

    recent = []
    for entry in history[:history_limit]:
        meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
        variant = meta.get("variant") if isinstance(meta.get("variant"), dict) else {}
        critique = entry.get("critique") if isinstance(entry.get("critique"), dict) else {}
        recent.append(
            {
                "candidate_id": entry.get("candidate_id"),
                "status": entry.get("status"),
                "variant_name": variant.get("variant_name"),
                "policy_profile": (meta.get("policy_profile") or {}).get("name")
                if isinstance(meta.get("policy_profile"), dict)
                else None,
                "summary": critique.get("summary"),
                "pruned": bool(entry.get("pruned")),
            }
        )

    top_failure = None
    if repeated_summaries:
        top_failure = max(repeated_summaries.items(), key=lambda item: item[1])

    return {
        "problem": problem,
        "status_counts": knowledge.get("status_counts"),
        "failure_counts": knowledge.get("failure_counts"),
        "policy_signal_counts": knowledge.get("policy_signal_counts"),
        "policy_profile_counts": knowledge.get("policy_profile_counts"),
        "top_repeated_failure": {
            "summary": top_failure[0],
            "count": top_failure[1],
        }
        if top_failure
        else None,
        "recent_history": recent,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--history-limit", type=int, default=8)
    args = parser.parse_args()

    payload = summarize_problem(
        repo=Path(args.repo).resolve(),
        problem=args.problem,
        history_limit=args.history_limit,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
