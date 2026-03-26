from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from agent_loop.profile_rocprof import (
    OPTIONAL_PROFILE_SHAPES,
    PMC_GROUPS,
    PROFILE_SHAPE_ORDER,
    RocprofArtifact,
    VISIBLE_PROFILE_SHAPES,
    encode_artifact_text,
    summarize_profile_cases,
)


def run_rocprofv3_profiling(logger, tests: list[Any]) -> int:
    rocprof_binary = shutil.which("rocprofv3")
    if not rocprof_binary:
        logger.log("failure_kind", "missing_rocprofv3")
        logger.log("failure_signature", "rocprofv3-not-found")
        logger.log("check", "fail")
        return 113

    selected_cases = _select_profile_cases(tests)
    if not selected_cases:
        logger.log("failure_kind", "missing_profile_cases")
        logger.log("failure_signature", "no-visible-profile-cases")
        logger.log("check", "fail")
        return 113

    import submission as submission_module

    repo_root = Path(__file__).resolve().parent
    submission_root = Path(submission_module.__file__).resolve().parent
    work_root = Path(tempfile.mkdtemp(prefix="mxfp4_rocprofv3_"))
    case_payloads: list[dict[str, Any]] = []
    raw_artifacts: list[RocprofArtifact] = []

    try:
        logger.log("profile.case_count", len(selected_cases))
        logger.log("profile.case_shapes", ",".join(case["shape"] for case in selected_cases))
        for index, case in enumerate(selected_cases):
            logger.log(f"profile.case.{index}.spec", case["spec"])
            case_root = work_root / case["case_id"]
            case_root.mkdir(parents=True, exist_ok=True)

            counter_groups: dict[str, str] = {}
            for group_name, counters in PMC_GROUPS.items():
                group_root = case_root / group_name
                group_root.mkdir(parents=True, exist_ok=True)
                driver_path = group_root / "driver.py"
                driver_path.write_text(
                    _driver_script(repo_root, submission_root, case["args"]),
                    encoding="utf-8",
                )
                output_prefix = f"{case['case_id']}_{group_name}"
                command = [
                    rocprof_binary,
                    "--pmc",
                    ",".join(counters),
                    "--output-format",
                    "csv",
                    "--output-file",
                    output_prefix,
                    "-d",
                    str(group_root),
                    "--",
                    sys.executable,
                    str(driver_path),
                ]
                completed = subprocess.run(
                    command,
                    cwd=str(group_root),
                    capture_output=True,
                    text=True,
                    check=False,
                    env=_profile_env(),
                )
                (group_root / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
                (group_root / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
                if completed.returncode != 0:
                    logger.log("failure_kind", "rocprofv3_run_failed")
                    logger.log(
                        "failure_signature",
                        f"{case['case_id']}:{group_name}:rc={completed.returncode}",
                    )
                    logger.log("profile.failed_command", " ".join(command))
                    logger.log("check", "fail")
                    return 112

                csv_paths = sorted(group_root.rglob("*counter_collection.csv"))
                if not csv_paths:
                    logger.log("failure_kind", "rocprofv3_missing_csv")
                    logger.log(
                        "failure_signature",
                        f"{case['case_id']}:{group_name}:missing-counter-collection",
                    )
                    logger.log("profile.failed_command", " ".join(command))
                    logger.log("check", "fail")
                    return 112

                combined_csv = _merge_counter_csv_texts(csv_paths)
                counter_groups[group_name] = combined_csv
                for csv_path in csv_paths:
                    rel_path = Path("profile") / "raw" / case["case_id"] / group_name / csv_path.name
                    raw_artifacts.append(
                        RocprofArtifact(
                            relative_path=rel_path.as_posix(),
                            text=csv_path.read_text(encoding="utf-8"),
                        )
                    )

            case_payloads.append(
                {
                    "case_id": case["case_id"],
                    "shape": case["shape"],
                    "spec": case["spec"],
                    "args": dict(case["args"]),
                    "counter_groups": counter_groups,
                }
            )

        profile_summary, candidate_cards = summarize_profile_cases(case_payloads)
        _emit_profile_payload(logger, profile_summary, candidate_cards, raw_artifacts)
        logger.log("check", "pass")
        return 0
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def _select_profile_cases(tests: list[Any]) -> list[dict[str, Any]]:
    requested = {
        item.strip()
        for item in os.environ.get("POPCORN_PROFILE_CASES", "").split(",")
        if item.strip()
    }
    allowed_shapes = set(VISIBLE_PROFILE_SHAPES)
    if "m8" in requested or "8" in requested:
        allowed_shapes.update(OPTIONAL_PROFILE_SHAPES)
    selected: list[dict[str, Any]] = []
    shape_counts: dict[str, int] = {}
    for test in tests:
        args = getattr(test, "args", {})
        spec = getattr(test, "spec", "")
        try:
            m = int(args["m"])
            n = int(args["n"])
            k = int(args["k"])
        except (KeyError, TypeError, ValueError):
            continue
        shape = f"m{m}"
        if m not in allowed_shapes:
            continue
        if shape != "m32" and shape_counts.get(shape, 0) >= 1:
            continue
        case_index = shape_counts.get(shape, 0)
        if shape == "m32" and case_index >= 2:
            continue
        case_id = shape if shape != "m32" else f"{shape}_case{case_index}"
        if requested and not ({shape, str(m), case_id} & requested):
            continue
        shape_counts[shape] = case_index + 1
        selected.append(
            {
                "case_id": case_id,
                "shape": shape,
                "args": dict(args),
                "spec": str(spec),
                "m": m,
                "n": n,
                "k": k,
            }
        )

    selected.sort(
        key=lambda item: (
            PROFILE_SHAPE_ORDER.index(item["shape"]) if item["shape"] in PROFILE_SHAPE_ORDER else 999,
            item["case_id"],
        )
    )
    return selected


def _driver_script(repo_root: Path, submission_root: Path, args: dict[str, Any]) -> str:
    return f"""import json
import sys
import torch
from pathlib import Path

repo_root = Path({json.dumps(str(repo_root))})
submission_root = Path({json.dumps(str(submission_root))})
for candidate in (submission_root, repo_root):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from reference import generate_input
import submission
from submission import custom_kernel

args = json.loads({json.dumps(json.dumps(args, sort_keys=True))})
module_loader = getattr(submission, "_module", None)
if callable(module_loader):
    module_loader()
data = generate_input(**args)
torch.cuda.synchronize()
out = custom_kernel(data)
torch.cuda.synchronize()
del out
"""


def _profile_env() -> dict[str, str]:
    env = os.environ.copy()
    env["MXFP4_ROCTX_ENABLE"] = "1"
    env["POPCORN_PROFILE_BACKEND"] = "rocprofv3"
    return env


def _merge_counter_csv_texts(csv_paths: list[Path]) -> str:
    merged_lines: list[str] = []
    header: str | None = None
    for csv_path in csv_paths:
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        if header is None:
            header = lines[0]
            merged_lines.append(header)
            merged_lines.extend(lines[1:])
            continue
        merged_lines.extend(lines[1:])
    return "\n".join(merged_lines)


def _emit_profile_payload(
    logger,
    profile_summary: dict[str, Any],
    candidate_cards: list[dict[str, Any]],
    raw_artifacts: list[RocprofArtifact],
) -> None:
    logger.log("profile.backend", "rocprofv3")
    logger.log("profile.summary.relative_path", "profile/profile_summary.json")
    logger.log(
        "profile.summary.gzip_b64",
        encode_artifact_text(json.dumps(profile_summary, indent=2, sort_keys=True)),
    )
    logger.log("profile.candidate_cards.relative_path", "profile/candidate_cards.json")
    logger.log(
        "profile.candidate_cards.gzip_b64",
        encode_artifact_text(json.dumps(candidate_cards, indent=2, sort_keys=True)),
    )
    logger.log("profile.artifact_count", len(raw_artifacts))
    for index, artifact in enumerate(raw_artifacts):
        logger.log(f"profile.artifact.{index}.relative_path", artifact.relative_path)
        logger.log(
            f"profile.artifact.{index}.gzip_b64",
            encode_artifact_text(artifact.text),
        )
