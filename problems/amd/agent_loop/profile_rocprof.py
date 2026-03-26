from __future__ import annotations

import base64
import csv
import gzip
import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import zipfile


PMC_GROUPS: dict[str, tuple[str, ...]] = {
    "waves_busy": (
        "SPI_CSN_WAVE",
        "SQ_WAVES",
        "SQ_CYCLES",
        "SQ_BUSY_CYCLES",
        "SPI_CSN_BUSY",
        "SPI_CSN_NUM_THREADGROUPS",
        "SQ_WAIT_INST_ANY",
        "SQ_ACTIVE_INST_ANY",
        "SQ_ACTIVE_INST_VMEM",
    ),
    "memory_lds": (
        "SQ_LDS_BANK_CONFLICT",
        "SQ_INSTS_LDS",
        "TCC_REQ_sum",
        "TCC_HIT_sum",
        "TCC_MISS_sum",
        "TCP_TOTAL_ACCESSES_sum",
        "TCP_TOTAL_READ_sum",
        "TCP_TOTAL_WRITE_sum",
    ),
}


COUNTER_ALIASES = {
    "Counter_Name": ("Counter_Name", "CounterName"),
    "Counter_Value": ("Counter_Value", "CounterValue", "Value"),
    "Kernel_Name": ("Kernel_Name", "KernelName", "Name"),
    "Dispatch_Id": ("Dispatch_Id", "DispatchID", "DispatchId", "Dispatch Id"),
}

VISIBLE_PROFILE_SHAPES = (4, 16, 32, 64, 256)
OPTIONAL_PROFILE_SHAPES = (8,)
PROFILE_SHAPE_ORDER = ("m4", "m8", "m16", "m32", "m64", "m256")
PROFILE_SECTION_RE = re.compile(
    r"##\s+Profiling\s+k:\s*(?P<k>\d+);\s*m:\s*(?P<m>\d+);\s*n:\s*(?P<n>\d+);\s*seed:\s*(?P<seed>\d+):\s*```(?P<table>.*?)```",
    re.DOTALL,
)
TABLE_SPLIT_RE = re.compile(r"\s{2,}")


@dataclass(frozen=True)
class RocprofArtifact:
    relative_path: str
    text: str


def encode_artifact_text(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return base64.b64encode(gzip.compress(raw)).decode("ascii")


def decode_artifact_text(encoded: str) -> bytes:
    return gzip.decompress(base64.b64decode(encoded.encode("ascii")))


def materialize_encoded_artifacts(
    metrics: dict[str, Any],
    artifacts_dir: Path,
) -> dict[str, Any]:
    profile_metrics: dict[str, Any] = {}
    artifact_count = _coerce_int(metrics.pop("profile.artifact_count", 0))
    raw_paths: list[str] = []
    if artifact_count > 0:
        for index in range(artifact_count):
            rel_path = str(metrics.pop(f"profile.artifact.{index}.relative_path", "")).strip()
            encoded = str(metrics.pop(f"profile.artifact.{index}.gzip_b64", "")).strip()
            if not rel_path or not encoded:
                continue
            output_path = artifacts_dir / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(decode_artifact_text(encoded))
            raw_paths.append(str(output_path))
    if raw_paths:
        profile_metrics["profile_raw_artifact_paths"] = raw_paths

    summary_path = _materialize_single_encoded_payload(
        metrics,
        artifacts_dir,
        key_prefix="profile.summary",
    )
    if summary_path is not None:
        profile_metrics["profile_summary_path"] = str(summary_path)

    candidate_cards_path = _materialize_single_encoded_payload(
        metrics,
        artifacts_dir,
        key_prefix="profile.candidate_cards",
    )
    if candidate_cards_path is not None:
        profile_metrics["candidate_cards_path"] = str(candidate_cards_path)
    return profile_metrics


def materialize_kernelbot_profile_fallback(
    *,
    result_text: str,
    artifacts_dir: Path,
) -> dict[str, Any]:
    summary, candidate_cards, zip_paths = summarize_kernelbot_profile_result(
        result_text=result_text,
        run_dir=artifacts_dir.parents[1],
    )
    if summary is None:
        return {}

    profile_dir = artifacts_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    summary_path = profile_dir / "profile_summary.json"
    cards_path = profile_dir / "candidate_cards.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    cards_path.write_text(json.dumps(candidate_cards, indent=2, sort_keys=True), encoding="utf-8")

    payload: dict[str, Any] = {
        "profile_summary_path": str(summary_path),
        "candidate_cards_path": str(cards_path),
    }
    if zip_paths:
        payload["profile_raw_artifact_paths"] = [str(path) for path in zip_paths]
    return payload


def summarize_profile_cases(case_payloads: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_summaries: list[dict[str, Any]] = []
    for payload in case_payloads:
        case_id = str(payload["case_id"])
        shape = str(payload["shape"])
        counter_groups = payload.get("counter_groups", {})
        merged_rows: list[dict[str, str]] = []
        raw_artifacts: list[str] = []
        for group_name, csv_text in counter_groups.items():
            raw_artifacts.append(str(group_name))
            merged_rows.extend(_parse_counter_rows(str(csv_text)))
        counters, dispatch_ids, kernel_names = _aggregate_counters(merged_rows)
        derived = _derive_metrics(counters)
        case_summaries.append(
            {
                "case_id": case_id,
                "shape": shape,
                "spec": str(payload["spec"]),
                "args": dict(payload["args"]),
                "dispatch_count": len(dispatch_ids),
                "kernel_names": sorted(kernel_names),
                "counter_groups": sorted(raw_artifacts),
                "counters": {key: counters[key] for key in sorted(counters)},
                "derived": derived,
            }
        )

    shapes: dict[str, dict[str, Any]] = {}
    for case_summary in case_summaries:
        shape = str(case_summary["shape"])
        entry = shapes.setdefault(shape, {"shape": shape, "cases": []})
        entry["cases"].append(case_summary)

    candidate_cards: list[dict[str, Any]] = []
    for shape in PROFILE_SHAPE_ORDER:
        shape_entry = shapes.get(shape)
        if shape_entry is None:
            continue
        cases = list(shape_entry["cases"])
        card = _candidate_for_shape(shape, cases)
        shape_entry["candidate_card"] = card
        candidate_cards.append(card)

    summary = {
        "generated_by": "rocprofv3_profile_lane",
        "shape_count": len(shapes),
        "profile_shapes": [shape for shape in PROFILE_SHAPE_ORDER if shape in shapes],
        "cases": case_summaries,
        "shapes": {shape: shapes[shape] for shape in PROFILE_SHAPE_ORDER if shape in shapes},
    }
    return summary, candidate_cards


def summarize_kernelbot_profile_result(
    *,
    result_text: str,
    run_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[Path]]:
    sections = _parse_kernelbot_profile_sections(result_text)
    if not sections:
        return None, [], []

    zip_paths = sorted(path for path in run_dir.glob("profile_*.zip") if path.is_file())
    trace_metadata = _collect_trace_metadata(zip_paths)

    shapes: dict[str, dict[str, Any]] = {}
    for section in sections:
        shape = str(section["shape"])
        case_id = str(section["case_id"])
        entry = {
            "case_id": case_id,
            "shape": shape,
            "spec": str(section["spec"]),
            "args": dict(section["args"]),
            "kernel_rows": list(section["kernel_rows"]),
            "cost_buckets": dict(section["cost_buckets"]),
            "total_self_cuda_us": float(section["total_self_cuda_us"]),
            "total_kernel_self_cuda_us": float(section["total_kernel_self_cuda_us"]),
        }
        shape_entry = shapes.setdefault(shape, {"shape": shape, "cases": []})
        shape_entry["cases"].append(entry)

    candidate_cards: list[dict[str, Any]] = []
    for shape in PROFILE_SHAPE_ORDER:
        shape_entry = shapes.get(shape)
        if shape_entry is None:
            continue
        shape_entry["trace_metadata"] = trace_metadata.get(shape, [])
        card = _kernelbot_candidate_for_shape(shape, list(shape_entry["cases"]))
        shape_entry["candidate_card"] = card
        candidate_cards.append(card)

    summary = {
        "generated_by": "kernelbot_profile_fallback",
        "source": "kernelbot_profile_markdown+trace_zip",
        "shape_count": len(shapes),
        "profile_shapes": [shape for shape in PROFILE_SHAPE_ORDER if shape in shapes],
        "shapes": {shape: shapes[shape] for shape in PROFILE_SHAPE_ORDER if shape in shapes},
    }
    return summary, candidate_cards, zip_paths


def _materialize_single_encoded_payload(
    metrics: dict[str, Any],
    artifacts_dir: Path,
    *,
    key_prefix: str,
) -> Path | None:
    rel_path = str(metrics.pop(f"{key_prefix}.relative_path", "")).strip()
    encoded = str(metrics.pop(f"{key_prefix}.gzip_b64", "")).strip()
    if not rel_path or not encoded:
        return None
    output_path = artifacts_dir / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(decode_artifact_text(encoded))
    return output_path


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_counter_rows(csv_text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    rows: list[dict[str, str]] = []
    for row in reader:
        normalized = _normalize_counter_row(row)
        if normalized is not None:
            rows.append(normalized)
    return rows


def _normalize_counter_row(row: dict[str, str]) -> dict[str, str] | None:
    normalized: dict[str, str] = {}
    for canonical, aliases in COUNTER_ALIASES.items():
        value = None
        for alias in aliases:
            if alias in row and row[alias] not in {None, ""}:
                value = row[alias]
                break
        if value is None:
            if canonical in {"Kernel_Name", "Dispatch_Id"}:
                value = ""
            else:
                return None
        normalized[canonical] = str(value)
    return normalized


def _aggregate_counters(rows: list[dict[str, str]]) -> tuple[dict[str, float], set[str], set[str]]:
    counters: dict[str, float] = defaultdict(float)
    dispatch_ids: set[str] = set()
    kernel_names: set[str] = set()
    for row in rows:
        counter_name = row["Counter_Name"].strip()
        try:
            counter_value = float(row["Counter_Value"])
        except (TypeError, ValueError):
            continue
        counters[counter_name] += counter_value
        dispatch_id = row["Dispatch_Id"].strip()
        kernel_name = row["Kernel_Name"].strip()
        if dispatch_id:
            dispatch_ids.add(dispatch_id)
        if kernel_name:
            kernel_names.add(kernel_name)
    return dict(counters), dispatch_ids, kernel_names


def _derive_metrics(counters: dict[str, float]) -> dict[str, float]:
    return {
        "busy_fraction": _ratio(counters.get("SQ_BUSY_CYCLES", 0.0), counters.get("SQ_CYCLES", 0.0)),
        "wait_fraction": _ratio(counters.get("SQ_WAIT_INST_ANY", 0.0), counters.get("SQ_CYCLES", 0.0)),
        "active_vmem_fraction": _ratio(
            counters.get("SQ_ACTIVE_INST_VMEM", 0.0),
            counters.get("SQ_ACTIVE_INST_ANY", 0.0),
        ),
        "waves_per_workgroup": _ratio(
            counters.get("SPI_CSN_WAVE", 0.0),
            counters.get("SPI_CSN_NUM_THREADGROUPS", 0.0),
        ),
        "lds_conflict_rate": _ratio(
            counters.get("SQ_LDS_BANK_CONFLICT", 0.0),
            counters.get("SQ_INSTS_LDS", 0.0),
        ),
        "l2_hit_rate": _ratio(counters.get("TCC_HIT_sum", 0.0), counters.get("TCC_REQ_sum", 0.0)),
        "l2_miss_rate": _ratio(counters.get("TCC_MISS_sum", 0.0), counters.get("TCC_REQ_sum", 0.0)),
        "tcp_read_fraction": _ratio(
            counters.get("TCP_TOTAL_READ_sum", 0.0),
            counters.get("TCP_TOTAL_ACCESSES_sum", 0.0),
        ),
        "tcp_write_fraction": _ratio(
            counters.get("TCP_TOTAL_WRITE_sum", 0.0),
            counters.get("TCP_TOTAL_ACCESSES_sum", 0.0),
        ),
        "lds_insts": counters.get("SQ_INSTS_LDS", 0.0),
        "tcc_requests": counters.get("TCC_REQ_sum", 0.0),
        "tcp_total_accesses": counters.get("TCP_TOTAL_ACCESSES_sum", 0.0),
        "sq_waves": counters.get("SQ_WAVES", 0.0),
        "spi_waves": counters.get("SPI_CSN_WAVE", 0.0),
    }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _candidate_for_shape(shape: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    if shape == "m32" and len(cases) > 1:
        case_cards = [_candidate_for_case(shape, case["derived"]) for case in cases]
        unique_deletions = {
            str(card["deleted_cost_center"])
            for card in case_cards
            if str(card["deleted_cost_center"]) != "no_actionable_candidate"
        }
        if len(unique_deletions) == 1:
            merged = dict(case_cards[0])
            merged["why_larger_than_noise"] = (
                "Both visible m32 cases align on the same counter pattern: "
                + "; ".join(str(card["why_larger_than_noise"]) for card in case_cards)
            )
            return merged
        return _no_actionable_candidate(
            shape,
            "m32 profile cases disagree directionally, so the lane should not open a branch yet",
        )
    return _candidate_for_case(shape, cases[0]["derived"])


def _candidate_for_case(shape: str, derived: dict[str, float]) -> dict[str, Any]:
    busy = derived["busy_fraction"]
    wait = derived["wait_fraction"]
    active_vmem = derived["active_vmem_fraction"]
    waves_per_wg = derived["waves_per_workgroup"]
    lds_conflict = derived["lds_conflict_rate"]
    l2_hit = derived["l2_hit_rate"]

    if lds_conflict >= 0.05 and derived["lds_insts"] > 0:
        if shape == "m256":
            return _candidate_card(
                shape=shape,
                deleted_cost_center="bank-conflicted LDS staging/layout on exact m256",
                expected_upside_source="high LDS conflict rate indicates a real structural memory-serialization bucket",
                why_larger_than_noise=(
                    f"SQ_LDS_BANK_CONFLICT/SQ_INSTS_LDS={lds_conflict:.3f}; this is large enough to justify a gated m256-only "
                    "LDS-layout or staging investigation once direct-body deletions plateau"
                ),
                touched_symbols_or_regions=[
                    "mxfp4/exact_m256",
                    "mxfp4/exact_m256/kernel_launch",
                ],
                forbidden_edits=[
                    "do not roll LDS scheduling changes into m32 or m64",
                    "do not open CTA-order or wave-scheduling work before two direct-body m256 failures",
                ],
                success_gate="m256 < 27.8 us before any ranked spend",
            )
        return _no_actionable_candidate(
            shape,
            f"LDS conflict rate is {lds_conflict:.3f}, but current canon forbids LDS/staging work outside m256",
        )

    if busy <= 0.45 and active_vmem >= 0.20:
        if shape == "m4":
            return _candidate_card(
                shape=shape,
                deleted_cost_center="generic mxfp4_pack_a_fixed on exact m4",
                expected_upside_source="memory-side activity is high while compute busy is low, which points at the tiny-shape pack/materialize bucket",
                why_larger_than_noise=(
                    f"busy_fraction={busy:.3f}, active_vmem_fraction={active_vmem:.3f}; this suggests the exact m4 path is still paying "
                    "a whole A-pack/materialize bucket rather than being limited by MFMA throughput"
                ),
                touched_symbols_or_regions=["mxfp4/exact_m4", "mxfp4/exact_m4/a_pack"],
                forbidden_edits=[
                    "do not reopen the broken v79 launch structure",
                    "do not rewrite m8 or m16 in the same branch",
                ],
                success_gate="m4 < 18.5 us",
            )
        if shape == "m8":
            return _candidate_card(
                shape=shape,
                deleted_cost_center="generic mxfp4_pack_a_fixed on exact m8",
                expected_upside_source="memory-side pressure is dominating the exact tiny path instead of compute",
                why_larger_than_noise=(
                    f"busy_fraction={busy:.3f}, active_vmem_fraction={active_vmem:.3f}; exact m8 should not open until it deletes a full tiny-pack bucket"
                ),
                touched_symbols_or_regions=["mxfp4/exact_m8", "mxfp4/exact_m8/a_pack"],
                forbidden_edits=[
                    "do not benchmark m8 before higher-priority visible shapes are addressed",
                    "do not merge the branch with m4 or m16 changes",
                ],
                success_gate="test-green exact m8 path with no visible-shape regression",
            )
        if shape == "m16":
            return _candidate_card(
                shape=shape,
                deleted_cost_center="generic tiny-path B-scale row-major materialization on exact m16",
                expected_upside_source="the exact m16 path still looks memory-side dominated, which points at scale decode/materialization rather than MFMA throughput",
                why_larger_than_noise=(
                    f"busy_fraction={busy:.3f}, active_vmem_fraction={active_vmem:.3f}; this matches the cost center we have already isolated conceptually for exact m16"
                ),
                touched_symbols_or_regions=["mxfp4/exact_m16", "mxfp4/exact_m16/b_prep"],
                forbidden_edits=[
                    "do not retry vec-load-only body rewrites",
                    "do not spend another slot on wrapper polish without deleting this bucket",
                ],
                success_gate="m16 < 39.5 us",
            )
        if shape == "m32":
            return _candidate_card(
                shape=shape,
                deleted_cost_center="generic exact-wide B-pack/repack reused across exact m32",
                expected_upside_source="memory traffic dominates while compute busy stays low, pointing at exact-wide B materialization rather than the MFMA body",
                why_larger_than_noise=(
                    f"busy_fraction={busy:.3f}, active_vmem_fraction={active_vmem:.3f}; exact m32 should only reopen for a full B-pack/materialize deletion"
                ),
                touched_symbols_or_regions=["mxfp4/exact_m32", "mxfp4/exact_m32/b_prep"],
                forbidden_edits=[
                    "do not open another prep-only micro-variant",
                    "do not share the branch with m64",
                ],
                success_gate="both visible m32 cases beat or match 22.6 / 21.8 us",
            )
        if shape == "m64":
            return _candidate_card(
                shape=shape,
                deleted_cost_center="generic exact-wide B-pack/repack reused across exact m64",
                expected_upside_source="the m64 path is still memory-side dominated, which points at a whole B materialization bucket",
                why_larger_than_noise=(
                    f"busy_fraction={busy:.3f}, active_vmem_fraction={active_vmem:.3f}; this is bigger than a hoist-sized effect and fits the exact m64 B-pack bucket"
                ),
                touched_symbols_or_regions=["mxfp4/exact_m64", "mxfp4/exact_m64/b_prep"],
                forbidden_edits=[
                    "do not open another prep-only micro-variant",
                    "do not merge with m32 or m256 edits",
                ],
                success_gate="m64 < 35.0 us and stable on rerun",
            )
        if shape == "m256":
            return _candidate_card(
                shape=shape,
                deleted_cost_center="Python-side exact m256 wrapper materialization and exact-wide B repack",
                expected_upside_source="memory traffic dominates while compute busy is weak, pointing at the wrapper/materialize bucket on exact m256",
                why_larger_than_noise=(
                    f"busy_fraction={busy:.3f}, active_vmem_fraction={active_vmem:.3f}; exact m256 remains the clearest large undeleted wide bucket"
                ),
                touched_symbols_or_regions=["mxfp4/exact_m256", "mxfp4/exact_m256/b_prep"],
                forbidden_edits=[
                    "do not open CTA-order or wave scheduling before direct-body deletions fail twice",
                    "do not touch m64 in the same branch",
                ],
                success_gate="m256 < 27.8 us before any ranked spend",
            )

    if wait >= 0.25 and l2_hit <= 0.65:
        if shape == "m64":
            return _candidate_card(
                shape=shape,
                deleted_cost_center="per-iteration pointer and scale-block arithmetic inside the exact m64 body",
                expected_upside_source="high wait with weak cache tendency points at pointer-path and address-generation overhead rather than pure bandwidth",
                why_larger_than_noise=(
                    f"wait_fraction={wait:.3f}, l2_hit_rate={l2_hit:.3f}; this fits the exact m64 direct-body arithmetic bucket better than another prep edit"
                ),
                touched_symbols_or_regions=["mxfp4/exact_m64", "mxfp4/exact_m64/kernel_launch"],
                forbidden_edits=[
                    "do not add a new prep kernel",
                    "do not share the body change with m32",
                ],
                success_gate="m64 < 35.0 us and stable on rerun",
            )
        if shape == "m32":
            return _candidate_card(
                shape=shape,
                deleted_cost_center="dead exact-path epilogue and bounds work in the exact m32 body",
                expected_upside_source="high wait with weak cache hit tendency suggests the remaining cost is in dead exact-path control work",
                why_larger_than_noise=(
                    f"wait_fraction={wait:.3f}, l2_hit_rate={l2_hit:.3f}; exact m32 should only reopen for a constant-body deletion, not another prep change"
                ),
                touched_symbols_or_regions=["mxfp4/exact_m32", "mxfp4/exact_m32/kernel_launch"],
                forbidden_edits=[
                    "do not add another standalone B-prep branch",
                    "do not touch m64 in the same branch",
                ],
                success_gate="both visible m32 cases beat or match 22.6 / 21.8 us",
            )
        if shape == "m16":
            return _candidate_card(
                shape=shape,
                deleted_cost_center="dead lane and row work in the exact m16 dense body",
                expected_upside_source="wait-heavy behavior with modest cache reuse suggests constant-m16 body slimming is still open",
                why_larger_than_noise=(
                    f"wait_fraction={wait:.3f}, l2_hit_rate={l2_hit:.3f}; this is the structural m16 body bucket, not another wrapper polish"
                ),
                touched_symbols_or_regions=["mxfp4/exact_m16", "mxfp4/exact_m16/kernel_launch"],
                forbidden_edits=[
                    "do not retry the failed vec-load-only rewrite",
                    "do not combine with m4 or m8 changes",
                ],
                success_gate="m16 < 39.5 us",
            )
        if shape == "m256":
            return _candidate_card(
                shape=shape,
                deleted_cost_center="per-iteration pointer and scale-block arithmetic inside the exact m256 body",
                expected_upside_source="the direct m256 body still looks wait-heavy enough to justify a constant-body clone before scheduling work",
                why_larger_than_noise=(
                    f"wait_fraction={wait:.3f}, l2_hit_rate={l2_hit:.3f}; this is larger than noise because it targets a whole exact-body bucket"
                ),
                touched_symbols_or_regions=["mxfp4/exact_m256", "mxfp4/exact_m256/kernel_launch"],
                forbidden_edits=[
                    "do not open CTA-order or wave scheduling yet",
                    "do not merge with m64 edits",
                ],
                success_gate="m256 < 27.8 us before any ranked spend",
            )

    if busy <= 0.55 and waves_per_wg <= 2.0:
        if shape == "m16":
            return _candidate_card(
                shape=shape,
                deleted_cost_center="constant-m16 direct body/control overhead on the exact path",
                expected_upside_source="low useful wave density and middling busy fraction point at a remaining constant-body bucket",
                why_larger_than_noise=(
                    f"busy_fraction={busy:.3f}, waves_per_workgroup={waves_per_wg:.3f}; exact m16 should only reopen for a body-level deletion"
                ),
                touched_symbols_or_regions=["mxfp4/exact_m16", "mxfp4/exact_m16/kernel_launch"],
                forbidden_edits=[
                    "do not open another wrapper-only branch",
                    "do not change m4 or m8 in the same candidate",
                ],
                success_gate="m16 < 39.5 us",
            )
        if shape == "m32":
            return _candidate_card(
                shape=shape,
                deleted_cost_center="constant-m32 exact body overhead",
                expected_upside_source="low wave density points at a whole control/epilogue bucket inside the exact m32 body",
                why_larger_than_noise=(
                    f"busy_fraction={busy:.3f}, waves_per_workgroup={waves_per_wg:.3f}; exact m32 should only reopen for a constant-body deletion"
                ),
                touched_symbols_or_regions=["mxfp4/exact_m32", "mxfp4/exact_m32/kernel_launch"],
                forbidden_edits=[
                    "do not add another standalone hoist patch",
                    "do not share with m64",
                ],
                success_gate="both visible m32 cases beat or match 22.6 / 21.8 us",
            )
        if shape == "m64":
            return _candidate_card(
                shape=shape,
                deleted_cost_center="constant-m64 exact body overhead",
                expected_upside_source="low useful wave density suggests the remaining body cost is larger than a micro-hoist-sized effect",
                why_larger_than_noise=(
                    f"busy_fraction={busy:.3f}, waves_per_workgroup={waves_per_wg:.3f}; exact m64 should move to a constant-body clone, not another prep tweak"
                ),
                touched_symbols_or_regions=["mxfp4/exact_m64", "mxfp4/exact_m64/kernel_launch"],
                forbidden_edits=[
                    "do not add another prep-only kernel",
                    "do not touch m32 or m256",
                ],
                success_gate="m64 < 35.0 us and stable on rerun",
            )
        if shape == "m256":
            return _candidate_card(
                shape=shape,
                deleted_cost_center="constant-m256 exact body overhead",
                expected_upside_source="wave density is low enough that an exact body clone is still a live direct-body cost-center deletion",
                why_larger_than_noise=(
                    f"busy_fraction={busy:.3f}, waves_per_workgroup={waves_per_wg:.3f}; this is the right pre-scheduling m256 branch class"
                ),
                touched_symbols_or_regions=["mxfp4/exact_m256", "mxfp4/exact_m256/kernel_launch"],
                forbidden_edits=[
                    "do not open CTA-order or LDS scheduling yet",
                    "do not merge with any other shape",
                ],
                success_gate="m256 < 27.8 us before any ranked spend",
            )

    return _no_actionable_candidate(
        shape,
        (
            f"profile evidence is ambiguous for current canon: busy_fraction={busy:.3f}, wait_fraction={wait:.3f}, "
            f"active_vmem_fraction={active_vmem:.3f}, l2_hit_rate={l2_hit:.3f}, waves_per_workgroup={waves_per_wg:.3f}"
        ),
    )


def _parse_kernelbot_profile_sections(result_text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for match in PROFILE_SECTION_RE.finditer(result_text):
        m_value = int(match.group("m"))
        shape = f"m{m_value}"
        if m_value not in VISIBLE_PROFILE_SHAPES and m_value not in OPTIONAL_PROFILE_SHAPES:
            continue
        rows = _parse_kernelbot_profile_table(match.group("table"))
        if not rows:
            continue
        sections.append(
            {
                "case_id": f"k{match.group('k')}_m{match.group('m')}_n{match.group('n')}_seed{match.group('seed')}",
                "shape": shape,
                "spec": f"k: {match.group('k')}; m: {match.group('m')}; n: {match.group('n')}; seed: {match.group('seed')}",
                "args": {
                    "k": int(match.group("k")),
                    "m": m_value,
                    "n": int(match.group("n")),
                    "seed": int(match.group("seed")),
                },
                "kernel_rows": rows,
                "cost_buckets": _kernelbot_cost_buckets(rows),
                "total_self_cuda_us": sum(float(row["self_cuda_us"]) for row in rows),
                "total_kernel_self_cuda_us": sum(
                    float(row["self_cuda_us"]) for row in rows if row["bucket"] != "other"
                ),
            }
        )
    return sections


def _parse_kernelbot_profile_table(table_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in table_text.splitlines():
        line = raw_line.rstrip()
        if not line or set(line.strip()) == {"-"}:
            continue
        if "Name" in line and "Self CUDA" in line:
            continue
        parts = [part for part in TABLE_SPLIT_RE.split(line.strip()) if part]
        if len(parts) < 11:
            continue
        name = parts[0]
        if not any(token in name for token in ("mxfp4_", "aten::", "hipLaunchKernel", "hipDeviceSynchronize", "hipMalloc")):
            continue
        row = {
            "name": name,
            "self_cuda_us": _duration_to_us(parts[6]),
            "self_cuda_pct": _percent_to_float(parts[7]),
            "cuda_total_us": _duration_to_us(parts[8]),
            "cuda_time_avg_us": _duration_to_us(parts[9]),
            "call_count": _coerce_int(parts[10]),
            "bucket": _kernelbot_bucket_for_name(name),
        }
        rows.append(row)
    return rows


def _kernelbot_bucket_for_name(name: str) -> str:
    if "mxfp4_pack_a_fixed_kernel" in name:
        return "a_pack"
    if "mxfp4_unshuffle_b_scale_kernel" in name:
        return "b_scale_decode"
    if "mxfp4_pack_b_m32_direct_with_scale_kernel" in name:
        return "b_pack"
    if "mxfp4_mm_kernel" in name:
        return "kernel"
    return "other"


def _kernelbot_cost_buckets(rows: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, float] = defaultdict(float)
    for row in rows:
        buckets[str(row["bucket"])] += float(row["self_cuda_us"])
    total = sum(buckets.values())
    summary = {key: float(value) for key, value in buckets.items()}
    if total > 0.0:
        for key, value in list(summary.items()):
            summary[f"{key}_share"] = value / total
        summary["total_self_cuda_us"] = total
    else:
        summary["total_self_cuda_us"] = 0.0
    return summary


def _percent_to_float(token: str) -> float:
    text = str(token).strip()
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text) / 100.0
    except ValueError:
        return 0.0


def _duration_to_us(token: str) -> float:
    text = str(token).strip()
    if not text:
        return 0.0
    unit_multipliers = {
        "ns": 1e-3,
        "us": 1.0,
        "µs": 1.0,
        "ms": 1e3,
        "s": 1e6,
    }
    for suffix, factor in unit_multipliers.items():
        if text.endswith(suffix):
            try:
                return float(text[: -len(suffix)]) * factor
            except ValueError:
                return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _collect_trace_metadata(zip_paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    by_shape: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for zip_path in zip_paths:
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for name in archive.namelist():
                    if not name.endswith("kernel_trace.csv"):
                        continue
                    csv_text = archive.read(name).decode("utf-8", errors="replace")
                    reader = csv.DictReader(io.StringIO(csv_text))
                    for row in reader:
                        kernel_name = str(row.get("Kernel_Name", ""))
                        shape = _shape_from_kernel_name(kernel_name)
                        if shape is None:
                            continue
                        try:
                            duration_ns = int(str(row.get("End_Timestamp", "0"))) - int(
                                str(row.get("Start_Timestamp", "0"))
                            )
                        except ValueError:
                            duration_ns = 0
                        by_shape[shape].append(
                            {
                                "kernel_name": kernel_name,
                                "duration_ns": duration_ns,
                                "vgpr_count": _coerce_int(row.get("VGPR_Count")),
                                "accum_vgpr_count": _coerce_int(row.get("Accum_VGPR_Count")),
                                "sgpr_count": _coerce_int(row.get("SGPR_Count")),
                                "lds_block_size": _coerce_int(row.get("LDS_Block_Size")),
                                "workgroup_size": [
                                    _coerce_int(row.get("Workgroup_Size_X")),
                                    _coerce_int(row.get("Workgroup_Size_Y")),
                                    _coerce_int(row.get("Workgroup_Size_Z")),
                                ],
                                "grid_size": [
                                    _coerce_int(row.get("Grid_Size_X")),
                                    _coerce_int(row.get("Grid_Size_Y")),
                                    _coerce_int(row.get("Grid_Size_Z")),
                                ],
                                "artifact": str(zip_path),
                            }
                        )
        except zipfile.BadZipFile:
            continue
    return dict(by_shape)


def _shape_from_kernel_name(kernel_name: str) -> str | None:
    name = str(kernel_name)
    if "mxfp4_mm_kernel_mfma_scale_exact_m4_dense" in name:
        return "m4"
    if "mxfp4_mm_kernel_mfma_scale_exact_m16_dense" in name:
        return "m16"
    if "mxfp4_mm_kernel_mfma_scale_exact_m32_m64plus" in name:
        return "m64"
    if "mxfp4_mm_kernel_mfma_scale_exact_m32" in name:
        return "m32_or_m256"
    if "mxfp4_pack_a_fixed_kernel" in name:
        return "shared_pack"
    if "mxfp4_pack_b_m32_direct_with_scale_kernel" in name:
        return "shared_b_pack"
    if "mxfp4_unshuffle_b_scale_kernel" in name:
        return "shared_b_scale_decode"
    return None


def _kernelbot_candidate_for_shape(shape: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    if shape == "m32" and len(cases) > 1:
        cards = [_kernelbot_candidate_for_case(shape, case) for case in cases]
        deletions = {
            str(card["deleted_cost_center"])
            for card in cards
            if str(card["deleted_cost_center"]) != "no_actionable_candidate"
        }
        if len(deletions) == 1:
            merged = dict(cards[0])
            merged["why_larger_than_noise"] = (
                "Both visible m32 cases agree: "
                + "; ".join(str(card["why_larger_than_noise"]) for card in cards)
            )
            return merged
        return _no_actionable_candidate(
            shape,
            "kernelbot profile shows divergent m32 case composition, so the lane should stay closed until a stronger structural hypothesis exists",
        )
    return _kernelbot_candidate_for_case(shape, cases[0])


def _kernelbot_candidate_for_case(shape: str, case: dict[str, Any]) -> dict[str, Any]:
    buckets = dict(case.get("cost_buckets", {}))
    total = float(buckets.get("total_self_cuda_us", 0.0))
    if total <= 0.0:
        return _no_actionable_candidate(shape, "kernelbot profile did not expose usable CUDA-time buckets")

    a_pack_share = float(buckets.get("a_pack_share", 0.0))
    b_scale_share = float(buckets.get("b_scale_decode_share", 0.0))
    b_pack_share = float(buckets.get("b_pack_share", 0.0))
    kernel_share = float(buckets.get("kernel_share", 0.0))

    if shape == "m4" and (a_pack_share + b_scale_share) >= 0.70:
        return _candidate_card(
            shape=shape,
            deleted_cost_center="generic mxfp4_pack_a_fixed on exact m4",
            expected_upside_source="kernelbot profile shows the exact m4 path is dominated by tiny-path materialization, not by the MFMA body",
            why_larger_than_noise=(
                f"a_pack_share={a_pack_share:.3f}, b_scale_decode_share={b_scale_share:.3f}, kernel_share={kernel_share:.3f}; "
                "the prep buckets dwarf the compute body, so deleting exact m4 A-pack is larger than noise"
            ),
            touched_symbols_or_regions=["mxfp4/exact_m4", "mxfp4/exact_m4/a_pack"],
            forbidden_edits=[
                "do not reopen the broken v79 launch structure",
                "do not rewrite m8 or m16 in the same branch",
            ],
            success_gate="m4 < 18.5 us",
        )

    if shape == "m16" and (a_pack_share + b_scale_share) >= 0.70:
        return _candidate_card(
            shape=shape,
            deleted_cost_center="generic tiny-path B-scale row-major materialization on exact m16",
            expected_upside_source="kernelbot profile shows exact m16 still spends most of its CUDA time in tiny-path scale decode/materialization rather than MFMA",
            why_larger_than_noise=(
                f"a_pack_share={a_pack_share:.3f}, b_scale_decode_share={b_scale_share:.3f}, kernel_share={kernel_share:.3f}; "
                "this is a whole-bucket deletion candidate, not wrapper polish"
            ),
            touched_symbols_or_regions=["mxfp4/exact_m16", "mxfp4/exact_m16/b_prep"],
            forbidden_edits=[
                "do not retry vec-load-only body rewrites",
                "do not spend another slot on wrapper polish without deleting this bucket",
            ],
            success_gate="m16 < 39.5 us",
        )

    if shape == "m32" and (a_pack_share + b_pack_share) >= 0.60:
        return _candidate_card(
            shape=shape,
            deleted_cost_center="generic exact-wide B-pack/repack reused across exact m32",
            expected_upside_source="kernelbot profile shows exact m32 still splits most of its CUDA time across A-pack, B-pack, and kernel, leaving a large reusable B materialization bucket",
            why_larger_than_noise=(
                f"a_pack_share={a_pack_share:.3f}, b_pack_share={b_pack_share:.3f}, kernel_share={kernel_share:.3f}; "
                "deleting exact-wide B-pack is larger than another m32 prep micro-variant"
            ),
            touched_symbols_or_regions=["mxfp4/exact_m32", "mxfp4/exact_m32/b_prep"],
            forbidden_edits=[
                "do not open another prep-only micro-variant",
                "do not share the branch with m64",
            ],
            success_gate="both visible m32 cases beat or match 22.6 / 21.8 us",
        )

    if shape == "m64" and (a_pack_share + b_pack_share) >= 0.60:
        return _candidate_card(
            shape=shape,
            deleted_cost_center="generic exact-wide B-pack/repack reused across exact m64",
            expected_upside_source="kernelbot profile shows exact m64 still burns most of its CUDA time outside the MFMA body, with B-pack as the clearest undeleted bucket",
            why_larger_than_noise=(
                f"a_pack_share={a_pack_share:.3f}, b_pack_share={b_pack_share:.3f}, kernel_share={kernel_share:.3f}; "
                "this is a whole-bucket deletion candidate, not another prep tweak"
            ),
            touched_symbols_or_regions=["mxfp4/exact_m64", "mxfp4/exact_m64/b_prep"],
            forbidden_edits=[
                "do not open another prep-only micro-variant",
                "do not merge with m32 or m256 edits",
            ],
            success_gate="m64 < 35.0 us and stable on rerun",
        )

    if shape == "m256" and (a_pack_share + b_pack_share) >= 0.60:
        return _candidate_card(
            shape=shape,
            deleted_cost_center="Python-side exact m256 wrapper materialization and exact-wide B repack",
            expected_upside_source="kernelbot profile shows exact m256 still spends as much time packing inputs as it does in the MFMA body",
            why_larger_than_noise=(
                f"a_pack_share={a_pack_share:.3f}, b_pack_share={b_pack_share:.3f}, kernel_share={kernel_share:.3f}; "
                "exact m256 still has a large direct-entry/materialize bucket to delete before any scheduling work"
            ),
            touched_symbols_or_regions=["mxfp4/exact_m256", "mxfp4/exact_m256/b_prep"],
            forbidden_edits=[
                "do not open CTA-order or wave scheduling before direct-body deletions fail twice",
                "do not touch m64 in the same branch",
            ],
            success_gate="m256 < 27.8 us before any ranked spend",
        )

    if shape == "m8":
        return _candidate_card(
            shape=shape,
            deleted_cost_center="generic thin shared compute/wrapper behavior on exact m8",
            expected_upside_source="exact m8 still needs a true isolated path before tiny-shape work can stop leaking",
            why_larger_than_noise="m8 is not part of the visible profile set yet, so the right first move is structural separation rather than more generic sharing",
            touched_symbols_or_regions=["mxfp4/exact_m8"],
            forbidden_edits=[
                "do not benchmark m8 before higher-priority visible shapes are addressed",
                "do not merge the branch with m4 or m16 changes",
            ],
            success_gate="test-green exact m8 path with no visible-shape regression",
        )

    return _no_actionable_candidate(
        shape,
        (
            f"kernelbot profile did not isolate a legal whole-bucket deletion: "
            f"a_pack_share={a_pack_share:.3f}, b_scale_decode_share={b_scale_share:.3f}, "
            f"b_pack_share={b_pack_share:.3f}, kernel_share={kernel_share:.3f}"
        ),
    )


def _candidate_card(
    *,
    shape: str,
    deleted_cost_center: str,
    expected_upside_source: str,
    why_larger_than_noise: str,
    touched_symbols_or_regions: list[str],
    forbidden_edits: list[str],
    success_gate: str,
) -> dict[str, Any]:
    return {
        "shape": shape,
        "deleted_cost_center": deleted_cost_center,
        "expected_upside_source": expected_upside_source,
        "why_larger_than_noise": why_larger_than_noise,
        "touched_symbols_or_regions": touched_symbols_or_regions,
        "forbidden_edits": forbidden_edits,
        "success_gate": success_gate,
    }


def _no_actionable_candidate(shape: str, reason: str) -> dict[str, Any]:
    return _candidate_card(
        shape=shape,
        deleted_cost_center="no_actionable_candidate",
        expected_upside_source="profile evidence does not isolate a legal whole-bucket deletion yet",
        why_larger_than_noise=reason,
        touched_symbols_or_regions=[f"mxfp4/exact_{shape}"],
        forbidden_edits=["do not open a branch without a named whole cost center"],
        success_gate="none",
    )
