from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
import ast
import fcntl
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from .config import AppConfig
from .harness import KernelHarness
from .preflight_worker import (
    PROBLEM_DIR_BY_KEY,
    PreflightCheck,
    PreflightReport,
    run_host_preflight,
)


TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S.%f%z"
PROBLEM_KEY = "moe_mxfp4"
LANE_VALUES = {
    "dispatch_pack",
    "stage1_core",
    "stage2_reduce",
    "shared_expert",
    "full_pipeline",
    "unknown",
}
HOT_PATH_STATES = {"anchor-backed", "partial-native", "native", "unknown"}
REGIME_TAG_VALUES = {
    "re256_de256_bs16_topk8",
    "re256_de256_bs128_topk8",
    "re256_de256_bs512_topk8",
    "re32_de512_bs16_topk8",
    "re32_de512_bs128_topk8",
    "re32_de512_bs512_topk8",
    "re32_de2048_bs512_topk8",
    "mixed",
    "unknown",
}
PREFLIGHT_PROFILES = {"amd-parity-full", "amd-compile-fast"}
REMOTE_STAGE_VALUES = {"test", "benchmark", "leaderboard"}
SHARED_TEST_BUCKET_STAGES = {"test", "benchmark"}
CONTAINER_PLATFORM = "linux/amd64"
CUSTOM_KERNEL_RE = re.compile(
    r"def\s+custom_kernel\s*\([^)]*\)\s*:\s*\n(?P<body>(?:[ \t]+.*\n?)*)",
    re.MULTILINE,
)
AGENT_LOOP_META_RE = re.compile(r"^# AGENT_LOOP_META:\s*(\{.*\})\s*$", re.MULTILINE)
CANDIDATE_CARD_FIELD_MAP = {
    "shape": "regime_tag",
    "lane": "lane",
    "deleted cost center": "deleted_cost_center",
    "expected upside source": "expected_upside_source",
    "why larger than noise": "why_larger_than_noise",
    "forbidden edits": "forbidden_edits",
    "success gate": "success_gate",
    "motivation refs": "motivation_refs",
    "retrieval queries": "retrieval_queries",
}


def _utc_now() -> str:
    return datetime.now(UTC).strftime(TIMESTAMP_FMT)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, TIMESTAMP_FMT)
    except ValueError:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None


def _mode_objective_to_us(objective: float | None) -> float | None:
    if objective is None:
        return None
    return float(objective) / 1000.0


def _resolve_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported type for json serialization: {type(value)!r}")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _extract_agent_loop_meta(source: str) -> dict[str, Any]:
    match = AGENT_LOOP_META_RE.search(source)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _candidate_card_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    card: dict[str, Any] = {}
    raw_card = meta.get("candidate_card")
    if isinstance(raw_card, dict):
        card.update(raw_card)
        if not card.get("retrieval_queries"):
            retrieval_pack = raw_card.get("retrieval_pack")
            if isinstance(retrieval_pack, dict):
                card["retrieval_queries"] = _coerce_string_list(retrieval_pack.get("queries"))
    variant = meta.get("variant")
    if isinstance(variant, dict):
        mapping = {
            "deleted_cost_center": "DELETED_COST_CENTER",
            "expected_upside_source": "EXPECTED_UPSIDE_SOURCE",
            "why_larger_than_noise": "WHY_LARGER_THAN_NOISE",
            "success_gate": "SUCCESS_GATE",
            "lane": "LANE",
            "regime_tag": "REGIME_HINT",
        }
        for target_key, source_key in mapping.items():
            value = variant.get(source_key)
            if value and not card.get(target_key):
                card[target_key] = value
        if not card.get("forbidden_edits"):
            card["forbidden_edits"] = _coerce_string_list(variant.get("FORBIDDEN_EDITS"))
        if not card.get("motivation_refs"):
            card["motivation_refs"] = _coerce_string_list(variant.get("MOTIVATION_REFS"))
        if not card.get("retrieval_queries"):
            card["retrieval_queries"] = _coerce_string_list(variant.get("RETRIEVAL_PACK"))
    return card


def _split_candidate_card_list(value: str) -> list[str]:
    if not value.strip():
        return []
    return [item.strip() for item in re.split(r"[;,]", value) if item.strip()]


def _candidate_card_from_comments(source: str) -> dict[str, Any]:
    card: dict[str, Any] = {}
    in_block = False
    for line in source.splitlines():
        stripped = line.lstrip()
        if not in_block:
            if stripped.startswith("# Candidate Card:"):
                in_block = True
            continue
        if not stripped.startswith("#"):
            break
        content = stripped[1:].strip()
        if not content:
            continue
        if content.startswith("AGENT_LOOP_META:"):
            break
        if ":" not in content:
            continue
        raw_key, raw_value = content.split(":", 1)
        mapped_key = CANDIDATE_CARD_FIELD_MAP.get(raw_key.strip().lower())
        value = raw_value.strip()
        if not mapped_key or not value:
            continue
        if mapped_key in {"forbidden_edits", "motivation_refs", "retrieval_queries"}:
            card[mapped_key] = _split_candidate_card_list(value)
        else:
            card[mapped_key] = value
    return card


def _candidate_card_from_source(source: str) -> dict[str, Any]:
    card = _candidate_card_from_meta(_extract_agent_loop_meta(source))
    comment_card = _candidate_card_from_comments(source)
    for key, value in comment_card.items():
        if isinstance(value, list):
            if value and not card.get(key):
                card[key] = value
            continue
        if value and not card.get(key):
            card[key] = value
    return card


def _custom_kernel_body(source: str) -> str | None:
    match = CUSTOM_KERNEL_RE.search(source)
    if not match:
        return None
    return match.group("body")


def _custom_kernel_identifiers(source: str) -> set[str]:
    try:
        module_ast = ast.parse(source)
    except SyntaxError:
        return set()
    for node in module_ast.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "custom_kernel":
            continue
        return {
            item.id
            for item in ast.walk(node)
            if isinstance(item, ast.Name)
        }
    return set()


def _infer_variant_name(source_path: Path, label: str | None) -> str:
    if source_path.name == "submission.py":
        parent = source_path.parent.name
        if parent and parent not in {"moe", "amd"}:
            return parent
    if label:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", label).strip("_") or "unknown"
    stem = source_path.stem
    return stem if stem else "unknown"


def _manifest_variant(payload: dict[str, Any]) -> str | None:
    explicit = payload.get("variant")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    source_path = payload.get("source_path")
    if not source_path:
        return None
    label = str(payload.get("label") or "")
    return _infer_variant_name(Path(str(source_path)).expanduser().resolve(), label)


def _infer_lane(name: str, source: str) -> str:
    lowered = f"{name}\n{source}".lower()
    if "fused_moe(" in source:
        return "full_pipeline"
    if "shared" in lowered:
        return "shared_expert"
    if "dispatch" in lowered or "sorting" in lowered or "pack" in lowered or "route" in lowered:
        return "dispatch_pack"
    if "stage2" in lowered or "reduce" in lowered or "weighted" in lowered:
        return "stage2_reduce"
    if "stage1" in lowered or "swiglu" in lowered or "gate_up" in lowered:
        return "stage1_core"
    return "full_pipeline"


def _infer_hot_path_state(source: str) -> str:
    body = _custom_kernel_body(source) or source
    if "fused_moe(" in body:
        return "anchor-backed"
    if "load_inline(" in source or "torch.utils.cpp_extension" in source:
        return "native"
    if "triton.jit" in source or "torch.argsort(" in source or "index_add_(" in body:
        return "partial-native"
    return "unknown"


def _infer_regime_tag(name: str, per_case_times: dict[str, float]) -> str:
    lowered = name.lower()
    if "sparse32" in lowered or "re32" in lowered:
        if "2048" in lowered:
            return "re32_de2048_bs512_topk8"
        if "bs16" in lowered:
            return "re32_de512_bs16_topk8"
        if "bs128" in lowered:
            return "re32_de512_bs128_topk8"
        if "bs512" in lowered:
            return "re32_de512_bs512_topk8"
    if "256" in lowered and "sparse32" not in lowered and "re32" not in lowered:
        if "bs16" in lowered:
            return "re256_de256_bs16_topk8"
        if "bs128" in lowered:
            return "re256_de256_bs128_topk8"
        if "bs512" in lowered:
            return "re256_de256_bs512_topk8"
    if len(per_case_times) == 1:
        return next(iter(per_case_times))
    return "mixed" if per_case_times else "unknown"


def _shape_regression_fraction(
    record_cases: dict[str, float],
    baseline_cases: dict[str, float],
) -> float:
    worst = 0.0
    for key, baseline in baseline_cases.items():
        current = record_cases.get(key)
        if current is None or baseline <= 0.0:
            continue
        regression = (current - baseline) / baseline
        if regression > worst:
            worst = regression
    return worst


def _append_optional_shared_suffix(
    key: str,
    *,
    nsharedexperts: int | None,
    shared_counts_seen: set[int],
) -> str:
    if nsharedexperts is None or len(shared_counts_seen) <= 1:
        return key
    return f"{key}_se{nsharedexperts}"


@dataclass
class RemoteEvent:
    stage: str
    requested_at: str
    run_dir: str
    status: str
    objective_us: float | None = None
    workflow_url: str | None = None
    failure_kind: str | None = None
    failure_signature: str | None = None
    finished_at: str | None = None


@dataclass
class ExperimentRecord:
    variant: str
    lane: str
    hot_path_state: str
    replaced_stages: list[str]
    regime_tag: str
    hypothesis: str
    expected_gain: str
    next_patch: str
    deleted_cost_center: str
    expected_upside_source: str
    why_larger_than_noise: str
    forbidden_edits: list[str]
    success_gate: str
    source_path: str
    baseline_variant: str
    motivation_refs: list[str]
    retrieval_queries: list[str]
    remote_cost: dict[str, int]
    purity_status: str
    preflight_status: str
    test_status: str
    benchmark_status: str
    leaderboard_status: str
    benchmark_geomean_us: float | None
    benchmark_reruns_us: list[float]
    leaderboard_geomean_us: float | None
    per_case_times: dict[str, float]
    delta_vs_working_baseline_us: float | None
    delta_vs_ranked_anchor_us: float | None
    decision: str
    created_at: str
    updated_at: str
    notes: list[str] = field(default_factory=list)
    preflight_report_path: str | None = None
    remote_history: list[RemoteEvent] = field(default_factory=list)
    failure_kind: str | None = None
    failure_signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["remote_history"] = [asdict(item) for item in self.remote_history]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentRecord":
        history = [
            RemoteEvent(**item)
            for item in payload.get("remote_history", [])
            if isinstance(item, dict)
        ]
        return cls(
            variant=str(payload["variant"]),
            lane=str(payload.get("lane", "unknown")),
            hot_path_state=str(payload.get("hot_path_state", "unknown")),
            replaced_stages=[str(item) for item in payload.get("replaced_stages", [])],
            regime_tag=str(payload.get("regime_tag", "unknown")),
            hypothesis=str(payload.get("hypothesis", "")),
            expected_gain=str(payload.get("expected_gain", "")),
            next_patch=str(payload.get("next_patch", "")),
            deleted_cost_center=str(payload.get("deleted_cost_center", "")),
            expected_upside_source=str(payload.get("expected_upside_source", "")),
            why_larger_than_noise=str(payload.get("why_larger_than_noise", "")),
            forbidden_edits=[str(item) for item in payload.get("forbidden_edits", [])],
            success_gate=str(payload.get("success_gate", "")),
            source_path=str(payload.get("source_path", "")),
            baseline_variant=str(payload.get("baseline_variant", "ranked_anchor_2026_03_10")),
            motivation_refs=[str(item) for item in payload.get("motivation_refs", [])],
            retrieval_queries=[str(item) for item in payload.get("retrieval_queries", [])],
            remote_cost=dict(payload.get("remote_cost", {"test": 1, "benchmark": 1, "leaderboard": 1})),
            purity_status=str(payload.get("purity_status", "pending")),
            preflight_status=str(payload.get("preflight_status", "pending")),
            test_status=str(payload.get("test_status", "pending")),
            benchmark_status=str(payload.get("benchmark_status", "pending")),
            leaderboard_status=str(payload.get("leaderboard_status", "pending")),
            benchmark_geomean_us=(
                float(payload["benchmark_geomean_us"])
                if payload.get("benchmark_geomean_us") is not None
                else None
            ),
            benchmark_reruns_us=[
                float(item) for item in payload.get("benchmark_reruns_us", []) if item is not None
            ],
            leaderboard_geomean_us=(
                float(payload["leaderboard_geomean_us"])
                if payload.get("leaderboard_geomean_us") is not None
                else None
            ),
            per_case_times={
                str(key): float(value)
                for key, value in dict(payload.get("per_case_times", {})).items()
            },
            delta_vs_working_baseline_us=(
                float(payload["delta_vs_working_baseline_us"])
                if payload.get("delta_vs_working_baseline_us") is not None
                else None
            ),
            delta_vs_ranked_anchor_us=(
                float(payload["delta_vs_ranked_anchor_us"])
                if payload.get("delta_vs_ranked_anchor_us") is not None
                else None
            ),
            decision=str(payload.get("decision", "pending")),
            created_at=str(payload.get("created_at", _utc_now())),
            updated_at=str(payload.get("updated_at", _utc_now())),
            notes=[str(item) for item in payload.get("notes", [])],
            preflight_report_path=(
                str(payload["preflight_report_path"])
                if payload.get("preflight_report_path")
                else None
            ),
            remote_history=history,
            failure_kind=(
                str(payload["failure_kind"])
                if payload.get("failure_kind")
                else None
            ),
            failure_signature=(
                str(payload["failure_signature"])
                if payload.get("failure_signature")
                else None
            ),
        )


class CoordinatorLock:
    def __init__(self, path: Path):
        self.path = path
        self._handle = None

    def __enter__(self) -> "CoordinatorLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


class MoeClosedLoopCoordinator:
    ranked_anchor_variant = "ranked_anchor_2026_03_10"
    ranked_anchor_geomean_us = 185.1214567536387
    ranked_anchor_per_case_us = {
        "re256_de256_bs16_topk8": 138.0,
        "re256_de256_bs128_topk8": 223.0,
        "re256_de256_bs512_topk8": 254.0,
        "re32_de512_bs16_topk8": 95.7,
        "re32_de512_bs128_topk8": 131.0,
        "re32_de512_bs512_topk8": 216.0,
        "re32_de2048_bs512_topk8": 352.0,
    }
    repo_baseline_variant = "repo_submission"

    @staticmethod
    def _record_requires_candidate_card(record: ExperimentRecord) -> bool:
        return record.variant not in {
            MoeClosedLoopCoordinator.ranked_anchor_variant,
            MoeClosedLoopCoordinator.repo_baseline_variant,
        }

    @staticmethod
    def _missing_candidate_card_fields(record: ExperimentRecord) -> list[str]:
        missing: list[str] = []
        if record.lane == "unknown":
            missing.append("lane")
        if record.regime_tag in {"unknown", "mixed"}:
            missing.append("regime_tag")
        if not record.deleted_cost_center.strip():
            missing.append("deleted_cost_center")
        if not record.expected_upside_source.strip():
            missing.append("expected_upside_source")
        if not record.why_larger_than_noise.strip():
            missing.append("why_larger_than_noise")
        if not record.forbidden_edits:
            missing.append("forbidden_edits")
        if not record.success_gate.strip():
            missing.append("success_gate")
        return missing

    @staticmethod
    def _missing_candidate_evidence(record: ExperimentRecord) -> list[str]:
        missing: list[str] = []
        if not record.motivation_refs:
            missing.append("motivation_refs")
        if not record.retrieval_queries:
            missing.append("retrieval_queries")
        return missing

    def __init__(self, config: AppConfig):
        self.config = config
        self.repo_root = config.repo_root
        self.root = config.workspace.root / "closed_loop" / PROBLEM_KEY
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.root / "experiment_ledger.jsonl"
        self.lock_path = self.root / "coordinator.lock"
        self.reports_dir = self.root / "reports"
        self.preflight_dir = self.root / "preflight"
        self.bootstrap_dir = self.root / "bootstrap"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.preflight_dir.mkdir(parents=True, exist_ok=True)
        self.bootstrap_dir.mkdir(parents=True, exist_ok=True)
        self.harness = KernelHarness(config)
        self.safe_baseline_source = (self.repo_root / "moe" / "submission.py").resolve()
        self.venv_python = self.repo_root / ".venv" / "bin" / "python"
        self._ensure_seed_records()
        self._import_existing_harness_runs()

    def status(self) -> dict[str, Any]:
        records = self._latest_records()
        working = self.working_baseline_record(records)
        return {
            "problem": PROBLEM_KEY,
            "venv_python": str(self.venv_python),
            "safe_baseline_variant": self.repo_baseline_variant,
            "safe_baseline_source": str(self.safe_baseline_source),
            "ranked_anchor_variant": self.ranked_anchor_variant,
            "ranked_anchor_geomean_us": self.ranked_anchor_geomean_us,
            "working_baseline_variant": working.variant if working else self.ranked_anchor_variant,
            "working_baseline_geomean_us": (
                working.benchmark_geomean_us if working and working.benchmark_geomean_us is not None else self.ranked_anchor_geomean_us
            ),
            "record_count": len(records),
            "budget": self.budget_status(),
            "latest_variants": {
                key: {
                    "lane": value.lane,
                    "hot_path_state": value.hot_path_state,
                    "regime_tag": value.regime_tag,
                    "candidate_card_complete": (
                        not self._record_requires_candidate_card(value)
                        or (
                            not self._missing_candidate_card_fields(value)
                            and not self._missing_candidate_evidence(value)
                        )
                    ),
                    "deleted_cost_center": value.deleted_cost_center,
                    "success_gate": value.success_gate,
                    "decision": value.decision,
                    "test_status": value.test_status,
                    "benchmark_status": value.benchmark_status,
                    "leaderboard_status": value.leaderboard_status,
                    "benchmark_geomean_us": value.benchmark_geomean_us,
                    "benchmark_reruns_us": value.benchmark_reruns_us,
                    "delta_vs_working_baseline_us": value.delta_vs_working_baseline_us,
                    "updated_at": value.updated_at,
                }
                for key, value in sorted(records.items())
            },
        }

    def report(self) -> dict[str, Any]:
        records = self._latest_records()
        ordered = sorted(
            records.values(),
            key=lambda item: (
                item.benchmark_geomean_us is None,
                item.benchmark_geomean_us if item.benchmark_geomean_us is not None else float("inf"),
                item.updated_at,
            ),
        )
        return {
            "problem": PROBLEM_KEY,
            "venv_python": str(self.venv_python),
            "ranked_anchor_variant": self.ranked_anchor_variant,
            "ranked_anchor_geomean_us": self.ranked_anchor_geomean_us,
            "records": [record.to_dict() for record in ordered],
        }

    def register_candidate(
        self,
        *,
        variant: str,
        source_path: Path,
        lane: str,
        hot_path_state: str,
        regime_tag: str,
        replaced_stages: list[str] | None,
        hypothesis: str,
        expected_gain: str,
        next_patch: str,
        deleted_cost_center: str = "",
        expected_upside_source: str = "",
        why_larger_than_noise: str = "",
        forbidden_edits: list[str] | None = None,
        success_gate: str = "",
        notes: list[str] | None = None,
        motivation_refs: list[str] | None = None,
        retrieval_queries: list[str] | None = None,
    ) -> ExperimentRecord:
        with CoordinatorLock(self.lock_path):
            record = self._ensure_record(
                variant=variant,
                source_path=source_path,
                lane=lane,
                hot_path_state=hot_path_state,
                regime_tag=regime_tag,
                replaced_stages=replaced_stages,
                hypothesis=hypothesis,
                expected_gain=expected_gain,
                next_patch=next_patch,
                deleted_cost_center=deleted_cost_center,
                expected_upside_source=expected_upside_source,
                why_larger_than_noise=why_larger_than_noise,
                forbidden_edits=forbidden_edits,
                success_gate=success_gate,
                motivation_refs=motivation_refs,
                retrieval_queries=retrieval_queries,
            )
            if self._record_requires_candidate_card(record):
                missing = self._missing_candidate_card_fields(record)
                if missing:
                    raise SystemExit(f"candidate card is incomplete; missing {', '.join(missing)}")
                evidence_missing = self._missing_candidate_evidence(record)
                if evidence_missing:
                    raise SystemExit(f"candidate evidence pack is incomplete; missing {', '.join(evidence_missing)}")
            if notes:
                record.notes.extend(str(item) for item in notes)
            self._refresh_decision_fields({record.variant: record})
            record.updated_at = _utc_now()
            self._append_snapshot(record)
            return record

    def preflight(
        self,
        *,
        variant: str,
        source_path: Path,
        lane: str,
        hot_path_state: str,
        regime_tag: str,
        replaced_stages: list[str] | None,
        hypothesis: str,
        expected_gain: str,
        next_patch: str,
        deleted_cost_center: str = "",
        expected_upside_source: str = "",
        why_larger_than_noise: str = "",
        forbidden_edits: list[str] | None = None,
        success_gate: str = "",
        motivation_refs: list[str] | None = None,
        retrieval_queries: list[str] | None = None,
        profile: str = "amd-parity-full",
        runtime: str = "none",
        build_image: bool = False,
    ) -> dict[str, Any]:
        if profile not in PREFLIGHT_PROFILES:
            raise SystemExit(f"unknown preflight profile {profile!r}")
        with CoordinatorLock(self.lock_path):
            record = self._ensure_record(
                variant=variant,
                source_path=source_path,
                lane=lane,
                hot_path_state=hot_path_state,
                regime_tag=regime_tag,
                replaced_stages=replaced_stages,
                hypothesis=hypothesis,
                expected_gain=expected_gain,
                next_patch=next_patch,
                deleted_cost_center=deleted_cost_center,
                expected_upside_source=expected_upside_source,
                why_larger_than_noise=why_larger_than_noise,
                forbidden_edits=forbidden_edits,
                success_gate=success_gate,
                motivation_refs=motivation_refs,
                retrieval_queries=retrieval_queries,
            )
            report = self._run_preflight_with_optional_container(
                source_path=source_path,
                profile=profile,
                runtime=runtime,
                build_image=build_image,
            )
            source_text = _read_text(source_path)
            self._apply_moe_contract_checks(
                report=report,
                source_text=source_text,
                record=record,
            )
            record.preflight_status = report.status
            record.purity_status = report.purity_status
            record.hot_path_state = _infer_hot_path_state(source_text)
            report_path = self.preflight_dir / f"{variant}-{profile}.json"
            report_path.write_text(
                json.dumps(report.to_dict(), indent=2, sort_keys=True, default=_json_default),
                encoding="utf-8",
            )
            record.preflight_report_path = str(report_path)
            if report.notes:
                record.notes.extend(report.notes)
            self._refresh_decision_fields({record.variant: record})
            record.updated_at = _utc_now()
            self._append_snapshot(record)
            return {
                "variant": variant,
                "lane": record.lane,
                "hot_path_state": record.hot_path_state,
                "regime_tag": record.regime_tag,
                "deleted_cost_center": record.deleted_cost_center,
                "expected_upside_source": record.expected_upside_source,
                "why_larger_than_noise": record.why_larger_than_noise,
                "forbidden_edits": record.forbidden_edits,
                "success_gate": record.success_gate,
                "motivation_refs": record.motivation_refs,
                "retrieval_queries": record.retrieval_queries,
                "source_path": str(source_path.resolve()),
                "profile": profile,
                "runtime": report.runtime,
                "status": report.status,
                "purity_status": report.purity_status,
                "report_path": str(report_path),
                "checks": [asdict(item) for item in report.checks],
                "notes": report.notes,
            }

    def submit(
        self,
        *,
        variant: str,
        source_path: Path,
        lane: str,
        hot_path_state: str,
        regime_tag: str,
        replaced_stages: list[str] | None,
        hypothesis: str,
        expected_gain: str,
        next_patch: str,
        deleted_cost_center: str = "",
        expected_upside_source: str = "",
        why_larger_than_noise: str = "",
        forbidden_edits: list[str] | None = None,
        success_gate: str = "",
        motivation_refs: list[str] | None = None,
        retrieval_queries: list[str] | None = None,
        stage: str,
        label: str = "",
        continue_after_fail: bool = False,
    ) -> dict[str, Any]:
        if stage not in REMOTE_STAGE_VALUES:
            raise SystemExit(f"unknown stage {stage!r}")
        with CoordinatorLock(self.lock_path):
            record = self._ensure_record(
                variant=variant,
                source_path=source_path,
                lane=lane,
                hot_path_state=hot_path_state,
                regime_tag=regime_tag,
                replaced_stages=replaced_stages,
                hypothesis=hypothesis,
                expected_gain=expected_gain,
                next_patch=next_patch,
                deleted_cost_center=deleted_cost_center,
                expected_upside_source=expected_upside_source,
                why_larger_than_noise=why_larger_than_noise,
                forbidden_edits=forbidden_edits,
                success_gate=success_gate,
                motivation_refs=motivation_refs,
                retrieval_queries=retrieval_queries,
            )
            self._sync_record_from_harness(record)
            self._refresh_decision_fields({record.variant: record})
            allowed, rationale = self._check_submission_policy(record, stage)
            if not allowed:
                raise SystemExit(rationale)
            requested_at = _utc_now()
            run_label = label or f"{variant}-{stage}"
            run_dir = self.harness.create_run(
                PROBLEM_KEY,
                source_path=source_path,
                stages=[stage],
                family="moe_closed_loop",
                label=run_label,
                variant=variant,
            )
            record.remote_history.append(
                RemoteEvent(
                    stage=stage,
                    requested_at=requested_at,
                    run_dir=str(run_dir),
                    status="requested",
                )
            )
            record.updated_at = requested_at
            self._append_snapshot(record)
            summary = self.harness.resume_run(
                run_dir,
                continue_after_fail=continue_after_fail,
            )
            self._sync_record_from_harness(record, limit_run_dir=run_dir)
            self._refresh_decision_fields({record.variant: record})
            record.updated_at = _utc_now()
            self._append_snapshot(record)
            return {
                "variant": variant,
                "lane": record.lane,
                "stage": stage,
                "policy": rationale,
                "run_dir": str(run_dir),
                "summary": summary.to_dict(),
                "record": record.to_dict(),
            }

    def budget_status(self, now: datetime | None = None) -> dict[str, Any]:
        current = now or datetime.now(UTC)
        records = self._latest_records()
        shared_events: dict[tuple[str, str], RemoteEvent] = {}
        leaderboard_events: dict[tuple[str, str], RemoteEvent] = {}
        test_events: dict[tuple[str, str], RemoteEvent] = {}
        benchmark_events: dict[tuple[str, str], RemoteEvent] = {}
        for record in records.values():
            for event in record.remote_history:
                event_time = _parse_ts(event.requested_at)
                if event_time is None or current - event_time > timedelta(hours=1):
                    continue
                event_key = (event.run_dir, event.stage)
                if event.stage in SHARED_TEST_BUCKET_STAGES:
                    shared_events[event_key] = event
                    if event.stage == "test":
                        test_events[event_key] = event
                    elif event.stage == "benchmark":
                        benchmark_events[event_key] = event
                elif event.stage == "leaderboard":
                    leaderboard_events[event_key] = event
        shared_limit = 16
        reserve = 0
        usable_shared = shared_limit - reserve
        return {
            "generated_at": current.isoformat(),
            "shared_test_bucket_limit_per_hour": shared_limit,
            "shared_test_bucket_reserved": reserve,
            "shared_test_bucket_usable": usable_shared,
            "shared_test_bucket_used": len(shared_events),
            "shared_test_bucket_remaining_before_reserve": max(0, usable_shared - len(shared_events)),
            "test_stage_limit_per_hour": 10,
            "test_stage_used": len(test_events),
            "test_stage_remaining": max(0, 10 - len(test_events)),
            "benchmark_stage_limit_per_hour": 6,
            "benchmark_stage_used": len(benchmark_events),
            "benchmark_stage_remaining": max(0, 6 - len(benchmark_events)),
            "leaderboard_limit_per_hour": 1,
            "leaderboard_used": len(leaderboard_events),
            "leaderboard_remaining": max(0, 1 - len(leaderboard_events)),
        }

    def current_best_record(self, records: dict[str, ExperimentRecord] | None = None) -> ExperimentRecord | None:
        snapshot = records or self._latest_records()
        candidates = [
            record
            for record in snapshot.values()
            if record.variant != self.ranked_anchor_variant
            and record.benchmark_status == "ok"
            and record.benchmark_geomean_us is not None
        ]
        if not candidates:
            ranked = snapshot.get(self.ranked_anchor_variant)
            return ranked
        return min(candidates, key=lambda item: item.benchmark_geomean_us or float("inf"))

    def working_baseline_record(self, records: dict[str, ExperimentRecord] | None = None) -> ExperimentRecord | None:
        return self.current_best_record(records)

    def _ensure_seed_records(self) -> None:
        records = self._latest_records()
        changed = False
        if self.ranked_anchor_variant not in records:
            ranked = ExperimentRecord(
                variant=self.ranked_anchor_variant,
                lane="full_pipeline",
                hot_path_state="anchor-backed",
                replaced_stages=["ranked_anchor"],
                regime_tag="mixed",
                hypothesis="promoted ranked fused_moe anchor",
                expected_gain="0.0 us ranked anchor reference",
                next_patch="build a non-anchor lane-local MoE kernel",
                deleted_cost_center="none ranked reference",
                expected_upside_source="ranked promoted baseline",
                why_larger_than_noise="seed record only",
                forbidden_edits=[],
                success_gate="reference only",
                source_path=str(self.safe_baseline_source),
                baseline_variant=self.ranked_anchor_variant,
                motivation_refs=[],
                retrieval_queries=[],
                remote_cost={"test": 1, "benchmark": 1, "leaderboard": 1},
                purity_status="ok",
                preflight_status="ok",
                test_status="ok",
                benchmark_status="ok",
                leaderboard_status="ok",
                benchmark_geomean_us=self.ranked_anchor_geomean_us,
                benchmark_reruns_us=[self.ranked_anchor_geomean_us],
                leaderboard_geomean_us=self.ranked_anchor_geomean_us,
                per_case_times=dict(self.ranked_anchor_per_case_us),
                delta_vs_working_baseline_us=0.0,
                delta_vs_ranked_anchor_us=0.0,
                decision="keep",
                created_at=_utc_now(),
                updated_at=_utc_now(),
                notes=["seeded from team_results/ranked/2026-03-10/summary.md"],
            )
            records[ranked.variant] = ranked
            changed = True
        if self.repo_baseline_variant not in records:
            repo_record = ExperimentRecord(
                variant=self.repo_baseline_variant,
                lane="full_pipeline",
                hot_path_state=_infer_hot_path_state(_read_text(self.safe_baseline_source)),
                replaced_stages=["repo_submission"],
                regime_tag="unknown",
                hypothesis="current repo moe/submission.py baseline",
                expected_gain="bootstrap current repo state before non-anchor lanes",
                next_patch="register the first dispatch_pack or stage1_core native lane",
                deleted_cost_center="none repo control",
                expected_upside_source="working tree control path",
                why_larger_than_noise="seed record only",
                forbidden_edits=[],
                success_gate="control only",
                source_path=str(self.safe_baseline_source),
                baseline_variant=self.ranked_anchor_variant,
                motivation_refs=[],
                retrieval_queries=[],
                remote_cost={"test": 1, "benchmark": 1, "leaderboard": 1},
                purity_status="pending",
                preflight_status="pending",
                test_status="pending",
                benchmark_status="pending",
                leaderboard_status="pending",
                benchmark_geomean_us=None,
                benchmark_reruns_us=[],
                leaderboard_geomean_us=None,
                per_case_times={},
                delta_vs_working_baseline_us=None,
                delta_vs_ranked_anchor_us=None,
                decision="pending",
                created_at=_utc_now(),
                updated_at=_utc_now(),
                notes=["seeded from repo moe/submission.py after branch sync"],
            )
            records[repo_record.variant] = repo_record
            changed = True
        if changed:
            self._refresh_decision_fields(records)
            for record in records.values():
                if record.variant in {self.ranked_anchor_variant, self.repo_baseline_variant}:
                    self._append_snapshot(record)

    def _latest_records(self) -> dict[str, ExperimentRecord]:
        records: dict[str, ExperimentRecord] = {}
        if not self.ledger_path.exists():
            return records
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            record = ExperimentRecord.from_dict(payload)
            records[record.variant] = record
        return records

    def _ensure_record(
        self,
        *,
        variant: str,
        source_path: Path,
        lane: str,
        hot_path_state: str,
        regime_tag: str,
        replaced_stages: list[str] | None,
        hypothesis: str,
        expected_gain: str,
        next_patch: str,
        deleted_cost_center: str,
        expected_upside_source: str,
        why_larger_than_noise: str,
        forbidden_edits: list[str] | None,
        success_gate: str,
        motivation_refs: list[str] | None,
        retrieval_queries: list[str] | None,
    ) -> ExperimentRecord:
        records = self._latest_records()
        source_text = _read_text(source_path)
        source_card = _candidate_card_from_source(source_text)
        inferred_lane = _infer_lane(variant, source_text)
        inferred_hot_path_state = _infer_hot_path_state(source_text)
        meta_lane = str(source_card.get("lane", "")).strip()
        meta_regime = str(source_card.get("regime_tag", "")).strip()
        normalized_lane = lane if lane in LANE_VALUES else (meta_lane if meta_lane in LANE_VALUES else inferred_lane)
        normalized_hot_path = (
            hot_path_state if hot_path_state in HOT_PATH_STATES and hot_path_state != "unknown" else inferred_hot_path_state
        )
        if normalized_hot_path == "anchor-backed":
            normalized_lane = "full_pipeline"
        normalized_regime = regime_tag if regime_tag in REGIME_TAG_VALUES else (meta_regime if meta_regime in REGIME_TAG_VALUES else "unknown")
        normalized_stages = [str(item) for item in (replaced_stages or []) if str(item).strip()]
        if not normalized_stages and normalized_hot_path != "anchor-backed" and normalized_lane != "unknown":
            normalized_stages = [normalized_lane]
        normalized_deleted_cost_center = deleted_cost_center or str(source_card.get("deleted_cost_center", ""))
        normalized_expected_upside_source = expected_upside_source or str(source_card.get("expected_upside_source", ""))
        normalized_why_larger_than_noise = why_larger_than_noise or str(source_card.get("why_larger_than_noise", ""))
        normalized_forbidden_edits = [
            str(item)
            for item in (forbidden_edits or _coerce_string_list(source_card.get("forbidden_edits")))
            if str(item).strip()
        ]
        normalized_success_gate = success_gate or str(source_card.get("success_gate", ""))
        normalized_motivation_refs = [str(item) for item in motivation_refs or _coerce_string_list(source_card.get("motivation_refs"))]
        normalized_retrieval_queries = [str(item) for item in retrieval_queries or _coerce_string_list(source_card.get("retrieval_queries"))]
        if variant in records:
            record = records[variant]
            record.source_path = _resolve_path(source_path)
            record.lane = normalized_lane
            record.hot_path_state = normalized_hot_path
            if normalized_regime != "unknown":
                record.regime_tag = normalized_regime
            if normalized_stages:
                record.replaced_stages = normalized_stages
            if hypothesis:
                record.hypothesis = hypothesis
            if expected_gain:
                record.expected_gain = expected_gain
            if next_patch:
                record.next_patch = next_patch
            if normalized_deleted_cost_center:
                record.deleted_cost_center = normalized_deleted_cost_center
            if normalized_expected_upside_source:
                record.expected_upside_source = normalized_expected_upside_source
            if normalized_why_larger_than_noise:
                record.why_larger_than_noise = normalized_why_larger_than_noise
            if normalized_forbidden_edits:
                record.forbidden_edits = normalized_forbidden_edits
            if normalized_success_gate:
                record.success_gate = normalized_success_gate
            if normalized_motivation_refs:
                record.motivation_refs = normalized_motivation_refs
            if normalized_retrieval_queries:
                record.retrieval_queries = normalized_retrieval_queries
            return record
        created = _utc_now()
        baseline = self.working_baseline_record(records)
        record = ExperimentRecord(
            variant=variant,
            lane=normalized_lane,
            hot_path_state=normalized_hot_path,
            replaced_stages=normalized_stages,
            regime_tag=normalized_regime,
            hypothesis=hypothesis,
            expected_gain=expected_gain,
            next_patch=next_patch,
            deleted_cost_center=normalized_deleted_cost_center,
            expected_upside_source=normalized_expected_upside_source,
            why_larger_than_noise=normalized_why_larger_than_noise,
            forbidden_edits=normalized_forbidden_edits,
            success_gate=normalized_success_gate,
            source_path=_resolve_path(source_path),
            baseline_variant=baseline.variant if baseline else self.ranked_anchor_variant,
            motivation_refs=normalized_motivation_refs,
            retrieval_queries=normalized_retrieval_queries,
            remote_cost={"test": 1, "benchmark": 1, "leaderboard": 1},
            purity_status="pending",
            preflight_status="pending",
            test_status="pending",
            benchmark_status="pending",
            leaderboard_status="pending",
            benchmark_geomean_us=None,
            benchmark_reruns_us=[],
            leaderboard_geomean_us=None,
            per_case_times={},
            delta_vs_working_baseline_us=None,
            delta_vs_ranked_anchor_us=None,
            decision="pending",
            created_at=created,
            updated_at=created,
            notes=[],
            remote_history=[],
        )
        self._sync_record_from_harness(record)
        return record

    def _check_submission_policy(self, record: ExperimentRecord, stage: str) -> tuple[bool, str]:
        budget = self.budget_status()
        if record.purity_status == "fail":
            return False, "candidate purity scan failed; fix purity before remote submission"
        if record.preflight_status not in {"ok", "warn"}:
            return False, "local preflight is required before remote submission"
        if self._record_requires_candidate_card(record):
            missing = self._missing_candidate_card_fields(record)
            if missing:
                return False, f"candidate card is incomplete; missing {', '.join(missing)}"
            evidence_missing = self._missing_candidate_evidence(record)
            if evidence_missing:
                return False, f"candidate evidence pack is incomplete; missing {', '.join(evidence_missing)}"
        if stage == "test":
            if budget["shared_test_bucket_remaining_before_reserve"] <= 0:
                return False, "benchmark/test shared quota is exhausted for this hour"
            if budget["test_stage_remaining"] <= 0:
                return False, "test-stage hourly coordinator budget is exhausted"
            return True, "test allowed after preflight"
        if stage == "benchmark":
            if record.test_status != "ok":
                return False, "benchmark requires a passing remote test result"
            if budget["shared_test_bucket_remaining_before_reserve"] <= 0:
                return False, "benchmark/test shared quota is exhausted for this hour"
            if budget["benchmark_stage_remaining"] <= 0:
                return False, "benchmark-stage hourly coordinator budget is exhausted"
            return True, "benchmark allowed after passing test"
        if stage == "leaderboard":
            if budget["leaderboard_remaining"] <= 0:
                return False, "leaderboard hourly budget is exhausted"
            if record.hot_path_state == "anchor-backed":
                return False, "leaderboard quota is reserved for non-anchor MoE candidates"
            if record.lane != "full_pipeline":
                return False, "leaderboard quota is reserved for full_pipeline candidates"
            if record.test_status != "ok" or record.benchmark_status != "ok":
                return False, "leaderboard requires passing remote test and benchmark"
            if record.benchmark_geomean_us is None:
                return False, "leaderboard requires a benchmark geomean"
            if len(record.benchmark_reruns_us) < 2:
                return False, "leaderboard requires two benchmark reruns for the same variant"
            stable = max(record.benchmark_reruns_us) <= min(record.benchmark_reruns_us) * 1.01
            if not stable:
                return False, "leaderboard requires benchmark reruns that agree within 1.0%"
            fast_enough = (
                record.benchmark_geomean_us <= 125.0
                or record.benchmark_geomean_us <= self.ranked_anchor_geomean_us * 0.70
            )
            if not fast_enough:
                return False, "leaderboard requires <=125us or at least 30% improvement versus the ranked anchor"
            baseline = self.working_baseline_record()
            if baseline is not None:
                worst_regression = _shape_regression_fraction(record.per_case_times, baseline.per_case_times)
                if worst_regression > 0.07 and record.benchmark_geomean_us > 115.0:
                    return False, "leaderboard blocks per-case regressions worse than 7% unless the candidate is already <=115us"
            return True, "leaderboard allowed after stable benchmark reruns and score gate"
        return False, f"unsupported stage {stage}"

    def _sync_record_from_harness(
        self,
        record: ExperimentRecord,
        *,
        limit_run_dir: Path | None = None,
    ) -> None:
        runs_root = self.config.workspace.root / "harness_runs" / PROBLEM_KEY
        if not runs_root.exists():
            return
        target_source = _resolve_path(record.source_path)
        manifests = sorted(runs_root.glob("*/manifest.json"))
        seen_events = {
            (event.run_dir, event.stage): event
            for event in record.remote_history
        }
        for manifest_path in manifests:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_path = payload.get("source_path")
            if source_path and _resolve_path(str(source_path)) != target_source:
                continue
            manifest_variant = _manifest_variant(payload)
            if manifest_variant and manifest_variant != record.variant:
                continue
            run_dir = manifest_path.parent.resolve()
            if limit_run_dir is not None and run_dir != limit_run_dir.resolve():
                continue
            for stage in payload.get("stages", []):
                if not isinstance(stage, dict):
                    continue
                stage_name = str(stage.get("name", ""))
                if stage_name not in REMOTE_STAGE_VALUES:
                    continue
                key = (str(run_dir), stage_name)
                objective_us = _mode_objective_to_us(
                    float(stage["objective"]) if stage.get("objective") is not None else None
                )
                event = seen_events.get(key)
                if event is None:
                    event = RemoteEvent(
                        stage=stage_name,
                        requested_at=str(stage.get("started_at") or payload.get("created_at") or _utc_now()),
                        run_dir=str(run_dir),
                        status=str(stage.get("status", "pending")),
                    )
                    record.remote_history.append(event)
                    seen_events[key] = event
                event.status = str(stage.get("status", event.status))
                event.finished_at = str(stage.get("finished_at")) if stage.get("finished_at") else event.finished_at
                event.objective_us = objective_us if objective_us is not None else event.objective_us
                event.workflow_url = str(stage.get("workflow_url")) if stage.get("workflow_url") else event.workflow_url
                event.failure_kind = str(stage.get("failure_kind")) if stage.get("failure_kind") else event.failure_kind
                event.failure_signature = (
                    str(stage.get("failure_signature"))
                    if stage.get("failure_signature")
                    else event.failure_signature
                )
                self._apply_stage_to_record(record, stage_name, stage, objective_us)
        record.remote_history.sort(key=lambda item: item.requested_at)

    def _apply_stage_to_record(
        self,
        record: ExperimentRecord,
        stage_name: str,
        stage_payload: dict[str, Any],
        objective_us: float | None,
    ) -> None:
        status = str(stage_payload.get("status", "pending"))
        failure_kind = str(stage_payload.get("failure_kind")) if stage_payload.get("failure_kind") else None
        failure_signature = str(stage_payload.get("failure_signature")) if stage_payload.get("failure_signature") else None
        record.failure_kind = failure_kind or record.failure_kind
        record.failure_signature = failure_signature or record.failure_signature
        if stage_name == "test":
            record.test_status = status
        elif stage_name == "benchmark":
            record.benchmark_status = status
            if status == "ok" and objective_us is not None:
                metrics = self._load_metrics(stage_payload.get("parsed_metrics_path"))
                if metrics:
                    per_case = self._extract_per_case_times(metrics)
                    if per_case:
                        record.per_case_times = per_case
                        inferred_regime = _infer_regime_tag(record.variant, per_case)
                        if record.regime_tag == "unknown":
                            record.regime_tag = inferred_regime
                ordered_events = sorted(
                    (
                        event
                        for event in record.remote_history
                        if event.stage == "benchmark" and event.status == "ok" and event.objective_us is not None
                    ),
                    key=lambda item: item.requested_at,
                )
                reruns: list[float] = []
                seen: set[float] = set()
                for event in ordered_events:
                    value = round(float(event.objective_us), 6)
                    if value in seen:
                        continue
                    seen.add(value)
                    reruns.append(value)
                current_value = round(objective_us, 6)
                if current_value not in seen:
                    reruns.append(current_value)
                record.benchmark_reruns_us = reruns
                record.benchmark_geomean_us = reruns[-1]
        elif stage_name == "leaderboard":
            record.leaderboard_status = status
            if status == "ok" and objective_us is not None:
                record.leaderboard_geomean_us = objective_us

    def _load_metrics(self, path_value: Any) -> dict[str, Any] | None:
        if not path_value:
            return None
        path = Path(str(path_value))
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def _extract_per_case_times(self, metrics: dict[str, Any]) -> dict[str, float]:
        result: dict[str, float] = {}
        cases = metrics.get("benchmarks")
        if not isinstance(cases, list):
            return result
        shared_counts_seen = {
            int(item["nsharedexperts"])
            for item in cases
            if isinstance(item, dict) and item.get("status") == "pass" and item.get("nsharedexperts") is not None
        }
        for item in cases:
            if not isinstance(item, dict):
                continue
            if item.get("status") != "pass":
                continue
            try:
                nroutedexperts = int(item["nroutedexperts"])
                dexpert = int(item["dexpert"])
                bs = int(item["bs"])
                nexpertspertoken = int(item["nexpertspertoken"])
                mean_us = float(item["mean_ns"]) / 1000.0
            except (KeyError, TypeError, ValueError):
                continue
            shared = None
            try:
                if item.get("nsharedexperts") is not None:
                    shared = int(item["nsharedexperts"])
            except (TypeError, ValueError):
                shared = None
            key = f"re{nroutedexperts}_de{dexpert}_bs{bs}_topk{nexpertspertoken}"
            key = _append_optional_shared_suffix(key, nsharedexperts=shared, shared_counts_seen=shared_counts_seen)
            result[key] = mean_us
        return result

    def _apply_moe_contract_checks(
        self,
        *,
        report: PreflightReport,
        source_text: str,
        record: ExperimentRecord,
    ) -> None:
        identifiers = _custom_kernel_identifiers(source_text)
        topk_visible = "topk_ids" in identifiers and "topk_weights" in identifiers
        report.checks.append(
            PreflightCheck(
                "moe_topk_contract",
                "ok" if topk_visible else "fail",
                "topk_ids/topk_weights are visible in custom_kernel" if topk_visible else "custom_kernel must keep topk_ids and topk_weights visible",
            )
        )
        if not topk_visible:
            report.status = "fail"
            report.purity_status = "fail"
            return

        uses_anchor = "fused_moe(" in source_text
        if record.lane != "full_pipeline" and uses_anchor:
            report.checks.append(
                PreflightCheck(
                    "moe_anchor_hot_path",
                    "fail",
                    f"lane {record.lane} must not route the hot path through fused_moe(",
                )
            )
            report.status = "fail"
            report.purity_status = "fail"
            return
        report.checks.append(
            PreflightCheck(
                "moe_anchor_hot_path",
                "ok",
                "anchor usage matches lane policy",
            )
        )

        stage_hints = {
            "dispatch_pack": ("argsort(", "sorted_token", "sorted_expert", "bincount(", "unique(", "route"),
            "stage1_core": ("gate_up", "stage1", "gate =", "up =", "chunk(", "@"),
            "stage2_reduce": ("down_weight", "stage2", "index_add_", "weighted", "expert_out"),
            "shared_expert": ("shared_expert", "nsharedexperts", "shared_ids", "shared_weight"),
        }
        if record.lane in stage_hints:
            visible = any(token in source_text for token in stage_hints[record.lane])
            report.checks.append(
                PreflightCheck(
                    "moe_stage_ownership",
                    "ok" if visible else "fail",
                    f"candidate visibly owns lane {record.lane}" if visible else f"candidate does not visibly own lane {record.lane}",
                )
            )
            if not visible:
                report.status = "fail"
                report.purity_status = "fail"
                return

        if self._record_requires_candidate_card(record):
            missing = self._missing_candidate_card_fields(record)
            report.checks.append(
                PreflightCheck(
                    "moe_candidate_card",
                    "ok" if not missing else "fail",
                    "candidate card is complete" if not missing else f"candidate card is missing {', '.join(missing)}",
                )
            )
            if missing:
                report.status = "fail"
                report.purity_status = "fail"
                return
            evidence_missing = self._missing_candidate_evidence(record)
            report.checks.append(
                PreflightCheck(
                    "moe_candidate_evidence",
                    "ok" if not evidence_missing else "warn",
                    "motivation refs and retrieval queries are present"
                    if not evidence_missing
                    else f"candidate evidence is missing {', '.join(evidence_missing)}",
                )
            )
            if evidence_missing:
                report.notes.append(
                    f"candidate evidence is incomplete: {', '.join(evidence_missing)}; submit will be blocked until they are present"
                )
                if report.status == "ok":
                    report.status = "warn"

        lower_source = source_text.lower()
        all_expert_rebuild = (
            (
                re.search(r"for\s+\w+\s+in\s+range\(\s*experts\s*\)", lower_source)
                or re.search(r"for\s+\w+\s+in\s+range\(\s*num_experts\s*\)", lower_source)
                or "torch.stack(gate_w)" in source_text
                or "torch.stack(up_w)" in source_text
                or "torch.stack(down_w)" in source_text
            )
            and "unique_experts" not in lower_source
            and "torch.unique(" not in lower_source
        )
        if record.hot_path_state != "anchor-backed" and all_expert_rebuild:
            report.checks.append(
                PreflightCheck(
                    "moe_all_expert_rebuild",
                    "fail",
                    "non-anchor candidates must not eagerly dequantize or rebuild all experts in Python",
                )
            )
            report.status = "fail"
            report.purity_status = "fail"
            return
        report.checks.append(
            PreflightCheck(
                "moe_all_expert_rebuild",
                "ok",
                "no eager all-expert Python rebuild pattern detected",
            )
        )

    def _refresh_decision_fields(self, records: dict[str, ExperimentRecord]) -> None:
        latest = self._latest_records()
        latest.update(records)
        baseline = self.current_best_record(latest)
        baseline_per_case = baseline.per_case_times if baseline is not None else self.ranked_anchor_per_case_us
        baseline_variant = baseline.variant if baseline is not None else self.ranked_anchor_variant
        baseline_geomean = (
            baseline.benchmark_geomean_us
            if baseline is not None and baseline.benchmark_geomean_us is not None
            else self.ranked_anchor_geomean_us
        )
        for record in latest.values():
            if record.benchmark_geomean_us is not None:
                record.delta_vs_working_baseline_us = record.benchmark_geomean_us - baseline_geomean
                record.delta_vs_ranked_anchor_us = record.benchmark_geomean_us - self.ranked_anchor_geomean_us
                record.baseline_variant = baseline_variant
            else:
                record.delta_vs_working_baseline_us = None
                record.delta_vs_ranked_anchor_us = None
            if record.benchmark_status == "ok" and record.benchmark_geomean_us is not None:
                worst_regression = _shape_regression_fraction(record.per_case_times, baseline_per_case)
                if record.variant == baseline_variant:
                    record.decision = "keep"
                elif record.benchmark_geomean_us <= baseline_geomean * 0.985 and worst_regression <= 0.05:
                    record.decision = "keep"
                elif (
                    record.regime_tag in record.per_case_times
                    and record.regime_tag in baseline_per_case
                    and record.per_case_times[record.regime_tag] <= baseline_per_case[record.regime_tag] * 0.97
                    and abs(record.benchmark_geomean_us - baseline_geomean) <= baseline_geomean * 0.01
                ):
                    record.decision = "stash"
                else:
                    record.decision = "discard"
            elif record.preflight_status in {"ok", "warn"} or record.test_status == "ok":
                record.decision = "pending"
            else:
                record.decision = record.decision if record.decision != "pending" else "pending"
        records.clear()
        records.update(latest)

    def _append_snapshot(self, record: ExperimentRecord) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True, default=_json_default))
            handle.write("\n")

    def _import_existing_harness_runs(self) -> None:
        runs_root = self.config.workspace.root / "harness_runs" / PROBLEM_KEY
        if not runs_root.exists():
            return
        with CoordinatorLock(self.lock_path):
            records = self._latest_records()
            before_snapshots = {
                variant: json.dumps(record.to_dict(), sort_keys=True, default=_json_default)
                for variant, record in records.items()
            }
            for manifest_path in sorted(runs_root.glob("*/manifest.json")):
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                source_path_raw = payload.get("source_path")
                if not source_path_raw:
                    continue
                source_path = Path(str(source_path_raw)).expanduser().resolve()
                label = str(payload.get("label") or "")
                variant = _manifest_variant(payload)
                if not variant:
                    continue
                source_text = _read_text(source_path)
                if variant in records:
                    record = records[variant]
                else:
                    record = ExperimentRecord(
                        variant=variant,
                        lane=_infer_lane(label or variant, source_text),
                        hot_path_state=_infer_hot_path_state(source_text),
                        replaced_stages=[],
                        regime_tag="unknown",
                        hypothesis=f"imported from harness run {label or variant}",
                        expected_gain="imported historical MoE harness evidence",
                        next_patch="promote a real non-anchor lane if this stays best",
                        deleted_cost_center="",
                        expected_upside_source="",
                        why_larger_than_noise="",
                        forbidden_edits=[],
                        success_gate="",
                        source_path=str(source_path),
                        baseline_variant=self.ranked_anchor_variant,
                        motivation_refs=[],
                        retrieval_queries=[],
                        remote_cost={"test": 1, "benchmark": 1, "leaderboard": 1},
                        purity_status="pending",
                        preflight_status="pending",
                        test_status="pending",
                        benchmark_status="pending",
                        leaderboard_status="pending",
                        benchmark_geomean_us=None,
                        benchmark_reruns_us=[],
                        leaderboard_geomean_us=None,
                        per_case_times={},
                        delta_vs_working_baseline_us=None,
                        delta_vs_ranked_anchor_us=None,
                        decision="pending",
                        created_at=_utc_now(),
                        updated_at=_utc_now(),
                        notes=[f"imported from harness run label={label}"],
                    )
                    records[variant] = record
                before = json.dumps(record.to_dict(), sort_keys=True, default=_json_default)
                record.source_path = str(source_path)
                record.hot_path_state = _infer_hot_path_state(source_text)
                source_card = _candidate_card_from_source(source_text)
                if not record.deleted_cost_center:
                    record.deleted_cost_center = str(source_card.get("deleted_cost_center", ""))
                if not record.expected_upside_source:
                    record.expected_upside_source = str(source_card.get("expected_upside_source", ""))
                if not record.why_larger_than_noise:
                    record.why_larger_than_noise = str(source_card.get("why_larger_than_noise", ""))
                if not record.forbidden_edits:
                    record.forbidden_edits = _coerce_string_list(source_card.get("forbidden_edits"))
                if not record.success_gate:
                    record.success_gate = str(source_card.get("success_gate", ""))
                if not record.motivation_refs:
                    record.motivation_refs = _coerce_string_list(source_card.get("motivation_refs"))
                if not record.retrieval_queries:
                    record.retrieval_queries = _coerce_string_list(source_card.get("retrieval_queries"))
                if record.hot_path_state == "anchor-backed":
                    record.lane = "full_pipeline"
                elif record.lane == "unknown":
                    source_lane = str(source_card.get("lane", ""))
                    record.lane = source_lane if source_lane in LANE_VALUES else _infer_lane(label or variant, source_text)
                self._sync_record_from_harness(record, limit_run_dir=manifest_path.parent)
                if record.regime_tag == "unknown":
                    source_regime = str(source_card.get("regime_tag", ""))
                    if source_regime in REGIME_TAG_VALUES:
                        record.regime_tag = source_regime
                    else:
                        record.regime_tag = _infer_regime_tag(label or variant, record.per_case_times)
                record.updated_at = _utc_now()
            self._refresh_decision_fields(records)
            for variant, record in sorted(records.items()):
                after = json.dumps(record.to_dict(), sort_keys=True, default=_json_default)
                if after != before_snapshots.get(variant):
                    self._append_snapshot(record)

    def _run_preflight_with_optional_container(
        self,
        *,
        source_path: Path,
        profile: str,
        runtime: str,
        build_image: bool,
    ) -> PreflightReport:
        runtime_cmd = self._resolve_container_runtime(runtime)
        if runtime_cmd is None:
            report = run_host_preflight(
                repo_root=self.repo_root,
                config_path=self.config.config_path,
                problem_key=PROBLEM_KEY,
                source_path=source_path,
                compile_jit=False,
                runtime_label="host-static-only",
                static_only=True,
            )
            report.notes.append("remote-first mode: Docker parity preflight skipped")
            return report

        image_tag = f"agent-loop/{PROBLEM_KEY}:{profile}"
        dockerfile = self.repo_root / "agent_loop" / "docker" / f"Dockerfile.{profile}"
        if not dockerfile.exists():
            report = run_host_preflight(
                repo_root=self.repo_root,
                config_path=self.config.config_path,
                problem_key=PROBLEM_KEY,
                source_path=source_path,
                compile_jit=False,
                runtime_label="host-fallback",
            )
            report.status = "warn" if report.status == "ok" else report.status
            report.notes.append(f"preflight Dockerfile missing: {dockerfile}")
            return report
        if build_image or not self._container_image_exists(runtime_cmd, image_tag):
            try:
                subprocess.run(
                    [
                        runtime_cmd,
                        "build",
                        "--platform",
                        CONTAINER_PLATFORM,
                        "-f",
                        str(dockerfile),
                        "-t",
                        image_tag,
                        str(self.repo_root),
                    ],
                    cwd=str(self.repo_root),
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                report = run_host_preflight(
                    repo_root=self.repo_root,
                    config_path=self.config.config_path,
                    problem_key=PROBLEM_KEY,
                    source_path=source_path,
                    compile_jit=False,
                    runtime_label="host-fallback",
                )
                report.status = "warn" if report.status == "ok" else report.status
                report.notes.append(f"container image build failed: {exc}")
                return report
        worker_cmd = [
            runtime_cmd,
            "run",
            "--rm",
            "--platform",
            CONTAINER_PLATFORM,
            "-v",
            f"{self.repo_root}:/workspace",
            "-w",
            "/workspace",
            image_tag,
            "python3",
            "-m",
            "agent_loop.preflight_worker",
            "--config",
            "/workspace/agent_loop.toml",
            "--problem",
            PROBLEM_KEY,
            "--source",
            f"/workspace/{source_path.resolve().relative_to(self.repo_root)}",
            "--compile-jit",
        ]
        try:
            completed = subprocess.run(
                worker_cmd,
                cwd=str(self.repo_root),
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            report = run_host_preflight(
                repo_root=self.repo_root,
                config_path=self.config.config_path,
                problem_key=PROBLEM_KEY,
                source_path=source_path,
                compile_jit=False,
                runtime_label="host-fallback",
            )
            report.status = "warn" if report.status == "ok" else report.status
            report.notes.append(f"container preflight execution failed: {exc}")
            return report
        if completed.returncode != 0:
            report = run_host_preflight(
                repo_root=self.repo_root,
                config_path=self.config.config_path,
                problem_key=PROBLEM_KEY,
                source_path=source_path,
                compile_jit=False,
                runtime_label="host-fallback",
            )
            report.status = "warn" if report.status == "ok" else report.status
            report.notes.append(f"container preflight failed: {completed.stderr.strip() or completed.stdout.strip()}")
            return report
        payload = json.loads(completed.stdout)
        return PreflightReport.from_dict(payload)

    def _resolve_container_runtime(self, runtime: str) -> str | None:
        if runtime == "none":
            return None
        if runtime in {"docker", "podman"}:
            return runtime
        for candidate in ("docker", "podman"):
            if subprocess.run(
                [candidate, "--version"],
                text=True,
                capture_output=True,
                check=False,
            ).returncode == 0:
                return candidate
        return None

    def _container_image_exists(self, runtime_cmd: str, image_tag: str) -> bool:
        return (
            subprocess.run(
                [runtime_cmd, "image", "inspect", image_tag],
                text=True,
                capture_output=True,
                check=False,
            ).returncode
            == 0
        )


def _problem_dir(config: AppConfig) -> Path:
    return (config.repo_root.parent / PROBLEM_DIR_BY_KEY[PROBLEM_KEY]).resolve()


def default_source_path(config: AppConfig, source_arg: str | None) -> Path:
    if source_arg:
        return Path(source_arg).expanduser().resolve()
    return MoeClosedLoopCoordinator(config).safe_baseline_source
