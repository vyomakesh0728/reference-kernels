from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .config import AppConfig
from .harness import KernelHarness
from .preflight_worker import (
    PROBLEM_DIR_BY_KEY,
    PreflightReport,
    run_host_preflight,
)


TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S.%f%z"
PROBLEM_KEY = "mxfp4_mm"
LANE_VALUES = {"A", "B", "A+B", "unknown"}
PREFLIGHT_PROFILES = {"amd-parity-full", "amd-compile-fast"}
REMOTE_STAGE_VALUES = {"test", "benchmark", "leaderboard"}
SHARED_TEST_BUCKET_STAGES = {"test", "benchmark"}
CONTAINER_PLATFORM = "linux/amd64"


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
    hypothesis: str
    expected_gain: str
    remote_cost: dict[str, int]
    purity_status: str
    preflight_status: str
    test_status: str
    benchmark_status: str
    leaderboard_status: str
    benchmark_geomean: float | None
    per_shape_times: dict[str, float]
    decision: str
    next_patch: str
    source_path: str
    baseline_variant: str
    created_at: str
    updated_at: str
    notes: list[str] = field(default_factory=list)
    preflight_report_path: str | None = None
    remote_history: list[RemoteEvent] = field(default_factory=list)

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
            hypothesis=str(payload.get("hypothesis", "")),
            expected_gain=str(payload.get("expected_gain", "")),
            remote_cost=dict(payload.get("remote_cost", {"test": 1, "benchmark": 1, "leaderboard": 1})),
            purity_status=str(payload.get("purity_status", "pending")),
            preflight_status=str(payload.get("preflight_status", "pending")),
            test_status=str(payload.get("test_status", "pending")),
            benchmark_status=str(payload.get("benchmark_status", "pending")),
            leaderboard_status=str(payload.get("leaderboard_status", "pending")),
            benchmark_geomean=(
                float(payload["benchmark_geomean"])
                if payload.get("benchmark_geomean") is not None
                else None
            ),
            per_shape_times={
                str(key): float(value)
                for key, value in dict(payload.get("per_shape_times", {})).items()
            },
            decision=str(payload.get("decision", "pending")),
            next_patch=str(payload.get("next_patch", "")),
            source_path=str(payload.get("source_path", "")),
            baseline_variant=str(payload.get("baseline_variant", "v44")),
            created_at=str(payload.get("created_at", _utc_now())),
            updated_at=str(payload.get("updated_at", _utc_now())),
            notes=[str(item) for item in payload.get("notes", [])],
            preflight_report_path=(
                str(payload["preflight_report_path"])
                if payload.get("preflight_report_path")
                else None
            ),
            remote_history=history,
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


class Mxfp4ClosedLoopCoordinator:
    promoted_baseline_variant = "v44"
    promoted_baseline_geomean_us = 46.964308646121994
    promoted_baseline_per_shape_us = {
        "m4_n2880_k512": 26.9,
        "m16_n2112_k7168": 108.0,
        "m32_n4096_k512": 30.8,
        "m32_n2880_k512": 29.1,
        "m64_n7168_k2048": 79.4,
        "m256_n3072_k1536": 51.9,
    }

    def __init__(self, config: AppConfig):
        self.config = config
        self.repo_root = config.repo_root
        self.root = config.workspace.root / "closed_loop" / PROBLEM_KEY
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.root / "experiment_ledger.jsonl"
        self.lock_path = self.root / "coordinator.lock"
        self.reports_dir = self.root / "reports"
        self.preflight_dir = self.root / "preflight"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.preflight_dir.mkdir(parents=True, exist_ok=True)
        self.harness = KernelHarness(config)
        self.safe_baseline_source = (
            self.repo_root
            / ".agent-loop/manual/native_scaled_pure_compiled_bscale_v44/submission.py"
        ).resolve()
        self._ensure_baseline_snapshot()

    def status(self) -> dict[str, Any]:
        records = self._latest_records()
        budget = self.budget_status()
        best_record = self.current_best_record()
        return {
            "problem": PROBLEM_KEY,
            "safe_baseline_variant": self.promoted_baseline_variant,
            "safe_baseline_source": str(self.safe_baseline_source),
            "record_count": len(records),
            "current_best_variant": best_record.variant if best_record else self.promoted_baseline_variant,
            "current_best_geomean_us": (
                best_record.benchmark_geomean
                if best_record and best_record.benchmark_geomean is not None
                else self.promoted_baseline_geomean_us
            ),
            "budget": budget,
            "latest_variants": {
                key: {
                    "lane": value.lane,
                    "decision": value.decision,
                    "test_status": value.test_status,
                    "benchmark_status": value.benchmark_status,
                    "leaderboard_status": value.leaderboard_status,
                    "benchmark_geomean": value.benchmark_geomean,
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
                item.benchmark_geomean is None,
                item.benchmark_geomean if item.benchmark_geomean is not None else float("inf"),
                item.updated_at,
            ),
        )
        return {
            "problem": PROBLEM_KEY,
            "safe_baseline_variant": self.promoted_baseline_variant,
            "safe_baseline_geomean_us": self.promoted_baseline_geomean_us,
            "records": [record.to_dict() for record in ordered],
        }

    def register_candidate(
        self,
        *,
        variant: str,
        source_path: Path,
        lane: str,
        hypothesis: str,
        expected_gain: str,
        next_patch: str,
        notes: list[str] | None = None,
    ) -> ExperimentRecord:
        with CoordinatorLock(self.lock_path):
            record = self._ensure_record(
                variant=variant,
                source_path=source_path,
                lane=lane,
                hypothesis=hypothesis,
                expected_gain=expected_gain,
                next_patch=next_patch,
            )
            if notes:
                record.notes.extend(str(item) for item in notes)
            record.updated_at = _utc_now()
            self._append_snapshot(record)
            return record

    def preflight(
        self,
        *,
        variant: str,
        source_path: Path,
        lane: str,
        hypothesis: str,
        expected_gain: str,
        next_patch: str,
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
                hypothesis=hypothesis,
                expected_gain=expected_gain,
                next_patch=next_patch,
            )
            purity = run_host_preflight(
                repo_root=self.repo_root,
                config_path=self.config.config_path,
                problem_key=PROBLEM_KEY,
                source_path=source_path,
                compile_jit=False,
                runtime_label="host-static",
            )
            record.purity_status = purity.purity_status
            report = self._run_preflight_with_optional_container(
                source_path=source_path,
                profile=profile,
                runtime=runtime,
                build_image=build_image,
            )
            record.preflight_status = report.status
            if "fail" in {purity.purity_status, report.purity_status}:
                record.purity_status = "fail"
            elif "warn" in {purity.purity_status, report.purity_status}:
                record.purity_status = "warn"
            else:
                record.purity_status = "ok"
            report_path = self.preflight_dir / f"{variant}-{profile}.json"
            report_path.write_text(
                json.dumps(report.to_dict(), indent=2, sort_keys=True, default=_json_default),
                encoding="utf-8",
            )
            record.preflight_report_path = str(report_path)
            if report.notes:
                record.notes.extend(report.notes)
            record.updated_at = _utc_now()
            self._append_snapshot(record)
            return {
                "variant": variant,
                "lane": lane,
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
        hypothesis: str,
        expected_gain: str,
        next_patch: str,
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
                hypothesis=hypothesis,
                expected_gain=expected_gain,
                next_patch=next_patch,
            )
            self._sync_record_from_harness(record)
            allowed, rationale = self._check_submission_policy(record, stage)
            if not allowed:
                raise SystemExit(rationale)
            requested_at = _utc_now()
            run_label = label or f"{variant}-{stage}"
            run_dir = self.harness.create_run(
                PROBLEM_KEY,
                source_path=source_path,
                stages=[stage],
                family="closed_loop_coordinator",
                label=run_label,
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
            record.updated_at = _utc_now()
            self._append_snapshot(record)
            return {
                "variant": variant,
                "lane": lane,
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
        shared_limit = 6
        reserve = 2
        usable_shared = shared_limit - reserve
        next_lb_earliest = current.replace(minute=45, second=0, microsecond=0)
        if current.minute >= 45:
            next_lb_earliest = current
        return {
            "generated_at": current.isoformat(),
            "shared_test_bucket_limit_per_hour": shared_limit,
            "shared_test_bucket_reserved": reserve,
            "shared_test_bucket_usable": usable_shared,
            "shared_test_bucket_used": len(shared_events),
            "shared_test_bucket_remaining_before_reserve": max(0, usable_shared - len(shared_events)),
            "test_stage_limit_per_hour": 2,
            "test_stage_used": len(test_events),
            "test_stage_remaining": max(0, 2 - len(test_events)),
            "benchmark_stage_limit_per_hour": 2,
            "benchmark_stage_used": len(benchmark_events),
            "benchmark_stage_remaining": max(0, 2 - len(benchmark_events)),
            "leaderboard_limit_per_hour": 1,
            "leaderboard_used": len(leaderboard_events),
            "leaderboard_remaining": max(0, 1 - len(leaderboard_events)),
            "leaderboard_policy_ready_after": next_lb_earliest.isoformat(),
        }

    def current_best_record(self) -> ExperimentRecord | None:
        records = self._latest_records()
        candidates = [
            record
            for record in records.values()
            if record.benchmark_status == "ok" and record.benchmark_geomean is not None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item.benchmark_geomean or float("inf"))

    def _append_snapshot(self, record: ExperimentRecord) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True, default=_json_default))
            handle.write("\n")

    def _ensure_baseline_snapshot(self) -> None:
        records = self._latest_records()
        if self.promoted_baseline_variant in records:
            return
        baseline = ExperimentRecord(
            variant=self.promoted_baseline_variant,
            lane="A+B",
            hypothesis="promoted safe pure baseline",
            expected_gain="0.0 us baseline anchor",
            remote_cost={"test": 1, "benchmark": 1, "leaderboard": 1},
            purity_status="ok",
            preflight_status="ok",
            test_status="ok",
            benchmark_status="ok",
            leaderboard_status="pending",
            benchmark_geomean=self.promoted_baseline_geomean_us,
            per_shape_times=dict(self.promoted_baseline_per_shape_us),
            decision="keep",
            next_patch="thin direct-contract data movement",
            source_path=str(self.safe_baseline_source),
            baseline_variant=self.promoted_baseline_variant,
            created_at=_utc_now(),
            updated_at=_utc_now(),
            notes=["seeded from promoted team summary baseline"],
            remote_history=[],
        )
        self._append_snapshot(baseline)

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
        hypothesis: str,
        expected_gain: str,
        next_patch: str,
    ) -> ExperimentRecord:
        records = self._latest_records()
        if variant in records:
            record = records[variant]
            record.source_path = _resolve_path(source_path)
            if lane in LANE_VALUES:
                record.lane = lane
            if hypothesis:
                record.hypothesis = hypothesis
            if expected_gain:
                record.expected_gain = expected_gain
            if next_patch:
                record.next_patch = next_patch
            return record
        created = _utc_now()
        record = ExperimentRecord(
            variant=variant,
            lane=lane if lane in LANE_VALUES else "unknown",
            hypothesis=hypothesis,
            expected_gain=expected_gain,
            remote_cost={"test": 1, "benchmark": 1, "leaderboard": 1},
            purity_status="pending",
            preflight_status="pending",
            test_status="pending",
            benchmark_status="pending",
            leaderboard_status="pending",
            benchmark_geomean=None,
            per_shape_times={},
            decision="pending",
            next_patch=next_patch,
            source_path=_resolve_path(source_path),
            baseline_variant=self.promoted_baseline_variant,
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
            now = datetime.now(UTC)
            if now.minute < 45:
                return False, "leaderboard submissions are blocked before minute 45 of the hour"
            if budget["leaderboard_remaining"] <= 0:
                return False, "leaderboard hourly budget is exhausted"
            if record.test_status != "ok" or record.benchmark_status != "ok":
                return False, "leaderboard requires passing remote test and benchmark"
            if record.benchmark_geomean is None:
                return False, "leaderboard requires a benchmark geomean"
            if record.benchmark_geomean > self.promoted_baseline_geomean_us * 0.90:
                return False, "leaderboard requires at least a 10% geomean improvement over the promoted safe baseline"
            if self._shape_regression_exceeds(record, threshold=0.05) and record.benchmark_geomean > self.promoted_baseline_geomean_us * 0.85:
                return False, "leaderboard blocks shape regressions worse than 5% unless overall geomean improves by at least 15%"
            return True, "leaderboard allowed under conservative promotion policy"
        return False, f"unsupported stage {stage}"

    def _shape_regression_exceeds(self, record: ExperimentRecord, threshold: float) -> bool:
        for shape, baseline in self.promoted_baseline_per_shape_us.items():
            current = record.per_shape_times.get(shape)
            if current is None:
                continue
            if current > baseline * (1.0 + threshold):
                return True
        return False

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
        if stage_name == "test":
            record.test_status = status
        elif stage_name == "benchmark":
            record.benchmark_status = status
            if status == "ok" and objective_us is not None:
                record.benchmark_geomean = objective_us
                metrics = self._load_metrics(stage_payload.get("parsed_metrics_path"))
                if metrics:
                    record.per_shape_times = self._extract_per_shape_times(metrics)
                if objective_us < self.promoted_baseline_geomean_us:
                    record.decision = "keep"
                else:
                    record.decision = "discard"
        elif stage_name == "leaderboard":
            record.leaderboard_status = status

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

    def _extract_per_shape_times(self, metrics: dict[str, Any]) -> dict[str, float]:
        result: dict[str, float] = {}
        cases = metrics.get("benchmarks")
        if not isinstance(cases, list):
            return result
        for item in cases:
            if not isinstance(item, dict):
                continue
            if item.get("status") != "pass":
                continue
            try:
                m = int(item["m"])
                n = int(item["n"])
                k = int(item["k"])
                mean_us = float(item["mean_ns"]) / 1000.0
            except (KeyError, TypeError, ValueError):
                continue
            result[f"m{m}_n{n}_k{k}"] = mean_us
        return result

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


def build_record_from_args(args) -> dict[str, Any]:
    return {
        "variant": str(args.variant),
        "source_path": Path(args.source).expanduser().resolve(),
        "lane": str(args.lane),
        "hypothesis": str(args.hypothesis),
        "expected_gain": str(args.expected_gain),
        "next_patch": str(args.next_patch),
    }


def _problem_dir(config: AppConfig) -> Path:
    return (config.repo_root.parent / PROBLEM_DIR_BY_KEY[PROBLEM_KEY]).resolve()


def default_source_path(config: AppConfig, source_arg: str | None) -> Path:
    if source_arg:
        return Path(source_arg).expanduser().resolve()
    return Mxfp4ClosedLoopCoordinator(config).safe_baseline_source
