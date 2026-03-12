from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil

from .config import AppConfig
from .evaluator import EvaluationResult, run_popcorn_submission


TARGET_BY_STAGE = {
    "test": "kernelbot-test",
    "benchmark": "kernelbot-benchmark",
    "leaderboard": "kernelbot-ranked",
}


@dataclass(frozen=True)
class HarnessSummary:
    run_dir: Path
    problem: str
    source_path: Path
    stages: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        completed = [stage for stage in self.stages if stage.get("status") not in {"pending", "running"}]
        ok_count = sum(1 for stage in completed if stage.get("status") == "ok")
        last_stage = self.stages[-1] if self.stages else {}
        return {
            "run_dir": str(self.run_dir),
            "problem": self.problem,
            "source_path": str(self.source_path),
            "stage_count": len(self.stages),
            "completed_stage_count": len(completed),
            "ok_stage_count": ok_count,
            "last_stage": last_stage,
            "stages": self.stages,
        }


class KernelHarness:
    def __init__(self, config: AppConfig):
        self.config = config

    def create_run(
        self,
        problem_key: str,
        *,
        source_path: Path,
        stages: list[str],
        family: str | None = None,
        label: str = "",
    ) -> Path:
        problem = self.config.require_problem(problem_key)
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        safe_label = _slugify(label or source_path.stem or "run")
        run_dir = self._runs_root(problem_key) / f"{timestamp}-{safe_label}"
        run_dir.mkdir(parents=True, exist_ok=False)

        copied_source = run_dir / "submission.py"
        shutil.copy2(source_path, copied_source)

        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "problem": problem_key,
            "family": family,
            "label": label,
            "source_path": str(source_path.resolve()),
            "submission_copy": str(copied_source),
            "leaderboard": problem.leaderboard,
            "gpu": problem.gpu,
            "harness_style": "kernelbench-v3-inspired",
            "stages": [
                {
                    "name": stage,
                    "target": TARGET_BY_STAGE.get(stage, f"kernelbot-{stage}"),
                    "status": "pending",
                }
                for stage in stages
            ],
        }
        self._write_manifest(run_dir, manifest)
        return run_dir

    def resume_run(self, run_dir: Path, *, continue_after_fail: bool = False) -> HarnessSummary:
        manifest = self._read_manifest(run_dir)
        problem_key = str(manifest["problem"])
        problem = self.config.require_problem(problem_key)
        submission_path = Path(str(manifest["submission_copy"]))

        stages = manifest.get("stages", [])
        if not isinstance(stages, list):
            raise RuntimeError(f"invalid harness manifest in {run_dir}")

        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                continue
            status = str(stage.get("status", "pending"))
            if status == "ok":
                continue
            if status not in {"pending", "running"} and not continue_after_fail:
                break

            stage_name = str(stage["name"])
            stage_dir = run_dir / "stages" / f"{index + 1:02d}_{stage_name}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            stage["status"] = "running"
            stage["started_at"] = datetime.now(UTC).isoformat()
            self._write_manifest(run_dir, manifest)

            result = run_popcorn_submission(
                workspace=self.config.workspace,
                problem=problem,
                submission_path=submission_path,
                artifacts_dir=stage_dir,
                mode=stage_name,
            )
            self._record_stage_result(stage, stage_dir, result)
            stage["finished_at"] = datetime.now(UTC).isoformat()
            self._write_manifest(run_dir, manifest)

            if result.status != "ok" and not continue_after_fail:
                break

        return self.summary(run_dir)

    def summary(self, run_dir: Path) -> HarnessSummary:
        manifest = self._read_manifest(run_dir)
        problem_key = str(manifest["problem"])
        source_path = Path(str(manifest["source_path"]))
        stages = manifest.get("stages", [])
        stage_payloads = [stage for stage in stages if isinstance(stage, dict)]
        return HarnessSummary(
            run_dir=run_dir,
            problem=problem_key,
            source_path=source_path,
            stages=stage_payloads,
        )

    def latest_run_dir(self, problem_key: str) -> Path | None:
        root = self._runs_root(problem_key)
        if not root.exists():
            return None
        candidates = [path for path in root.iterdir() if path.is_dir()]
        if not candidates:
            return None
        return sorted(candidates)[-1]

    def _runs_root(self, problem_key: str) -> Path:
        return self.config.workspace.root / "harness_runs" / problem_key

    def _manifest_path(self, run_dir: Path) -> Path:
        return run_dir / "manifest.json"

    def _read_manifest(self, run_dir: Path) -> dict[str, object]:
        return json.loads(self._manifest_path(run_dir).read_text(encoding="utf-8"))

    def _write_manifest(self, run_dir: Path, manifest: dict[str, object]) -> None:
        self._manifest_path(run_dir).write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _record_stage_result(
        self,
        stage: dict[str, object],
        stage_dir: Path,
        result: EvaluationResult,
    ) -> None:
        metrics = result.metrics
        stage.update(
            {
                "status": result.status,
                "return_code": result.return_code,
                "objective": result.objective,
                "workflow_url": metrics.get("workflow_url"),
                "failure_kind": metrics.get("failure_kind"),
                "failure_signature": metrics.get("failure_signature"),
                "mismatch_count": metrics.get("mismatch_count"),
                "wall_time_seconds": metrics.get("wall_time_seconds"),
                "result_path": str(stage_dir / "result.txt"),
                "stdout_path": str(stage_dir / "stdout.txt"),
                "stderr_path": str(stage_dir / "stderr.txt"),
                "parsed_metrics_path": str(stage_dir / "parsed_metrics.json"),
            }
        )


def _slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in text.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.strip("-")
    return cleaned or "run"
