from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
import shlex
import shutil
import subprocess

from .config import AppConfig, ProblemConfig
from .critic import build_critique, load_critique, write_critique
from .evaluator import EvaluationResult, run_popcorn_submission
from .storage import ExperimentStore, new_candidate_id, sha256_text


META_RE = re.compile(r"^# AGENT_LOOP_META:\s*(\{.*\})\s*$", re.MULTILINE)


@dataclass(frozen=True)
class LoopSummary:
    problem: str
    candidate_id: str
    status: str
    objective: float | None
    improved: bool
    promoted: bool


class ClosedLoopRunner:
    def __init__(self, config: AppConfig):
        self.config = config
        self.store = ExperimentStore(config.workspace.root)

    def close(self) -> None:
        self.store.close()

    def ensure_baseline(self, problem_key: str) -> str:
        problem = self.config.require_problem(problem_key)
        latest = self.store.latest_candidate(problem_key)
        current_source = problem.submission_path.read_text(encoding="utf-8")
        current_hash = sha256_text(current_source)
        if latest and latest.source_sha256 == current_hash:
            return latest.candidate_id

        candidate_id = new_candidate_id()
        candidate_dir = self.store.candidate_dir(problem_key, candidate_id)
        source_path = candidate_dir / "submission.py"
        shutil.copy2(problem.submission_path, source_path)
        self.store.add_candidate(
            candidate_id=candidate_id,
            problem_key=problem_key,
            parent_id=None,
            kind="baseline",
            hypothesis="current repository submission",
            source_path=source_path,
            source_sha256=current_hash,
        )
        return candidate_id

    def evaluate_candidate(self, problem_key: str, candidate_id: str, mode: str | None = None) -> EvaluationResult:
        problem = self.config.require_problem(problem_key)
        candidate = self._require_candidate(candidate_id)
        artifacts_dir = self.store.candidate_dir(problem_key, candidate_id) / "evaluation"
        result = run_popcorn_submission(
            workspace=self.config.workspace,
            problem=problem,
            submission_path=Path(candidate.source_path),
            artifacts_dir=artifacts_dir,
            mode=mode,
        )
        self.store.record_evaluation(
            candidate_id=candidate_id,
            mode=mode or problem.mode,
            return_code=result.return_code,
            status=result.status,
            objective=result.objective,
            metrics=result.metrics,
            stdout_path=artifacts_dir / "stdout.txt",
            stderr_path=artifacts_dir / "stderr.txt",
            result_path=artifacts_dir / "result.txt",
            command=result.command,
        )
        critique = build_critique(problem_key, candidate_id, result)
        write_critique(artifacts_dir / "critique.json", critique)
        self._write_problem_memory(problem_key)

        if result.status == "ok" and result.objective is not None:
            best = self.store.get_best_candidate(problem_key)
            if self._is_improvement(problem, best.score if best else None, result.objective):
                self.store.set_best(
                    problem_key=problem_key,
                    candidate_id=candidate_id,
                    objective=result.objective,
                )

        return result

    def run_baseline(self, problem_key: str, mode: str | None = None) -> LoopSummary:
        candidate_id = self.ensure_baseline(problem_key)
        result = self.evaluate_candidate(problem_key, candidate_id, mode=mode)
        improved = False
        promoted = False
        if result.status == "ok" and result.objective is not None:
            best = self.store.get_best_candidate(problem_key)
            improved = best is not None and best.candidate_id == candidate_id
            if improved and self.config.workspace.promote_on_improve:
                self.promote_candidate(problem_key, candidate_id)
                promoted = True
        return LoopSummary(
            problem=problem_key,
            candidate_id=candidate_id,
            status=result.status,
            objective=result.objective,
            improved=improved,
            promoted=promoted,
        )

    def run_iteration(
        self,
        problem_key: str,
        hypothesis: str | None = None,
        mutator_command: str | None = None,
    ) -> LoopSummary:
        problem = self.config.require_problem(problem_key)
        parent = self.store.get_best_candidate(problem_key)
        if parent is None:
            parent = self._require_candidate(self.ensure_baseline(problem_key))

        candidate_id = new_candidate_id()
        candidate_dir = self.store.candidate_dir(problem_key, candidate_id)
        output_path = candidate_dir / "submission.py"
        context_path = candidate_dir / "context.json"
        memory_path = self._write_problem_memory(problem_key)

        hypothesis_text = hypothesis or "llm/generated mutation"
        context = {
            "problem": {
                "key": problem.key,
                "leaderboard": problem.leaderboard,
                "gpu": problem.gpu,
                "mode": problem.mode,
                "objective": problem.objective,
            },
            "parent": {
                "candidate_id": parent.candidate_id,
                "source_path": parent.source_path,
                "score": parent.score,
                "status": parent.status,
            },
            "repo_submission_path": str(problem.submission_path),
            "workspace_root": str(self.config.workspace.root),
            "history": self._candidate_history(problem_key, limit=12),
            "knowledge_path": str(memory_path),
        }
        context_path.write_text(json.dumps(context, indent=2, sort_keys=True), encoding="utf-8")

        command = mutator_command or problem.mutator_command
        if command:
            self._run_mutator(
                problem,
                command,
                parent.source_path,
                output_path,
                context_path,
                candidate_dir,
            )
        else:
            raise RuntimeError(
                f"problem '{problem.key}' does not define mutator_command in the config"
            )

        source_text = output_path.read_text(encoding="utf-8")
        self.store.add_candidate(
            candidate_id=candidate_id,
            problem_key=problem_key,
            parent_id=parent.candidate_id,
            kind="mutation",
            hypothesis=hypothesis_text,
            source_path=output_path,
            source_sha256=sha256_text(source_text),
        )

        result = self.evaluate_candidate(problem_key, candidate_id)
        improved = False
        promoted = False
        best = self.store.get_best_candidate(problem_key)
        if best is not None and best.candidate_id == candidate_id and result.status == "ok":
            improved = True
            if self.config.workspace.promote_on_improve:
                self.promote_candidate(problem_key, candidate_id)
                promoted = True

        return LoopSummary(
            problem=problem_key,
            candidate_id=candidate_id,
            status=result.status,
            objective=result.objective,
            improved=improved,
            promoted=promoted,
        )

    def promote_candidate(self, problem_key: str, candidate_id: str) -> None:
        problem = self.config.require_problem(problem_key)
        candidate = self._require_candidate(candidate_id)
        shutil.copy2(candidate.source_path, problem.submission_path)
        self.store.mark_promoted(problem_key=problem_key, candidate_id=candidate_id)

    def problem_snapshot(self, problem_key: str) -> dict[str, object]:
        problem = self.config.require_problem(problem_key)
        best = self.store.get_best_candidate(problem_key)
        state = self.store.get_problem_state(problem_key)
        return {
            "problem": problem.key,
            "submission_path": str(problem.submission_path),
            "leaderboard": problem.leaderboard,
            "best_candidate_id": best.candidate_id if best else None,
            "best_objective": best.score if best else None,
            "promoted_candidate_id": state.promoted_candidate_id if state else None,
            "knowledge_path": str(self._problem_dir(problem_key) / "knowledge.json"),
            "recent_candidates": [asdict(row) for row in self.store.recent_candidates(problem_key)],
        }

    def _is_improvement(
        self,
        problem: ProblemConfig,
        current_best: float | None,
        candidate_score: float,
    ) -> bool:
        if current_best is None:
            return True
        eps = self.config.workspace.improvement_epsilon
        if problem.lower_is_better:
            return candidate_score < (current_best - eps)
        return candidate_score > (current_best + eps)

    def _run_mutator(
        self,
        problem: ProblemConfig,
        command_template: str,
        parent_path: str,
        output_path: Path,
        context_path: Path,
        candidate_dir: Path,
    ) -> None:
        placeholders = {
            "problem": problem.key,
            "parent": shlex.quote(parent_path),
            "output": shlex.quote(str(output_path)),
            "context": shlex.quote(str(context_path)),
            "candidate_dir": shlex.quote(str(candidate_dir)),
            "repo_root": shlex.quote(str(self.config.repo_root)),
            "submission": shlex.quote(str(problem.submission_path)),
        }
        command = command_template.format(**placeholders)
        completed = subprocess.run(
            command,
            cwd=str(self.config.repo_root),
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        (candidate_dir / "mutator.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (candidate_dir / "mutator.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        (candidate_dir / "mutator.command.txt").write_text(command, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"mutator command failed for problem '{problem.key}' with rc={completed.returncode}"
            )
        if not output_path.exists():
            raise RuntimeError(
                f"mutator command completed but did not create {output_path}"
            )

    def _require_candidate(self, candidate_id: str):
        candidate = self.store.get_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"unknown candidate '{candidate_id}'")
        return candidate

    def _candidate_history(self, problem_key: str, limit: int) -> list[dict[str, object]]:
        history: list[dict[str, object]] = []
        for row in self.store.recent_candidates(problem_key, limit=limit):
            source_path = Path(row.source_path)
            critique_path = source_path.parent / "evaluation" / "critique.json"
            entry: dict[str, object] = {
                "candidate_id": row.candidate_id,
                "created_at": row.created_at,
                "hypothesis": row.hypothesis,
                "kind": row.kind,
                "parent_id": row.parent_id,
                "promoted": row.promoted,
                "score": row.score,
                "source_path": row.source_path,
                "status": row.status,
            }
            meta = self._load_candidate_meta(source_path)
            if meta is not None:
                entry["meta"] = meta
            critique = load_critique(critique_path)
            if critique is not None:
                entry["critique"] = critique
            history.append(entry)
        return history

    def _problem_dir(self, problem_key: str) -> Path:
        path = self.store.root / "problems" / problem_key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_problem_memory(self, problem_key: str) -> Path:
        history = self._candidate_history(problem_key, limit=20)
        status_counts: dict[str, int] = {}
        failure_counts: dict[str, int] = {}
        for entry in history:
            status = str(entry.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            critique = entry.get("critique")
            if isinstance(critique, dict):
                failure_kind = critique.get("failure_kind")
                if isinstance(failure_kind, str) and failure_kind:
                    failure_counts[failure_kind] = failure_counts.get(failure_kind, 0) + 1

        payload = {
            "problem": problem_key,
            "status_counts": status_counts,
            "failure_counts": failure_counts,
            "history": history,
        }
        path = self._problem_dir(problem_key) / "knowledge.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _load_candidate_meta(self, source_path: Path) -> dict[str, object] | None:
        if not source_path.exists():
            return None
        match = META_RE.search(source_path.read_text(encoding="utf-8"))
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
