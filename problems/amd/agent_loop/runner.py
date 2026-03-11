from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from difflib import unified_diff
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
RESULTS_TSV_HEADER = [
    "experiment",
    "tag",
    "problem",
    "parent_candidate_id",
    "candidate_id",
    "decision",
    "status",
    "objective_ns",
    "objective_us",
    "delta_ns_vs_parent",
    "workflow_url",
    "variant_name",
    "policy_profile",
    "added_lines",
    "deleted_lines",
    "lines_changed",
    "description",
]


@dataclass(frozen=True)
class LoopSummary:
    problem: str
    candidate_id: str
    mode: str
    status: str
    objective: float | None
    improved: bool
    promoted: bool
    variant_family: str | None = None
    variant_name: str | None = None
    policy_profile_name: str | None = None


class ClosedLoopRunner:
    def __init__(self, config: AppConfig):
        self.config = config
        self.store = ExperimentStore(config.workspace.root)

    def close(self) -> None:
        self.store.close()

    def ensure_baseline(self, problem_key: str) -> str:
        problem = self.config.require_problem(problem_key)
        current_source = problem.submission_path.read_text(encoding="utf-8")
        current_hash = sha256_text(current_source)
        existing = self.store.latest_candidate_by_hash(problem_key, current_hash)
        if existing is not None:
            return existing.candidate_id

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

    def reset_problem_workspace(self, problem_key: str) -> dict[str, object]:
        self.config.require_problem(problem_key)
        problem_dir = self._problem_dir(problem_key)
        removed: list[str] = []
        for name in ["candidates", "pruned", "working", "knowledge.json"]:
            target = problem_dir / name
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
                removed.append(name)
            elif target.is_file():
                target.unlink(missing_ok=True)
                removed.append(name)
        self.store.reset_problem(problem_key)
        return {
            "problem": problem_key,
            "removed": removed,
            "reset": True,
        }

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
        if self._should_prune_candidate(problem_key, candidate, result):
            self._prune_candidate_artifacts(problem_key, candidate, result, critique)
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
        problem = self.config.require_problem(problem_key)
        selected_mode = mode or problem.mode
        candidate = self._require_candidate(candidate_id)
        candidate_dir = Path(candidate.source_path).parent
        candidate_meta = self._load_candidate_meta(Path(self._require_candidate(candidate_id).source_path)) or {}
        variant = candidate_meta.get("variant")
        variant_family = variant.get("family") if isinstance(variant, dict) else None
        variant_name = variant.get("variant_name") if isinstance(variant, dict) else None
        policy_profile = candidate_meta.get("policy_profile")
        policy_profile_name = policy_profile.get("name") if isinstance(policy_profile, dict) else None
        improved = False
        promoted = False
        if result.status == "ok" and result.objective is not None:
            best = self.store.get_best_candidate(problem_key)
            improved = best is not None and best.candidate_id == candidate_id
            if improved and self.config.workspace.promote_on_improve:
                self.promote_candidate(problem_key, candidate_id)
                promoted = True
        self._sync_working_snapshot(problem_key, candidate_id, candidate_dir)
        self._ensure_baseline_logged(problem_key, candidate, result, candidate_meta)
        return LoopSummary(
            problem=problem_key,
            candidate_id=candidate_id,
            mode=selected_mode,
            status=result.status,
            objective=result.objective,
            improved=improved,
            promoted=promoted,
            variant_family=variant_family if isinstance(variant_family, str) else None,
            variant_name=variant_name if isinstance(variant_name, str) else None,
            policy_profile_name=policy_profile_name if isinstance(policy_profile_name, str) else None,
        )

    def bootstrap_problem(self, problem_key: str, mode: str | None = None) -> LoopSummary:
        problem = self.config.require_problem(problem_key)
        selected_mode = mode or problem.mode
        candidate_id = self.ensure_baseline(problem_key)
        cached_eval = self.store.latest_evaluation_for_candidate(candidate_id, selected_mode)
        if cached_eval is None:
            return self.run_baseline(problem_key, mode=selected_mode)

        candidate = self._require_candidate(candidate_id)
        if cached_eval.status == "ok" and cached_eval.objective is not None:
            best = self.store.get_best_candidate(problem_key)
            if best is None and self._is_improvement(problem, None, cached_eval.objective):
                self.store.set_best(
                    problem_key=problem_key,
                    candidate_id=candidate_id,
                    objective=cached_eval.objective,
                )

        candidate_meta = self._load_candidate_meta(Path(candidate.source_path)) or {}
        variant = candidate_meta.get("variant")
        variant_family = variant.get("family") if isinstance(variant, dict) else None
        variant_name = variant.get("variant_name") if isinstance(variant, dict) else None
        policy_profile = candidate_meta.get("policy_profile")
        policy_profile_name = policy_profile.get("name") if isinstance(policy_profile, dict) else None
        best = self.store.get_best_candidate(problem_key)
        improved = (
            cached_eval.status == "ok"
            and cached_eval.objective is not None
            and best is not None
            and best.candidate_id == candidate_id
        )
        candidate_dir = Path(candidate.source_path).parent
        self._sync_working_snapshot(problem_key, candidate_id, candidate_dir)
        cached_result = self._load_cached_evaluation(candidate)
        if cached_result is not None:
            self._ensure_baseline_logged(problem_key, candidate, cached_result, candidate_meta)
        return LoopSummary(
            problem=problem_key,
            candidate_id=candidate_id,
            mode=selected_mode,
            status="cached_baseline",
            objective=cached_eval.objective,
            improved=improved,
            promoted=False,
            variant_family=variant_family if isinstance(variant_family, str) else None,
            variant_name=variant_name if isinstance(variant_name, str) else None,
            policy_profile_name=policy_profile_name if isinstance(policy_profile_name, str) else None,
        )

    def run_iteration(
        self,
        problem_key: str,
        hypothesis: str | None = None,
        mutator_command: str | None = None,
        mode: str | None = None,
        desired_family: str | None = None,
    ) -> LoopSummary:
        problem = self.config.require_problem(problem_key)
        selected_mode = mode or problem.mode
        parent = self.store.get_best_candidate(problem_key)
        if parent is None:
            baseline_summary = self.run_baseline(problem_key, mode=selected_mode)
            parent = self._require_candidate(baseline_summary.candidate_id)
        parent = self._select_parent_candidate(problem_key, fallback=parent)
        parent_dir = Path(parent.source_path).parent

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
                "mode": selected_mode,
                "objective": problem.objective,
            },
            "edit_budget": {
                "max_changed_lines": problem.max_changed_lines,
                "max_edit_hunks": problem.max_edit_hunks,
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
            "desired_family": desired_family,
            "working_submission_path": str(self._working_dir(problem_key) / "submission.py"),
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

        scope_check = self._check_candidate_scope(
            problem=problem,
            parent_path=Path(parent.source_path),
            candidate_path=output_path,
            candidate_dir=candidate_dir,
        )
        self._sync_attempt_snapshot(problem_key, candidate_id, candidate_dir)

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
        candidate_meta = self._load_candidate_meta(output_path) or {}
        variant = candidate_meta.get("variant")
        variant_family = variant.get("family") if isinstance(variant, dict) else None
        variant_name = variant.get("variant_name") if isinstance(variant, dict) else None
        policy_profile = candidate_meta.get("policy_profile")
        policy_profile_name = policy_profile.get("name") if isinstance(policy_profile, dict) else None

        if not scope_check["passed"]:
            return self._record_scope_reject(
                problem_key=problem_key,
                selected_mode=selected_mode,
                parent=parent,
                candidate=self._require_candidate(candidate_id),
                candidate_dir=candidate_dir,
                hypothesis=hypothesis_text,
                candidate_meta=candidate_meta,
                scope_check=scope_check,
                variant_family=variant_family if isinstance(variant_family, str) else None,
                variant_name=variant_name if isinstance(variant_name, str) else None,
                policy_profile_name=policy_profile_name if isinstance(policy_profile_name, str) else None,
            )

        result = self.evaluate_candidate(problem_key, candidate_id, mode=selected_mode)
        self._sync_attempt_snapshot(problem_key, candidate_id, candidate_dir)
        improved = False
        promoted = False
        decision = self._decide_keep_or_revert(problem_key, parent, result)
        if decision == "keep":
            improved = True
            if self.config.workspace.promote_on_improve:
                self.promote_candidate(problem_key, candidate_id)
                promoted = True
            self._sync_working_snapshot(problem_key, candidate_id, candidate_dir)
        else:
            self._sync_working_snapshot(problem_key, parent.candidate_id, parent_dir)
        self._append_experiment_log(
            problem_key=problem_key,
            parent=parent,
            candidate=self._require_candidate(candidate_id),
            result=result,
            decision=decision,
            hypothesis=hypothesis_text,
            candidate_meta=candidate_meta,
        )

        return LoopSummary(
            problem=problem_key,
            candidate_id=candidate_id,
            mode=selected_mode,
            status=result.status,
            objective=result.objective,
            improved=improved,
            promoted=promoted,
            variant_family=variant_family if isinstance(variant_family, str) else None,
            variant_name=variant_name if isinstance(variant_name, str) else None,
            policy_profile_name=policy_profile_name if isinstance(policy_profile_name, str) else None,
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

    def compact_failed_candidates(
        self,
        problem_key: str | None = None,
        *,
        stale_pending_hours: float | None = None,
    ) -> dict[str, int]:
        scanned = 0
        pruned = 0
        pruned_stale_pending = 0
        skipped_missing = 0
        stale_cutoff = None
        if stale_pending_hours is not None and stale_pending_hours > 0:
            stale_cutoff = datetime.now(UTC).timestamp() - (stale_pending_hours * 3600.0)
        for candidate in self.store.all_candidates(problem_key):
            scanned += 1
            if candidate.kind != "mutation":
                continue
            candidate_dir = Path(candidate.source_path).parent
            if not candidate_dir.exists():
                skipped_missing += 1
                continue
            if candidate.status is None:
                if stale_cutoff is None:
                    continue
                created = self._parse_created_at(candidate.created_at)
                if created is None or created.timestamp() >= stale_cutoff:
                    continue
                if candidate.promoted:
                    continue
                best = self.store.get_best_candidate(candidate.problem_key)
                if best is not None and best.candidate_id == candidate.candidate_id:
                    continue
                critique = {
                    "failure_kind": "stale_pending",
                    "summary": (
                        f"candidate never reached evaluation and was pruned after exceeding "
                        f"the {stale_pending_hours:g}h stale threshold"
                    ),
                    "policy_signal": "submission_repair",
                    "next_action": "retry with the updated mutator and current problem memory",
                }
                self._prune_candidate_artifacts(
                    candidate.problem_key,
                    candidate,
                    EvaluationResult(
                        command=[],
                        return_code=0,
                        status="stale_pending",
                        objective=None,
                        metrics={},
                        stdout="",
                        stderr="",
                        result_text="",
                    ),
                    critique,
                )
                pruned += 1
                pruned_stale_pending += 1
                continue
            if candidate.status == "ok":
                continue
            evaluation = self._load_cached_evaluation(candidate)
            if evaluation is None:
                skipped_missing += 1
                continue
            critique = load_critique(candidate_dir / "evaluation" / "critique.json")
            if critique is None:
                critique = asdict(build_critique(candidate.problem_key, candidate.candidate_id, evaluation))
            self._prune_candidate_artifacts(candidate.problem_key, candidate, evaluation, critique)
            pruned += 1
        if problem_key is not None:
            self._write_problem_memory(problem_key)
        else:
            for key in sorted(self.config.problems):
                self._write_problem_memory(key)
        return {
            "scanned": scanned,
            "pruned": pruned,
            "pruned_stale_pending": pruned_stale_pending,
            "skipped_missing": skipped_missing,
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

    def _decide_keep_or_revert(self, problem_key: str, parent, result: EvaluationResult) -> str:
        problem = self.config.require_problem(problem_key)
        if not problem.keep_revert:
            best = self.store.get_best_candidate(problem_key)
            if best is not None and best.candidate_id == parent.candidate_id:
                if result.status == "ok" and result.objective is not None:
                    return "keep" if self._is_improvement(problem, parent.score, result.objective) else "revert"
            return "keep" if result.status == "ok" and result.objective is not None else "revert"
        if result.status != "ok" or result.objective is None:
            return "revert"
        return "keep" if self._is_improvement(problem, parent.score, result.objective) else "revert"

    def _select_parent_candidate(self, problem_key: str, fallback):
        problem = self.config.require_problem(problem_key)
        if problem.keep_revert:
            working = self._current_working_candidate(problem_key)
            if working is not None and Path(working.source_path).exists():
                return working
        if problem.parent_strategy == "latest":
            latest = self.store.latest_candidate(problem_key)
            if latest is not None and Path(latest.source_path).exists():
                return latest
        return fallback

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
            critique = load_critique(critique_path)
            pruned_summary = None
            if meta is None or critique is None:
                pruned_summary = self._load_pruned_candidate_summary(problem_key, row.candidate_id)
            if meta is None and isinstance(pruned_summary, dict):
                pruned_meta = pruned_summary.get("meta")
                if isinstance(pruned_meta, dict):
                    meta = pruned_meta
            if meta is not None:
                entry["meta"] = meta
            if critique is None and isinstance(pruned_summary, dict):
                pruned_critique = pruned_summary.get("critique")
                if isinstance(pruned_critique, dict):
                    critique = pruned_critique
            if critique is not None:
                entry["critique"] = critique
            if isinstance(pruned_summary, dict):
                entry["pruned"] = True
            history.append(entry)
        return history

    def _problem_dir(self, problem_key: str) -> Path:
        path = self.store.root / "problems" / problem_key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _working_dir(self, problem_key: str) -> Path:
        path = self._problem_dir(problem_key) / "working"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _attempt_dir(self, problem_key: str) -> Path:
        path = self._working_dir(problem_key) / "attempt"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _results_tsv_path(self, problem_key: str) -> Path:
        return self._working_dir(problem_key) / "results.tsv"

    def _journal_jsonl_path(self, problem_key: str) -> Path:
        return self._working_dir(problem_key) / "journal.jsonl"

    def _state_json_path(self, problem_key: str) -> Path:
        return self._working_dir(problem_key) / "state.json"

    def _current_working_candidate(self, problem_key: str):
        path = self._working_dir(problem_key) / "current_candidate.txt"
        if not path.exists():
            return None
        candidate_id = path.read_text(encoding="utf-8").strip()
        if not candidate_id:
            return None
        candidate = self.store.get_candidate(candidate_id)
        if candidate is None:
            return None
        return candidate

    def _ensure_results_header(self, problem_key: str) -> None:
        path = self._results_tsv_path(problem_key)
        if path.exists():
            return
        path.write_text("\t".join(RESULTS_TSV_HEADER) + "\n", encoding="utf-8")

    def _next_experiment_number(self, problem_key: str) -> int:
        path = self._journal_jsonl_path(problem_key)
        if not path.exists():
            return 0
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count

    def _diff_stats(self, parent_path: Path, candidate_path: Path) -> tuple[int, int, int]:
        return self._diff_stats_from_lines(self._diff_lines(parent_path, candidate_path))

    def _diff_lines(self, parent_path: Path, candidate_path: Path) -> list[str]:
        if not parent_path.exists() or not candidate_path.exists():
            return []
        return list(
            unified_diff(
                parent_path.read_text(encoding="utf-8").splitlines(),
                candidate_path.read_text(encoding="utf-8").splitlines(),
                fromfile=str(parent_path),
                tofile=str(candidate_path),
                lineterm="",
            )
        )

    def _diff_stats_from_lines(self, diff_lines: list[str]) -> tuple[int, int, int]:
        if not diff_lines:
            return (0, 0, 0)
        added = 0
        deleted = 0
        for line in diff_lines:
            if line.startswith(("+++", "---", "@@")):
                continue
            if line.startswith("+"):
                added += 1
            elif line.startswith("-"):
                deleted += 1
        return added, deleted, added + deleted

    def _diff_hunk_count(self, diff_lines: list[str]) -> int:
        return sum(1 for line in diff_lines if line.startswith("@@"))

    def _check_candidate_scope(
        self,
        *,
        problem: ProblemConfig,
        parent_path: Path,
        candidate_path: Path,
        candidate_dir: Path,
    ) -> dict[str, object]:
        diff_lines = self._diff_lines(parent_path, candidate_path)
        diff_text = "\n".join(diff_lines).rstrip()
        (candidate_dir / "candidate.diff").write_text(
            f"{diff_text}\n" if diff_text else "",
            encoding="utf-8",
        )
        added, deleted, changed = self._diff_stats_from_lines(diff_lines)
        hunk_count = self._diff_hunk_count(diff_lines)
        violations: list[str] = []
        if problem.max_changed_lines is not None and changed > problem.max_changed_lines:
            violations.append(
                f"changed lines {changed} exceed budget {problem.max_changed_lines}"
            )
        if problem.max_edit_hunks is not None and hunk_count > problem.max_edit_hunks:
            violations.append(
                f"edit hunks {hunk_count} exceed budget {problem.max_edit_hunks}"
            )
        payload = {
            "problem": problem.key,
            "passed": not violations,
            "added_lines": added,
            "deleted_lines": deleted,
            "lines_changed": changed,
            "edit_hunks": hunk_count,
            "max_changed_lines": problem.max_changed_lines,
            "max_edit_hunks": problem.max_edit_hunks,
            "violations": violations,
        }
        (candidate_dir / "scope_check.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return payload

    def _workflow_url_from_result(self, result: EvaluationResult) -> str | None:
        workflow_url = result.metrics.get("workflow_url")
        return workflow_url if isinstance(workflow_url, str) and workflow_url else None

    def _update_working_state(
        self,
        problem_key: str,
        *,
        current_candidate_id: str,
        decision: str,
    ) -> None:
        journal_path = self._journal_jsonl_path(problem_key)
        experiments = 0
        kept = 0
        reverted = 0
        if journal_path.exists():
            for line in journal_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                experiments += 1
                entry = json.loads(line)
                if entry.get("decision") == "keep":
                    kept += 1
                elif entry.get("decision") == "revert":
                    reverted += 1
        candidate = self.store.get_candidate(current_candidate_id)
        payload = {
            "problem": problem_key,
            "current_candidate_id": current_candidate_id,
            "current_objective": candidate.score if candidate is not None else None,
            "experiments": experiments,
            "kept": kept,
            "reverted": reverted,
            "last_decision": decision,
        }
        self._state_json_path(problem_key).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _ensure_baseline_logged(
        self,
        problem_key: str,
        candidate,
        result: EvaluationResult,
        candidate_meta: dict[str, object],
    ) -> None:
        if self._journal_jsonl_path(problem_key).exists():
            return
        self._append_experiment_log(
            problem_key=problem_key,
            parent=None,
            candidate=candidate,
            result=result,
            decision="keep",
            hypothesis="baseline",
            candidate_meta=candidate_meta,
            tag="baseline",
        )

    def _append_experiment_log(
        self,
        *,
        problem_key: str,
        parent,
        candidate,
        result: EvaluationResult,
        decision: str,
        hypothesis: str,
        candidate_meta: dict[str, object],
        tag: str | None = None,
    ) -> None:
        self._ensure_results_header(problem_key)
        experiment = self._next_experiment_number(problem_key)
        variant = candidate_meta.get("variant") if isinstance(candidate_meta, dict) else None
        variant_name = variant.get("variant_name") if isinstance(variant, dict) else ""
        policy_profile = candidate_meta.get("policy_profile") if isinstance(candidate_meta, dict) else None
        policy_name = policy_profile.get("name") if isinstance(policy_profile, dict) else ""
        parent_path = Path(parent.source_path) if parent is not None else Path(candidate.source_path)
        added, deleted, changed = self._diff_stats(parent_path, Path(candidate.source_path))
        delta_ns = None
        if parent is not None and isinstance(parent.score, (int, float)) and isinstance(result.objective, (int, float)):
            delta_ns = float(result.objective) - float(parent.score)
        row = [
            str(experiment),
            tag or decision,
            problem_key,
            parent.candidate_id if parent is not None else "",
            candidate.candidate_id,
            decision,
            result.status,
            "" if result.objective is None else f"{float(result.objective):.6f}",
            "" if result.objective is None else f"{float(result.objective) / 1_000.0:.6f}",
            "" if delta_ns is None else f"{delta_ns:.6f}",
            self._workflow_url_from_result(result) or "",
            str(variant_name or ""),
            str(policy_name or ""),
            str(added),
            str(deleted),
            str(changed),
            hypothesis.replace("\t", " ").replace("\n", " "),
        ]
        with self._results_tsv_path(problem_key).open("a", encoding="utf-8") as handle:
            handle.write("\t".join(row) + "\n")
        payload = {
            "experiment": experiment,
            "problem": problem_key,
            "parent_candidate_id": parent.candidate_id if parent is not None else None,
            "candidate_id": candidate.candidate_id,
            "decision": decision,
            "status": result.status,
            "objective_ns": result.objective,
            "objective_us": None if result.objective is None else float(result.objective) / 1_000.0,
            "delta_ns_vs_parent": delta_ns,
            "workflow_url": self._workflow_url_from_result(result),
            "variant_name": variant_name,
            "policy_profile": policy_name,
            "added_lines": added,
            "deleted_lines": deleted,
            "lines_changed": changed,
            "description": hypothesis,
            "created_at": datetime.now(UTC).isoformat(),
        }
        with self._journal_jsonl_path(problem_key).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        kept_candidate_id = candidate.candidate_id if decision == "keep" else (parent.candidate_id if parent is not None else candidate.candidate_id)
        self._update_working_state(problem_key, current_candidate_id=kept_candidate_id, decision=decision)

    def _record_scope_reject(
        self,
        *,
        problem_key: str,
        selected_mode: str,
        parent,
        candidate,
        candidate_dir: Path,
        hypothesis: str,
        candidate_meta: dict[str, object],
        scope_check: dict[str, object],
        variant_family: str | None,
        variant_name: str | None,
        policy_profile_name: str | None,
    ) -> LoopSummary:
        evaluation_dir = candidate_dir / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        violation_text = "; ".join(
            str(item) for item in scope_check.get("violations", []) if str(item)
        ) or "candidate exceeded the focused-edit budget"
        result_text = (
            "AutoKernel-style scope check rejected this candidate before remote submission.\n"
            f"{violation_text}\n"
        )
        metrics = {
            "failure_kind": "scope_budget_exceeded",
            "failure_signature": violation_text,
            "scope_check": scope_check,
        }
        stdout_path = evaluation_dir / "stdout.txt"
        stderr_path = evaluation_dir / "stderr.txt"
        result_path = evaluation_dir / "result.txt"
        parsed_metrics_path = evaluation_dir / "parsed_metrics.json"
        critique_path = evaluation_dir / "critique.json"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(result_text, encoding="utf-8")
        result_path.write_text(result_text, encoding="utf-8")
        parsed_metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = EvaluationResult(
            command=[],
            return_code=1,
            status="scope_reject",
            objective=None,
            metrics=metrics,
            stdout="",
            stderr=result_text,
            result_text=result_text,
        )
        critique = build_critique(problem_key, candidate.candidate_id, result)
        write_critique(critique_path, critique)
        self.store.record_evaluation(
            candidate_id=candidate.candidate_id,
            mode=selected_mode,
            return_code=1,
            status="scope_reject",
            objective=None,
            metrics=metrics,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            result_path=result_path,
            command=[],
        )
        self._write_problem_memory(problem_key)
        self._sync_attempt_snapshot(problem_key, candidate.candidate_id, candidate_dir)
        self._sync_working_snapshot(problem_key, parent.candidate_id, Path(parent.source_path).parent)
        self._append_experiment_log(
            problem_key=problem_key,
            parent=parent,
            candidate=candidate,
            result=result,
            decision="revert",
            hypothesis=hypothesis,
            candidate_meta=candidate_meta,
        )
        return LoopSummary(
            problem=problem_key,
            candidate_id=candidate.candidate_id,
            mode=selected_mode,
            status="scope_reject",
            objective=None,
            improved=False,
            promoted=False,
            variant_family=variant_family,
            variant_name=variant_name,
            policy_profile_name=policy_profile_name,
        )

    def _sync_attempt_snapshot(self, problem_key: str, candidate_id: str, candidate_dir: Path) -> None:
        attempt_dir = self._attempt_dir(problem_key)
        if attempt_dir.exists():
            shutil.rmtree(attempt_dir, ignore_errors=True)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        snapshot_files = [
            "submission.py",
            "context.json",
            "candidate.diff",
            "prompt.system.txt",
            "prompt.user.txt",
            "scope_check.json",
            "mutator.command.txt",
            "mutator.stdout.txt",
            "mutator.stderr.txt",
            "evaluation/result.txt",
            "evaluation/parsed_metrics.json",
            "evaluation/critique.json",
            "evaluation/stdout.txt",
            "evaluation/stderr.txt",
        ]
        for relative_name in snapshot_files:
            source = candidate_dir / relative_name
            target = attempt_dir / relative_name
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        (self._working_dir(problem_key) / "attempt_candidate.txt").write_text(candidate_id, encoding="utf-8")

    def _sync_working_snapshot(self, problem_key: str, candidate_id: str, candidate_dir: Path) -> None:
        working_dir = self._working_dir(problem_key)
        evaluation_dir = working_dir / "evaluation"
        if evaluation_dir.exists():
            shutil.rmtree(evaluation_dir, ignore_errors=True)
        snapshot_files = [
            "submission.py",
            "evaluation/result.txt",
            "evaluation/parsed_metrics.json",
            "evaluation/critique.json",
            "evaluation/stdout.txt",
            "evaluation/stderr.txt",
        ]
        for relative_name in snapshot_files:
            source = candidate_dir / relative_name
            target = working_dir / relative_name
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        (working_dir / "current_candidate.txt").write_text(candidate_id, encoding="utf-8")

    def _write_problem_memory(self, problem_key: str) -> Path:
        history = self._candidate_history(problem_key, limit=20)
        status_counts: dict[str, int] = {}
        failure_counts: dict[str, int] = {}
        policy_signal_counts: dict[str, int] = {}
        policy_profile_counts: dict[str, int] = {}
        for entry in history:
            status = str(entry.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            meta = entry.get("meta")
            if isinstance(meta, dict):
                policy_profile = meta.get("policy_profile")
                if isinstance(policy_profile, dict):
                    profile_name = policy_profile.get("name")
                    if isinstance(profile_name, str) and profile_name:
                        policy_profile_counts[profile_name] = policy_profile_counts.get(profile_name, 0) + 1
            critique = entry.get("critique")
            if isinstance(critique, dict):
                failure_kind = critique.get("failure_kind")
                if isinstance(failure_kind, str) and failure_kind:
                    failure_counts[failure_kind] = failure_counts.get(failure_kind, 0) + 1
                policy_signal = critique.get("policy_signal")
                if isinstance(policy_signal, str) and policy_signal:
                    policy_signal_counts[policy_signal] = policy_signal_counts.get(policy_signal, 0) + 1

        payload = {
            "problem": problem_key,
            "status_counts": status_counts,
            "failure_counts": failure_counts,
            "policy_signal_counts": policy_signal_counts,
            "policy_profile_counts": policy_profile_counts,
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

    def _load_cached_evaluation(self, candidate) -> EvaluationResult | None:
        candidate_dir = Path(candidate.source_path).parent
        evaluation_dir = candidate_dir / "evaluation"
        metrics_path = evaluation_dir / "parsed_metrics.json"
        result_path = evaluation_dir / "result.txt"
        stdout_path = evaluation_dir / "stdout.txt"
        stderr_path = evaluation_dir / "stderr.txt"
        if not metrics_path.exists():
            return None
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        result_text = result_path.read_text(encoding="utf-8") if result_path.exists() else ""
        stdout = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
        stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
        objective = candidate.score if isinstance(candidate.score, (int, float)) else None
        return EvaluationResult(
            command=[],
            return_code=0,
            status=str(candidate.status),
            objective=objective,
            metrics=metrics,
            stdout=stdout,
            stderr=stderr,
            result_text=result_text,
        )

    def _parse_created_at(self, raw: str | None) -> datetime | None:
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    def _should_prune_candidate(
        self,
        problem_key: str,
        candidate,
        result: EvaluationResult,
    ) -> bool:
        if not self.config.require_problem(problem_key).prune_failures:
            return False
        if candidate.kind != "mutation":
            return False
        if result.status == "ok":
            return False
        if candidate.promoted:
            return False
        best = self.store.get_best_candidate(problem_key)
        if best is not None and best.candidate_id == candidate.candidate_id:
            return False
        return True

    def _pruned_summary_path(self, problem_key: str, candidate_id: str) -> Path:
        path = self._problem_dir(problem_key) / "pruned" / f"{candidate_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _load_pruned_candidate_summary(
        self,
        problem_key: str,
        candidate_id: str,
    ) -> dict[str, object] | None:
        path = self._pruned_summary_path(problem_key, candidate_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _prune_candidate_artifacts(
        self,
        problem_key: str,
        candidate,
        result: EvaluationResult,
        critique,
    ) -> None:
        source_path = Path(candidate.source_path)
        candidate_dir = source_path.parent
        meta = self._load_candidate_meta(source_path) or {}
        critique_payload = critique if isinstance(critique, dict) else asdict(critique)
        summary = {
            "candidate_id": candidate.candidate_id,
            "problem_key": problem_key,
            "parent_id": candidate.parent_id,
            "kind": candidate.kind,
            "hypothesis": candidate.hypothesis,
            "source_sha256": candidate.source_sha256,
            "status": result.status,
            "objective": result.objective,
            "meta": meta,
            "critique": critique_payload,
            "metrics": result.metrics,
            "pruned_at": datetime.now(UTC).isoformat(),
        }
        self._pruned_summary_path(problem_key, candidate.candidate_id).write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        shutil.rmtree(candidate_dir, ignore_errors=True)
