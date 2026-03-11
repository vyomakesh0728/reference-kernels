from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
import json
import sqlite3
import uuid


TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"


def utc_now() -> str:
    return datetime.now(UTC).strftime(TIMESTAMP_FORMAT)


def new_candidate_id() -> str:
    return uuid.uuid4().hex[:12]


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateRow:
    candidate_id: str
    problem_key: str
    parent_id: str | None
    kind: str
    hypothesis: str
    source_path: str
    source_sha256: str
    promoted: bool
    created_at: str
    score: float | None
    status: str | None


@dataclass(frozen=True)
class ProblemState:
    problem_key: str
    best_candidate_id: str | None
    best_objective: float | None
    promoted_candidate_id: str | None
    updated_at: str


class ExperimentStore:
    def __init__(self, root: Path):
        self.root = root
        self.db_path = root / "state.sqlite3"
        self.root.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=60.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def candidate_dir(self, problem_key: str, candidate_id: str) -> Path:
        path = self.root / "problems" / problem_key / "candidates" / candidate_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def add_candidate(
        self,
        *,
        candidate_id: str,
        problem_key: str,
        parent_id: str | None,
        kind: str,
        hypothesis: str,
        source_path: Path,
        source_sha256: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO candidates (
                candidate_id, problem_key, parent_id, kind, hypothesis,
                source_path, source_sha256, promoted, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                candidate_id,
                problem_key,
                parent_id,
                kind,
                hypothesis,
                str(source_path),
                source_sha256,
                utc_now(),
            ),
        )
        self.conn.commit()

    def record_evaluation(
        self,
        *,
        candidate_id: str,
        mode: str,
        return_code: int,
        status: str,
        objective: float | None,
        metrics: dict[str, object],
        stdout_path: Path,
        stderr_path: Path,
        result_path: Path,
        command: list[str],
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO evaluations (
                candidate_id, mode, return_code, status, objective,
                metrics_json, stdout_path, stderr_path, result_path,
                command_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                mode,
                return_code,
                status,
                objective,
                json.dumps(metrics, sort_keys=True),
                str(stdout_path),
                str(stderr_path),
                str(result_path),
                json.dumps(command),
                utc_now(),
            ),
        )
        self.conn.execute(
            "UPDATE candidates SET score = ?, status = ? WHERE candidate_id = ?",
            (objective, status, candidate_id),
        )
        self.conn.commit()

    def set_best(
        self,
        *,
        problem_key: str,
        candidate_id: str,
        objective: float,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO problem_state (
                problem_key, best_candidate_id, best_objective,
                promoted_candidate_id, updated_at
            ) VALUES (?, ?, ?, NULL, ?)
            ON CONFLICT(problem_key) DO UPDATE SET
                best_candidate_id = excluded.best_candidate_id,
                best_objective = excluded.best_objective,
                updated_at = excluded.updated_at
            """,
            (problem_key, candidate_id, objective, utc_now()),
        )
        self.conn.commit()

    def mark_promoted(self, *, problem_key: str, candidate_id: str) -> None:
        self.conn.execute(
            "UPDATE candidates SET promoted = 1 WHERE candidate_id = ?",
            (candidate_id,),
        )
        self.conn.execute(
            """
            INSERT INTO problem_state (
                problem_key, best_candidate_id, best_objective,
                promoted_candidate_id, updated_at
            )
            VALUES (
                ?, NULL, NULL, ?, ?
            )
            ON CONFLICT(problem_key) DO UPDATE SET
                promoted_candidate_id = excluded.promoted_candidate_id,
                updated_at = excluded.updated_at
            """,
            (problem_key, candidate_id, utc_now()),
        )
        self.conn.commit()

    def get_problem_state(self, problem_key: str) -> ProblemState | None:
        row = self.conn.execute(
            "SELECT * FROM problem_state WHERE problem_key = ?",
            (problem_key,),
        ).fetchone()
        if row is None:
            return None
        return ProblemState(
            problem_key=row["problem_key"],
            best_candidate_id=row["best_candidate_id"],
            best_objective=row["best_objective"],
            promoted_candidate_id=row["promoted_candidate_id"],
            updated_at=row["updated_at"],
        )

    def get_candidate(self, candidate_id: str) -> CandidateRow | None:
        row = self.conn.execute(
            "SELECT * FROM candidates WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            return None
        return self._candidate_from_row(row)

    def get_best_candidate(self, problem_key: str) -> CandidateRow | None:
        state = self.get_problem_state(problem_key)
        if state is None or not state.best_candidate_id:
            return None
        return self.get_candidate(state.best_candidate_id)

    def latest_candidate(self, problem_key: str) -> CandidateRow | None:
        row = self.conn.execute(
            """
            SELECT * FROM candidates
            WHERE problem_key = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (problem_key,),
        ).fetchone()
        if row is None:
            return None
        return self._candidate_from_row(row)

    def latest_candidate_by_hash(self, problem_key: str, source_sha256: str) -> CandidateRow | None:
        row = self.conn.execute(
            """
            SELECT * FROM candidates
            WHERE problem_key = ? AND source_sha256 = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (problem_key, source_sha256),
        ).fetchone()
        if row is None:
            return None
        return self._candidate_from_row(row)

    def recent_candidates(self, problem_key: str, limit: int = 10) -> list[CandidateRow]:
        rows = self.conn.execute(
            """
            SELECT * FROM candidates
            WHERE problem_key = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (problem_key, limit),
        ).fetchall()
        return [self._candidate_from_row(row) for row in rows]

    def _candidate_from_row(self, row: sqlite3.Row) -> CandidateRow:
        return CandidateRow(
            candidate_id=row["candidate_id"],
            problem_key=row["problem_key"],
            parent_id=row["parent_id"],
            kind=row["kind"],
            hypothesis=row["hypothesis"],
            source_path=row["source_path"],
            source_sha256=row["source_sha256"],
            promoted=bool(row["promoted"]),
            created_at=row["created_at"],
            score=row["score"],
            status=row["status"],
        )

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY,
                problem_key TEXT NOT NULL,
                parent_id TEXT,
                kind TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                promoted INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                score REAL,
                status TEXT
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                return_code INTEGER NOT NULL,
                status TEXT NOT NULL,
                objective REAL,
                metrics_json TEXT NOT NULL,
                stdout_path TEXT NOT NULL,
                stderr_path TEXT NOT NULL,
                result_path TEXT NOT NULL,
                command_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
            );

            CREATE TABLE IF NOT EXISTS problem_state (
                problem_key TEXT PRIMARY KEY,
                best_candidate_id TEXT,
                best_objective REAL,
                promoted_candidate_id TEXT,
                updated_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()
