"""Local record of what the server did for each project.

The ledger stores facts that cannot be looked up later: which project a job
belongs to, why it was submitted, which commit it ran, where a project lives
on each cluster. Live state (is the job still running?) is fetched from Slurm
when asked for and cached here with the time it was checked, so a reader can
tell how fresh it is.

SQLite keeps it safe for several MCP servers (for example a Claude Code and a
Codex session) writing at the same time. Each call opens its own connection.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .slurm import TERMINAL_JOB_STATES

SCHEMA_VERSION = 1

# "missing" means the task folder is gone, so its state can never be read again.
TERMINAL_TASK_STATES = frozenset({"succeeded", "failed", "lost", "missing"})

# Marker for "do not filter on project", since None means "records without a project".
ANY = object()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    project TEXT,
    cluster TEXT,
    kind TEXT NOT NULL,
    summary TEXT NOT NULL,
    data TEXT
);
CREATE INDEX IF NOT EXISTS events_by_project ON events (project, id);
CREATE TABLE IF NOT EXISTS jobs (
    cluster TEXT NOT NULL,
    job_id TEXT NOT NULL,
    project TEXT,
    job_name TEXT,
    profile TEXT,
    script TEXT,
    purpose TEXT,
    git_commit TEXT,
    git_dirty INTEGER,
    submitted_at TEXT NOT NULL,
    details TEXT,
    state TEXT,
    exit_code TEXT,
    elapsed TEXT,
    state_checked_at TEXT,
    terminal INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (cluster, job_id)
);
CREATE INDEX IF NOT EXISTS jobs_by_project ON jobs (project, submitted_at);
CREATE TABLE IF NOT EXISTS tasks (
    cluster TEXT NOT NULL,
    task_id TEXT NOT NULL,
    project TEXT,
    kind TEXT NOT NULL,
    description TEXT,
    started_at TEXT NOT NULL,
    details TEXT,
    state TEXT,
    exit_code INTEGER,
    state_checked_at TEXT,
    terminal INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (cluster, task_id)
);
CREATE TABLE IF NOT EXISTS locations (
    project TEXT NOT NULL,
    cluster TEXT NOT NULL,
    details TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (project, cluster)
);
"""


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_job_state(state: str | None) -> str | None:
    """'CANCELLED by 1234' -> 'CANCELLED'; '' -> None."""
    if not state:
        return None
    return state.split()[0].rstrip("+").upper()


def is_terminal_job_state(state: str | None) -> bool:
    return normalize_job_state(state) in TERMINAL_JOB_STATES


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._ready = False

    # -- plumbing -------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if not self._ready:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        if not self._ready:
            with conn:
                conn.executescript(_SCHEMA)
                conn.execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
                )
            self._ready = True
        return conn

    def _write(self, sql: str, params: Iterable[Any] = ()) -> int:
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(sql, tuple(params))
            return int(cursor.lastrowid or 0)

    def _read(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            return [_row(row) for row in conn.execute(sql, tuple(params)).fetchall()]

    # -- events ---------------------------------------------------------------------

    def add_event(
        self,
        kind: str,
        summary: str,
        *,
        project: str | None = None,
        cluster: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> int:
        return self._write(
            "INSERT INTO events (ts, project, cluster, kind, summary, data) VALUES (?, ?, ?, ?, ?, ?)",
            (now(), project, cluster, kind, summary, _dump(data)),
        )

    def events(
        self,
        *,
        project: Any = ANY,
        cluster: str | None = None,
        kind: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        where, params = _filters(project=project, cluster=cluster)
        if kind:
            where.append("kind = ?")
            params.append(kind)
        sql = "SELECT * FROM events" + _where(where) + " ORDER BY id DESC LIMIT ?"
        return self._read(sql, [*params, limit])

    # -- jobs -----------------------------------------------------------------------

    def add_job(
        self,
        *,
        cluster: str,
        job_id: str,
        project: str | None,
        job_name: str | None,
        profile: str | None,
        script: str | None,
        purpose: str | None,
        git_commit: str | None,
        git_dirty: bool | None,
        details: dict[str, Any] | None,
    ) -> None:
        self._write(
            """
            INSERT INTO jobs (cluster, job_id, project, job_name, profile, script, purpose, git_commit, git_dirty,
                              submitted_at, details, state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'SUBMITTED')
            ON CONFLICT (cluster, job_id) DO UPDATE SET
                project = excluded.project, job_name = excluded.job_name, profile = excluded.profile,
                script = excluded.script, purpose = excluded.purpose, git_commit = excluded.git_commit,
                git_dirty = excluded.git_dirty, details = excluded.details
            """,
            (
                cluster,
                job_id,
                project,
                job_name,
                profile,
                script,
                purpose,
                git_commit,
                None if git_dirty is None else int(git_dirty),
                now(),
                _dump(details),
            ),
        )

    def update_job_states(self, cluster: str, states: dict[str, dict[str, Any]]) -> int:
        """Cache live states for jobs already in the ledger; unknown job ids are ignored.

        The first time a job is seen in a finished state, a job_finished event is logged.
        """
        if not states:
            return 0
        checked = now()
        updated = 0
        with closing(self._connect()) as conn, conn:
            for job_id, info in states.items():
                state = normalize_job_state(info.get("state"))
                if state is None:
                    continue
                row = conn.execute(
                    "SELECT project, terminal, job_name FROM jobs WHERE cluster = ? AND job_id = ?", (cluster, job_id)
                ).fetchone()
                if row is None:
                    continue
                terminal = state in TERMINAL_JOB_STATES
                exit_code = info.get("exit_code") or None
                elapsed = info.get("elapsed") or None
                conn.execute(
                    """
                    UPDATE jobs SET state = ?, exit_code = COALESCE(?, exit_code), elapsed = COALESCE(?, elapsed),
                                    state_checked_at = ?, terminal = ?
                    WHERE cluster = ? AND job_id = ?
                    """,
                    (state, exit_code, elapsed, checked, int(terminal), cluster, job_id),
                )
                updated += 1
                if terminal and not row["terminal"]:
                    detail = ", ".join(part for part in (f"exit {exit_code}" if exit_code else "", elapsed or "") if part)
                    conn.execute(
                        "INSERT INTO events (ts, project, cluster, kind, summary, data) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            checked,
                            row["project"],
                            cluster,
                            "job_finished",
                            f"Job {job_id} ({row['job_name'] or 'unnamed'}) {state}" + (f" ({detail})" if detail else ""),
                            _dump({"job_id": job_id, "state": state, "exit_code": exit_code, "elapsed": elapsed,
                                   "array_tasks": info.get("array_tasks")}),
                        ),
                    )
        return updated

    def get_job(self, cluster: str, job_id: str) -> dict[str, Any] | None:
        rows = self._read("SELECT * FROM jobs WHERE cluster = ? AND job_id = ?", (cluster, job_id))
        return rows[0] if rows else None

    def job_projects(self, cluster: str, job_ids: Iterable[str]) -> dict[str, str | None]:
        ids = sorted(set(job_ids))
        found: dict[str, str | None] = {}
        # Stay well under SQLite's limit on bound parameters.
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._read(
                f"SELECT job_id, project FROM jobs WHERE cluster = ? AND job_id IN ({placeholders})", [cluster, *chunk]
            )
            found.update({row["job_id"]: row["project"] for row in rows})
        return found

    def jobs(
        self,
        *,
        project: Any = ANY,
        cluster: str | None = None,
        active_only: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        where, params = _filters(project=project, cluster=cluster)
        if active_only:
            where.append("terminal = 0")
        sql = "SELECT * FROM jobs" + _where(where) + " ORDER BY submitted_at DESC, job_id DESC LIMIT ?"
        return self._read(sql, [*params, limit])

    def job_state_counts(self, *, project: Any = ANY, cluster: str | None = None) -> dict[str, int]:
        where, params = _filters(project=project, cluster=cluster)
        rows = self._read("SELECT state, COUNT(*) AS n FROM jobs" + _where(where) + " GROUP BY state", params)
        return {row["state"] or "UNKNOWN": row["n"] for row in rows}

    # -- background tasks -------------------------------------------------------------

    def add_task(
        self,
        *,
        cluster: str,
        task_id: str,
        project: str | None,
        kind: str,
        description: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._write(
            """
            INSERT OR REPLACE INTO tasks (cluster, task_id, project, kind, description, started_at, details, state)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'running')
            """,
            (cluster, task_id, project, kind, description, now(), _dump(details)),
        )

    def update_task_state(self, cluster: str, task_id: str, state: str, exit_code: int | None) -> int:
        """Cache a background task's state; logs a task_finished event the first time it ends."""
        checked = now()
        terminal = state in TERMINAL_TASK_STATES
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT project, terminal, description FROM tasks WHERE cluster = ? AND task_id = ?", (cluster, task_id)
            ).fetchone()
            if row is None:
                return 0
            conn.execute(
                """
                UPDATE tasks SET state = ?, exit_code = ?, state_checked_at = ?, terminal = ?
                WHERE cluster = ? AND task_id = ?
                """,
                (state, exit_code, checked, int(terminal), cluster, task_id),
            )
            if terminal and not row["terminal"]:
                conn.execute(
                    "INSERT INTO events (ts, project, cluster, kind, summary, data) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        checked,
                        row["project"],
                        cluster,
                        "task_finished",
                        f"Task {task_id} ({row['description'] or 'task'}) {state}"
                        + (f" (exit {exit_code})" if exit_code is not None else ""),
                        _dump({"task_id": task_id, "state": state, "exit_code": exit_code}),
                    ),
                )
            return 1

    def tasks(
        self,
        *,
        project: Any = ANY,
        cluster: str | None = None,
        active_only: bool = False,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        where, params = _filters(project=project, cluster=cluster)
        if active_only:
            where.append("terminal = 0")
        sql = "SELECT * FROM tasks" + _where(where) + " ORDER BY started_at DESC, task_id DESC LIMIT ?"
        return self._read(sql, [*params, limit])

    # -- locations --------------------------------------------------------------------

    def update_location(self, project: str, cluster: str, **fields: Any) -> None:
        """Merge facts about where a project lives on a cluster (paths, commit, venv)."""
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT details FROM locations WHERE project = ? AND cluster = ?", (project, cluster)
            ).fetchone()
            details = json.loads(row["details"]) if row else {}
            details.update({key: value for key, value in fields.items() if value is not None})
            conn.execute(
                """
                INSERT INTO locations (project, cluster, details, updated_at) VALUES (?, ?, ?, ?)
                ON CONFLICT (project, cluster) DO UPDATE SET details = excluded.details, updated_at = excluded.updated_at
                """,
                (project, cluster, json.dumps(details, sort_keys=True), now()),
            )

    def locations(self, project: str) -> dict[str, dict[str, Any]]:
        rows = self._read("SELECT cluster, details, updated_at FROM locations WHERE project = ?", (project,))
        return {row["cluster"]: {**row["details"], "updated_at": row["updated_at"]} for row in rows}

    # -- summaries --------------------------------------------------------------------

    def clusters_for(self, project: str | None) -> list[str]:
        where, params = _filters(project=project)
        clause = _where(where)
        rows = self._read(
            f"SELECT cluster FROM jobs{clause} UNION SELECT cluster FROM tasks{clause} "
            + ("UNION SELECT cluster FROM locations WHERE project = ?" if project is not None else ""),
            [*params, *params, *([project] if project is not None else [])],
        )
        return sorted(row["cluster"] for row in rows if row["cluster"])

    def projects(self) -> list[dict[str, Any]]:
        """Projects with any recorded activity, most recently active first."""
        rows = self._read(
            """
            SELECT project,
                   MAX(ts) AS last_activity,
                   (SELECT COUNT(*) FROM jobs j WHERE j.project = e.project) AS jobs,
                   (SELECT COUNT(*) FROM jobs j WHERE j.project = e.project AND j.terminal = 0) AS unfinished_jobs,
                   (SELECT GROUP_CONCAT(DISTINCT cluster) FROM events c WHERE c.project = e.project) AS clusters
            FROM events e
            WHERE project IS NOT NULL
            GROUP BY project
            ORDER BY last_activity DESC
            """
        )
        for row in rows:
            row["clusters"] = sorted((row["clusters"] or "").split(",")) if row["clusters"] else []
        return rows


def _dump(data: dict[str, Any] | None) -> str | None:
    return None if data is None else json.dumps(data, sort_keys=True, default=str)


def _row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    for key in ("data", "details"):
        if isinstance(result.get(key), str):
            try:
                result[key] = json.loads(result[key])
            except ValueError:
                pass
    if "git_dirty" in result and result["git_dirty"] is not None:
        result["git_dirty"] = bool(result["git_dirty"])
    if "terminal" in result:
        result["terminal"] = bool(result["terminal"])
    return result


def _filters(*, project: Any = ANY, cluster: str | None = None) -> tuple[list[str], list[Any]]:
    where: list[str] = []
    params: list[Any] = []
    if project is not ANY:
        if project is None:
            where.append("project IS NULL")
        else:
            where.append("project = ?")
            params.append(project)
    if cluster is not None:
        where.append("cluster = ?")
        params.append(cluster)
    return where, params


def _where(clauses: list[str]) -> str:
    return " WHERE " + " AND ".join(clauses) if clauses else ""
