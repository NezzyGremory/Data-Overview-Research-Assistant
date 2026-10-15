"""Project management, SQLite versioning, run history, and graph branching."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """Compute deterministic SHA256 hash for a pandas DataFrame."""
    try:
        col_bytes = str(tuple(df.columns)).encode("utf-8")
        row_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        return hashlib.sha256(col_bytes + row_bytes).hexdigest()
    except Exception:
        return hashlib.sha256(repr(df.values).encode("utf-8")).hexdigest()


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """Compute SHA256 of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_current_library_versions() -> Dict[str, str]:
    """Capture current versions of core ML dependencies."""
    versions = {}
    for pkg in ("sklearn", "numpy", "pandas", "scipy", "xgboost", "lightgbm", "catboost", "imblearn"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


class GraphHistoryStack:
    """In-memory undo/redo stack for node editor actions (separated from version history)."""

    def __init__(self, max_depth: int = 50):
        self.max_depth = max_depth
        self._undo_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []

    def push(self, state: Dict[str, Any]) -> None:
        """Push a snapshot of graph state (clears redo stack)."""
        self._undo_stack.append(json.loads(json.dumps(state)))
        if len(self._undo_stack) > self.max_depth:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 1

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def undo(self) -> Optional[Dict[str, Any]]:
        """Pop current state into redo and return previous state."""
        if not self.can_undo():
            return None
        current = self._undo_stack.pop()
        self._redo_stack.append(current)
        return json.loads(json.dumps(self._undo_stack[-1]))

    def redo(self) -> Optional[Dict[str, Any]]:
        """Pop redo state and restore to undo."""
        if not self.can_redo():
            return None
        state = self._redo_stack.pop()
        self._undo_stack.append(state)
        return json.loads(json.dumps(state))

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()


class ProjectManager:
    """SQLite-backed manager for ML projects, dataset metadata, graph versions, and run tracking."""

    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        self.db_path = str(db_path)
        self._memory_conn: Optional[sqlite3.Connection] = None
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            self._memory_conn = sqlite3.connect(":memory:")
            self._memory_conn.row_factory = sqlite3.Row
            self._memory_conn.execute("PRAGMA foreign_keys = ON;")
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    file_path TEXT DEFAULT '',
                    file_hash TEXT NOT NULL,
                    row_count INTEGER DEFAULT 0,
                    column_count INTEGER DEFAULT 0,
                    columns_json TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS graph_versions (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    parent_version_id TEXT,
                    version_number INTEGER NOT NULL,
                    branch_name TEXT DEFAULT 'main',
                    graph_json TEXT NOT NULL,
                    message TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY(parent_version_id) REFERENCES graph_versions(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    graph_version_id TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    status TEXT DEFAULT 'success',
                    metrics_json TEXT DEFAULT '{}',
                    confusion_matrix_json TEXT DEFAULT '[]',
                    duration_ms REAL DEFAULT 0.0,
                    random_state INTEGER DEFAULT 42,
                    library_versions_json TEXT DEFAULT '{}',
                    notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY(graph_version_id) REFERENCES graph_versions(id) ON DELETE CASCADE
                );
            """)

    # -------------------------------------------------------------
    # PROJECT OPERATIONS
    # -------------------------------------------------------------

    def create_project(self, name: str, description: str = "") -> Dict[str, Any]:
        project_id = str(uuid.uuid4())[:8]
        now = _current_timestamp()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO projects (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (project_id, name, description, now, now),
            )
        return {
            "id": project_id,
            "name": name,
            "description": description,
            "created_at": now,
            "updated_at": now,
        }

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            return dict(row) if row else None

    def list_projects(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
            return [dict(r) for r in rows]

    def delete_project(self, project_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            return cursor.rowcount > 0

    # -------------------------------------------------------------
    # DATASET TRACKING
    # -------------------------------------------------------------

    def register_dataset(
        self,
        project_id: str,
        data: Union[pd.DataFrame, str, Path],
        file_path: str = "",
    ) -> Dict[str, Any]:
        """Register dataset with SHA256 hash to track data versioning."""
        if isinstance(data, pd.DataFrame):
            file_hash = compute_dataframe_hash(data)
            row_count = len(data)
            col_count = len(data.columns)
            columns = list(data.columns)
        else:
            p = Path(data)
            file_path = str(p)
            file_hash = compute_file_hash(p)
            # Try reading first few rows to get columns
            try:
                sample = pd.read_csv(p, nrows=2)
                columns = list(sample.columns)
                col_count = len(columns)
                # Count total rows
                with open(p, "rb") as f:
                    row_count = max(0, sum(1 for _ in f) - 1)
            except Exception:
                columns = []
                row_count = 0
                col_count = 0

        dataset_id = str(uuid.uuid4())[:8]
        now = _current_timestamp()
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO datasets (id, project_id, file_path, file_hash, row_count, column_count, columns_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (dataset_id, project_id, file_path, file_hash, row_count, col_count, json.dumps(columns), now),
            )
        return {
            "id": dataset_id,
            "project_id": project_id,
            "file_path": file_path,
            "file_hash": file_hash,
            "row_count": row_count,
            "column_count": col_count,
            "columns": columns,
            "created_at": now,
        }

    # -------------------------------------------------------------
    # GRAPH VERSIONING & BRANCHING
    # -------------------------------------------------------------

    def save_graph_version(
        self,
        project_id: str,
        graph_dict: Dict[str, Any],
        message: str = "",
        parent_version_id: Optional[str] = None,
        branch_name: str = "main",
    ) -> Dict[str, Any]:
        """Save a new graph version with auto-incremented version number per branch."""
        now = _current_timestamp()
        version_id = str(uuid.uuid4())[:8]

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT MAX(version_number) as max_v FROM graph_versions WHERE project_id = ? AND branch_name = ?",
                (project_id, branch_name),
            ).fetchone()
            next_version = (row["max_v"] or 0) + 1

            conn.execute(
                """INSERT INTO graph_versions (id, project_id, parent_version_id, version_number, branch_name, graph_json, message, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (version_id, project_id, parent_version_id, next_version, branch_name, json.dumps(graph_dict), message, now),
            )
            conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))

        return {
            "id": version_id,
            "project_id": project_id,
            "parent_version_id": parent_version_id,
            "version_number": next_version,
            "branch_name": branch_name,
            "graph": graph_dict,
            "message": message,
            "created_at": now,
        }

    def branch_graph(
        self,
        project_id: str,
        source_version_id: str,
        new_branch_name: str,
        message: str = "",
    ) -> Dict[str, Any]:
        """Duplicate/branch graph from an existing version into a new branch."""
        source = self.get_graph_version(source_version_id)
        if not source:
            raise ValueError(f"Version ID '{source_version_id}' tidak ditemukan.")

        msg = message or f"Branched from {source['branch_name']} v{source['version_number']}"
        return self.save_graph_version(
            project_id=project_id,
            graph_dict=source["graph"],
            message=msg,
            parent_version_id=source_version_id,
            branch_name=new_branch_name,
        )

    def get_graph_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM graph_versions WHERE id = ?", (version_id,)).fetchone()
            if not row:
                return None
            res = dict(row)
            res["graph"] = json.loads(res["graph_json"])
            return res

    def list_graph_versions(self, project_id: str, branch_name: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            if branch_name:
                rows = conn.execute(
                    "SELECT * FROM graph_versions WHERE project_id = ? AND branch_name = ? ORDER BY version_number ASC",
                    (project_id, branch_name),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM graph_versions WHERE project_id = ? ORDER BY branch_name, version_number ASC",
                    (project_id),
                ).fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["graph"] = json.loads(item["graph_json"])
                results.append(item)
            return results

    # -------------------------------------------------------------
    # RUN HISTORY & COMPARISON
    # -------------------------------------------------------------

    def record_run(
        self,
        project_id: str,
        graph_version_id: str,
        dataset_hash: str,
        metrics: Dict[str, Any],
        confusion_matrix: Optional[List[Any]] = None,
        duration_ms: float = 0.0,
        random_state: int = 42,
        library_versions: Optional[Dict[str, str]] = None,
        notes: str = "",
        status: str = "success",
    ) -> Dict[str, Any]:
        """Record an execution run with its metrics, duration, seed, and library versions."""
        run_id = str(uuid.uuid4())[:8]
        now = _current_timestamp()
        lib_vers = library_versions or get_current_library_versions()

        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO runs (id, project_id, graph_version_id, dataset_hash, status, metrics_json, confusion_matrix_json, duration_ms, random_state, library_versions_json, notes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    project_id,
                    graph_version_id,
                    dataset_hash,
                    status,
                    json.dumps(metrics),
                    json.dumps(confusion_matrix or []),
                    duration_ms,
                    random_state,
                    json.dumps(lib_vers),
                    notes,
                    now,
                ),
            )
            conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))

        return {
            "id": run_id,
            "project_id": project_id,
            "graph_version_id": graph_version_id,
            "dataset_hash": dataset_hash,
            "status": status,
            "metrics": metrics,
            "confusion_matrix": confusion_matrix or [],
            "duration_ms": duration_ms,
            "random_state": random_state,
            "library_versions": lib_vers,
            "notes": notes,
            "created_at": now,
        }

    def list_runs(self, project_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["metrics"] = json.loads(item["metrics_json"])
                item["confusion_matrix"] = json.loads(item["confusion_matrix_json"])
                item["library_versions"] = json.loads(item["library_versions_json"])
                results.append(item)
            return results

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return None
            res = dict(row)
            res["metrics"] = json.loads(res["metrics_json"])
            res["confusion_matrix"] = json.loads(res["confusion_matrix_json"])
            res["library_versions"] = json.loads(res["library_versions_json"])
            return res

    def check_dataset_hash(self, run_id: str, current_dataset_hash: str) -> Tuple[bool, str]:
        """Check if dataset hash changed compared to a previous run."""
        run = self.get_run(run_id)
        if not run:
            return False, f"Run ID '{run_id}' tidak ditemukan."
        if run["dataset_hash"] != current_dataset_hash:
            return True, (
                f"Peringatan: Hash dataset berbeda dengan run {run_id}. "
                f"Data mungkin telah dimodifikasi (Run: {run['dataset_hash'][:8]} vs Sekarang: {current_dataset_hash[:8]})."
            )
        return False, "Dataset hash cocok."

    def compare_runs(self, run_id_a: str, run_id_b: str) -> Dict[str, Any]:
        """Side-by-side comparison between two runs."""
        run_a = self.get_run(run_id_a)
        run_b = self.get_run(run_id_b)
        if not run_a or not run_b:
            raise ValueError("Satu atau kedua Run ID tidak ditemukan.")

        all_metric_keys = sorted(set(run_a["metrics"].keys()) | set(run_b["metrics"].keys()))
        metric_diffs = {}
        for k in all_metric_keys:
            val_a = run_a["metrics"].get(k)
            val_b = run_b["metrics"].get(k)
            delta = None
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                delta = round(val_b - val_a, 4)
            metric_diffs[k] = {
                "run_a": val_a,
                "run_b": val_b,
                "delta": delta,
            }

        return {
            "run_a": {
                "id": run_a["id"],
                "version_id": run_a["graph_version_id"],
                "dataset_hash": run_a["dataset_hash"],
                "duration_ms": run_a["duration_ms"],
                "random_state": run_a["random_state"],
                "created_at": run_a["created_at"],
            },
            "run_b": {
                "id": run_b["id"],
                "version_id": run_b["graph_version_id"],
                "dataset_hash": run_b["dataset_hash"],
                "duration_ms": run_b["duration_ms"],
                "random_state": run_b["random_state"],
                "created_at": run_b["created_at"],
            },
            "same_dataset": run_a["dataset_hash"] == run_b["dataset_hash"],
            "dataset_warning": "" if run_a["dataset_hash"] == run_b["dataset_hash"] else "Dataset berbeda antar run!",
            "metrics_comparison": metric_diffs,
            "duration_delta_ms": round(run_b["duration_ms"] - run_a["duration_ms"], 2),
        }
