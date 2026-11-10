"""SQLite persistence isolated to Vision Intelligence records."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.storage.database import Database


class VisionAnalysisRepository:
    """Store paths and analysis metadata without embedding source media."""

    def __init__(self, database: Database):
        self.database = database
        self._initialize()

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _decode(row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        record = dict(row)
        for source, target, default in (
            ("dataset_metadata_json", "dataset_metadata", {}),
            ("analysis_config_json", "analysis_config", {}),
            ("results_json", "results", {}),
            ("recommendations_json", "recommendations", {}),
            ("error_log_json", "error_log", []),
        ):
            try:
                record[target] = json.loads(record.get(source) or "")
            except (json.JSONDecodeError, TypeError):
                record[target] = default
        return record

    def _initialize(self) -> None:
        connection = self.database.connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS vision_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    dataset_path TEXT NOT NULL,
                    dataset_type TEXT NOT NULL CHECK (dataset_type IN ('image', 'video')),
                    dataset_metadata_json TEXT NOT NULL DEFAULT '{}',
                    analysis_config_json TEXT NOT NULL DEFAULT '{}',
                    results_json TEXT NOT NULL DEFAULT '{}',
                    recommendations_json TEXT NOT NULL DEFAULT '{}',
                    error_log_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_vision_analyses_project
                    ON vision_analyses(project_id, id DESC);
                CREATE INDEX IF NOT EXISTS idx_vision_analyses_path
                    ON vision_analyses(dataset_path COLLATE NOCASE, dataset_type, id DESC);
                """
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def normalize_path(path: str | Path) -> str:
        try:
            return os.path.normcase(str(Path(path).expanduser().resolve()))
        except (OSError, RuntimeError):
            return os.path.normcase(str(path))

    def save(
        self,
        *,
        project_id: int | None,
        dataset_path: str | Path,
        dataset_type: str,
        dataset_metadata: dict[str, Any],
        analysis_config: dict[str, Any],
        results: dict[str, Any],
        recommendations: dict[str, Any],
        error_log: list[dict[str, Any]] | None = None,
    ) -> int:
        if dataset_type not in {"image", "video"}:
            raise ValueError("dataset_type harus image atau video.")
        connection = self.database.connect()
        try:
            cursor = connection.execute(
                """
                INSERT INTO vision_analyses (
                    project_id, dataset_path, dataset_type, dataset_metadata_json,
                    analysis_config_json, results_json, recommendations_json, error_log_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    self.normalize_path(dataset_path),
                    dataset_type,
                    self._json(dataset_metadata),
                    self._json(analysis_config),
                    self._json(results),
                    self._json(recommendations),
                    self._json(error_log or []),
                ),
            )
            if project_id is not None:
                connection.execute(
                    "UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (project_id,),
                )
            connection.commit()
            return int(cursor.lastrowid)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def update_recommendations(
        self,
        analysis_id: int,
        analysis_config: dict[str, Any],
        recommendations: dict[str, Any],
    ) -> None:
        self.database.execute(
            """
            UPDATE vision_analyses
            SET analysis_config_json = ?, recommendations_json = ?
            WHERE id = ?
            """,
            (self._json(analysis_config), self._json(recommendations), int(analysis_id)),
        )

    def get_latest_by_path(self, dataset_path: str | Path, dataset_type: str) -> dict[str, Any] | None:
        row = self.database.fetch_one(
            """
            SELECT * FROM vision_analyses
            WHERE dataset_path = ? COLLATE NOCASE AND dataset_type = ?
            ORDER BY id DESC LIMIT 1
            """,
            (self.normalize_path(dataset_path), dataset_type),
        )
        return self._decode(row)

    def get_latest_for_project(
        self,
        project_id: int,
        dataset_type: str | None = None,
    ) -> dict[str, Any] | None:
        if dataset_type is not None and dataset_type not in {"image", "video"}:
            raise ValueError("dataset_type harus image atau video.")
        if dataset_type is None:
            row = self.database.fetch_one(
                """
                SELECT * FROM vision_analyses WHERE project_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (int(project_id),),
            )
        else:
            row = self.database.fetch_one(
                """
                SELECT * FROM vision_analyses WHERE project_id = ? AND dataset_type = ?
                ORDER BY id DESC LIMIT 1
                """,
                (int(project_id), dataset_type),
            )
        return self._decode(row)

