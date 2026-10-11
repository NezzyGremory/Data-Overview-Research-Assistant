from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import get_data_dir


class Database:
    """
    SQLite database manager untuk Dataset Research.

    Database menyimpan:
    - projects
    - research_results
    - papers

    Connection SQLite selalu ditutup secara eksplisit
    setelah operasi selesai.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
    ):
        self.db_path = Path(db_path) if db_path is not None else get_data_dir() / "dataset_research.db"

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize()

    # ============================================================
    # CONNECTION
    # ============================================================

    def connect(self) -> sqlite3.Connection:
        """
        Membuat koneksi SQLite baru.

        Connection DIKEMBALIKAN ke caller dan caller wajib
        menutupnya menggunakan connection.close().
        """

        connection = sqlite3.connect(
            str(self.db_path),
            timeout=30,
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        return connection

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def _initialize(self) -> None:
        """
        Membuat seluruh tabel database jika belum tersedia.
        """

        connection = self.connect()

        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    dataset_name TEXT,
                    dataset_path TEXT,
                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS research_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    project_id INTEGER NOT NULL UNIQUE,

                    status TEXT,

                    keywords_json TEXT,
                    domain_json TEXT,
                    queries_json TEXT,
                    ml_result_json TEXT,
                    landscape_json TEXT,
                    trend_json TEXT,
                    gaps_json TEXT,
                    summary_json TEXT,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    updated_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (project_id)
                        REFERENCES projects(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS project_workflows (
                    project_id INTEGER PRIMARY KEY,
                    analysis_result_json TEXT,
                    ml_task_result_json TEXT,
                    ml_methods_result_json TEXT,
                    research_result_json TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id)
                        REFERENCES projects(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    project_id INTEGER NOT NULL,

                    title TEXT NOT NULL,
                    authors_json TEXT,
                    abstract TEXT,
                    year INTEGER,
                    doi TEXT,
                    venue TEXT,
                    url TEXT,
                    citation_count INTEGER DEFAULT 0,
                    source TEXT,
                    external_id TEXT,

                    keywords_json TEXT,

                    relevance_score REAL,

                    score_breakdown_json TEXT,
                    reasons_json TEXT,

                    ranking_status TEXT,
                    rank INTEGER,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (project_id)
                        REFERENCES projects(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS
                    idx_papers_project_id
                ON papers(project_id);

                CREATE INDEX IF NOT EXISTS
                    idx_papers_project_rank
                ON papers(project_id, rank);

                CREATE INDEX IF NOT EXISTS
                    idx_research_project_id
                ON research_results(project_id);

                -- ============================================
                -- DATASET VERSIONING
                -- ============================================

                CREATE TABLE IF NOT EXISTS dataset_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,

                    version INTEGER NOT NULL DEFAULT 0,

                    label TEXT NOT NULL
                        DEFAULT 'Raw Dataset',

                    file_path TEXT NOT NULL,

                    row_count INTEGER,
                    column_count INTEGER,
                    columns_json TEXT,
                    dtypes_json TEXT,
                    file_hash TEXT,
                    file_size_bytes INTEGER,

                    is_current INTEGER NOT NULL DEFAULT 0,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (project_id)
                        REFERENCES projects(id)
                        ON DELETE CASCADE,

                    UNIQUE(project_id, version)
                );

                CREATE INDEX IF NOT EXISTS
                    idx_versions_project
                ON dataset_versions(project_id, version);

                CREATE INDEX IF NOT EXISTS
                    idx_versions_current
                ON dataset_versions(project_id, is_current);

                -- ============================================
                -- TRANSFORMATION TRAIL
                -- ============================================

                CREATE TABLE IF NOT EXISTS transformation_trail (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,

                    from_version INTEGER NOT NULL,
                    to_version INTEGER NOT NULL,

                    operation TEXT NOT NULL,

                    parameters_json TEXT NOT NULL
                        DEFAULT '{}',

                    impact_json TEXT NOT NULL
                        DEFAULT '{}',

                    description TEXT,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (project_id)
                        REFERENCES projects(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS
                    idx_trail_project
                ON transformation_trail(
                    project_id, from_version
                );

                -- ============================================
                -- JOURNAL SCREENING & ACCREDITATION PERSISTENCE
                -- ============================================

                CREATE TABLE IF NOT EXISTS journal_screening (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doi TEXT,
                    paper_key TEXT,
                    journal_name TEXT,
                    issn_matched TEXT,
                    rank TEXT,
                    status TEXT NOT NULL,
                    period_start INTEGER,
                    period_end INTEGER,
                    period_confirmed INTEGER DEFAULT 0,
                    checked_for_year INTEGER,
                    verification_source TEXT,
                    source_url TEXT,
                    is_ambiguous INTEGER DEFAULT 0,
                    ambiguity_reason TEXT,
                    notes TEXT,
                    screening_json TEXT,
                    checked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS
                    idx_screening_doi
                ON journal_screening(doi);

                CREATE INDEX IF NOT EXISTS
                    idx_screening_paper_key
                ON journal_screening(paper_key);
                """
            )

            cursor = connection.cursor()
            cursor.execute("PRAGMA table_info(papers)")
            columns = [row[1] for row in cursor.fetchall()]
            if "issn_json" not in columns:
                cursor.execute("ALTER TABLE papers ADD COLUMN issn_json TEXT")
            if "sinta_screening_json" not in columns:
                cursor.execute("ALTER TABLE papers ADD COLUMN sinta_screening_json TEXT")

            cursor.execute("PRAGMA table_info(journal_screening)")
            screening_columns = {row[1] for row in cursor.fetchall()}
            screening_migrations = {
                "issn_matched": "TEXT",
                "rank": "TEXT",
                "period_start": "INTEGER",
                "period_end": "INTEGER",
                "period_confirmed": "INTEGER DEFAULT 0",
                "checked_for_year": "INTEGER",
                "verification_source": "TEXT",
                "source_url": "TEXT",
                "is_ambiguous": "INTEGER DEFAULT 0",
                "ambiguity_reason": "TEXT",
                "notes": "TEXT",
                "screening_json": "TEXT",
                "checked_at": "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
            }
            for column, definition in screening_migrations.items():
                if column not in screening_columns:
                    cursor.execute(
                        f"ALTER TABLE journal_screening ADD COLUMN {column} {definition}"
                    )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            # SANGAT PENTING:
            # SQLite connection harus ditutup secara eksplisit.
            connection.close()

    # ============================================================
    # EXECUTE
    # ============================================================

    def execute(
        self,
        query: str,
        parameters: tuple[Any, ...] = (),
    ) -> int:
        """
        Menjalankan INSERT / UPDATE / DELETE.

        Returns:
            lastrowid jika tersedia.
        """

        connection = self.connect()

        try:
            cursor = connection.execute(
                query,
                parameters,
            )

            lastrowid = cursor.lastrowid

            connection.commit()

            return int(lastrowid or 0)

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    # ============================================================
    # FETCH ONE
    # ============================================================

    def fetch_one(
        self,
        query: str,
        parameters: tuple[Any, ...] = (),
    ) -> sqlite3.Row | None:
        """
        Mengambil satu row dari database.
        """

        connection = self.connect()

        try:
            cursor = connection.execute(
                query,
                parameters,
            )

            row = cursor.fetchone()

            return row

        finally:
            connection.close()

    # ============================================================
    # FETCH ALL
    # ============================================================

    def fetch_all(
        self,
        query: str,
        parameters: tuple[Any, ...] = (),
    ) -> list[sqlite3.Row]:
        """
        Mengambil seluruh row dari database.
        """

        connection = self.connect()

        try:
            cursor = connection.execute(
                query,
                parameters,
            )

            rows = cursor.fetchall()

            return rows

        finally:
            connection.close()

    # ============================================================
    # DELETE DATABASE
    # ============================================================

    def delete_database(self) -> None:
        """
        Menghapus database.

        Digunakan untuk testing/reset database.
        """

        # Pastikan tidak ada connection dari object ini
        # yang masih aktif.
        #
        # Karena setiap operasi database menggunakan
        # connection lokal dan selalu close() di finally,
        # file aman untuk dihapus.

        if self.db_path.exists():
            self.db_path.unlink()

        self._initialize()
