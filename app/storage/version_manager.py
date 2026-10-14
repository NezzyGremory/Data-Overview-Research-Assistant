from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.storage.database import Database
from app.storage.version_models import (
    DatasetVersion,
    TransformationRecord,
)


class DatasetVersionManager:
    """
    Mengelola dataset versioning dan transformation trail.

    Tanggung jawab:
    - Menyimpan dataset original (immutable) saat upload.
    - Membuat versi baru setiap kali transformasi dilakukan.
    - Mencatat setiap operasi ke transformation trail.
    - Memuat DataFrame dari versi tertentu.
    - Rollback ke versi sebelumnya tanpa menghapus data.

    Struktur folder per project:

        data/projects/{project_id}/
        ├── raw/
        │   └── original.csv
        └── versions/
            ├── v0.parquet
            ├── v1.parquet
            └── ...
    """

    def __init__(
        self,
        database: Database,
        data_dir: str | Path = "data",
    ):
        self.database = database
        self.data_dir = Path(data_dir)

    # ============================================================
    # PATH HELPERS
    # ============================================================

    def _project_dir(self, project_id: int) -> Path:
        return self.data_dir / "projects" / str(project_id)

    def _raw_dir(self, project_id: int) -> Path:
        return self._project_dir(project_id) / "raw"

    def _versions_dir(self, project_id: int) -> Path:
        return self._project_dir(project_id) / "versions"

    @staticmethod
    def _has_parquet_support() -> bool:
        """
        Mengecek apakah engine parquet (pyarrow/fastparquet) tersedia.
        """
        return any(
            importlib.util.find_spec(engine) is not None
            for engine in ("pyarrow", "fastparquet")
        )

    def _version_file(
        self,
        project_id: int,
        version: int,
    ) -> Path:
        extension = ".parquet" if self._has_parquet_support() else ".csv"
        return (
            self._versions_dir(project_id)
            / f"v{version}{extension}"
        )

    def _save_dataframe(
        self,
        dataframe: pd.DataFrame,
        file_path: Path,
    ) -> None:
        """
        Menyimpan DataFrame ke Parquet (jika didukung) atau CSV.
        """
        if file_path.suffix.lower() == ".parquet":
            dataframe.to_parquet(file_path, index=False)
        else:
            dataframe.to_csv(file_path, index=False)

    def _read_dataframe(
        self,
        file_path: Path,
    ) -> pd.DataFrame:
        """
        Membaca DataFrame dari Parquet atau CSV sesuai ekstensi.
        """
        if file_path.suffix.lower() == ".parquet":
            return pd.read_parquet(file_path)
        return pd.read_csv(file_path)

    # ============================================================
    # HASHING
    # ============================================================

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """
        Menghitung SHA-256 hash dari file.
        """

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    # ============================================================
    # JSON HELPERS
    # ============================================================

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    def _from_json(
        value: str | None,
        default: Any = None,
    ) -> Any:
        if value is None:
            return default

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default

    # ============================================================
    # CREATE INITIAL VERSION (v0)
    # ============================================================

    def create_initial_version(
        self,
        project_id: int,
        csv_path: str | Path | None = None,
        dataframe: pd.DataFrame | None = None,
    ) -> DatasetVersion:
        """
        Membuat versi awal (v0) dari dataset.

        Dipanggil saat dataset pertama kali di-upload.

        Proses:
        1. Simpan CSV original ke raw/original.csv
        2. Simpan DataFrame sebagai v0.parquet (atau v0.csv jika fallback)
        3. Catat metadata ke database
        """
        if csv_path is not None:
            csv_path = Path(csv_path)

        if dataframe is None:
            if csv_path is None or not csv_path.exists():
                raise FileNotFoundError(
                    f"File CSV tidak ditemukan: {csv_path}"
                )

        # Pastikan belum ada versi untuk project ini
        existing = self._get_latest_version_number(
            project_id
        )

        if existing is not None:
            raise ValueError(
                f"Project {project_id} sudah memiliki "
                f"dataset version. Gunakan create_version() "
                f"untuk membuat versi baru."
            )

        # Buat folder project
        raw_dir = self._raw_dir(project_id)
        versions_dir = self._versions_dir(project_id)

        raw_dir.mkdir(parents=True, exist_ok=True)
        versions_dir.mkdir(parents=True, exist_ok=True)

        # Copy CSV original (IMMUTABLE)
        raw_copy = raw_dir / "original.csv"
        if csv_path is not None and csv_path.exists():
            shutil.copy2(str(csv_path), str(raw_copy))
            if dataframe is None:
                dataframe = pd.read_csv(
                    raw_copy,
                    engine="python",
                )
        else:
            dataframe.to_csv(raw_copy, index=False)

        # Simpan DataFrame v0
        version_file = self._version_file(
            project_id, 0
        )

        self._save_dataframe(
            dataframe,
            version_file,
        )


        # Hitung metadata
        file_hash = self._compute_file_hash(
            version_file
        )

        file_size = version_file.stat().st_size

        columns = list(dataframe.columns)

        dtypes = {
            col: str(dtype)
            for col, dtype in dataframe.dtypes.items()
        }

        # Simpan ke database
        version_id = self.database.execute(
            """
            INSERT INTO dataset_versions (
                project_id,
                version,
                label,
                file_path,
                row_count,
                column_count,
                columns_json,
                dtypes_json,
                file_hash,
                file_size_bytes,
                is_current
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                project_id,
                0,
                "Raw Dataset",
                str(version_file),
                len(dataframe),
                len(columns),
                self._json(columns),
                self._json(dtypes),
                file_hash,
                file_size,
            ),
        )

        return DatasetVersion(
            id=version_id,
            project_id=project_id,
            version=0,
            label="Raw Dataset",
            file_path=str(version_file),
            row_count=len(dataframe),
            column_count=len(columns),
            columns=columns,
            dtypes=dtypes,
            file_hash=file_hash,
            file_size_bytes=file_size,
            is_current=True,
        )

    # ============================================================
    # CREATE NEW VERSION
    # ============================================================

    def create_version(
        self,
        project_id: int,
        dataframe: pd.DataFrame,
        operation: str,
        parameters: Dict[str, Any] | None = None,
        impact: Dict[str, Any] | None = None,
        description: str | None = None,
        label: str | None = None,
    ) -> DatasetVersion:
        """
        Membuat versi baru dari dataset setelah transformasi.

        Args:
            project_id: ID project.
            dataframe: DataFrame hasil transformasi.
            operation: Nama operasi (e.g. 'remove_duplicates').
            parameters: Parameter operasi (JSON-serializable).
            impact: Dampak operasi (JSON-serializable).
            description: Deskripsi operasi.
            label: Label versi (default: capitalize operation).
        """

        if parameters is None:
            parameters = {}

        if impact is None:
            impact = {}

        # Tentukan nomor versi berikutnya
        latest = self._get_latest_version_number(
            project_id
        )

        if latest is None:
            raise ValueError(
                f"Project {project_id} belum memiliki "
                f"dataset version. Gunakan "
                f"create_initial_version() terlebih dahulu."
            )

        new_version = latest + 1

        if label is None:
            label = operation.replace("_", " ").title()

        # Simpan DataFrame versi baru
        versions_dir = self._versions_dir(project_id)
        versions_dir.mkdir(parents=True, exist_ok=True)

        version_file = self._version_file(
            project_id, new_version
        )

        self._save_dataframe(
            dataframe,
            version_file,
        )


        # Hitung metadata
        file_hash = self._compute_file_hash(
            version_file
        )

        file_size = version_file.stat().st_size

        columns = list(dataframe.columns)

        dtypes = {
            col: str(dtype)
            for col, dtype in dataframe.dtypes.items()
        }

        # Update is_current: matikan semua versi lama
        self.database.execute(
            """
            UPDATE dataset_versions
            SET is_current = 0
            WHERE project_id = ?
            """,
            (project_id,),
        )

        # Insert versi baru
        version_id = self.database.execute(
            """
            INSERT INTO dataset_versions (
                project_id,
                version,
                label,
                file_path,
                row_count,
                column_count,
                columns_json,
                dtypes_json,
                file_hash,
                file_size_bytes,
                is_current
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                project_id,
                new_version,
                label,
                str(version_file),
                len(dataframe),
                len(columns),
                self._json(columns),
                self._json(dtypes),
                file_hash,
                file_size,
            ),
        )

        # Catat di transformation trail
        self.database.execute(
            """
            INSERT INTO transformation_trail (
                project_id,
                from_version,
                to_version,
                operation,
                parameters_json,
                impact_json,
                description
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                latest,
                new_version,
                operation,
                self._json(parameters),
                self._json(impact),
                description,
            ),
        )

        return DatasetVersion(
            id=version_id,
            project_id=project_id,
            version=new_version,
            label=label,
            file_path=str(version_file),
            row_count=len(dataframe),
            column_count=len(columns),
            columns=columns,
            dtypes=dtypes,
            file_hash=file_hash,
            file_size_bytes=file_size,
            is_current=True,
        )

    # ============================================================
    # GET VERSION
    # ============================================================

    def get_version(
        self,
        project_id: int,
        version: int,
    ) -> DatasetVersion | None:
        """
        Mengambil metadata versi tertentu.
        """

        row = self.database.fetch_one(
            """
            SELECT *
            FROM dataset_versions
            WHERE project_id = ?
              AND version = ?
            """,
            (project_id, version),
        )

        if row is None:
            return None

        return self._row_to_version(row)

    def get_current_version(
        self,
        project_id: int,
    ) -> DatasetVersion | None:
        """
        Mengambil metadata versi yang sedang aktif.
        """

        row = self.database.fetch_one(
            """
            SELECT *
            FROM dataset_versions
            WHERE project_id = ?
              AND is_current = 1
            """,
            (project_id,),
        )

        if row is None:
            return None

        return self._row_to_version(row)

    def set_current_version(
        self,
        project_id: int,
        version: int,
    ) -> DatasetVersion:
        """Mark an existing version as current without creating a new copy."""
        if self.get_version(project_id, version) is None:
            raise ValueError(
                f"Version {version} tidak ditemukan untuk project {project_id}."
            )

        self.database.execute(
            """
            UPDATE dataset_versions
            SET is_current = CASE WHEN version = ? THEN 1 ELSE 0 END
            WHERE project_id = ?
            """,
            (version, project_id),
        )
        current = self.get_current_version(project_id)
        if current is None:
            raise RuntimeError("Versi aktif tidak dapat diperbarui.")
        return current

    # ============================================================
    # LOAD DATAFRAME
    # ============================================================

    def load_dataframe(
        self,
        project_id: int,
        version: int,
    ) -> pd.DataFrame:
        """
        Memuat DataFrame dari versi tertentu.
        """

        ver = self.get_version(project_id, version)

        if ver is None:
            raise ValueError(
                f"Version {version} tidak ditemukan "
                f"untuk project {project_id}."
            )

        file_path = Path(ver.file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File version tidak ditemukan: "
                f"{file_path}"
            )

        return self._read_dataframe(file_path)


    def load_current_dataframe(
        self,
        project_id: int,
    ) -> pd.DataFrame:
        """
        Memuat DataFrame dari versi yang sedang aktif.
        """

        ver = self.get_current_version(project_id)

        if ver is None:
            raise ValueError(
                f"Project {project_id} belum memiliki "
                f"dataset version."
            )

        return self.load_dataframe(
            project_id,
            ver.version,
        )

    # ============================================================
    # LIST VERSIONS
    # ============================================================

    def list_versions(
        self,
        project_id: int,
    ) -> List[DatasetVersion]:
        """
        Menampilkan semua versi dataset untuk project.
        """

        rows = self.database.fetch_all(
            """
            SELECT *
            FROM dataset_versions
            WHERE project_id = ?
            ORDER BY version ASC
            """,
            (project_id,),
        )

        return [
            self._row_to_version(row)
            for row in rows
        ]

    # ============================================================
    # TRAIL
    # ============================================================

    def get_trail(
        self,
        project_id: int,
    ) -> List[TransformationRecord]:
        """
        Menampilkan seluruh transformation trail
        untuk project, terurut kronologis.
        """

        rows = self.database.fetch_all(
            """
            SELECT *
            FROM transformation_trail
            WHERE project_id = ?
            ORDER BY from_version ASC, id ASC
            """,
            (project_id,),
        )

        return [
            self._row_to_trail(row)
            for row in rows
        ]

    # ============================================================
    # ROLLBACK
    # ============================================================

    def rollback_to(
        self,
        project_id: int,
        version: int,
    ) -> DatasetVersion:
        """
        Mengubah versi aktif ke versi tertentu.

        TIDAK menghapus versi setelahnya.
        Semua versi tetap tersimpan untuk audit trail.
        """

        ver = self.get_version(project_id, version)

        if ver is None:
            raise ValueError(
                f"Version {version} tidak ditemukan "
                f"untuk project {project_id}."
            )

        # Matikan semua is_current
        self.database.execute(
            """
            UPDATE dataset_versions
            SET is_current = 0
            WHERE project_id = ?
            """,
            (project_id,),
        )

        # Aktifkan versi target
        self.database.execute(
            """
            UPDATE dataset_versions
            SET is_current = 1
            WHERE project_id = ?
              AND version = ?
            """,
            (project_id, version),
        )

        ver.is_current = True

        return ver

    # ============================================================
    # VERSION COUNT
    # ============================================================

    def version_count(
        self,
        project_id: int,
    ) -> int:
        """
        Menghitung jumlah versi untuk project.
        """

        row = self.database.fetch_one(
            """
            SELECT COUNT(*) AS count
            FROM dataset_versions
            WHERE project_id = ?
            """,
            (project_id,),
        )

        if row is None:
            return 0

        return int(row["count"])

    def has_versions(
        self,
        project_id: int,
    ) -> bool:
        """
        Mengecek apakah project memiliki dataset version.
        """

        return self.version_count(project_id) > 0

    # ============================================================
    # DELETE PROJECT VERSIONS
    # ============================================================

    def delete_project_versions(
        self,
        project_id: int,
    ) -> None:
        """
        Menghapus semua versi dan file untuk project.

        Dipanggil saat project dihapus.
        """

        self.delete_project_files(project_id)

        # Hapus record database
        # (CASCADE dari projects juga akan menghapus,
        #  tetapi eksplisit lebih aman)
        self.database.execute(
            """
            DELETE FROM transformation_trail
            WHERE project_id = ?
            """,
            (project_id,),
        )

        self.database.execute(
            """
            DELETE FROM dataset_versions
            WHERE project_id = ?
            """,
            (project_id,),
        )

    def delete_project_files(self, project_id: int) -> None:
        """Remove only the app-managed files for a project."""
        project_dir = self._project_dir(project_id)

        if project_dir.exists():
            shutil.rmtree(str(project_dir))

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _get_latest_version_number(
        self,
        project_id: int,
    ) -> Optional[int]:
        """
        Mengambil nomor versi tertinggi untuk project.
        Returns None jika belum ada versi.
        """

        row = self.database.fetch_one(
            """
            SELECT MAX(version) AS max_version
            FROM dataset_versions
            WHERE project_id = ?
            """,
            (project_id,),
        )

        if row is None:
            return None

        return row["max_version"]

    def _row_to_version(self, row) -> DatasetVersion:
        """
        Konversi sqlite3.Row ke DatasetVersion.
        """

        return DatasetVersion(
            id=row["id"],
            project_id=row["project_id"],
            version=row["version"],
            label=row["label"],
            file_path=row["file_path"],
            row_count=row["row_count"],
            column_count=row["column_count"],
            columns=self._from_json(
                row["columns_json"], []
            ),
            dtypes=self._from_json(
                row["dtypes_json"], {}
            ),
            file_hash=row["file_hash"],
            file_size_bytes=row["file_size_bytes"],
            is_current=bool(row["is_current"]),
            created_at=row["created_at"],
        )

    def _row_to_trail(
        self, row
    ) -> TransformationRecord:
        """
        Konversi sqlite3.Row ke TransformationRecord.
        """

        return TransformationRecord(
            id=row["id"],
            project_id=row["project_id"],
            from_version=row["from_version"],
            to_version=row["to_version"],
            operation=row["operation"],
            parameters=self._from_json(
                row["parameters_json"], {}
            ),
            impact=self._from_json(
                row["impact_json"], {}
            ),
            description=row["description"],
            created_at=row["created_at"],
        )
