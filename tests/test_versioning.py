import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.storage.database import Database
from app.storage.project_repository import ProjectRepository
from app.storage.version_manager import DatasetVersionManager


@contextmanager
def temp_workspace():
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = Path(temp_dir) / "test.db"
        database = Database(db_path=db_path)
        repository = ProjectRepository(database=database)
        version_manager = DatasetVersionManager(
            database=database,
            data_dir=Path(temp_dir) / "data",
        )

        project_id = repository.create_project(
            name="Test Titanic Project",
            description="Testing dataset versioning",
        )

        sample_csv = Path(temp_dir) / "sample_titanic.csv"
        sample_df = pd.DataFrame(
            {
                "PassengerId": [1, 2, 3, 4, 4],  # duplicate row 4
                "Survived": [0, 1, 1, 0, 0],
                "Pclass": [3, 1, 3, 1, 1],
                "Name": ["Braund", "Cumings", "Heikkinen", "Futrelle", "Futrelle"],
                "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
                "Fare": [7.25, 71.28, 7.92, 53.1, 53.1],
            }
        )
        sample_df.to_csv(sample_csv, index=False)

        yield {
            "temp_dir": temp_dir,
            "database": database,
            "repository": repository,
            "version_manager": version_manager,
            "project_id": project_id,
            "sample_csv": sample_csv,
            "sample_df": sample_df,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_create_initial_version():
    with temp_workspace() as ws:
        vm: DatasetVersionManager = ws["version_manager"]
        project_id = ws["project_id"]
        csv_path = ws["sample_csv"]

        v0 = vm.create_initial_version(project_id, csv_path)

        assert v0.version == 0
        assert v0.label == "Raw Dataset"
        assert v0.is_current is True
        assert v0.row_count == 5
        assert v0.column_count == 6
        assert "PassengerId" in v0.columns
        assert v0.file_hash is not None
        assert Path(v0.file_path).exists()

        # Raw immutable copy must exist
        raw_copy = vm._raw_dir(project_id) / "original.csv"
        assert raw_copy.exists()
        assert raw_copy.read_bytes() == Path(csv_path).read_bytes()

        print("  - create_initial_version: PASSED")


def test_cannot_create_duplicate_initial_version():
    with temp_workspace() as ws:
        vm: DatasetVersionManager = ws["version_manager"]
        project_id = ws["project_id"]
        csv_path = ws["sample_csv"]

        vm.create_initial_version(project_id, csv_path)

        try:
            vm.create_initial_version(project_id, csv_path)
            assert False, "Should raise ValueError when v0 already exists"
        except ValueError:
            pass

        print("  - cannot_create_duplicate_initial_version: PASSED")


def test_raw_dataset_immutability():
    with temp_workspace() as ws:
        vm: DatasetVersionManager = ws["version_manager"]
        project_id = ws["project_id"]
        csv_path = ws["sample_csv"]

        vm.create_initial_version(project_id, csv_path)
        raw_copy = vm._raw_dir(project_id) / "original.csv"
        original_bytes = raw_copy.read_bytes()

        # Corrupt outside source file
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("corrupted,data\n1,2\n")

        # Project raw original must remain unchanged
        assert raw_copy.read_bytes() == original_bytes

        print("  - raw_dataset_immutability: PASSED")


def test_create_version_and_trail():
    with temp_workspace() as ws:
        vm: DatasetVersionManager = ws["version_manager"]
        project_id = ws["project_id"]
        csv_path = ws["sample_csv"]

        v0 = vm.create_initial_version(project_id, csv_path)
        assert v0.is_current is True

        # Transform: remove duplicates
        df = vm.load_current_dataframe(project_id)
        assert len(df) == 5

        df_cleaned = df.drop_duplicates()
        assert len(df_cleaned) == 4

        v1 = vm.create_version(
            project_id=project_id,
            dataframe=df_cleaned,
            operation="remove_duplicates",
            parameters={"subset": None, "keep": "first"},
            impact={"rows_before": 5, "rows_after": 4, "rows_removed": 1},
            description="Removed 1 duplicate row.",
        )

        assert v1.version == 1
        assert v1.is_current is True
        assert v1.row_count == 4

        # v0 is no longer current
        v0_refreshed = vm.get_version(project_id, 0)
        assert v0_refreshed.is_current is False

        # Trail check
        trail = vm.get_trail(project_id)
        assert len(trail) == 1
        assert trail[0].from_version == 0
        assert trail[0].to_version == 1
        assert trail[0].operation == "remove_duplicates"
        assert trail[0].impact["rows_removed"] == 1

        print("  - create_version_and_trail: PASSED")


def test_set_current_version_switches_without_creating_version():
    with temp_workspace() as ws:
        vm: DatasetVersionManager = ws["version_manager"]
        project_id = ws["project_id"]
        original = ws["sample_df"]
        vm.create_initial_version(project_id, dataframe=original)
        vm.create_version(
            project_id=project_id,
            dataframe=original.iloc[:3],
            operation="drop_missing_values",
            label="Cleaned Dataset",
        )

        selected = vm.set_current_version(project_id, 0)

        assert selected.version == 0
        assert selected.is_current is True
        assert len(vm.list_versions(project_id)) == 2
        assert vm.get_version(project_id, 1).is_current is False


def test_load_dataframes_by_version():
    with temp_workspace() as ws:
        vm: DatasetVersionManager = ws["version_manager"]
        project_id = ws["project_id"]
        csv_path = ws["sample_csv"]

        vm.create_initial_version(project_id, csv_path)

        df0 = vm.load_dataframe(project_id, 0)
        assert len(df0) == 5

        df_clean = df0.drop_duplicates()
        vm.create_version(
            project_id=project_id,
            dataframe=df_clean,
            operation="remove_duplicates",
        )

        df1 = vm.load_dataframe(project_id, 1)
        assert len(df1) == 4

        # load_current_dataframe should return latest (v1)
        df_curr = vm.load_current_dataframe(project_id)
        assert len(df_curr) == 4

        print("  - load_dataframes_by_version: PASSED")


def test_rollback():
    with temp_workspace() as ws:
        vm: DatasetVersionManager = ws["version_manager"]
        project_id = ws["project_id"]
        csv_path = ws["sample_csv"]

        vm.create_initial_version(project_id, csv_path)

        df0 = vm.load_current_dataframe(project_id)
        df_v1 = df0.drop_duplicates()
        vm.create_version(
            project_id=project_id,
            dataframe=df_v1,
            operation="remove_duplicates",
        )

        assert vm.get_current_version(project_id).version == 1

        # Rollback to v0
        rolled_back = vm.rollback_to(project_id, 0)
        assert rolled_back.version == 0
        assert rolled_back.is_current is True

        # Current version is now v0
        assert vm.get_current_version(project_id).version == 0

        # v1 still exists in database and on disk
        v1 = vm.get_version(project_id, 1)
        assert v1 is not None
        assert v1.is_current is False
        assert Path(v1.file_path).exists()

        print("  - rollback: PASSED")


def test_list_versions():
    with temp_workspace() as ws:
        vm: DatasetVersionManager = ws["version_manager"]
        project_id = ws["project_id"]
        csv_path = ws["sample_csv"]

        vm.create_initial_version(project_id, csv_path)
        df = vm.load_current_dataframe(project_id)

        vm.create_version(
            project_id=project_id,
            dataframe=df.drop_duplicates(),
            operation="remove_duplicates",
        )

        versions = vm.list_versions(project_id)
        assert len(versions) == 2
        assert [v.version for v in versions] == [0, 1]
        assert vm.version_count(project_id) == 2

        print("  - list_versions: PASSED")


def test_delete_project_versions():
    with temp_workspace() as ws:
        vm: DatasetVersionManager = ws["version_manager"]
        project_id = ws["project_id"]
        csv_path = ws["sample_csv"]

        vm.create_initial_version(project_id, csv_path)
        project_dir = vm._project_dir(project_id)
        assert project_dir.exists()

        vm.delete_project_versions(project_id)

        assert not project_dir.exists()
        assert vm.list_versions(project_id) == []
        assert vm.get_trail(project_id) == []

        print("  - delete_project_versions: PASSED")


if __name__ == "__main__":
    print("=== DATASET VERSIONING & RESEARCH TRAIL TESTS ===")
    test_create_initial_version()
    test_cannot_create_duplicate_initial_version()
    test_raw_dataset_immutability()
    test_create_version_and_trail()
    test_load_dataframes_by_version()
    test_rollback()
    test_list_versions()
    test_delete_project_versions()
    print("==================================================")
    print("Semua Dataset Versioning test BERHASIL! (8/8 passed)")
    print("==================================================")
