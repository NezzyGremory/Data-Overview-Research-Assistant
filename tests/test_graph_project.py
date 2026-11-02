"""Tests for Stage 3: Project + History (SQLite)."""

import pandas as pd
import pytest

from app.core.exceptions import DatasetLoadError
from app.ml.graph.project import (
    GraphHistoryStack,
    ProjectManager,
    compute_dataframe_hash,
    compute_file_hash,
)


@pytest.fixture
def pm():
    # Use SQLite in-memory for testing
    return ProjectManager(":memory:")


def test_project_crud(pm):
    p = pm.create_project("Experiment Alpha", "First test project")
    assert p["name"] == "Experiment Alpha"
    assert "id" in p

    fetched = pm.get_project(p["id"])
    assert fetched is not None
    assert fetched["name"] == "Experiment Alpha"

    all_p = pm.list_projects()
    assert len(all_p) == 1

    deleted = pm.delete_project(p["id"])
    assert deleted is True
    assert pm.get_project(p["id"]) is None


def test_dataset_registration_and_hash(pm):
    p = pm.create_project("Project Data")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    reg = pm.register_dataset(p["id"], df)

    assert reg["row_count"] == 3
    assert reg["column_count"] == 2
    assert reg["columns"] == ["a", "b"]
    assert len(reg["file_hash"]) == 64

    # Hash should be deterministic
    h2 = compute_dataframe_hash(df)
    assert reg["file_hash"] == h2


@pytest.mark.parametrize("suffix", ["csv", "jsonl", "parquet"])
def test_file_registration_uses_shared_loader_for_tabular_formats(pm, tmp_path, suffix):
    project = pm.create_project("Imported dataset")
    frame = pd.DataFrame({"feature": [1, 2, 3], "target": ["a", "b", "a"]})
    path = tmp_path / f"example.{suffix}"
    if suffix == "csv":
        frame.to_csv(path, index=False)
    elif suffix == "jsonl":
        frame.to_json(path, orient="records", lines=True)
    else:
        frame.to_parquet(path, index=False)

    registration = pm.register_dataset(project["id"], path)

    assert registration["row_count"] == 3
    assert registration["column_count"] == 2
    assert registration["columns"] == ["feature", "target"]
    assert registration["file_hash"] == compute_file_hash(path)


def test_file_registration_rejects_unreadable_dataset(pm, tmp_path):
    project = pm.create_project("Invalid dataset")
    path = tmp_path / "invalid.parquet"
    path.write_text("not parquet", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Gagal membaca dataset"):
        pm.register_dataset(project["id"], path)

    with pm._get_connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM datasets WHERE project_id = ?", (project["id"],)
        ).fetchone()[0]
    assert count == 0


def test_graph_versioning_and_branching(pm):
    p = pm.create_project("Versioned ML")
    graph_v1 = {"nodes": [{"id": "load"}], "connections": []}
    v1 = pm.save_graph_version(p["id"], graph_v1, message="Initial graph")
    assert v1["version_number"] == 1
    assert v1["branch_name"] == "main"

    graph_v2 = {"nodes": [{"id": "load"}, {"id": "split"}], "connections": []}
    v2 = pm.save_graph_version(p["id"], graph_v2, message="Added split", parent_version_id=v1["id"])
    assert v2["version_number"] == 2
    assert v2["parent_version_id"] == v1["id"]

    # Branching
    v_branch = pm.branch_graph(p["id"], v2["id"], new_branch_name="experiment-xgboost")
    assert v_branch["branch_name"] == "experiment-xgboost"
    assert v_branch["version_number"] == 1
    assert v_branch["parent_version_id"] == v2["id"]


def test_run_recording_and_comparison(pm):
    p = pm.create_project("Runs Project")
    v1 = pm.save_graph_version(p["id"], {"nodes": []}, message="v1")
    df1 = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    hash1 = compute_dataframe_hash(df1)

    run1 = pm.record_run(
        project_id=p["id"],
        graph_version_id=v1["id"],
        dataset_hash=hash1,
        metrics={"accuracy": 0.85, "f1": 0.82},
        confusion_matrix=[[10, 2], [1, 7]],
        duration_ms=120.5,
        random_state=42,
        notes="Baseline decision tree",
    )
    assert run1["status"] == "success"
    assert run1["metrics"]["accuracy"] == 0.85

    # Run 2 with improved metrics
    run2 = pm.record_run(
        project_id=p["id"],
        graph_version_id=v1["id"],
        dataset_hash=hash1,
        metrics={"accuracy": 0.92, "f1": 0.90},
        duration_ms=210.0,
        random_state=42,
        notes="Hyperparameter tuned",
    )

    # Compare runs
    comp = pm.compare_runs(run1["id"], run2["id"])
    assert comp["same_dataset"] is True
    assert comp["metrics_comparison"]["accuracy"]["delta"] == 0.07
    assert comp["metrics_comparison"]["f1"]["delta"] == 0.08
    assert comp["duration_delta_ms"] == pytest.approx(89.5, 0.1)


def test_dataset_hash_change_warning(pm):
    p = pm.create_project("Hash Test")
    v1 = pm.save_graph_version(p["id"], {"nodes": []})
    hash_old = "a" * 64
    run = pm.record_run(p["id"], v1["id"], hash_old, {"accuracy": 0.8})

    # Same hash -> no warning
    changed, msg = pm.check_dataset_hash(run["id"], hash_old)
    assert changed is False

    # Different hash -> warning flagged
    hash_new = "b" * 64
    changed, msg = pm.check_dataset_hash(run["id"], hash_new)
    assert changed is True
    assert "Peringatan" in msg


def test_graph_history_stack_undo_redo():
    stack = GraphHistoryStack(max_depth=5)
    s1 = {"nodes": ["a"]}
    s2 = {"nodes": ["a", "b"]}
    s3 = {"nodes": ["a", "b", "c"]}

    stack.push(s1)
    stack.push(s2)
    stack.push(s3)

    assert stack.can_undo() is True
    assert stack.can_redo() is False

    # Undo to s2
    res2 = stack.undo()
    assert res2 == s2
    assert stack.can_redo() is True

    # Undo to s1
    res1 = stack.undo()
    assert res1 == s1

    # Redo to s2
    redo2 = stack.redo()
    assert redo2 == s2

