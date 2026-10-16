"""Tests for Stage 6: Multi-format export engine and standalone execution verification."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from app.ml.graph.exporter import (
    export_jupyter_notebook,
    export_model_pkl,
    export_project_zip,
    export_python_script,
    export_runs_csv,
)


@pytest.fixture
def sample_graph():
    return {
        "nodes": [
            {"id": "load", "spec_id": "load_dataset"},
            {"id": "split", "spec_id": "split", "parameters": {"test_size": 0.25, "random_state": 42}},
            {"id": "prep", "spec_id": "preprocess", "parameters": {"scaler": "standard"}},
            {"id": "model", "spec_id": "model:decision_tree_classifier", "parameters": {"max_depth": 5, "random_state": 42}},
            {"id": "train", "spec_id": "train", "parameters": {"target": "target"}},
            {"id": "eval", "spec_id": "evaluate", "parameters": {"target": "target"}},
        ],
        "connections": [
            ["load", "table", "split", "table"],
            ["split", "train", "prep", "train"],
            ["split", "test", "prep", "test"],
            ["prep", "train", "train", "train"],
            ["model", "estimator", "train", "estimator"],
            ["train", "model", "eval", "model"],
            ["prep", "test", "eval", "test"],
        ],
    }


def test_export_python_script_execution_standalone(sample_graph, tmp_path):
    # 1. Create synthetic dataset CSV
    df = pd.DataFrame({
        "feature1": [float(i) for i in range(40)],
        "feature2": [float(i * 1.5) for i in range(40)],
        "target": [0] * 20 + [1] * 20,
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    # 2. Export Python script
    script_content = export_python_script(sample_graph, target_column="target", task="classification")
    script_path = tmp_path / "run_pipeline.py"
    script_path.write_text(script_content, encoding="utf-8")

    # 3. Execute standalone script via subprocess (python script.py --data dataset.csv)
    proc = subprocess.run(
        [sys.executable, str(script_path), "--data", str(csv_path), "--target", "target"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"Script execution failed: {proc.stderr}"

    # 4. Verify metrics in output
    metrics = json.loads(proc.stdout.strip())
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_export_jupyter_notebook(sample_graph):
    nb = export_jupyter_notebook(sample_graph, target_column="target", task="classification")
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 5
    # Verify markdown and code cells present
    types = [c["cell_type"] for c in nb["cells"]]
    assert "markdown" in types
    assert "code" in types


def test_export_model_pkl(tmp_path):
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit([[1, 2], [3, 4], [5, 6]], [0, 1, 0])

    pkl_path = tmp_path / "model.pkl"
    saved_path = export_model_pkl(clf, pkl_path)

    loaded = joblib.load(saved_path)
    assert hasattr(loaded, "predict")
    assert loaded.max_depth == 3


def test_export_runs_csv(tmp_path):
    runs = [
        {
            "id": "run_01",
            "graph_version_id": "v1",
            "status": "success",
            "duration_ms": 120.0,
            "random_state": 42,
            "metrics": {"accuracy": 0.88, "f1": 0.85},
            "notes": "baseline",
        },
        {
            "id": "run_02",
            "graph_version_id": "v2",
            "status": "success",
            "duration_ms": 140.0,
            "random_state": 42,
            "metrics": {"accuracy": 0.94, "f1": 0.92},
            "notes": "tuned",
        },
    ]
    out_csv = tmp_path / "runs.csv"
    df = export_runs_csv(runs, out_csv)
    assert len(df) == 2
    assert "metric_accuracy" in df.columns
    assert out_csv.exists()


def test_export_project_zip(sample_graph, tmp_path):
    project = {"id": "p1", "name": "Test Project"}
    runs = [{"id": "r1", "status": "success", "metrics": {"accuracy": 0.9}}]
    data_df = pd.DataFrame({"x": [1, 2], "target": [0, 1]})

    zip_path = tmp_path / "project_export.zip"
    res_path = export_project_zip(
        project_dict=project,
        graph_dict=sample_graph,
        runs=runs,
        output_zip_path=zip_path,
        dataset_df=data_df,
    )

    assert res_path.exists()
    with zipfile.ZipFile(res_path, "r") as zf:
        names = zf.namelist()
        assert "project.json" in names
        assert "graph.json" in names
        assert "pipeline.py" in names
        assert "notebook.ipynb" in names
        assert "runs_summary.csv" in names
        assert "dataset.csv" in names

