"""Tests for Stage 5: Task detection + dynamic template graphs."""

import numpy as np
import pandas as pd
import pytest

from app.ml.graph.engine import GraphEngine
from app.ml.graph.task_detector import (
    delete_custom_template,
    detect_task,
    generate_template_graphs,
    list_custom_templates,
    profile_dataset,
    save_custom_template,
)


def test_profile_dataset():
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 100.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "cat": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    prof = profile_dataset(df, "target")
    assert prof["row_count"] == 12
    assert prof["col_count"] == 3
    assert "num" in prof["numeric_cols"]
    assert "cat" in prof["categorical_cols"]
    assert prof["target_info"]["target_type"] == "binary"
    assert prof["outlier_count"] >= 1


def test_detect_task_binary_classification():
    df = pd.DataFrame({
        "feature": range(20),
        "label": [0] * 10 + [1] * 10,
    })
    res = detect_task(df, "label")
    assert res["task_type"] == "binary_classification"
    assert res["confidence"] > 0.9


def test_detect_task_multiclass():
    df = pd.DataFrame({
        "feature": range(30),
        "label": ["setosa"] * 10 + ["versicolor"] * 10 + ["virginica"] * 10,
    })
    res = detect_task(df, "label")
    assert res["task_type"] == "multiclass_classification"


def test_detect_task_regression():
    np.random.seed(42)
    df = pd.DataFrame({
        "feat1": np.random.randn(50),
        "price": np.random.randn(50) * 1000 + 5000,
    })
    res = detect_task(df, "price")
    assert res["task_type"] == "regression"


def test_detect_task_clustering_no_target():
    df = pd.DataFrame({"x": range(20), "y": range(20)})
    res = detect_task(df)
    assert res["task_type"] == "clustering"


def test_detect_task_time_series():
    dates = pd.date_range("2024-01-01", periods=30)
    df = pd.DataFrame({
        "date": dates,
        "sales": np.random.rand(30) * 100,
    })
    res = detect_task(df, "sales")
    assert res["task_type"] == "time_series"


def test_generate_template_graphs_and_run():
    # Generate binary classification templates
    df = pd.DataFrame({
        "f1": [float(i) for i in range(40)],
        "f2": [float(i * 2) for i in range(40)],
        "target": [0] * 20 + [1] * 20,
    })
    templates = generate_template_graphs(df, "target")
    assert len(templates) >= 2

    # Check baseline template execution
    baseline = templates[0]
    engine = GraphEngine()
    for n in baseline["graph"]["nodes"]:
        engine.add_node(n["spec_id"], n["id"], n.get("parameters"), tuple(n.get("position", (0, 0))))
    for conn in baseline["graph"]["connections"]:
        engine.connect(*conn)

    results = engine.run(df)
    assert "eval" in results
    assert "metrics" in results["eval"]["metrics"]


def test_custom_templates_lifecycle():
    sample_graph = {"nodes": [{"id": "load"}], "connections": []}
    saved = save_custom_template("Custom RF Pipeline", sample_graph, "My custom template")
    assert saved["id"] == "custom_custom_rf_pipeline"

    templates = list_custom_templates()
    assert any(t["id"] == saved["id"] for t in templates)

    deleted = delete_custom_template(saved["id"])
    assert deleted is True
