import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_graph_nodes_endpoint():
    response = client.get("/api/graph/nodes")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert len(data["nodes"]) >= 30


def test_validate_graph_endpoint_detects_leakage():
    # Preprocessing directly from load_dataset before split = data leakage!
    payload = {
        "nodes": [
            {"id": "load", "spec_id": "load_dataset"},
            {"id": "prep", "spec_id": "preprocess"},
            {"id": "split", "spec_id": "split"},
        ],
        "connections": [
            ["load", "table", "prep", "train"],
            ["prep", "train", "split", "table"],
        ],
    }
    response = client.post("/api/graph/validate", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert len(res["leakages"]) >= 1
    assert "Data leakage terdeteksi" in res["leakages"][0]["message"]
    assert len(res["warnings"]) >= 1


def test_run_graph_endpoint_with_dataframe():
    # Build valid pipeline: load -> split -> train -> evaluate
    payload = {
        "graph": {
            "nodes": [
                {"id": "load", "spec_id": "load_dataset"},
                {"id": "split", "spec_id": "split", "parameters": {"test_size": 0.25, "random_state": 42}},
                {"id": "model", "spec_id": "model:logistic_regression"},
                {"id": "train", "spec_id": "train", "parameters": {"target": "target", "task": "classification"}},
                {"id": "eval", "spec_id": "evaluate", "parameters": {"target": "target", "task": "classification"}},
            ],
            "connections": [
                ["load", "table", "split", "table"],
                ["split", "train", "train", "train"],
                ["model", "estimator", "train", "estimator"],
                ["train", "model", "eval", "model"],
                ["split", "test", "eval", "test"],
            ],
        },
        "dataframe": [
            {"feat1": 1.0, "feat2": 2.0, "target": 0},
            {"feat1": 1.2, "feat2": 2.1, "target": 0},
            {"feat1": 1.1, "feat2": 1.9, "target": 0},
            {"feat1": 1.3, "feat2": 2.2, "target": 0},
            {"feat1": 5.0, "feat2": 6.0, "target": 1},
            {"feat1": 5.2, "feat2": 5.9, "target": 1},
            {"feat1": 5.1, "feat2": 6.1, "target": 1},
            {"feat1": 5.3, "feat2": 6.2, "target": 1},
        ],
        "random_state": 42,
    }
    response = client.post("/api/graph/run", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert "eval" in res["results"]
    metrics = res["results"]["eval"]["metrics"]["metrics"]
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert res["random_state"] == 42


def test_run_graph_with_auto_fix_leakage():
    # Pipeline with leakage: load -> preprocess -> split -> train -> evaluate
    # Auto-fix should rewire preprocess after split
    payload = {
        "graph": {
            "nodes": [
                {"id": "load", "spec_id": "load_dataset"},
                {"id": "prep", "spec_id": "preprocess", "parameters": {"scaler": "standard"}},
                {"id": "split", "spec_id": "split", "parameters": {"test_size": 0.25, "random_state": 42}},
                {"id": "model", "spec_id": "model:decision_tree_classifier"},
                {"id": "train", "spec_id": "train", "parameters": {"target": "target"}},
                {"id": "eval", "spec_id": "evaluate", "parameters": {"target": "target"}},
            ],
            "connections": [
                ["load", "table", "prep", "train"],
                ["load", "table", "split", "table"],
                ["split", "train", "train", "train"],
                ["model", "estimator", "train", "estimator"],
                ["train", "model", "eval", "model"],
                ["split", "test", "eval", "test"],
            ],
        },
        "dataframe": [
            {"feat1": i * 1.0, "feat2": i * 2.0, "target": i % 2}
            for i in range(20)
        ],
        "auto_fix_leakage": True,
    }
    response = client.post("/api/graph/run", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["fixes_applied"] >= 1

