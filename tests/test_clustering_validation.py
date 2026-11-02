"""Scientific guards for internal clustering validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

from app.ml.evaluation import MLEvaluator


class _FixedClusterer:
    def __init__(self, labels):
        self.labels = np.asarray(labels)

    def fit_predict(self, data):
        assert len(data) == len(self.labels)
        return self.labels


@pytest.mark.parametrize("labels", [[-1] * 7, [0] * 7])
def test_undefined_cluster_metrics_are_not_ranked(monkeypatch, labels):
    evaluator = MLEvaluator()
    frame = pd.DataFrame({"x": range(7), "y": range(7, 14)})
    monkeypatch.setattr(evaluator, "_get_model_instance", lambda _: _FixedClusterer(labels))

    result = evaluator._evaluate_clustering(frame, ["fixed_clusterer"])

    assert result["status"] == "FAILED"
    assert result["results"] == []
    assert result["result_count"] == 0
    assert "2 cluster" in result["failed_methods"][0]["error"]


def test_noise_is_excluded_from_internal_cluster_scores(monkeypatch):
    evaluator = MLEvaluator()
    frame = pd.DataFrame({
        "x": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 50.0],
        "y": [0.0, 0.2, 0.1, 5.0, 5.2, 5.1, 50.0],
    })
    labels = np.array([0, 0, 0, 1, 1, 1, -1])
    monkeypatch.setattr(evaluator, "_get_model_instance", lambda _: _FixedClusterer(labels))

    result = evaluator._evaluate_clustering(frame, ["fixed_clusterer"])

    transformed = evaluator._build_preprocessor(frame, requires_scaling=True).fit_transform(frame)
    expected = silhouette_score(transformed[:6], labels[:6])
    assert result["status"] == "EVALUATION"
    assert result["ranking_metric"] == "silhouette_score"
    assert result["metrics_used"] == ["silhouette_score", "calinski_harabasz", "davies_bouldin"]
    metrics = result["results"][0]["metrics"]
    assert metrics["noise_count"] == 1
    assert metrics["evaluated_rows"] == 6
    assert metrics["n_clusters"] == 2
    assert metrics["silhouette_score"] == round(expected, 4)
    assert "cluster_score" not in metrics
