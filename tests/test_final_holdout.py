"""The final supervised score must be separate from model selection."""

import numpy as np
import pandas as pd

from app.ml.evaluation import MLEvaluator
from app.reports.generator import ReportGenerator


def test_classification_has_independent_final_test():
    rng = np.random.default_rng(7)
    values = rng.normal(size=100)
    data = pd.DataFrame({"feature": values, "target": (values > 0).astype(int)})

    result = MLEvaluator().evaluate(
        data, target="target", task="binary_classification",
        methods=["logistic_regression"],
    )

    assert result["status"] == "EVALUATION"
    assert result["validation"] == "stratified_k_fold"
    assert result["dataset"]["selection_rows"] == 80
    assert result["dataset"]["test_rows"] == 20
    assert result["final_test"]["status"] == "EVALUATED"
    assert result["final_test"]["method_id"] == "logistic_regression"
    assert "f1_score" in result["final_test"]["metrics"]


def test_regression_has_independent_final_test():
    values = np.arange(100, dtype=float)
    data = pd.DataFrame({"feature": values, "target": values * 2.0 + 1.0})

    result = MLEvaluator().evaluate(
        data, target="target", task="regression", methods=["linear_regression"],
    )

    assert result["final_test"]["status"] == "EVALUATED"
    assert result["final_test"]["test_rows"] == 20
    assert result["final_test"]["metrics"]["mae"] == 0.0


def test_small_dataset_marks_final_test_unavailable():
    data = pd.DataFrame({"feature": range(20), "target": [0, 1] * 10})

    result = MLEvaluator().evaluate(
        data, target="target", task="binary_classification",
        methods=["logistic_regression"],
    )

    assert result["status"] == "EVALUATION"
    assert result["final_test"]["status"] == "NOT_AVAILABLE"
    assert result["dataset"]["test_rows"] == 0


def test_report_uses_actual_fold_count_and_final_test():
    evaluation = {
        "validation": "stratified_k_fold", "folds": 3,
        "ranking_metric": "f1_score",
        "results": [{"method_id": "logistic_regression", "method": "Logistic Regression",
                     "metrics": {"f1_score": 90.0, "accuracy": 91.0}, "folds": 3}],
        "final_test": {"status": "EVALUATED", "method": "Logistic Regression",
                       "train_rows": 80, "test_rows": 20,
                       "metrics": {"f1_score": 88.0, "accuracy": 89.0}},
    }

    html = ReportGenerator().generate(dataset_name="Fold Report", ml_data={"evaluation": evaluation})

    assert "3-Fold Cross Validation (stratified)" in html
    assert "F1 tertimbang: 90.0000" in html
    assert "F1 tertimbang: 88.0000" in html
    assert "20 baris uji" in html


def test_target_copy_is_excluded_and_infinite_feature_is_imputed():
    values = np.arange(100, dtype=float)
    values[12] = np.inf
    target = (np.arange(100) > 50).astype(int)
    data = pd.DataFrame({"feature": values, "target_copy": target, "target": target})

    result = MLEvaluator().evaluate(
        data, target="target", task="binary_classification",
        methods=["logistic_regression"],
    )

    assert result["status"] == "EVALUATION"
    assert result["dataset"]["features"] == 1
    assert result["excluded_target_copies"] == ["target_copy"]
    assert result["final_test"]["status"] == "EVALUATED"


def test_only_target_copy_is_rejected():
    target = [0, 1] * 25
    data = pd.DataFrame({"target_copy": target, "target": target})

    result = MLEvaluator().evaluate(
        data, target="target", task="binary_classification",
        methods=["logistic_regression"],
    )

    assert result["status"] == "ERROR"
    assert "identik dengan target" in result["message"]
