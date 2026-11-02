"""Manual ML choices reach the evaluator without changing the source dataset."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.analyzer.fingerprint import DatasetFingerprint
from app.ml.intelligence import MLIntelligenceEngine
from app.ui.ml_run_options_dialog import MLRunOptionsDialog


def _frame():
    x = np.arange(100, dtype=float)
    return pd.DataFrame({"useful": x, "unused": x[::-1], "target": (x > 50).astype(int)})


def test_manual_options_include_features_split_and_folds():
    app = QApplication.instance() or QApplication([])
    frame = _frame()
    dialog = MLRunOptionsDialog(frame, {"primary_task": {
        "task": "binary_classification", "target": "target", "label": "Klasifikasi biner"
    }})
    dialog.task_combo.setCurrentIndex(dialog.task_combo.findData("binary_classification"))
    dialog.target_combo.setCurrentIndex(dialog.target_combo.findData("target"))
    dialog.test_percent.setValue(25)
    dialog.cv_folds.setValue(4)
    for index in range(dialog.feature_list.count()):
        item = dialog.feature_list.item(index)
        if item.data(Qt.ItemDataRole.UserRole) == "unused":
            item.setCheckState(Qt.CheckState.Unchecked)

    options = dialog.options()

    assert options["target_override"] == "target"
    assert options["selected_features"] == ["useful"]
    assert options["test_size"] == 0.25
    assert options["cv_folds"] == 4
    dialog.close()
    assert app is not None


def test_clustering_does_not_select_suggested_label_as_feature():
    app = QApplication.instance() or QApplication([])
    frame = _frame()
    dialog = MLRunOptionsDialog(frame, {"primary_task": {
        "task": "binary_classification", "target": "target", "label": "Klasifikasi biner"
    }, "target_candidates": [{"name": "target"}]})
    dialog.task_combo.setCurrentIndex(dialog.task_combo.findData("clustering"))

    assert "target" not in dialog.options()["selected_features"]
    assert "useful" in dialog.options()["selected_features"]
    dialog.close()
    assert app is not None


def test_manual_engine_uses_selected_features_and_split():
    frame = _frame()
    fingerprint = {"representation": DatasetFingerprint().generate_representation(frame)}
    result = MLIntelligenceEngine().analyze(
        frame, fingerprint,
        task_override="binary_classification", target_override="target",
        selected_features=["useful"], test_size=0.25, cv_folds=4,
        requested_methods=["logistic_regression"],
    )

    assert result["status"] == "ML_INTELLIGENCE"
    evaluation = result["evaluation"]
    assert evaluation["dataset"]["features"] == 1
    assert evaluation["dataset"]["test_rows"] == 25
    assert evaluation["folds"] == 4
    assert evaluation["config"]["selected_features"] == ["useful"]
    assert "unused" in frame.columns  # The input remains untouched.
