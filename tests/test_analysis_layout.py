import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from PySide6.QtWidgets import QApplication, QLabel, QTableWidget

from app.ui.main_window import AnalysisPage


class _MainWindowStub:
    def __init__(self, dataframe):
        self.current_dataset = dataframe
        self.ml_page = None


def _make_page(dataframe):
    app = QApplication.instance() or QApplication([])
    return app, AnalysisPage(_MainWindowStub(dataframe))


def test_profiling_card_contains_size_metrics_and_five_row_preview_only():
    dataframe = pd.DataFrame({"age": range(8), "target": [0, 1] * 4})
    app, page = _make_page(dataframe)

    page.add_cell_profiling(
        {"rows": 8, "columns": 2, "numeric_columns": 2, "categorical_columns": 0, "memory_usage": 128},
        {"schema": "legacy schema"},
        [{"column": "age"}],
    )

    card = page.result_layout.itemAt(page.result_layout.count() - 1).widget()
    tables = card.findChildren(QTableWidget)
    labels = [label.text().lower() for label in card.findChildren(QLabel)]
    assert len(tables) == 1
    assert tables[0].rowCount() == 5
    assert tables[0].columnCount() == 2
    assert not any("schema" in label or "target" in label for label in labels)

    page.close()
    app.processEvents()


def test_readiness_layout_keeps_quality_summary_in_overview_and_correlations_in_hints():
    dataframe = pd.DataFrame({"feature_a": [1, 2, 3, 4], "feature_b": [2, 4, 6, 8]})
    app, page = _make_page(dataframe)
    for method_name in (
        "add_cell_data_quality",
        "add_cell_missing_values",
        "add_cell_outliers",
        "add_versioning_section",
        "add_cell_gemini",
    ):
        setattr(page, method_name, lambda *_args, **_kwargs: None)

    readiness = {
        "problem_definition": {"summary": "Target not detected", "status": "Warning"},
        "quality_summary": {"summary": "1 missing cell", "status": "Issue", "missing_cells": 1},
        "feature_selection": {
            "summary": "One correlated pair",
            "status": "Warning",
            "correlated_pairs": [
                {"left": "feature_a", "right": "feature_b", "correlation": 1.0}
            ],
        },
    }
    page.display_results(
        {
            "profile": {"rows": 4, "columns": 2, "numeric_columns": 2, "categorical_columns": 0},
            "statistics": [],
            "missing_values": [],
            "duplicates": {},
            "outliers": [],
            "correlations": pd.DataFrame({"feature_a": [1.0, 1.0], "feature_b": [1.0, 1.0]}),
            "fingerprint": {},
            "data_quality": [],
            "ml_readiness": readiness,
        }
    )

    labels = [label.text() for label in page.findChildren(QLabel)]
    assert labels.count("Kualitas data") == 1
    assert labels.count("Feature Selection Hints") == 1
    assert "quality_summary" in page._ml_readiness_cards
    assert "feature_selection" in page._ml_readiness_cards
    assert not any("correlation" == label.strip().lower() for label in labels)

    feature_card = page._ml_readiness_cards["feature_selection"]
    table = feature_card.findChild(QTableWidget)
    assert table is not None
    assert table.rowCount() == 1
    assert table.item(0, 1).text() == "feature_a / feature_b"

    page.close()
    app.processEvents()
