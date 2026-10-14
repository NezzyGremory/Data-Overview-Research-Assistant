import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from PySide6.QtWidgets import QApplication, QLabel

from app.ui.missing_values_dialogs import DropMissingOptionsDialog, DropMissingPreviewDialog
from app.analyzer.cleaner import drop_missing


def test_dialog_defaults_to_auto_and_exposes_all_strategies():
    app = QApplication.instance() or QApplication([])
    dialog = DropMissingOptionsDialog(pd.DataFrame({"reading": [0, 10, 20]}))

    options = dialog.options()

    assert options["zero_as_missing"] == "auto"
    assert options["zero_mode"] is None
    assert options["strategy"] == "drop_rows"
    assert {dialog.strategy_combo.itemData(i) for i in range(dialog.strategy_combo.count())} == {
        "drop_rows", "drop_cols", "impute", "report"
    }
    dialog.close()
    app.processEvents()


def test_preview_dialog_renders_report_and_warning_details():
    app = QApplication.instance() or QApplication([])
    result = drop_missing(
        pd.DataFrame({"reading": [0, 0, 100, 120, 140]}),
        zero_as_missing="auto",
    )
    dialog = DropMissingPreviewDialog(result)

    text = " ".join(label.text() for label in dialog.findChildren(QLabel))

    assert "Strategi: drop_rows" in text
    assert "nilai 0 diperlakukan" in text
    dialog.close()
    app.processEvents()
