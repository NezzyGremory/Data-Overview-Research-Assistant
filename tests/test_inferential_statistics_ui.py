import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from PySide6.QtWidgets import QApplication, QDialog, QTextBrowser

from app.analyzer.inferential_statistics import InferentialStatisticsAnalyzer
from app.ui import main_window as main_window_module
from app.ui.main_window import AnalysisPage, InferentialTestDialog


def _application():
    return QApplication.instance() or QApplication([])


def _dataframe():
    return pd.DataFrame({
        "measurement": [1.1, 1.8, 2.0, 2.2, 2.6, 2.9, 4.2, 4.5, 4.8, 5.0, 5.4, 5.8],
        "group": ["A"] * 6 + ["B"] * 6,
        "paired": [0.8, 1.5, 1.8, 2.1, 2.2, 2.7, 3.8, 4.1, 4.4, 4.6, 5.0, 5.3],
        "category": ["x", "y"] * 6,
    })


def test_inferential_dialog_runs_user_selected_test_and_shows_assumptions():
    app = _application()
    dialog = InferentialTestDialog(_dataframe(), InferentialStatisticsAnalyzer())
    dialog.test_combo.setCurrentIndex(dialog.test_combo.findData("welch_t"))
    dialog.column_combo.setCurrentIndex(dialog.column_combo.findData("measurement"))
    dialog.second_column_combo.setCurrentIndex(dialog.second_column_combo.findData("group"))

    dialog._run_test()

    assert dialog.result["status"] == "COMPLETED"
    assert dialog.result["test_id"] == "welch_t"
    assert dialog.save_button.isEnabled()
    assert "H0:" in dialog.result_browser.toPlainText()
    assert "Asumsi dan batas interpretasi" in dialog.result_browser.toPlainText()
    dialog.close()
    app.processEvents()


def test_dialog_disables_second_column_for_single_column_normality_test():
    app = _application()
    dialog = InferentialTestDialog(_dataframe(), InferentialStatisticsAnalyzer())
    dialog.test_combo.setCurrentIndex(dialog.test_combo.findData("shapiro"))

    assert dialog.second_column_combo.isHidden()
    assert dialog.column_combo.findData("measurement") >= 0
    assert dialog.column_combo.findData("category") == -1
    dialog.close()
    app.processEvents()


class _MainWindowStub:
    def __init__(self, dataframe):
        self.current_dataset = dataframe
        self.analysis_result = {}
        self.current_project_id = 42
        self.repository = Mock()
        self.ml_page = None


def test_analysis_page_saves_and_renders_accepted_result():
    app = _application()
    main = _MainWindowStub(_dataframe())
    page = AnalysisPage(main)
    result = {
        "status": "COMPLETED",
        "test_id": "shapiro",
        "test_name": "Shapiro-Wilk",
        "variables": ["measurement"],
        "sample_size": 12,
        "statistic": 0.98,
        "p_value": 0.8,
        "decision": "GAGAL_MENOLAK_H0",
        "hypotheses": {"null": "Normal.", "alternative": "Tidak normal."},
        "assumptions": [{"name": "Observasi", "status": "USER_ASSUMED", "detail": "Sampel."}],
    }

    class AcceptedDialog(InferentialTestDialog):
        def __init__(self, *_args):
            self.result = result

        def exec(self):
            return QDialog.Accepted

    with patch.object(main_window_module, "InferentialTestDialog", AcceptedDialog):
        page.run_inferential_test()

    assert main.analysis_result["inferential_statistics"] == [result]
    main.repository.save_workflow_section.assert_called_once_with(
        42, "analysis", main.analysis_result
    )
    summary = page.findChild(QTextBrowser, "inferentialResultSummary")
    assert summary is not None
    assert "Shapiro-Wilk" in summary.toPlainText()
    page.close()
    app.processEvents()
