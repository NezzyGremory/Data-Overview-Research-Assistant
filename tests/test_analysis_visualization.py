import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from matplotlib.axes import Axes
from PySide6.QtWidgets import QApplication, QComboBox

from app.analyzer.relationships import RelationshipAnalyzer
from app.ui.main_window import AnalysisPage


class _MainWindowStub:
    def __init__(self, dataframe):
        self.current_dataset = dataframe


def _make_page(dataframe):
    app = QApplication.instance() or QApplication([])
    page = AnalysisPage(_MainWindowStub(dataframe))
    return app, page


def test_histogram_uses_loaded_dataframe_values(monkeypatch):
    dataframe = pd.DataFrame({"measurement": [2, 2, 4, 8]})
    app, page = _make_page(dataframe)
    captured = {}

    def capture_hist(self, values, *args, **kwargs):
        captured["values"] = list(values)

    monkeypatch.setattr(Axes, "hist", capture_hist)
    page._visualization_dataframe = dataframe
    page._build_visualization_panel()
    page.chart_type.setCurrentIndex(page.chart_type.findData("histogram"))

    assert captured["values"] == dataframe["measurement"].tolist()
    page.close()
    app.processEvents()


def test_categorical_chart_handles_many_categories_and_actual_counts():
    dataframe = pd.DataFrame({"group": [f"group-{i % 30}" for i in range(90)]})
    app, page = _make_page(dataframe)
    page._visualization_dataframe = dataframe
    page._build_visualization_panel()
    page.chart_type.setCurrentIndex(page.chart_type.findData("bar"))

    axis = page.chart_figure.axes[0]
    assert len(axis.patches) == page.MAX_CATEGORIES
    assert sum(patch.get_height() for patch in axis.patches) == 60
    page.close()
    app.processEvents()


def test_incompatible_columns_show_clear_empty_state():
    dataframe = pd.DataFrame({"constant": [1, 1, 1], "label": ["only", "only", "only"]})
    app, page = _make_page(dataframe)
    page._visualization_dataframe = dataframe
    page._visualization_correlations = pd.DataFrame()
    page._visualization_missing_values = [
        {"column": "constant", "missing_count": 0}
    ]
    page._build_visualization_panel()

    assert page.chart_column.count() == 0
    assert "Tidak ada kolom numerik" in page.chart_message.text()

    page.chart_type.setCurrentIndex(page.chart_type.findData("correlation"))
    assert "sedikitnya dua kolom numerik" in page.chart_message.text()

    page.chart_type.setCurrentIndex(page.chart_type.findData("missing"))
    assert "Tidak ada nilai kosong" in page.chart_message.text()
    assert not page.chart_canvas.isVisible()
    page.close()
    app.processEvents()


def test_advanced_eda_plots_render_for_numeric_and_categorical_columns():
    dataframe = pd.DataFrame(
        {
            "x": [float(i) for i in range(1, 41)],
            "y": [float(i * 2 + (i % 3)) for i in range(1, 41)],
            "z": [float((i % 7) + i / 4) for i in range(1, 41)],
            "group": ["control" if i % 2 else "case" for i in range(1, 41)],
            "site": ["north" if i % 4 < 2 else "south" for i in range(1, 41)],
        }
    )
    app, page = _make_page(dataframe)
    page._visualization_dataframe = dataframe
    page._visualization_correlations = dataframe[["x", "y", "z"]].corr()
    relationships = RelationshipAnalyzer()
    page._visualization_covariance = relationships.covariance(dataframe)
    page._visualization_categorical_associations = relationships.categorical_associations(dataframe)
    page._visualization_missing_values = []
    page._build_visualization_panel()

    for chart_type in (
        "violin", "density", "ecdf", "qq", "scatter", "grouped_box", "pair",
        "covariance", "categorical_association",
    ):
        index = page.chart_type.findData(chart_type)
        assert index >= 0, f"Missing chart option: {chart_type}"
        page.chart_type.setCurrentIndex(index)
        assert page.chart_message.isHidden(), chart_type
        assert not page.chart_canvas.isHidden(), chart_type
        if chart_type == "pair":
            assert len(page.chart_figure.axes) == 9
        elif chart_type in ("covariance", "categorical_association"):
            assert len(page.chart_figure.axes) == 2  # plot plus colorbar
        else:
            assert len(page.chart_figure.axes) == 1

    page.chart_type.setCurrentIndex(page.chart_type.findData("scatter"))
    assert page.chart_column_label.text() == "X"
    page.close()
    app.processEvents()


def test_scatter_and_distribution_plots_skip_non_finite_values():
    dataframe = pd.DataFrame(
        {
            "x": [1.0, 2.0, float("inf"), 4.0, 5.0],
            "y": [2.0, 5.0, 8.0, float("-inf"), 11.0],
        }
    )
    app, page = _make_page(dataframe)
    page._visualization_dataframe = dataframe
    page._build_visualization_panel()

    histogram_index = page.chart_type.findData("histogram")
    page.chart_type.setCurrentIndex(histogram_index)
    assert page.chart_message.isHidden()
    assert len(page.chart_figure.axes[0].patches) > 0

    scatter_index = page.chart_type.findData("scatter")
    page.chart_type.setCurrentIndex(scatter_index)
    assert page.chart_message.isHidden()
    assert len(page.chart_figure.axes[0].collections) == 1
    page.close()
    app.processEvents()


def test_qq_plot_uses_the_full_bounded_plot_sample(monkeypatch):
    import app.ui.main_window as main_window_module

    dataframe = pd.DataFrame({"measurement": range(25_001)})
    app, page = _make_page(dataframe)
    page._visualization_dataframe = dataframe
    captured = {}

    def capture_probplot(values, *args, **kwargs):
        captured["values"] = values
        kwargs["plot"].plot([0, 1], [0, 1])
        kwargs["plot"].plot([0, 1], [0, 1])

    monkeypatch.setattr(main_window_module.scipy_stats, "probplot", capture_probplot)
    page._build_visualization_panel()
    page.chart_type.setCurrentIndex(page.chart_type.findData("qq"))

    assert len(captured["values"]) == len(page._plot_sample(dataframe))
    assert captured["values"][0] == 0
    assert captured["values"][-1] == page._plot_sample(dataframe)["measurement"].iloc[-1]
    page.close()
    app.processEvents()
