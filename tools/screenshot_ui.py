"""Capture a named DORA screen at the Figma reference dimensions."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QApplication, QWidget


def _preview_dataframe() -> pd.DataFrame:
    rows = [
        [1, 0, 3, "Braund, Mr. Owen Harris", "male", 22, 1, 0, "A/5 21171", 7.25, None, "S"],
        [2, 1, 1, "Lena, Mrs. John Bradley (Florence...) ", "female", 38, 1, 0, "PC 17599", 71.283, "C85", "C"],
        [3, 1, 3, "Heikinen, Miss. Liana", "female", 26, 0, 0, "STON/O2. 3101282", 7.925, None, "S"],
        [4, 1, 1, "Futrelle, Mrs. Jacques Heath", "female", 35, 1, 0, "113803", 53.1, "C123", "S"],
        [5, 0, 3, "Allen, Mr. William Henry", "male", 35, 0, 0, "373450", 8.05, None, "S"],
        [6, 0, 3, "Moran, Mr. James", "male", None, 0, 0, "330877", 8.453, None, "Q"],
        [7, 0, 1, "McCarty, Mr. Timothy J", "male", 54, 0, 0, "17463", 51.862, "E46", "S"],
    ]
    columns = [
        "PassengerId", "Survived", "Pclass", "Name", "Gender", "Age",
        "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    repeated = pd.concat([frame] * 128, ignore_index=True).iloc[:891].copy()
    repeated.reset_index(drop=True, inplace=True)
    repeated["PassengerId"] = range(1, len(repeated) + 1)
    age_values = np.random.default_rng(2026).normal(29, 14, len(repeated))
    repeated["Age"] = np.clip(age_values, 0, 80).round(1)
    repeated["Cabin"] = repeated["Cabin"].fillna("C42")
    repeated.iloc[:7, repeated.columns.get_loc("Cabin")] = frame["Cabin"].to_numpy()
    repeated.loc[7:182, "Age"] = pd.NA
    repeated.loc[7:689, "Cabin"] = pd.NA
    repeated.loc[7:8, "Embarked"] = pd.NA
    return repeated


def _write_popup_csv() -> Path:
    handle, filename = tempfile.mkstemp(prefix="dora_figma_preview_", suffix=".csv")
    os.close(handle)
    path = Path(filename)
    _preview_dataframe().to_csv(path, index=False)
    return path


def _analysis_preview_result(window, dataframe: pd.DataFrame) -> dict:
    analyzers = {
        "profile": window.profiler.profile,
        "statistics": window.statistics.analyze,
        "missing_values": window.missing_analyzer.analyze,
        "duplicates": window.duplicate_analyzer.analyze,
        "outliers": window.outlier_analyzer.analyze,
        "correlations": window.correlation_analyzer.analyze,
        "fingerprint": window.fingerprint_analyzer.generate,
    }
    result = {name: analyzer(dataframe) for name, analyzer in analyzers.items()}
    result["data_quality"] = window.quality_diagnoser.diagnose(
        dataframe=dataframe,
        profile=result["profile"],
        missing_values=result["missing_values"],
        duplicates=result["duplicates"],
        outliers=result["outliers"],
        fingerprint=result["fingerprint"],
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--page",
        required=True,
        choices=(
            "dataset-empty", "dataset-loaded", "upload-loaded", "dataset-search", "analysis",
            "overview", "overview-empty", "landscape", "about", "popup", "editor-analysis", "ml",
        ),
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--ui-scale", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug-layout", action="store_true")
    args = parser.parse_args()

    from app.ui.main_window import MainWindow, UploadDatasetDialog

    app = QApplication.instance() or QApplication(["dora-ui-screenshot"])
    window = MainWindow()
    if args.ui_scale is not None:
        # Keep screenshot captures independent from the user's persisted setting.
        window.scale_manager._manual_scale = args.ui_scale
        window.scale_manager._scale = args.ui_scale
        window._apply_ui_scale(args.ui_scale)
    window.resize(args.width, args.height)

    popup_path: Path | None = None
    preview_path: Path | None = None
    dialog = None
    if args.page == "dataset-empty":
        window.current_dataset = None
        window.dataset_preview_page.update_dataset(None)
        window.open_upload_page()
    elif args.page == "dataset-loaded":
        preview_path = _write_popup_csv()
        window.current_dataset = _preview_dataframe()
        window.current_file_path = preview_path
        window.dataset_preview_page.update_dataset(
            window.current_dataset, "Titanic-dataset.csv"
        )
        window.datasets_tabs.setCurrentWidget(window.dataset_preview_page)
        window._show_page("dataset")
    elif args.page == "upload-loaded":
        preview_path = _write_popup_csv()
        window.current_dataset = _preview_dataframe()
        window.current_file_path = preview_path
        window.dataset_preview_page.update_dataset(
            window.current_dataset, "Titanic-dataset.csv"
        )
        window.open_upload_page()
    elif args.page == "editor-analysis":
        window.current_dataset = _preview_dataframe()
        window.editor_analysis_page.set_dataset(
            window.current_dataset, "Titanic-dataset.csv"
        )
        window.open_editor_analysis_page()
        # The managed screenshot environment cannot write the user's AppData
        # autosave path; keep that environment-only warning out of the design capture.
        window.editor_analysis_page._autosave_timer.stop()
    elif args.page == "ml":
        window.current_dataset = _preview_dataframe()
        window.current_file_path = Path("Titanic-Dataset.csv")
        window.analysis_result = _analysis_preview_result(
            window, window.current_dataset
        )
        window.ml_page.update_dataset()
        window.open_ml_page()
    elif args.page == "dataset-search":
        window.open_dataset_search_page()
        from app.dataset_search.huggingface import DatasetFile, DatasetSearchResult

        preview_files = [
            DatasetFile(
                filename="student-performance.csv (Auto Extract)",
                size_bytes=106803,
            )
        ]
        window.dataset_search_page.search_input.setText("student")
        window.dataset_search_page.source_combo.setCurrentIndex(
            window.dataset_search_page.source_combo.findData("all")
        )
        window.dataset_search_page._on_search_finished([
            DatasetSearchResult(
                dataset_id="kaggle:whenamancodes/student-performance",
                author="Aman Chauhan",
                description="Predict student performance in secondary education (high school)",
                downloads=45339,
                likes=520,
                tags=["tabular"],
                files=preview_files,
                match_reason="Kaggle dataset (520 votes, usability 100%)",
            ),
            DatasetSearchResult(
                dataset_id="kaggle:spscientist/student-performance-in-exams",
                author="Spescientist",
                description="Marks secured by student in various subjects",
                downloads=280114,
                likes=3210,
                tags=["tabular"],
                files=preview_files,
                match_reason="Kaggle dataset (3120 votes, usability 100%)",
            ),
        ])
    elif args.page == "analysis":
        window.current_dataset = _preview_dataframe()
        window.current_file_path = Path("Titanic-Dataset.csv")
        window.analysis_result = _analysis_preview_result(
            window, window.current_dataset
        )
        window.open_analysis_page()
    elif args.page == "overview-empty":
        window.current_dataset = None
        window.current_file_path = None
        window.dashboard.update_dataset(None, "")
        window.dashboard.update_workflow_state(False, False, False)
        window.open_dashboard()
    elif args.page == "overview":
        overview_dataframe = _preview_dataframe()
        rng = np.random.default_rng(2026)
        overview_dataframe["Name"] = [f"Passenger {index:04d}" for index in range(len(overview_dataframe))]
        overview_dataframe["Ticket"] = [f"TKT-{index:07d}" for index in range(len(overview_dataframe))]
        overview_dataframe["Cabin"] = [f"C{index:03d}" for index in range(len(overview_dataframe))]
        overview_dataframe.loc[:6, "Cabin"] = _preview_dataframe().loc[:6, "Cabin"]
        overview_dataframe.loc[7:689, "Cabin"] = pd.NA
        overview_dataframe["Age"] = rng.integers(18, 66, len(overview_dataframe)).astype(float)
        overview_dataframe.loc[7:182, "Age"] = pd.NA
        overview_dataframe.loc[6, "Age"] = pd.NA
        overview_dataframe.loc[10, "Fare"] = 512.3
        window.current_dataset = overview_dataframe
        window.current_file_path = None
        window.analysis_result = _analysis_preview_result(window, overview_dataframe)
        window.dashboard._filename = "Titanic-dataset.csv"
        window.dashboard._preview_file_size = "60.3 KB"
        window.dashboard._preview_readiness_label = "Readiness 55/100"
        window.ml_page.last_task_result = {
            "primary_task": {
                "label": "Binary Clasification",
                "task": "binary classification",
                "confidence": 80,
                "target": "Survived",
            }
        }
        window.ml_page.last_method_result = {
            "recommendations": [
                {"method": "Random Forest", "score": 82.83, "score_type": "accuracy"},
                {"method": "Gradient Boosting", "score": 82.1, "score_type": "accuracy"},
                {"method": "Logistic Regression", "score": 80.7, "score_type": "accuracy"},
            ]
        }
        overview_research = {
            "papers": [{"year": 2026} for _ in range(83)],
            "gaps": {"summary": {"gap_count": 4}},
        }
        window.research_page.research_result = overview_research
        window.dashboard.update_dataset(overview_dataframe, "Titanic-dataset.csv")
        window.dashboard.update_workflow_state(True, True, True)
        window.open_dashboard()
    elif args.page == "landscape":
        window.research_page.research_result = {
            "papers": [],
            "landscape": {
                "paper_count": 83,
                "summary": {
                    "papers_analyzed": 83,
                    "latest_publication_year": 2026,
                    "dominant_method": "Random Forest",
                },
                "methods": [
                    {"name": "Random Forest", "percentage": 13.25},
                    {"name": "Desicion Tree", "percentage": 10.84},
                    {"name": "Logistic Regression", "percentage": 10.84},
                    {"name": "SVM", "percentage": 3.61},
                    {"name": "Gradient Boosting", "percentage": 2.41},
                    {"name": "Linear Regression", "percentage": 2.41},
                ],
                "top_venues": [
                    {"name": "Zenodo", "percentage": 8.43},
                    {"name": "arXiv", "percentage": 6.02},
                    {"name": "Rosa P", "percentage": 4.82},
                    {"name": "BCP Business & Management", "percentage": 2.41},
                ],
            },
        }
        window.open_landscape_tool()
    elif args.page == "about":
        window.open_about()
    elif args.page == "popup":
        popup_path = _write_popup_csv()
        dialog = UploadDatasetDialog(window, popup_path)
        window.open_upload_page()

    window.show()
    window.resize(args.width, args.height)
    window.move(0, 0)

    if dialog is not None:
        dialog.resize(dialog.dialog_width, dialog.dialog_height)
        dialog.findChild(QWidget, "uploadDialogFileName").setText("Titanic-dataset.csv")
        dialog.findChild(QWidget, "uploadDialogFileDetails").setText(
            "60.3 KB . 891 rows x 12 column detected"
        )
        dialog.findChild(QWidget, "uploadDialogInput").setText("Titanic-dataset.csv")
        dialog.move(
            (args.width - dialog.width() + 1) // 2,
            (args.height - dialog.height() + 1) // 2,
        )
        dialog.show()
        dialog.raise_()

    output = args.output or (PROJECT_ROOT / "screenshots" / f"app_{args.page}.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    def capture() -> None:
        if args.debug_layout and args.page == "overview":
            for name, widget in (
                ("topbar", window.topbar),
                ("overview_topbar_frame", window.overview_topbar_frame),
                ("overview_header", window.dashboard.findChild(QWidget, "overviewHeader")),
                ("overview_key_findings", window.dashboard.key_findings),
                ("overview_pipeline", window.dashboard.pipeline_card),
                ("overview_dataset_card", window.dashboard.summary_cards[0]),
            ):
                point = widget.mapTo(window, widget.rect().topLeft())
                print(
                    f"{name}: ({point.x()}, {point.y()}, {widget.width()}, "
                    f"{widget.height()}) visible={widget.isVisible()}"
                )
        if args.debug_layout and args.page == "dataset-empty":
            page = window.upload_page
            for index in range(page.upload_root_layout.count()):
                root_item = page.upload_root_layout.itemAt(index)
                print(f"root item {index}: {root_item.geometry().getRect()}")
                if index == 0 and root_item.layout():
                    header = root_item.layout()
                    for header_index in range(header.count()):
                        print(f"header item {header_index}: {header.itemAt(header_index).geometry().getRect()}")
                    title_block = page.upload_title_block
                    for title_index in range(title_block.count()):
                        print(f"title item {title_index}: {title_block.itemAt(title_index).geometry().getRect()}")
            for name in (
                "upload_card", "upload_dropzone", "upload_tips_card",
                "upload_search_card", "upload_button",
            ):
                widget = getattr(page, name, None)
                if widget is not None:
                    position = widget.mapTo(window, widget.rect().topLeft())
                    print(f"{name}: ({position.x()}, {position.y()}, {widget.width()}, {widget.height()})")
            for index, step in enumerate(page.upload_steps):
                position = step.mapTo(window, step.rect().topLeft())
                print(f"step {index + 1}: ({position.x()}, {position.y()}, {step.width()}, {step.height()})")
            for label in page.findChildren(QWidget):
                if label.objectName() in {
                    "pageTitle", "pageSubtitle", "uploadPageSectionTitle",
                    "uploadDropIcon", "uploadDropTitle", "uploadDropDescription",
                    "uploadDropSizeLimit", "uploadBrowseLink",
                }:
                    position = label.mapTo(window, label.rect().topLeft())
                    print(f"{label.objectName()}: ({position.x()}, {position.y()}, {label.width()}, {label.height()})")
            for card in (page.upload_search_card, page.upload_tips_card):
                for child in card.findChildren(QWidget):
                    if child.objectName() in {
                        "uploadSearchIcon", "cardTitle", "cardDescription",
                        "uploadSearchButton", "uploadTipsText",
                    }:
                        position = child.mapTo(window, child.rect().topLeft())
                        print(f"{child.objectName()}: ({position.x()}, {position.y()}, {child.width()}, {child.height()})")
            drop_layout = page.upload_dropzone.layout()
            for index in range(drop_layout.count()):
                print(f"drop item {index}: {drop_layout.itemAt(index).geometry().getRect()}")
        if args.debug_layout and args.page == "dataset-loaded":
            page = window.dataset_preview_page
            root_margins = page.root_layout.contentsMargins()
            container_margins = page.page_scroll.widget().contentsMargins()
            print(
                f"layout_margins: root=({root_margins.left()}, {root_margins.top()}, "
                f"{root_margins.right()}, {root_margins.bottom()}) "
                f"container=({container_margins.left()}, {container_margins.top()}, "
                f"{container_margins.right()}, {container_margins.bottom()}) "
                f"spacing={page.root_layout.spacing()}",
                flush=True,
            )
            for index in range(page.root_layout.count()):
                item = page.root_layout.itemAt(index)
                print(
                    f"root item {index}: {item.geometry().getRect()} "
                    f"widget={item.widget().objectName() if item.widget() else None}",
                    flush=True,
                )
            for name, widget in (
                ("page_stack", window.page_stack),
                ("datasets_page", window.datasets_page),
                ("datasets_tabs", window.datasets_tabs),
                ("datasets_tab_pages", window.datasets_tab_pages),
                ("page_scroll", page.page_scroll),
                ("page_container", page.page_scroll.widget()),
            ):
                point = widget.mapTo(window, widget.rect().topLeft())
                print(
                    f"{name}: ({point.x()}, {point.y()}, {widget.width()}, {widget.height()})",
                    flush=True,
                )
            for name in (
                "header", "overview_layout", "file_card", "tabs_scroll",
                "content_stack", "preview_card", "preview_title", "table",
                "table_heading_widget", "file_identity", "columns_card", "columns_scroll", "footer",
            ):
                widget = getattr(page, name, None)
                if widget is not None:
                    if isinstance(widget, QWidget):
                        point = widget.mapTo(window, widget.rect().topLeft())
                        geometry = (point.x(), point.y(), widget.width(), widget.height())
                    else:
                        geometry = widget.geometry().getRect()
                    print(f"{name}: {geometry}", flush=True)
            preview_layout = page.preview_card.layout()
            for index in range(preview_layout.count()):
                print(f"preview item {index}: {preview_layout.itemAt(index).geometry().getRect()}")
            print(f"empty state visible: {page.empty_state.isVisible()}")
            file_layout = page.file_card.layout()
            for index in range(file_layout.count()):
                print(f"file item {index}: {file_layout.itemAt(index).geometry().getRect()}")
        if args.debug_layout and args.page == "upload-loaded":
            page = window.upload_page.loaded_page
            print(f"ui_scale={QApplication.instance().property('uiScale')}", flush=True)
            sidebar = window.sidebar_container
            point = sidebar.mapTo(window, sidebar.rect().topLeft())
            print(
                f"sidebar: ({point.x()}, {point.y()}, {sidebar.width()}, {sidebar.height()}) "
                f"visible={sidebar.isVisible()}",
                flush=True,
            )
            for key in ("dashboard", "dataset", "analysis"):
                button = window.nav_buttons.get(key)
                if button is not None:
                    print(
                        f"nav_{key}: visible={button.isVisible()} text="
                        f"{button.findChild(QWidget, 'navText').text() if button.findChild(QWidget, 'navText') else ''}",
                        flush=True,
                    )
            for name in (
                "steps", "dropzone", "file_summary", "preview_panel", "table",
                "page_size_combo", "continue_button",
            ):
                widget = getattr(page, name, None)
                if widget is not None:
                    point = widget.mapTo(window, widget.rect().topLeft())
                    print(
                        f"{name}: ({point.x()}, {point.y()}, {widget.width()}, {widget.height()})",
                        flush=True,
                    )
        if args.debug_layout and args.page == "analysis":
            page = window.analysis_page
            for name, widget in (
                ("analysis_page", page),
                ("title", page._title_label),
                ("subtitle", page._subtitle_label),
                ("run_button", page.analyze_button),
                ("dataset_banner", page.dataset_label),
                ("overview", page.findChild(QWidget, "analysisOverview")),
                ("visualization_card", page.visualization_card),
                ("fingerprint_card", page.findChild(QWidget, "analysisFingerprintCard")),
                ("quality_card", page.findChild(QWidget, "analysisQualitySummaryCard")),
                ("chart_type", page.chart_type),
                ("chart_column", page.chart_column),
                ("chart_canvas", page.chart_canvas),
            ):
                if widget is not None:
                    position = widget.mapTo(window, widget.rect().topLeft())
                    print(f"{name}: ({position.x()}, {position.y()}, {widget.width()}, {widget.height()})")
            for widget in page.findChildren(QWidget):
                if widget.objectName() in {
                    "analysisPanelTitle", "analysisPanelSubtitle", "analysisQualitySubtitle",
                    "analysisFingerprintValue", "analysisQualityCritical",
                    "analysisQualityWarning", "analysisQualityGood",
                }:
                    position = widget.mapTo(window, widget.rect().topLeft())
                    print(f"{widget.objectName()}: ({position.x()}, {position.y()}, {widget.width()}, {widget.height()})")
        if args.debug_layout and args.page == "dataset-search":
            page = window.dataset_search_page
            print(f"dataset_search_viewport_width: {page.scroll.viewport().width()}")
            for key, button in window.nav_buttons.items():
                for name, widget in (
                    ("button", button),
                    ("icon", button.findChild(QWidget, "navIcon")),
                    ("text", button.findChild(QWidget, "navText")),
                ):
                    if widget is not None:
                        point = widget.mapTo(window, widget.rect().topLeft())
                        print(f"nav_{key}_{name}: ({point.x()}, {point.y()}, {widget.width()}, {widget.height()})")
            for key, widget in window.topbar_links.items():
                point = widget.mapTo(window, widget.rect().topLeft())
                print(f"topbar_{key}: ({point.x()}, {point.y()}, {widget.width()}, {widget.height()})")
            for name, widget in (
                ("sidebar_group_workspace", window.sidebar_group_cards[0]),
                ("sidebar_group_research", window.sidebar_group_cards[1]),
                ("topbar_divider", window.topbar_divider),
                ("topbar_version", window.topbar_version_container),
            ):
                point = widget.mapTo(window, widget.rect().topLeft())
                print(f"{name}: ({point.x()}, {point.y()}, {widget.width()}, {widget.height()})")
            for widget in window.topbar_version_container.findChildren(QWidget):
                if widget.objectName() in {"topbarVersionIcon", "topbarVersion"}:
                    point = widget.mapTo(window, widget.rect().topLeft())
                    print(f"{widget.objectName()}: ({point.x()}, {point.y()}, {widget.width()}, {widget.height()})")
            for widget in page.findChildren(QWidget):
                if widget.objectName() in {
                    "datasetSearchCard", "datasetSearchInput", "datasetProviderCombo",
                    "datasetSearchResults", "datasetResultCard", "datasetResultTitle",
                    "datasetResultId", "datasetResultDescription", "datasetResultMatch",
                    "datasetResultMeta", "datasetResultTag", "datasetFileSelect",
                    "datasetAnalyzeButton", "secondaryButton", "scoreBadge", "cardDescription",
                } or widget.property("datasetSearch"):
                    position = widget.mapTo(window, widget.rect().topLeft())
                    print(f"{widget.objectName()}: ({position.x()}, {position.y()}, {widget.width()}, {widget.height()})")
        if args.debug_layout and args.page == "editor-analysis":
            page = window.editor_analysis_page
            for name, widget in (
                ("editor_page", page),
                ("header", page.editor_header_container),
                ("title_block", page.title_block_container),
                ("info_banner", page.info_banner),
                ("info_banner_container", page.info_banner_container),
                ("workspace", page.editor_card),
                ("steps_card", page.steps_card),
                ("output_tabs", page.output_tabs),
            ):
                point = widget.mapTo(window, widget.rect().topLeft())
                print(
                    f"{name}: ({point.x()}, {point.y()}, {widget.width()}, {widget.height()}) "
                    f"min={widget.minimumWidth()}x{widget.minimumHeight()} "
                    f"max={widget.maximumWidth()}x{widget.maximumHeight()}"
                )
            margins = page.root_layout.contentsMargins()
            print(
                f"root_layout: spacing={page.root_layout.spacing()} margins="
                f"({margins.left()}, {margins.top()}, {margins.right()}, {margins.bottom()})"
            )
            for index in range(page.root_layout.count()):
                item = page.root_layout.itemAt(index)
                rect = item.geometry()
                print(
                    f"root_item_{index}: ({rect.x()}, {rect.y()}, {rect.width()}, {rect.height()}) "
                    f"hint={item.sizeHint().width()}x{item.sizeHint().height()}"
                )
        if args.debug_layout and dialog is not None:
            position = dialog.mapTo(window, dialog.rect().topLeft())
            print(f"dialog: ({position.x()}, {position.y()}, {dialog.width()}, {dialog.height()})")
            for widget in dialog.findChildren(QWidget):
                name = widget.objectName()
                if name.startswith("uploadDialog") or name in {"primaryButton", "uploadDatasetDialog"}:
                    point = widget.mapTo(dialog, widget.rect().topLeft())
                    print(f"{name}: ({point.x()}, {point.y()}, {widget.width()}, {widget.height()})")
        image = window.grab().toImage()
        if dialog is not None:
            painter = QPainter(image)
            painter.fillRect(image.rect(), QColor(0, 0, 0, 64))
            left = (args.width - dialog.width()) // 2
            top = (args.height - dialog.height()) // 2
            painter.drawPixmap(left, top, dialog.grab())
            painter.end()
        image.save(str(output))
        print(f"Screenshot saved: {output}", flush=True)
        for temporary_path in (popup_path, preview_path):
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        if dialog is not None:
            dialog.close()
        window._session_data_cleaned = True
        window.close()
        QTimer.singleShot(350, app.quit)

    QTimer.singleShot(700, capture)
    app.exec()
    if popup_path is not None:
        popup_path.unlink(missing_ok=True)
    if preview_path is not None:
        preview_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
