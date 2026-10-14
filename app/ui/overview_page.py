"""Overview page matching the DORA Figma frame."""

from pathlib import Path
import sys

import pandas as pd
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ui.style_utils import apply_card_shadow


def _asset_path(relative_path: str) -> Path:
    if getattr(sys, "frozen", False):
        root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        return root / "app" / "ui" / relative_path
    return Path(__file__).resolve().parent / relative_path


class DashboardPage(QWidget):
    """Overview dashboard with links to the existing application pages."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._dataframe = getattr(main_window, "current_dataset", None)
        self._filename = "Dataset"
        self._workflow = {
            "has_dataset": self._dataframe is not None,
            "analysis_done": False,
            "ml_done": False,
        }
        self._build_ui()
        self.refresh_summary()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setObjectName("overviewScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        canvas = QWidget()
        canvas.setObjectName("overviewCanvas")
        layout = QVBoxLayout(canvas)
        self._canvas_layout = layout
        layout.setContentsMargins(36, 4, 42, 24)
        layout.setSpacing(0)

        header = QWidget()
        header.setObjectName("overviewHeader")
        header.setFixedHeight(66)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 3, 0)
        header_layout.setSpacing(12)
        title_block = QVBoxLayout()
        title_block.setContentsMargins(0, 0, 0, 0)
        title_block.setSpacing(0)
        title = QLabel("Overview")
        title.setObjectName("overviewTitle")
        subtitle = QLabel("Everything about your current research in one")
        subtitle.setObjectName("overviewSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header_layout.addLayout(title_block, 1)
        self.report_button = QPushButton("Generated Research Report")
        self.report_button.setObjectName("overviewReportButton")
        self.report_button.setIcon(QIcon(str(_asset_path("assets/icons/overview-report.svg"))))
        self.report_button.setIconSize(QSize(30, 30))
        self.report_button.setFixedSize(401, 66)
        self.report_button.setCursor(Qt.PointingHandCursor)
        self.report_button.clicked.connect(self.main_window.open_report_tool)
        header_layout.addWidget(self.report_button, 0, Qt.AlignTop)
        layout.addWidget(header)
        layout.addSpacing(24)

        self.key_findings = QFrame()
        self.key_findings.setObjectName("overviewKeyFindings")
        self.key_findings.setFixedHeight(88)
        apply_card_shadow(self.key_findings)
        findings_layout = QHBoxLayout(self.key_findings)
        findings_layout.setContentsMargins(31, 12, 20, 12)
        findings_layout.setSpacing(14)
        info_icon = QLabel()
        info_icon.setObjectName("overviewInfoIcon")
        info_path = _asset_path("assets/icons/about.svg")
        if info_path.exists():
            info_icon.setPixmap(QPixmap(str(info_path)).scaled(38, 38, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            info_icon.setText("ⓘ")
        info_icon.setFixedSize(40, 40)
        findings_layout.addWidget(info_icon, 0, Qt.AlignVCenter)
        self.key_findings_text = QLabel()
        self.key_findings_text.setObjectName("overviewKeyFindingsText")
        self.key_findings_text.setWordWrap(True)
        self.key_findings_text.setTextFormat(Qt.RichText)
        findings_layout.addWidget(self.key_findings_text, 1, Qt.AlignVCenter)
        layout.addWidget(self.key_findings)
        layout.addSpacing(7)

        self.pipeline_card = QFrame()
        self.pipeline_card.setObjectName("overviewPipelineCard")
        self.pipeline_card.setMinimumHeight(185)
        apply_card_shadow(self.pipeline_card)
        pipeline_layout = QVBoxLayout(self.pipeline_card)
        pipeline_layout.setContentsMargins(18, 14, 18, 12)
        pipeline_layout.setSpacing(21)
        pipeline_layout.setAlignment(Qt.AlignTop)
        pipeline_header = QHBoxLayout()
        pipeline_header.setContentsMargins(0, 0, 0, 0)
        pipeline_header.setSpacing(12)
        pipeline_title = QLabel("Dataset workflow")
        pipeline_title.setObjectName("overviewPipelineTitle")
        pipeline_title.setFixedWidth(207)
        pipeline_header.addWidget(pipeline_title)
        self.pipeline_progress = QProgressBar()
        self.pipeline_progress.setObjectName("overviewPipelineProgress")
        self.pipeline_progress.setRange(0, 3)
        self.pipeline_progress.setValue(0)
        self.pipeline_progress.setTextVisible(False)
        self.pipeline_progress.setFixedHeight(10)
        self.pipeline_progress.setFixedWidth(402)
        pipeline_header.addWidget(self.pipeline_progress)
        self.pipeline_count = QLabel("0 of 7 step completed")
        self.pipeline_count.setObjectName("overviewPipelineCount")
        pipeline_header.addWidget(self.pipeline_count)
        pipeline_layout.addLayout(pipeline_header)

        self.pipeline_stage_titles = (
            "Dataset", "Analysis", "ML Intelligence",
        )
        self._pipeline_stage_weights = (1, 1, 1)
        self.pipeline_stages = [
            self._make_pipeline_stage(index + 1, title)
            for index, title in enumerate(self.pipeline_stage_titles)
        ]
        self.pipeline_stages[-1].setProperty("kind", "report")
        self.pipeline_stage_layout = QGridLayout()
        self.pipeline_stage_layout.setContentsMargins(0, 0, 0, 0)
        self.pipeline_stage_layout.setHorizontalSpacing(9)
        self.pipeline_stage_layout.setVerticalSpacing(9)
        pipeline_layout.addLayout(self.pipeline_stage_layout)
        layout.addWidget(self.pipeline_card)
        layout.addSpacing(24)

        self.summary_cards = [
            self._make_dataset_card(),
            self._make_quality_card(),
            self._make_ml_card(),
        ]
        self.summary_cards_layout = QGridLayout()
        self.summary_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.summary_cards_layout.setHorizontalSpacing(22)
        self.summary_cards_layout.setVerticalSpacing(16)
        layout.addLayout(self.summary_cards_layout)
        layout.addStretch(1)
        scroll.setWidget(canvas)
        outer.addWidget(scroll)
        self._summary_columns = None
        self._pipeline_columns = None
        self._apply_responsive_layout(self.width())

    @staticmethod
    def _make_pipeline_stage(number, title):
        stage = QFrame()
        stage.setObjectName("overviewPipelineStage")
        stage.setProperty("state", "waiting")
        stage.setMinimumHeight(81)
        row = QHBoxLayout(stage)
        row.setContentsMargins(8, 9, 6, 8)
        row.setSpacing(8)
        number_label = QLabel(f"{number:02d}")
        number_label.setObjectName("overviewStageNumber")
        number_label.setFixedSize(30, 30)
        number_label.setAlignment(Qt.AlignCenter)
        row.addWidget(number_label, 0, Qt.AlignTop)
        labels = QVBoxLayout()
        labels.setContentsMargins(0, 0, 0, 0)
        labels.setSpacing(0)
        title_label = QLabel(title)
        title_label.setObjectName("overviewStageTitle")
        title_label.setWordWrap(True)
        status = QLabel("Waiting")
        status.setObjectName("overviewStageStatus")
        labels.addWidget(title_label)
        labels.addWidget(status)
        row.addLayout(labels, 1)
        stage._status_label = status
        return stage

    @staticmethod
    def _make_card(object_name):
        card = QFrame()
        card.setObjectName(object_name)
        card.setMinimumHeight(258)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)
        apply_card_shadow(card)
        return card, layout

    @staticmethod
    def _heading_row(title, icon_path):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        icon = QLabel()
        icon.setObjectName("overviewCardIcon")
        icon.setFixedSize(32, 32)
        path = _asset_path(icon_path)
        if path.exists():
            icon.setPixmap(QPixmap(str(path)).scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(icon, 0, Qt.AlignVCenter)
        label = QLabel(title)
        label.setObjectName("overviewCardTitle")
        layout.addWidget(label, 1, Qt.AlignVCenter)
        return row, layout

    def _make_details_button(self, callback):
        button = QPushButton("View details")
        button.setObjectName("overviewDetailsButton")
        button.setIcon(QIcon(str(_asset_path("assets/icons/overview-arrow-right.svg"))))
        button.setIconSize(QSize(10, 10))
        button.setLayoutDirection(Qt.RightToLeft)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(callback)
        return button

    def _make_dataset_card(self):
        card, layout = self._make_card("overviewDatasetCard")
        heading, heading_layout = self._heading_row(
            "Dataset", "assets/icons/dataset.svg"
        )
        self.dataset_loaded_badge = QLabel("Loaded")
        self.dataset_loaded_badge.setObjectName("overviewLoadedBadge")
        heading_layout.addWidget(self.dataset_loaded_badge, 0, Qt.AlignVCenter)
        layout.addWidget(heading)
        self.dataset_name_label = QLabel("No dataset loaded")
        self.dataset_name_label.setObjectName("overviewDatasetName")
        self.dataset_name_label.setWordWrap(True)
        layout.addWidget(self.dataset_name_label)
        self.dataset_metric_values = {}
        metrics = QHBoxLayout()
        metrics.setSpacing(7)
        for key, title in (("rows", "ROWS"), ("columns", "COLUMNS"), ("size", "SIZE"), ("duplicates", "DUPLICATES")):
            tile = QFrame()
            tile.setObjectName("overviewDatasetMetric")
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(7, 4, 4, 3)
            tile_layout.setSpacing(0)
            caption = QLabel(title)
            caption.setObjectName("overviewMetricTitle")
            value = QLabel("—")
            value.setObjectName("overviewMetricValue")
            tile_layout.addWidget(caption)
            tile_layout.addWidget(value)
            metrics.addWidget(tile, 1)
            self.dataset_metric_values[key] = value
        layout.addLayout(metrics)
        column_title = QLabel("COLUMNS TYPE")
        column_title.setObjectName("overviewSmallHeading")
        layout.addWidget(column_title)
        self.column_type_bar = QWidget()
        self.column_type_bar.setObjectName("overviewColumnTypeBar")
        self.column_type_bar.setFixedHeight(8)
        self.column_type_bar_layout = QHBoxLayout(self.column_type_bar)
        self.column_type_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.column_type_bar_layout.setSpacing(0)
        self.column_type_segments = []
        self.column_type_legend_labels = []
        for key in ("id", "target", "numeric", "categorical", "text"):
            segment = QFrame()
            segment.setObjectName("overviewColumnTypeSegment")
            segment.setProperty("kind", key)
            self.column_type_bar_layout.addWidget(segment, 1)
            self.column_type_segments.append(segment)
            legend = QLabel()
            legend.setObjectName("overviewColumnLegend")
            legend.setProperty("kind", key)
            self.column_type_legend_labels.append(legend)
        layout.addWidget(self.column_type_bar)
        self.column_type_legend = QHBoxLayout()
        self.column_type_legend.setContentsMargins(0, 0, 0, 0)
        self.column_type_legend.setSpacing(5)
        for label in self.column_type_legend_labels:
            self.column_type_legend.addWidget(label)
        self.column_type_legend.addStretch(1)
        layout.addLayout(self.column_type_legend)
        layout.addStretch(1)
        self.dataset_details_button = self._make_details_button(
            self.main_window.open_datasets_page
        )
        layout.addWidget(self.dataset_details_button, 0, Qt.AlignLeft)
        return card

    def _make_quality_card(self):
        card, layout = self._make_card("overviewQualityCard")
        heading, heading_layout = self._heading_row(
            "Data Quality", "assets/icons/overview-quality-folder.svg"
        )
        self.quality_readiness_badge = QLabel("Readiness")
        self.quality_readiness_badge.setObjectName("overviewReadinessBadge")
        heading_layout.addWidget(self.quality_readiness_badge, 0, Qt.AlignVCenter)
        layout.addWidget(heading)
        self.quality_missing = QLabel("Missing values: —")
        self.quality_outliers = QLabel("Outliers detected in —")
        self.quality_duplicates = QLabel("Duplicate rows: —")
        for label in (self.quality_missing, self.quality_outliers, self.quality_duplicates):
            label.setObjectName("overviewQualityLine")
            label.setWordWrap(True)
            layout.addWidget(label)
        most_title = QLabel("MOST MISSING")
        most_title.setObjectName("overviewSmallHeading")
        layout.addWidget(most_title)
        self.quality_most_missing = QLabel("—")
        self.quality_most_missing.setObjectName("overviewMostMissing")
        self.quality_most_missing.setWordWrap(True)
        layout.addWidget(self.quality_most_missing)
        layout.addStretch(1)
        self.quality_details_button = self._make_details_button(
            self.main_window.open_analysis_page
        )
        layout.addWidget(self.quality_details_button, 0, Qt.AlignLeft)
        return card

    def _make_ml_card(self):
        card, layout = self._make_card("overviewMLCard")
        heading, _ = self._heading_row(
            "ML Intelligence", "assets/icons/ml-intelligence.svg"
        )
        layout.addWidget(heading)
        self.ml_tags = QHBoxLayout()
        self.ml_tags.setContentsMargins(0, 0, 0, 0)
        self.ml_tags.setSpacing(7)
        self.ml_task_badge = QLabel("—")
        self.ml_task_badge.setObjectName("overviewMLTaskBadge")
        self.ml_confidence_badge = QLabel("—")
        self.ml_confidence_badge.setObjectName("overviewConfidenceBadge")
        self.ml_target_badge = QLabel("Target: —")
        self.ml_target_badge.setObjectName("overviewTargetBadge")
        for badge in (self.ml_task_badge, self.ml_confidence_badge, self.ml_target_badge):
            self.ml_tags.addWidget(badge)
        self.ml_tags.addStretch(1)
        layout.addLayout(self.ml_tags)
        models_title = QLabel("TOP RECOMMENDED MODELS")
        models_title.setObjectName("overviewSmallHeading")
        layout.addWidget(models_title)
        self.model_rows = []
        for _ in range(3):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            name = QLabel("—")
            name.setObjectName("overviewModelName")
            bar = QProgressBar()
            bar.setObjectName("overviewModelProgress")
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            value = QLabel("—")
            value.setObjectName("overviewModelValue")
            row_layout.addWidget(name, 3)
            row_layout.addWidget(bar, 4)
            row_layout.addWidget(value, 0, Qt.AlignRight)
            layout.addWidget(row)
            self.model_rows.append((name, bar, value))
        layout.addStretch(1)
        self.ml_details_button = self._make_details_button(
            self.main_window.open_ml_page
        )
        layout.addWidget(self.ml_details_button, 0, Qt.AlignLeft)
        return card

    def _reflow_summary_cards(self, columns):
        while self.summary_cards_layout.count():
            self.summary_cards_layout.takeAt(0)
        for index, card in enumerate(self.summary_cards):
            self.summary_cards_layout.addWidget(card, index // columns, index % columns)
        for column in range(columns):
            weights = (471, 442, 465) if columns == 3 else (1,) * columns
            self.summary_cards_layout.setColumnStretch(column, weights[column])

    def _reflow_pipeline(self, columns):
        while self.pipeline_stage_layout.count():
            self.pipeline_stage_layout.takeAt(0)
        for index, stage in enumerate(self.pipeline_stages):
            row, column = divmod(index, columns)
            self.pipeline_stage_layout.addWidget(stage, row, column)
            self.pipeline_stage_layout.setColumnStretch(
                column,
                self._pipeline_stage_weights[index] if columns == len(self.pipeline_stages) else 1,
            )

    def _apply_responsive_layout(self, width):
        scale_manager = getattr(self.main_window, "scale_manager", None)
        scale = scale_manager.get_scale() if scale_manager is not None else 1.0
        px = scale_manager.px if scale_manager is not None else lambda value: round(value)
        logical_width = width / max(scale, 0.01)
        summary_columns = 3 if logical_width >= 1360 else 2 if logical_width >= 900 else 1
        if self._summary_columns != summary_columns:
            self._summary_columns = summary_columns
            self._reflow_summary_cards(summary_columns)
        pipeline_columns = 3 if logical_width >= 1100 else 2 if logical_width >= 620 else 1
        if self._pipeline_columns != pipeline_columns:
            self._pipeline_columns = pipeline_columns
            self._reflow_pipeline(pipeline_columns)
        compact = logical_width < 1000
        self._canvas_layout.setContentsMargins(
            px(18 if compact else 36), px(4), px(18 if compact else 42),
            px(20 if compact else 24),
        )
        if logical_width < 1100:
            self.report_button.setFixedSize(
                px(min(300, max(180, logical_width // 3))),
                px(56 if logical_width < 800 else 66),
            )
            self.report_button.setIconSize(QSize(px(24), px(24)))
        else:
            self.report_button.setFixedSize(px(401), px(66))
            self.report_button.setIconSize(QSize(px(30), px(30)))
        if pipeline_columns == len(self.pipeline_stages):
            self.pipeline_card.setMinimumHeight(px(185))
        else:
            rows = (len(self.pipeline_stages) + pipeline_columns - 1) // pipeline_columns
            self.pipeline_card.setMinimumHeight(px(38 + rows * 81 + (rows - 1) * 9 + 26))
        for card in self.summary_cards:
            card.setMinimumHeight(px(258 if summary_columns == 3 else 240))

    def apply_scale(self, _scale=None):
        """Reflow dashboard grids after the UI scale changes."""
        self._apply_responsive_layout(self.width())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    @staticmethod
    def _format_percent(value):
        try:
            return f"{float(value):.1f}".replace(".", ",")
        except (TypeError, ValueError):
            return "0,0"

    def update_dataset(self, dataframe, filename):
        if dataframe is None:
            return
        self._dataframe = dataframe
        self._filename = str(filename or "Dataset")
        self._workflow["has_dataset"] = True
        self._workflow["analysis_done"] = bool(getattr(self.main_window, "analysis_result", {}))
        self.refresh_summary()

    def update_workflow_state(self, has_dataset, analysis_done, ml_done):
        self._workflow.update({
            "has_dataset": bool(has_dataset),
            "analysis_done": bool(analysis_done),
            "ml_done": bool(ml_done),
        })
        self.refresh_summary()

    def _column_types(self, dataframe):
        counts = {"id": 0, "target": 0, "numeric": 0, "categorical": 0, "text": 0}
        if dataframe is None:
            return counts
        analysis = getattr(self.main_window, "analysis_result", {}) or {}
        fingerprint = analysis.get("fingerprint", {}) if isinstance(analysis, dict) else {}
        representation = fingerprint.get("representation", {}) if isinstance(fingerprint, dict) else {}
        candidates = representation.get("target_candidates", []) if isinstance(representation, dict) else []
        target = None
        ml_page = getattr(self.main_window, "ml_page", None)
        task_result = getattr(ml_page, "last_task_result", {}) if ml_page is not None else {}
        primary = task_result.get("primary_task", {}) if isinstance(task_result, dict) else {}
        if isinstance(primary, dict):
            target = primary.get("target")
        if not target and candidates:
            first_candidate = candidates[0]
            target = first_candidate.get("name") if isinstance(first_candidate, dict) else first_candidate
        for column in dataframe.columns:
            name = str(column)
            lowered = name.casefold()
            series = dataframe[column]
            if name == target:
                counts["target"] += 1
            elif any(token in lowered for token in ("id", "index", "uuid")):
                counts["id"] += 1
            elif pd.api.types.is_datetime64_any_dtype(series.dtype):
                counts["categorical"] += 1
            elif pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
                if "class" in lowered and series.nunique(dropna=True) <= 10:
                    counts["categorical"] += 1
                else:
                    counts["numeric"] += 1
            elif series.nunique(dropna=True) <= max(25, int(len(dataframe) * 0.2)):
                counts["categorical"] += 1
            else:
                counts["text"] += 1
        return counts

    def _update_dataset_card(self):
        dataframe = getattr(self.main_window, "current_dataset", None)
        if dataframe is None:
            dataframe = self._dataframe
        self.dataset_name_label.setText(self._filename if dataframe is not None else "No dataset loaded")
        self.dataset_loaded_badge.setText("Loaded" if dataframe is not None else "Not loaded")
        if dataframe is None:
            values = {"rows": "—", "columns": "—", "size": "—", "duplicates": "—"}
            counts = {"id": 0, "target": 0, "numeric": 0, "categorical": 0, "text": 0}
        else:
            values = {"rows": f"{len(dataframe):,}", "columns": f"{len(dataframe.columns):,}"}
            analysis = getattr(self.main_window, "analysis_result", {}) or {}
            duplicate_info = analysis.get("duplicates", {}) if isinstance(analysis, dict) else {}
            duplicate_count = duplicate_info.get("duplicate_count") if isinstance(duplicate_info, dict) else None
            if duplicate_count is None:
                duplicate_count = int(dataframe.duplicated().sum())
            values["duplicates"] = f"{int(duplicate_count):,}"
            file_path = getattr(self.main_window, "current_file_path", None)
            try:
                size = Path(file_path).stat().st_size if file_path and Path(file_path).is_file() else None
            except OSError:
                size = None
            preview_size = getattr(self, "_preview_file_size", None)
            values["size"] = preview_size or (f"{size / 1024:.1f} KB" if size is not None else "—")
            counts = self._column_types(dataframe)
        for key, value in values.items():
            self.dataset_metric_values[key].setText(str(value))
        for index, (segment, key, label) in enumerate(zip(self.column_type_segments, counts, self.column_type_legend_labels)):
            count = counts[key]
            segment.setVisible(count > 0)
            self.column_type_bar_layout.setStretch(index, max(1, count))
            label.setText(f"{key.title()} {count}")
            label.setVisible(count > 0)

    def _update_quality_card(self):
        dataframe = getattr(self.main_window, "current_dataset", None)
        if dataframe is None:
            dataframe = self._dataframe
        result = getattr(self.main_window, "analysis_result", {}) or {}
        if dataframe is None:
            self.quality_missing.setText("Missing values: —")
            self.quality_outliers.setText("Outliers detected in —")
            self.quality_duplicates.setText("Duplicate rows: —")
            self.quality_most_missing.setText("—")
            self.quality_readiness_badge.setText("Not analyzed")
            return
        missing = result.get("missing_values", []) if isinstance(result, dict) else []
        missing = [item for item in missing if isinstance(item, dict)] if isinstance(missing, list) else []
        if not result:
            self.quality_missing.setText("Run analysis to inspect missing values")
        else:
            missing.sort(key=lambda item: float(item.get("missing_percentage", 0) or 0), reverse=True)
            missing_count = sum(int(item.get("missing_count", 0) or 0) for item in missing)
            total_cells = max(1, len(dataframe) * len(dataframe.columns))
            missing_percent = self._format_percent((missing_count / total_cells) * 100)
            self.quality_missing.setText(f"Missing values: {missing_count:,} ({missing_percent}%)")
        outliers = result.get("outliers", []) if isinstance(result, dict) else []
        active_outliers = [item for item in outliers if isinstance(item, dict) and int(item.get("outlier_count", 0) or 0) > 0] if isinstance(outliers, list) else []
        outlier_names = ", ".join(str(item.get("column", "")) for item in active_outliers[:3])
        if outlier_names:
            self.quality_outliers.setText(f"Outliers detected in {outlier_names}")
        else:
            self.quality_outliers.setText("No outliers detected" if result else "Run analysis to inspect outliers")
        duplicate_info = result.get("duplicates", {}) if isinstance(result, dict) else {}
        duplicate_count = duplicate_info.get("duplicate_count", int(dataframe.duplicated().sum())) if isinstance(duplicate_info, dict) else int(dataframe.duplicated().sum())
        self.quality_duplicates.setText(f"Duplicate rows: {int(duplicate_count):,}")
        if result:
            most_missing = []
            for item in missing[:3]:
                if int(item.get("missing_count", 0) or 0) <= 0:
                    continue
                try:
                    percentage = f"{float(item.get('missing_percentage', 0)):.1f}"
                except (TypeError, ValueError):
                    percentage = "0.0"
                most_missing.append(f"{item.get('column', '—')} {percentage}%")
            self.quality_most_missing.setText(" | ".join(most_missing) if most_missing else "No missing values")
            self.quality_readiness_badge.setText(getattr(self, "_preview_readiness_label", "Readiness"))
        else:
            self.quality_most_missing.setText("Run analysis to see columns")
            self.quality_readiness_badge.setText("Not analyzed")

    def _update_ml_card(self):
        ml_page = getattr(self.main_window, "ml_page", None)
        task_result = getattr(ml_page, "last_task_result", {}) if ml_page is not None else {}
        method_result = getattr(ml_page, "last_method_result", {}) if ml_page is not None else {}
        primary = task_result.get("primary_task", {}) if isinstance(task_result, dict) else {}
        primary = primary if isinstance(primary, dict) else {}
        self.ml_task_badge.setText(str(primary.get("label") or primary.get("task") or "—"))
        score = primary.get("confidence", primary.get("score"))
        try:
            confidence = f"{float(score):.0f}% Confidence" if score is not None else "—"
        except (TypeError, ValueError):
            confidence = "—"
        self.ml_confidence_badge.setText(confidence)
        self.ml_target_badge.setText(f"Target: {primary.get('target') or '—'}")
        recommendations = method_result.get("recommendations", []) if isinstance(method_result, dict) else []
        recommendations = recommendations[:3] if isinstance(recommendations, list) else []
        for index, (name_label, progress, value_label) in enumerate(self.model_rows):
            if index >= len(recommendations):
                name_label.setText("—")
                progress.setValue(0)
                value_label.setText("—")
                continue
            model = recommendations[index]
            name = model.get("method") or model.get("name") or model.get("label") or "Model"
            score = model.get("score", model.get("accuracy", model.get("confidence", 0)))
            try:
                score = float(score)
                if score <= 1:
                    score *= 100
            except (TypeError, ValueError):
                score = 0
            name_label.setText(str(name))
            progress.setValue(max(0, min(100, round(score))))
            value_label.setText(f"{score:.1f}%")

    def _set_key_findings(self, analysis):
        dataframe = getattr(self.main_window, "current_dataset", None)
        if dataframe is None:
            dataframe = self._dataframe
        if dataframe is None:
            self.key_findings_text.setText("<b>Key findings:</b> upload and analyze a dataset to see findings here.")
            return
        missing = analysis.get("missing_values", []) if isinstance(analysis, dict) else []
        missing_count = sum(
            int(item.get("missing_count", 0) or 0)
            for item in missing if isinstance(item, dict)
        ) if isinstance(missing, list) else 0
        ml_page = getattr(self.main_window, "ml_page", None)
        task_result = getattr(ml_page, "last_task_result", {}) if ml_page is not None else {}
        method_result = getattr(ml_page, "last_method_result", {}) if ml_page is not None else {}
        primary = task_result.get("primary_task", {}) if isinstance(task_result, dict) else {}
        recommendations = method_result.get("recommendations", []) if isinstance(method_result, dict) else []
        if isinstance(primary, dict) and primary:
            task = str(primary.get("task") or primary.get("label") or "machine learning").replace("_", " ").lower()
            model = recommendations[0] if isinstance(recommendations, list) and recommendations else {}
            model_name = model.get("method") or model.get("name") or "a recommended model"
            score = model.get("score", model.get("accuracy"))
            try:
                score_value = float(score)
                score_text = f"{score_value * (100 if score_value <= 1 else 1):.2f}%"
            except (TypeError, ValueError):
                score_text = "N/A"
            finding = f"your dataset suits <b>{task}</b> ({model_name} performs best, {score_text})."
        else:
            finding = "ML Intelligence has not been run yet."
        self.key_findings_text.setText(
            f"<b>Key findings:</b> {finding} Fix {missing_count:,} missing values before modeling."
        )

    def _update_pipeline(self):
        completed = (
            self._workflow["has_dataset"],
            self._workflow["analysis_done"],
            self._workflow["ml_done"],
        )
        count = sum(bool(value) for value in completed)
        total = len(completed)
        self.pipeline_count.setText(f"{count} of {total} steps completed")
        self.pipeline_progress.setValue(count)
        for stage, is_complete in zip(self.pipeline_stages, completed):
            stage.setProperty("state", "complete" if is_complete else "waiting")
            stage._status_label.setProperty("complete", bool(is_complete))
            stage._status_label.setText("Done" if is_complete else "Waiting")
            stage.style().unpolish(stage)
            stage.style().polish(stage)
            stage._status_label.style().unpolish(stage._status_label)
            stage._status_label.style().polish(stage._status_label)

    def refresh_summary(self):
        if not hasattr(self, "dataset_name_label"):
            return
        self._update_dataset_card()
        self._update_quality_card()
        self._update_ml_card()
        analysis = getattr(self.main_window, "analysis_result", {}) or {}
        self._set_key_findings(analysis)
        self._update_pipeline()

    def reset_dataset(self):
        self._dataframe = None
        self._filename = "Dataset"
        self._workflow = {
            "has_dataset": False,
            "analysis_done": False,
            "ml_done": False,
        }
        self.refresh_summary()
