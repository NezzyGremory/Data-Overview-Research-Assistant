"""Spreadsheet-like, read-only preview of the active pandas dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QBoxLayout,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from app.ui.style_utils import apply_card_shadow


class DataFrameTableModel(QAbstractTableModel):
    """Expose a DataFrame to Qt without copying every cell into widgets."""

    def __init__(self, dataframe: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self._dataframe = dataframe if dataframe is not None else pd.DataFrame()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._dataframe.index)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._dataframe.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        value = self._dataframe.iat[index.row(), index.column()]
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal and 0 <= section < len(self._dataframe.columns):
            name = str(self._dataframe.columns[section])
            return "Gender" if name == "Sex" else name
        if orientation == Qt.Vertical and 0 <= section < len(self._dataframe.index):
            return str(section + 1)
        return None

    def set_dataframe(self, dataframe: pd.DataFrame | None):
        self.beginResetModel()
        self._dataframe = dataframe if dataframe is not None else pd.DataFrame()
        self.endResetModel()


class DatasetTableView(QTableView):
    """Read-only grid with spreadsheet-style selection and copy support."""

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            indexes = self.selectedIndexes()
            if indexes:
                rows = range(min(i.row() for i in indexes), max(i.row() for i in indexes) + 1)
                columns = range(min(i.column() for i in indexes), max(i.column() for i in indexes) + 1)
                selected = {(i.row(), i.column()) for i in indexes}
                lines = []
                for row in rows:
                    values = [
                        str(self.model().data(self.model().index(row, column), Qt.DisplayRole) or "")
                        if (row, column) in selected else ""
                        for column in columns
                    ]
                    lines.append("\t".join(values))
                QApplication.clipboard().setText("\n".join(lines))
                return
        super().keyPressEvent(event)


class DatasetPreviewPage(QWidget):
    """Dataset overview and preview backed by the existing read-only model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = parent
        self._dataset_name = ""
        self._dataframe: pd.DataFrame | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.page_scroll = QScrollArea()
        self.page_scroll.setObjectName("datasetPreviewScroll")
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setFrameShape(QFrame.NoFrame)
        self.page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        container.setMinimumWidth(0)
        root = QVBoxLayout(container)
        root.setSizeConstraint(QLayout.SetNoConstraint)
        self.root_layout = root
        root.setContentsMargins(36, 22, 28, 16)
        root.setSpacing(14)

        header = QFrame()
        header.setObjectName("datasetPreviewHeader")
        header_layout = QHBoxLayout(header)
        self.header_layout = header_layout
        header_layout.setContentsMargins(0, 0, 0, 2)
        title_block = QVBoxLayout()
        title_block.setSpacing(4)
        title = QLabel("Dataset")
        title.setObjectName("pageTitle")
        title_block.addWidget(title)
        self.summary = QLabel("Upload, preview and clean your CSV before analysis.")
        self.summary.setObjectName("pageSubtitle")
        self.summary.setWordWrap(True)
        self.summary.setMinimumWidth(0)
        self.summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        title_block.addWidget(self.summary)
        header_layout.addLayout(title_block, 1)
        self.loaded_badge = QLabel("●  LOADED")
        self.loaded_badge.setObjectName("datasetPreviewLoadedBadge")
        header_layout.addWidget(self.loaded_badge, 0, Qt.AlignTop)
        root.addWidget(header)

        overview = QWidget()
        overview.setObjectName("datasetPreviewOverview")
        self.overview_layout = QGridLayout(overview)
        self.overview_layout.setHorizontalSpacing(12)
        self.overview_layout.setVerticalSpacing(12)
        file_card = QFrame()
        self.file_card = file_card
        file_card.setObjectName("datasetPreviewFileCard")
        file_card.setMinimumHeight(132)
        file_layout = QHBoxLayout(file_card)
        self.file_layout = file_layout
        file_layout.setContentsMargins(20, 16, 20, 16)
        file_layout.setSpacing(16)
        file_icon = QLabel()
        file_icon.setObjectName("datasetPreviewFileIcon")
        file_icon.setFixedSize(60, 60)
        file_icon.setAlignment(Qt.AlignCenter)
        icon_path = Path(__file__).resolve().parent / "assets" / "icons" / "file-text.svg"
        file_icon.setPixmap(QPixmap(str(icon_path)).scaled(38, 38, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        file_icon.setToolTip("CSV")
        file_layout.addWidget(file_icon)
        file_text_layout = QVBoxLayout()
        file_text_layout.setSpacing(6)
        self.file_name = QLabel("No dataset loaded")
        self.file_name.setObjectName("datasetPreviewFileName")
        self.file_name.setWordWrap(True)
        self.file_details = QLabel("Select a CSV file to preview your dataset.")
        self.file_details.setObjectName("datasetPreviewFileDetails")
        self.file_details.setWordWrap(True)
        file_text_layout.addWidget(self.file_name)
        file_text_layout.addWidget(self.file_details)
        file_layout.addLayout(file_text_layout, 1)
        self.replace_button = QPushButton("Replace file")
        self.replace_button.setObjectName("secondaryButton")
        self.replace_button.setMinimumHeight(42)
        self.replace_button.clicked.connect(self._replace_file)
        file_layout.addWidget(self.replace_button)
        self.upload_new_button = QPushButton("Upload new")
        self.upload_new_button.setObjectName("secondaryButton")
        self.upload_new_button.setMinimumHeight(42)
        self.upload_new_button.clicked.connect(self._open_upload)
        file_layout.addWidget(self.upload_new_button)
        apply_card_shadow(file_card)
        self.overview_layout.addWidget(file_card, 0, 0, 1, 4)

        self.metric_cards = []
        self.rows_value = self._create_metric_card(self.overview_layout, "ROWS", "—")
        self.columns_value = self._create_metric_card(self.overview_layout, "COLUMNS", "—")
        self.missing_value = self._create_metric_card(self.overview_layout, "MISSING CELLS", "—")
        self.duplicates_value = self._create_metric_card(self.overview_layout, "DUPLICATES", "—")
        root.addWidget(overview)

        tabs = QFrame()
        tabs.setObjectName("datasetPreviewTabs")
        tabs_layout = QGridLayout(tabs)
        self.tabs_layout = tabs_layout
        tabs_layout.setContentsMargins(18, 0, 18, 0)
        tabs_layout.setSpacing(30)
        self.tab_buttons: list[QPushButton] = []
        for index, text in enumerate(("Data Preview", "Data Quality", "Columns", "Missing Values", "Outliers", "Correlation")):
            button = QPushButton(text)
            button.setObjectName("datasetPreviewTab")
            button.setProperty("active", index == 0)
            button.setCheckable(False)
            self.tab_buttons.append(button)
            tabs_layout.addWidget(button, 0, index)
            button.clicked.connect(lambda checked=False, tab=index: self._select_tab(tab))
        root.addWidget(tabs)

        body = QHBoxLayout()
        self.body_layout = body
        body.setSpacing(16)
        preview_card = QFrame()
        preview_card.setObjectName("datasetPreviewTableCard")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(18, 14, 18, 16)
        preview_layout.setSpacing(12)
        table_heading = QHBoxLayout()
        preview_title = QLabel("Data Preview")
        preview_title.setObjectName("datasetPreviewSectionTitle")
        table_heading.addWidget(preview_title)
        table_heading.addStretch(1)
        self.row_count_label = QLabel("0 rows")
        self.row_count_label.setObjectName("datasetPreviewCount")
        table_heading.addWidget(self.row_count_label)
        preview_layout.addLayout(table_heading)

        self.table = DatasetTableView()
        self.table.setObjectName("datasetPreviewTable")
        self.model = DataFrameTableModel(parent=self.table)
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.setCornerButtonEnabled(False)
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.verticalHeader().setMinimumWidth(54)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setDefaultSectionSize(145)
        self.table.horizontalHeader().setMinimumSectionSize(80)
        preview_layout.addWidget(self.table, 1)
        self.empty_state = QLabel("No dataset loaded")
        self.empty_state.setObjectName("emptyState")
        self.empty_state.setAlignment(Qt.AlignCenter)
        self.empty_state.setMinimumHeight(100)
        self.empty_state.hide()
        preview_layout.addWidget(self.empty_state, 1)
        apply_card_shadow(preview_card)
        body.addWidget(preview_card, 7)

        columns_card = QFrame()
        columns_card.setObjectName("datasetPreviewColumnsCard")
        columns_layout = QVBoxLayout(columns_card)
        columns_layout.setContentsMargins(18, 14, 18, 14)
        columns_layout.setSpacing(12)
        columns_heading = QHBoxLayout()
        columns_title = QLabel("Column Identification")
        columns_title.setObjectName("datasetPreviewSectionTitle")
        columns_heading.addWidget(columns_title)
        columns_heading.addStretch(1)
        self.column_count_label = QLabel("0 columns")
        self.column_count_label.setObjectName("datasetPreviewCount")
        columns_heading.addWidget(self.column_count_label)
        columns_layout.addLayout(columns_heading)
        column_header = QHBoxLayout()
        column_header.addWidget(self._small_label("COLUMN", "datasetPreviewColumnHeader"), 3)
        column_header.addWidget(self._small_label("TYPE", "datasetPreviewColumnHeader"), 2)
        column_header.addWidget(self._small_label("MISSING", "datasetPreviewColumnHeader"), 2, Qt.AlignRight)
        columns_layout.addLayout(column_header)
        self.columns_scroll = QScrollArea()
        self.columns_scroll.setObjectName("datasetPreviewColumnsScroll")
        self.columns_scroll.setWidgetResizable(True)
        self.columns_scroll.setFrameShape(QFrame.NoFrame)
        self.columns_content = QWidget()
        self.columns_content.setObjectName("datasetPreviewColumnsContent")
        self.columns_content_layout = QVBoxLayout(self.columns_content)
        self.columns_content_layout.setContentsMargins(0, 0, 0, 0)
        self.columns_content_layout.setSpacing(2)
        self.columns_scroll.setWidget(self.columns_content)
        columns_layout.addWidget(self.columns_scroll, 1)
        apply_card_shadow(columns_card)
        body.addWidget(columns_card, 4)

        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("datasetPreviewContentStack")
        preview_content = QWidget()
        preview_content.setObjectName("datasetPreviewBody")
        preview_content.setLayout(body)
        self.content_stack.addWidget(preview_content)
        self.analysis_tables = {}
        self.analysis_empty_states = {}
        self.quality_summary = None
        for title, description in (
            ("Data Quality", "Ringkasan kelengkapan dan konsistensi dataset."),
            ("Columns", "Tipe, jumlah nilai unik, dan contoh isi setiap kolom."),
            ("Missing Values", "Kolom yang memiliki nilai kosong."),
            ("Outliers", "Perkiraan outlier numerik berdasarkan rentang IQR."),
            ("Correlation", "Korelasi antar kolom numerik (Pearson)."),
        ):
            page, table, empty_state, summary = self._create_analysis_page(title, description)
            self.content_stack.addWidget(page)
            self.analysis_tables[title] = table
            self.analysis_empty_states[title] = empty_state
            if title == "Data Quality":
                self.quality_summary = summary
        root.addWidget(self.content_stack, 1)

        footer = QFrame()
        footer.setObjectName("datasetPreviewFooter")
        footer_layout = QHBoxLayout(footer)
        self.footer_layout = footer_layout
        footer_layout.setContentsMargins(0, 2, 0, 0)
        footer_label = QLabel("Quick Cleaning:")
        footer_label.setObjectName("datasetPreviewFooterText")
        footer_layout.addWidget(footer_label, 1)
        self.continue_button = QPushButton("Continue to Analysis  →")
        self.continue_button.setObjectName("primaryButton")
        self.continue_button.setMinimumHeight(46)
        if self._main_window is not None:
            self.continue_button.clicked.connect(self._main_window.open_analysis_page)
        footer_layout.addWidget(self.continue_button)
        root.addWidget(footer)
        self.update_dataset(None)
        self.page_scroll.setWidget(container)
        outer.addWidget(self.page_scroll)
        self._layout_columns = 0
        self._apply_responsive_layout(self.width())

    def _select_tab(self, index: int):
        self.content_stack.setCurrentIndex(index)
        for tab_index, button in enumerate(self.tab_buttons):
            button.setProperty("active", tab_index == index)
            button.style().unpolish(button)
            button.style().polish(button)

    def _create_analysis_page(self, title: str, description: str):
        page = QWidget()
        page.setObjectName("datasetPreviewAnalysisPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        card = QFrame()
        card.setObjectName("datasetPreviewAnalysisCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 14, 18, 16)
        card_layout.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("datasetPreviewSectionTitle")
        card_layout.addWidget(heading)
        description_label = QLabel(description)
        description_label.setObjectName("datasetPreviewAnalysisDescription")
        description_label.setWordWrap(True)
        card_layout.addWidget(description_label)
        summary = QLabel("") if title == "Data Quality" else None
        if summary is not None:
            summary.setObjectName("datasetPreviewAnalysisSummary")
            summary.setWordWrap(True)
            card_layout.addWidget(summary)
        table = DatasetTableView()
        table.setObjectName("datasetPreviewAnalysisTable")
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        card_layout.addWidget(table, 1)
        empty_state = QLabel("Muat dataset untuk melihat informasi ini.")
        empty_state.setObjectName("emptyState")
        empty_state.setAlignment(Qt.AlignCenter)
        empty_state.setMinimumHeight(120)
        card_layout.addWidget(empty_state, 1)
        layout.addWidget(card, 1)
        apply_card_shadow(card)
        return page, table, empty_state, summary

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_responsive_layout(self.width())

    def _apply_responsive_layout(self, width: int):
        compact = width < 760
        self.root_layout.setContentsMargins(
            16 if compact else 36,
            16 if compact else 22,
            16 if compact else 28,
            12 if compact else 16,
        )
        self.header_layout.setDirection(
            QBoxLayout.TopToBottom if width < 540 else QBoxLayout.LeftToRight
        )
        self.file_layout.setDirection(
            QBoxLayout.TopToBottom if width < 600 else QBoxLayout.LeftToRight
        )
        columns = 4 if width >= 1100 else 2 if width >= 600 else 1
        if columns != self._layout_columns:
            for column in range(self.overview_layout.columnCount()):
                self.overview_layout.setColumnMinimumWidth(column, 0)
                self.overview_layout.setColumnStretch(column, 0)
            self.overview_layout.removeWidget(self.file_card)
            for card in self.metric_cards:
                self.overview_layout.removeWidget(card)
            self.overview_layout.addWidget(self.file_card, 0, 0, 1, columns)
            for index, card in enumerate(self.metric_cards):
                self.overview_layout.addWidget(
                    card, 1 + index // columns, index % columns
                )
            for column in range(columns):
                self.overview_layout.setColumnStretch(column, 1)
            self.overview_layout.invalidate()
            self._layout_columns = columns

        self.body_layout.setDirection(
            QBoxLayout.TopToBottom if width < 1050 else QBoxLayout.LeftToRight
        )
        self.footer_layout.setDirection(
            QBoxLayout.TopToBottom if width < 520 else QBoxLayout.LeftToRight
        )
        tab_columns = 3 if width >= 900 else 2 if width >= 600 else 1
        for column in range(self.tabs_layout.columnCount()):
            self.tabs_layout.setColumnMinimumWidth(column, 0)
            self.tabs_layout.setColumnStretch(column, 0)
        for button in self.tab_buttons:
            self.tabs_layout.removeWidget(button)
        for index, button in enumerate(self.tab_buttons):
            self.tabs_layout.addWidget(button, index // tab_columns, index % tab_columns)
        for column in range(tab_columns):
            self.tabs_layout.setColumnStretch(column, 1)
        self.tabs_layout.invalidate()
        self.file_layout.invalidate()
        self.root_layout.invalidate()

    @staticmethod
    def _small_label(text: str, object_name: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName(object_name)
        return label

    def _create_metric_card(self, layout, title, value):
        card = QFrame()
        card.setObjectName("datasetPreviewMetricCard")
        card.setMinimumHeight(132)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 16)
        card_layout.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("datasetPreviewMetricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("datasetPreviewMetricValue")
        card_layout.addWidget(heading)
        card_layout.addWidget(value_label)
        card_layout.addStretch(1)
        self.metric_cards.append(card)
        layout.addWidget(card, 1, len(self.metric_cards) - 1)
        apply_card_shadow(card)
        return value_label

    def _replace_file(self):
        if self._main_window is not None:
            self._main_window.upload_page.select_dataset()

    def _open_upload(self):
        if self._main_window is not None:
            self._main_window.open_upload_page()

    @staticmethod
    def _column_type(column: str, series: pd.Series) -> str:
        name = column.lower()
        if name == "passengerid" or name.endswith("_id") or name == "id":
            return "ID"
        if name in {"target", "survived", "label", "class"}:
            return "Target"
        if name in {"name", "ticket", "cabin", "description", "text"}:
            return "Text"
        if pd.api.types.is_bool_dtype(series.dtype):
            return "Categorical"
        if pd.api.types.is_string_dtype(series.dtype):
            return "Categorical"
        if pd.api.types.is_numeric_dtype(series.dtype):
            unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
            unique_count = series.nunique(dropna=True)
            return "Categorical" if unique_count <= 10 and unique_ratio <= 0.05 else "Numeric"
        if pd.api.types.is_object_dtype(series.dtype) or isinstance(series.dtype, pd.CategoricalDtype):
            return "Categorical"
        return "Numeric"

    def _clear_column_rows(self):
        while self.columns_content_layout.count():
            item = self.columns_content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _populate_column_rows(self, dataframe: pd.DataFrame):
        self._clear_column_rows()
        row_count = max(len(dataframe.index), 1)
        for column in dataframe.columns:
            name = str(column)
            missing = float(dataframe[column].isna().sum()) / row_count * 100
            row = QFrame()
            row.setObjectName("datasetPreviewColumnRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 7, 8, 7)
            row_layout.setSpacing(8)
            display_name = "Gender" if name == "Sex" else name
            column_name = QLabel(display_name)
            column_name.setObjectName("datasetPreviewColumnName")
            column_name.setToolTip(name if display_name != name else "")
            row_layout.addWidget(column_name, 3)
            kind = QLabel(self._column_type(name, dataframe[column]))
            kind.setObjectName("datasetPreviewTypeBadge")
            kind.setAlignment(Qt.AlignCenter)
            row_layout.addWidget(kind, 2)
            missing_text = QLabel(f"{missing:.1f}%")
            missing_text.setObjectName("datasetPreviewMissingValue")
            missing_text.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row_layout.addWidget(missing_text, 2)
            self.columns_content_layout.addWidget(row)
        self.columns_content_layout.addStretch(1)

    def _set_analysis_table(self, title: str, dataframe: pd.DataFrame):
        table = self.analysis_tables[title]
        table.setModel(DataFrameTableModel(dataframe, table))
        has_data = not dataframe.empty and len(dataframe.columns) > 0
        table.setVisible(has_data)
        self.analysis_empty_states[title].setVisible(not has_data)
        if not has_data:
            messages = {
                "Missing Values": "Tidak ada nilai kosong pada dataset ini.",
                "Outliers": "Tidak ada kolom numerik yang cukup untuk menghitung outlier.",
                "Correlation": "Diperlukan minimal dua kolom numerik untuk melihat korelasi.",
            }
            self.analysis_empty_states[title].setText(
                messages.get(title, "Muat dataset untuk melihat informasi ini.")
            )

    def _update_analysis_views(self, dataframe: pd.DataFrame | None, duplicate_count: int = 0):
        if dataframe is None:
            for title in self.analysis_tables:
                self._set_analysis_table(title, pd.DataFrame())
            self.quality_summary.setText("")
            return

        row_count = len(dataframe.index)
        column_count = len(dataframe.columns)
        missing_by_column = dataframe.isna().sum()
        missing_percent = missing_by_column.div(max(row_count, 1)).mul(100)
        numeric = dataframe.select_dtypes(include="number")

        quality_rows = [
            ("Rows", f"{row_count:,}"),
            ("Columns", f"{column_count:,}"),
            ("Missing cells", f"{int(missing_by_column.sum()):,}"),
            ("Complete rows", f"{int((~dataframe.isna().any(axis=1)).sum()):,}"),
            ("Duplicate rows", f"{duplicate_count:,}"),
            ("Columns with missing values", f"{int((missing_by_column > 0).sum()):,}"),
        ]
        quality_table = pd.DataFrame(quality_rows, columns=["Measure", "Value"])
        self.quality_summary.setText(
            f"{row_count:,} baris · {column_count:,} kolom · "
            f"{int(missing_by_column.sum()):,} sel kosong · "
            f"{duplicate_count:,} baris duplikat"
        )

        column_rows = []
        missing_rows = []
        for column in dataframe.columns:
            series = dataframe[column]
            non_null = int(series.notna().sum())
            unique_count = int(series.nunique(dropna=True))
            sample_values = series.dropna().astype(str).head(3).tolist()
            column_rows.append((
                str(column), self._column_type(str(column), series), non_null,
                unique_count, ", ".join(sample_values) if sample_values else "—",
            ))
            missing_count = int(missing_by_column[column])
            if missing_count:
                missing_rows.append((str(column), missing_count, f"{missing_percent[column]:.1f}%"))

        outlier_rows = []
        for column in numeric.columns:
            values = numeric[column].dropna()
            if len(values) < 4:
                continue
            q1, q3 = values.quantile([0.25, 0.75]).tolist()
            spread = q3 - q1
            count = int(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum())
            outlier_rows.append((str(column), count, f"{count / max(len(values), 1) * 100:.1f}%"))

        # Bound correlation work for wide/large datasets while preserving a useful preview.
        correlation_columns = list(numeric.columns[:20])
        correlation_data = numeric.loc[:, correlation_columns]
        if len(correlation_data) > 5000:
            correlation_data = correlation_data.sample(n=5000, random_state=0)
        correlation = correlation_data.corr().round(3) if len(correlation_columns) >= 2 else pd.DataFrame()
        if not correlation.empty:
            correlation.insert(0, "Column", correlation.index.astype(str))
            correlation = correlation.reset_index(drop=True)

        self._set_analysis_table("Data Quality", quality_table)
        self._set_analysis_table(
            "Columns",
            pd.DataFrame(column_rows, columns=["Column", "Type", "Non-empty", "Unique", "Example values"]),
        )
        self._set_analysis_table(
            "Missing Values",
            pd.DataFrame(missing_rows, columns=["Column", "Missing", "Percent"]),
        )
        self._set_analysis_table(
            "Outliers",
            pd.DataFrame(outlier_rows, columns=["Numeric column", "Outlier rows", "Percent"]),
        )
        self._set_analysis_table("Correlation", correlation)

    def update_dataset(self, dataframe: pd.DataFrame | None, filename: str = ""):
        self._dataframe = dataframe
        self._dataset_name = filename or "Dataset"
        self.model.set_dataframe(dataframe)
        if dataframe is None:
            self.summary.setText("Upload, preview and clean your CSV before analysis.")
            self.file_name.setText("No dataset loaded")
            self.file_details.setText("Select a CSV file to preview your dataset.")
            self.rows_value.setText("—")
            self.columns_value.setText("—")
            self.missing_value.setText("—")
            self.duplicates_value.setText("—")
            self.row_count_label.setText("0 rows")
            self.column_count_label.setText("0 columns")
            self.loaded_badge.hide()
            self.table.hide()
            self.empty_state.show()
            self._clear_column_rows()
            self._update_analysis_views(None)
            return

        self.empty_state.hide()
        self.table.show()
        rows, columns = dataframe.shape
        missing_cells = int(dataframe.isna().sum().sum())
        duplicates = int(dataframe.duplicated().sum())
        self.file_name.setText(self._dataset_name)
        self.file_details.setText(f"{rows:,} rows  ×  {columns:,} columns  ·  CSV")
        self.rows_value.setText(f"{rows:,}")
        self.columns_value.setText(f"{columns:,}")
        self.missing_value.setText(f"{missing_cells:,}")
        self.duplicates_value.setText(f"{duplicates:,}")
        self.row_count_label.setText(f"{rows:,} rows")
        self.column_count_label.setText(f"{columns:,} columns")
        self.loaded_badge.show()
        self._populate_column_rows(dataframe)
        self._update_analysis_views(dataframe, duplicates)
        if columns <= 12 and rows <= 500:
            self.table.resizeColumnsToContents()
            for column in range(columns):
                self.table.setColumnWidth(column, min(self.table.columnWidth(column), 280))
