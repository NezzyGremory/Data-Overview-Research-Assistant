"""Spreadsheet-like, read-only preview of the active pandas dataset."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)


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
            return str(self._dataframe.columns[section])
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
    """Scrollable table showing every row and column in the active dataset."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = parent
        self._dataset_name = ""
        self._dataframe: pd.DataFrame | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 18)
        root.setSpacing(12)

        header = QFrame()
        header.setObjectName("datasetPreviewHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 4)
        header_layout.setSpacing(10)

        title_block = QVBoxLayout()
        title_block.setSpacing(3)

        title = QLabel("Dataset")
        title.setObjectName("pageTitle")
        title_block.addWidget(title)

        self.summary = QLabel("Upload, preview and clean your CSV before analysis.")
        self.summary.setObjectName("pageSubtitle")
        self.summary.setWordWrap(True)
        title_block.addWidget(self.summary)
        header_layout.addLayout(title_block, 1)

        self.loaded_badge = QLabel("●  LOADED")
        self.loaded_badge.setObjectName("datasetPreviewLoadedBadge")
        header_layout.addWidget(self.loaded_badge, 0, Qt.AlignTop)
        root.addWidget(header)

        overview = QHBoxLayout()
        overview.setSpacing(10)
        file_card = QFrame()
        file_card.setObjectName("datasetPreviewFileCard")
        file_layout = QVBoxLayout(file_card)
        file_layout.setContentsMargins(16, 14, 16, 14)
        file_layout.setSpacing(4)
        self.file_name = QLabel("No dataset loaded")
        self.file_name.setObjectName("datasetPreviewFileName")
        self.file_details = QLabel("Pilih atau unggah dataset untuk melihat isinya.")
        self.file_details.setObjectName("datasetPreviewFileDetails")
        self.file_details.setWordWrap(True)
        file_layout.addWidget(self.file_name)
        file_layout.addWidget(self.file_details)
        overview.addWidget(file_card, 2)
        self.rows_value = self._create_metric_card(overview, "ROWS", "—")
        self.columns_value = self._create_metric_card(overview, "COLUMNS", "—")
        root.addLayout(overview)

        body = QHBoxLayout()
        body.setSpacing(12)
        preview_card = QFrame()
        preview_card.setObjectName("datasetPreviewTableCard")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(14, 12, 14, 14)
        preview_layout.setSpacing(8)
        preview_title = QLabel("Data Preview")
        preview_title.setObjectName("sectionTitle")
        preview_layout.addWidget(preview_title)

        self.table = DatasetTableView()
        self.table.setObjectName("datasetPreviewTable")
        self.model = DataFrameTableModel(parent=self.table)
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.setCornerButtonEnabled(False)
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.verticalHeader().setMinimumWidth(54)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setDefaultSectionSize(150)
        self.table.horizontalHeader().setMinimumSectionSize(80)
        preview_layout.addWidget(self.table, 1)

        self.empty_state = QLabel("Belum ada dataset yang dimuat.")
        self.empty_state.setObjectName("emptyState")
        self.empty_state.setAlignment(Qt.AlignCenter)
        self.empty_state.setMinimumHeight(100)
        self.empty_state.hide()
        preview_layout.addWidget(self.empty_state, 1)
        body.addWidget(preview_card, 3)

        columns_card = QFrame()
        columns_card.setObjectName("datasetPreviewColumnsCard")
        columns_layout = QVBoxLayout(columns_card)
        columns_layout.setContentsMargins(14, 12, 14, 14)
        columns_layout.setSpacing(8)
        columns_title = QLabel("Columns")
        columns_title.setObjectName("sectionTitle")
        columns_layout.addWidget(columns_title)
        self.column_list = QLabel("Kolom akan tampil setelah dataset dimuat.")
        self.column_list.setObjectName("datasetPreviewColumnList")
        self.column_list.setWordWrap(True)
        self.column_list.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        columns_layout.addWidget(self.column_list, 1)
        body.addWidget(columns_card, 1)
        root.addLayout(body, 1)

        footer = QFrame()
        footer.setObjectName("datasetPreviewFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(14, 8, 14, 8)
        footer_label = QLabel("Quick Cleaning is available in Analysis.")
        footer_label.setObjectName("datasetPreviewFooterText")
        footer_layout.addWidget(footer_label, 1)
        self.continue_button = QPushButton("Continue to Analysis  →")
        self.continue_button.setObjectName("primaryButton")
        self.continue_button.setMinimumHeight(36)
        if self._main_window is not None:
            self.continue_button.clicked.connect(self._main_window.open_analysis_page)
        footer_layout.addWidget(self.continue_button)
        root.addWidget(footer)
        self.update_dataset(None)

    @staticmethod
    def _create_metric_card(layout, title, value):
        card = QFrame()
        card.setObjectName("datasetPreviewMetricCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 10, 14, 10)
        card_layout.setSpacing(3)
        heading = QLabel(title)
        heading.setObjectName("datasetPreviewMetricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("datasetPreviewMetricValue")
        card_layout.addWidget(heading)
        card_layout.addWidget(value_label)
        layout.addWidget(card, 1)
        return value_label

    def update_dataset(self, dataframe: pd.DataFrame | None, filename: str = ""):
        self._dataframe = dataframe
        self._dataset_name = filename or "Dataset"
        self.model.set_dataframe(dataframe)
        if dataframe is None:
            self.summary.setText("Pilih atau unggah dataset untuk melihat isinya.")
            self.file_name.setText("No dataset loaded")
            self.file_details.setText("Pilih atau unggah dataset untuk melihat isinya.")
            self.rows_value.setText("—")
            self.columns_value.setText("—")
            self.column_list.setText("Kolom akan tampil setelah dataset dimuat.")
            self.loaded_badge.hide()
            self.table.hide()
            self.empty_state.show()
            return

        self.empty_state.hide()
        self.table.show()
        rows, columns = dataframe.shape
        self.summary.setText(
            f"{self._dataset_name}  ·  {rows:,} baris  ·  {columns:,} kolom"
            "  ·  Hanya lihat"
        )
        self.summary.setText("Upload, preview and clean your CSV before analysis.")
        self.file_name.setText(self._dataset_name)
        self.file_details.setText(f"{rows:,} rows × {columns:,} columns  ·  Read only")
        self.rows_value.setText(f"{rows:,}")
        self.columns_value.setText(f"{columns:,}")
        self.column_list.setText("\n".join(str(column) for column in dataframe.columns))
        self.loaded_badge.show()
        if columns <= 12 and rows <= 500:
            self.table.resizeColumnsToContents()
            for column in range(columns):
                self.table.setColumnWidth(column, min(self.table.columnWidth(column), 280))
