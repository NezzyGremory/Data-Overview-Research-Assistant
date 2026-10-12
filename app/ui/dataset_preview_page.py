"""Spreadsheet-like, read-only preview of the active pandas dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSize, Qt
from PySide6.QtGui import QIcon, QKeySequence, QPixmap
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
        if isinstance(value, float) and value.is_integer():
            value = int(value)
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
        self.page_scroll.setAttribute(Qt.WA_StyledBackground, True)
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setFrameShape(QFrame.NoFrame)
        self.page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        container.setObjectName("datasetPreviewContainer")
        container.setAttribute(Qt.WA_StyledBackground, True)
        container.setMinimumWidth(0)
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root = QVBoxLayout(container)
        root.setSizeConstraint(QLayout.SetNoConstraint)
        self.root_layout = root
        root.setContentsMargins(36, 22, 25, 20)
        root.setSpacing(0)

        header = QFrame()
        self.header = header
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
        self.header_upload_button = QPushButton("Upload Dataset")
        self.header_upload_button.setObjectName("datasetPreviewUploadButton")
        self.header_upload_button.setText("")
        self.header_upload_button.setFixedSize(253, 61)
        self.header_upload_button.setAccessibleName("Upload Dataset")
        upload_button_layout = QHBoxLayout(self.header_upload_button)
        upload_button_layout.setContentsMargins(19, 0, 14, 0)
        upload_button_layout.setSpacing(16)
        upload_icon = QLabel()
        upload_icon.setObjectName("datasetPreviewUploadIcon")
        upload_icon.setFixedSize(32, 24)
        upload_icon_path = Path(__file__).resolve().parent / "assets" / "icons" / "upload.svg"
        upload_icon.setPixmap(QPixmap(str(upload_icon_path)).scaled(32, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        upload_button_layout.addWidget(upload_icon)
        upload_text = QLabel("Upload Dataset")
        upload_text.setObjectName("datasetPreviewUploadText")
        upload_button_layout.addWidget(upload_text)
        upload_button_layout.addStretch(1)
        self.header_upload_button.clicked.connect(self._open_upload)
        upload_action = QWidget()
        self.upload_action = upload_action
        upload_action.setFixedSize(253, 78)
        upload_action_layout = QVBoxLayout(upload_action)
        upload_action_layout.setContentsMargins(0, 10, 0, 0)
        upload_action_layout.setSpacing(0)
        upload_action_layout.addWidget(self.header_upload_button, 0, Qt.AlignTop)
        header_layout.addWidget(upload_action)
        header.setFixedHeight(78)
        root.addWidget(header)
        root.addSpacing(28)

        overview = QWidget()
        overview.setObjectName("datasetPreviewOverview")
        self.overview_layout = QGridLayout(overview)
        self.overview_layout.setContentsMargins(0, 0, 0, 0)
        self.overview_layout.setHorizontalSpacing(12)
        self.overview_layout.setVerticalSpacing(12)
        file_card = QFrame()
        self.file_card = file_card
        file_card.setObjectName("datasetPreviewFileCard")
        file_card.setMinimumHeight(176)
        file_layout = QVBoxLayout(file_card)
        self.file_layout = file_layout
        file_layout.setContentsMargins(35, 21, 34, 14)
        file_layout.setSpacing(17)
        file_identity = QFrame()
        self.file_identity = file_identity
        file_identity.setObjectName("datasetPreviewFileIdentity")
        file_identity.setFixedHeight(76)
        identity_layout = QHBoxLayout(file_identity)
        identity_layout.setContentsMargins(16, 0, 12, 0)
        identity_layout.setSpacing(14)
        file_icon = QLabel()
        file_icon.setObjectName("datasetPreviewFileIcon")
        file_icon.setFixedSize(40, 40)
        file_icon.setAlignment(Qt.AlignCenter)
        icon_path = Path(__file__).resolve().parent / "assets" / "icons" / "file-text.svg"
        file_icon.setPixmap(QPixmap(str(icon_path)).scaled(25, 25, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        file_icon.setToolTip("CSV")
        identity_layout.addWidget(file_icon, 0, Qt.AlignVCenter)
        file_text_layout = QVBoxLayout()
        file_text_layout.setSpacing(2)
        self.file_name = QLabel("No dataset loaded")
        self.file_name.setObjectName("datasetPreviewFileName")
        self.file_name.setWordWrap(True)
        self.file_details = QLabel("Select a CSV file to preview your dataset.")
        self.file_details.setObjectName("datasetPreviewFileDetails")
        self.file_details.setWordWrap(True)
        file_text_layout.addWidget(self.file_name)
        file_text_layout.addWidget(self.file_details)
        identity_layout.addLayout(file_text_layout, 1)
        self.loaded_badge = QLabel("● Loaded")
        self.loaded_badge.setObjectName("datasetPreviewLoadedBadge")
        identity_layout.addWidget(self.loaded_badge, 0, Qt.AlignVCenter)
        file_layout.addWidget(file_identity)
        file_actions = QHBoxLayout()
        self.file_actions = file_actions
        file_actions.setContentsMargins(0, 0, 0, 0)
        self.replace_button = QPushButton("Replace file")
        self.replace_button.setObjectName("secondaryButton")
        self.replace_button.setFixedSize(162, 46)
        self.replace_button.setIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "icons" / "reload.svg")))
        self.replace_button.setIconSize(QSize(17, 17))
        self.replace_button.clicked.connect(self._replace_file)
        file_actions.addWidget(self.replace_button)
        file_actions.addStretch(1)
        self.upload_new_button = QPushButton("Upload new")
        self.upload_new_button.setObjectName("secondaryButton")
        self.upload_new_button.setFixedSize(162, 46)
        self.upload_new_button.setIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "icons" / "upload-small.svg")))
        self.upload_new_button.setIconSize(QSize(17, 17))
        self.upload_new_button.clicked.connect(self._open_upload)
        file_actions.addWidget(self.upload_new_button)
        file_layout.addLayout(file_actions)
        apply_card_shadow(file_card)
        self.overview_layout.addWidget(file_card, 0, 0)

        self.metric_cards = []
        self.metric_descriptions = {}
        self.rows_value = self._create_metric_card(self.overview_layout, "ROWS", "—", "Records")
        self.columns_value = self._create_metric_card(self.overview_layout, "COLUMNS", "—", "Features")
        self.missing_value = self._create_metric_card(self.overview_layout, "MISSING CELLS", "—", "")
        self.duplicates_value = self._create_metric_card(self.overview_layout, "DUPLICATES", "—", "Identical rows")
        root.addWidget(overview)
        overview.setMinimumHeight(176)
        root.addSpacing(9)

        tabs = QFrame()
        tabs.setObjectName("datasetPreviewTabs")
        tabs_layout = QHBoxLayout(tabs)
        self.tabs_layout = tabs_layout
        tabs_layout.setContentsMargins(0, 0, 0, 0)
        tabs_layout.setSpacing(0)
        self.tabs_scroll = QScrollArea()
        self.tabs_scroll.setObjectName("datasetPreviewTabsScroll")
        self.tabs_scroll.setWidgetResizable(False)
        self.tabs_scroll.setFrameShape(QFrame.NoFrame)
        self.tabs_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tabs_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tab_strip = QWidget()
        self.tab_strip_layout = QHBoxLayout(self.tab_strip)
        self.tab_strip_layout.setContentsMargins(0, 0, 0, 0)
        self.tab_strip_layout.setSpacing(0)
        self.tab_buttons: list[QPushButton] = []
        for index, text in enumerate(("Data Preview", "Data Quality", "Columns", "Missing Values", "Outliers", "Correlation")):
            button = QPushButton(text)
            button.setObjectName("datasetPreviewTab")
            button.setProperty("active", index == 0)
            button.setCheckable(False)
            self.tab_buttons.append(button)
            self.tab_strip_layout.addWidget(button)
            button.clicked.connect(lambda checked=False, tab=index: self._select_tab(tab))
        self.tab_strip_layout.addStretch(1)
        self.tab_strip.adjustSize()
        self.tabs_scroll.setWidget(self.tab_strip)
        tabs_layout.addWidget(self.tabs_scroll)
        tabs.setFixedHeight(54)
        root.addWidget(tabs)
        root.addSpacing(15)

        body = QHBoxLayout()
        self.body_layout = body
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(21)
        preview_card = QFrame()
        self.preview_card = preview_card
        preview_card.setObjectName("datasetPreviewTableCard")
        preview_card.setMinimumWidth(0)
        preview_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(0, 20, 0, 16)
        preview_layout.setSpacing(24)
        table_heading_widget = QWidget()
        self.table_heading_widget = table_heading_widget
        table_heading_widget.setFixedHeight(30)
        table_heading = QHBoxLayout(table_heading_widget)
        table_heading.setContentsMargins(24, 0, 18, 0)
        table_heading.setSpacing(0)
        preview_title = QLabel("Data Preview")
        self.preview_title = preview_title
        preview_title.setObjectName("datasetPreviewSectionTitle")
        preview_title.setFixedHeight(30)
        table_heading.addWidget(preview_title)
        table_heading.addStretch(1)
        self.row_count_label = QLabel("0 rows")
        self.row_count_label.setObjectName("datasetPreviewCount")
        self.row_count_label.hide()
        preview_layout.addWidget(table_heading_widget)

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
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setDefaultSectionSize(123)
        self.table.horizontalHeader().setMinimumSectionSize(48)
        self.table.horizontalHeader().setFixedHeight(50)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.table.setMinimumHeight(220)
        self.table.setMinimumWidth(0)
        self.table.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        preview_layout.addWidget(self.table, 1)
        self.empty_state = QLabel("No dataset loaded")
        self.empty_state.setObjectName("emptyState")
        self.empty_state.setAlignment(Qt.AlignCenter)
        self.empty_state.setMinimumHeight(100)
        self.empty_state.hide()
        preview_layout.addWidget(self.empty_state, 0, Qt.AlignTop)
        apply_card_shadow(preview_card)
        body.addWidget(preview_card, 7)

        columns_card = QFrame()
        self.columns_card = columns_card
        columns_card.setObjectName("datasetPreviewColumnsCard")
        columns_card.setMinimumWidth(250)
        columns_card.setMaximumWidth(420)
        columns_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        columns_layout = QVBoxLayout(columns_card)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(0)
        columns_heading = QHBoxLayout()
        columns_title = QLabel("Column Identification")
        columns_title.setObjectName("datasetPreviewSectionTitle")
        columns_heading.addWidget(columns_title)
        columns_heading.addStretch(1)
        self.column_count_label = QLabel("Missing")
        self.column_count_label.setObjectName("datasetPreviewCount")
        columns_heading.addWidget(self.column_count_label)
        columns_heading.setContentsMargins(24, 6, 22, 8)
        columns_layout.addLayout(columns_heading)
        self.columns_scroll = QScrollArea()
        self.columns_scroll.setObjectName("datasetPreviewColumnsScroll")
        self.columns_scroll.setWidgetResizable(True)
        self.columns_scroll.setFrameShape(QFrame.NoFrame)
        self.columns_content = QWidget()
        self.columns_content.setObjectName("datasetPreviewColumnsContent")
        self.columns_content_layout = QVBoxLayout(self.columns_content)
        self.columns_content_layout.setContentsMargins(0, 0, 0, 0)
        self.columns_content_layout.setSpacing(0)
        self.columns_scroll.setWidget(self.columns_content)
        columns_layout.addWidget(self.columns_scroll, 1)
        apply_card_shadow(columns_card)
        body.addWidget(columns_card, 3)

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
        root.addSpacing(15)

        footer = QFrame()
        self.footer = footer
        footer.setObjectName("datasetPreviewFooter")
        footer_layout = QHBoxLayout(footer)
        self.footer_layout = footer_layout
        footer_layout.setContentsMargins(24, 0, 20, 0)
        footer_label = QLabel("Quick Cleaning:")
        footer_label.setObjectName("datasetPreviewFooterText")
        footer_layout.addWidget(footer_label, 1)
        self.continue_button = QPushButton("Continue to Analysis  →")
        self.continue_button.setObjectName("primaryButton")
        self.continue_button.setMinimumSize(247, 40)
        if self._main_window is not None:
            self.continue_button.clicked.connect(self._main_window.open_analysis_page)
        footer_layout.addWidget(self.continue_button)
        root.addWidget(footer)
        footer.setFixedHeight(61)
        apply_card_shadow(footer)
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
            12 if compact else 36,
            14 if compact else 22,
            12 if compact else 25,
            14 if compact else 29,
        )
        self.header_layout.setDirection(
            QBoxLayout.TopToBottom if width < 620 else QBoxLayout.LeftToRight
        )
        self.header.setFixedHeight(126 if width < 620 else 78)
        self.header_upload_button.setFixedSize(
            190 if width < 620 else 253,
            46 if width < 620 else 63,
        )
        self.upload_action.setFixedSize(190 if width < 620 else 253, 126 if width < 620 else 78)
        self.file_actions.setDirection(
            QBoxLayout.TopToBottom if width < 430 else QBoxLayout.LeftToRight
        )
        self.file_actions.setSpacing(8 if width < 430 else 0)
        self.file_card.setMinimumHeight(230 if width < 430 else 176)
        columns = 5 if width >= 1100 else 2 if width >= 620 else 1
        if columns != self._layout_columns:
            for column in range(self.overview_layout.columnCount()):
                self.overview_layout.setColumnMinimumWidth(column, 0)
                self.overview_layout.setColumnStretch(column, 0)
            self.overview_layout.removeWidget(self.file_card)
            for card in self.metric_cards:
                self.overview_layout.removeWidget(card)
            if columns == 5:
                self.overview_layout.addWidget(self.file_card, 0, 0)
                for index, card in enumerate(self.metric_cards, start=1):
                    self.overview_layout.addWidget(card, 0, index)
                self.overview_layout.setColumnStretch(0, 2995)
                for column in range(1, columns):
                    self.overview_layout.setColumnStretch(column, 1000)
                self.overview_layout.setHorizontalSpacing(14)
            else:
                self.overview_layout.addWidget(self.file_card, 0, 0, 1, columns)
                for index, card in enumerate(self.metric_cards):
                    self.overview_layout.addWidget(
                        card, 1 + index // columns, index % columns
                    )
                for column in range(columns):
                    self.overview_layout.setColumnStretch(column, 1)
                self.overview_layout.setHorizontalSpacing(12)
            self.overview_layout.invalidate()
            self._layout_columns = columns

        self.body_layout.setDirection(
            QBoxLayout.TopToBottom if width < 1000 else QBoxLayout.LeftToRight
        )
        if width < 1000:
            self.columns_card.setMaximumWidth(16777215)
            self.columns_card.setMinimumWidth(0)
            self.columns_card.setMinimumHeight(220)
        else:
            self.columns_card.setMinimumWidth(250)
            self.columns_card.setMaximumWidth(420)
            self.columns_card.setMinimumHeight(0)
        self.footer_layout.setDirection(
            QBoxLayout.TopToBottom if width < 520 else QBoxLayout.LeftToRight
        )
        self.footer.setFixedHeight(92 if width < 520 else 61)
        self.content_stack.setMinimumHeight(470 if width >= 1000 else 520)
        self.tabs_scroll.setMinimumHeight(54)
        self.tab_strip.adjustSize()
        self.file_layout.invalidate()
        self.root_layout.invalidate()

    @staticmethod
    def _small_label(text: str, object_name: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName(object_name)
        return label

    def _create_metric_card(self, layout, title, value, description):
        card = QFrame()
        card.setObjectName("datasetPreviewMetricCard")
        card.setMinimumHeight(176)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 24, 20, 18)
        card_layout.setSpacing(9)
        heading = QLabel(title)
        heading.setObjectName("datasetPreviewMetricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("datasetPreviewMetricValue")
        card_layout.addWidget(heading)
        card_layout.addWidget(value_label)
        description_label = QLabel(description)
        description_label.setObjectName("datasetPreviewMetricDescription")
        card_layout.addWidget(description_label)
        self.metric_descriptions[title] = description_label
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
            return "Categorical" if name == "pclass" else "Numeric"
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
        titanic_row_heights = (34, 36, 34, 36, 37, 36, 36, 34, 36, 36, 37, 36)
        is_titanic_preview = {
            "passengerid", "survived", "pclass"
        }.issubset({str(name).casefold() for name in dataframe.columns}) and len(dataframe.columns) >= 10
        for row_index, column in enumerate(dataframe.columns):
            name = str(column)
            missing = float(dataframe[column].isna().sum()) / row_count * 100
            row = QFrame()
            row.setObjectName("datasetPreviewColumnRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(24, 0, 22, 0)
            row_layout.setSpacing(10)
            display_name = "Gender" if name == "Sex" else name
            column_name = QLabel(display_name)
            column_name.setObjectName("datasetPreviewColumnName")
            column_name.setToolTip(name if display_name != name else "")
            column_name.setMinimumWidth(0)
            row_layout.addWidget(column_name, 1)
            kind = QLabel(self._column_type(name, dataframe[column]))
            kind.setObjectName("datasetPreviewTypeBadge")
            kind.setProperty("columnType", kind.text().lower())
            kind.setAlignment(Qt.AlignCenter)
            kind.setFixedWidth(83)
            row_layout.addWidget(kind)
            missing_text = QLabel(f"{missing:.1f}%")
            missing_text.setObjectName("datasetPreviewMissingValue")
            missing_text.setProperty("hasMissing", missing > 0)
            missing_text.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            missing_text.setFixedWidth(48)
            row_layout.addWidget(missing_text)
            row.setFixedHeight(
                titanic_row_heights[row_index]
                if is_titanic_preview and row_index < len(titanic_row_heights)
                else 35
            )
            self.columns_content_layout.addWidget(row)
        self.columns_content_layout.addStretch(1)

    def _configure_preview_columns(self, dataframe: pd.DataFrame):
        names = [str(name) for name in dataframe.columns]
        normalized = {name.casefold(): index for index, name in enumerate(names)}
        is_titanic = {
            "passengerid", "survived", "pclass"
        }.issubset(normalized) and len(names) >= 10
        hidden_names = {"sibsp", "parch", "ticket", "cabin"} if is_titanic else set()
        widths = {
            "passengerid": 123,
            "survived": 95,
            "pclass": 77,
            "name": 278,
            "sex": 83,
            "gender": 83,
            "age": 57,
            "fare": 131,
            "embarked": 103,
        }
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        for index, name in enumerate(names):
            lowered = name.casefold()
            self.table.setColumnHidden(index, lowered in hidden_names)
            if lowered in widths:
                self.table.setColumnWidth(index, widths[lowered])
            elif not is_titanic:
                self.table.setColumnWidth(index, 140)

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
        if self._main_window is not None and self._main_window.current_file_path:
            try:
                file_size = Path(self._main_window.current_file_path).stat().st_size
                file_details = f"{file_size / 1024:.1f} KB · Uploaded just now"
            except OSError:
                file_details = f"{rows:,} rows · {columns:,} columns"
        else:
            file_details = f"{rows:,} rows · {columns:,} columns"
        self.file_details.setText(file_details)
        self.rows_value.setText(f"{rows:,}")
        self.columns_value.setText(f"{columns:,}")
        self.missing_value.setText(f"{missing_cells:,}")
        self.duplicates_value.setText(f"{duplicates:,}")
        missing_percent = missing_cells / max(rows * columns, 1) * 100
        self.metric_descriptions["ROWS"].setText("Records")
        self.metric_descriptions["COLUMNS"].setText("Features")
        self.metric_descriptions["MISSING CELLS"].setText(f"{missing_percent:.1f}% of all data")
        self.metric_descriptions["DUPLICATES"].setText("Identical rows")
        self.row_count_label.setText(f"{rows:,} rows")
        self.column_count_label.setText("Missing")
        self.loaded_badge.show()
        self._populate_column_rows(dataframe)
        self._update_analysis_views(dataframe, duplicates)
        self._configure_preview_columns(dataframe)
