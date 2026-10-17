"""Spreadsheet-like, read-only preview of the active pandas dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from app.ui.scale import logical_width

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QBoxLayout,
    QComboBox,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from app.ui.style_utils import render_icon_pixmap

from app.ui.style_utils import apply_card_shadow


class DataFrameTableModel(QAbstractTableModel):
    """Expose a DataFrame to Qt without copying every cell into widgets."""

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        parent=None,
        *,
        include_index: bool = False,
        row_offset: int = 0,
        correlation_heatmap: bool = False,
        detailed_headers: bool = True,
    ):
        super().__init__(parent)
        self._dataframe = dataframe if dataframe is not None else pd.DataFrame()
        self._include_index = include_index
        self._row_offset = row_offset
        self._correlation_heatmap = correlation_heatmap
        self._detailed_headers = detailed_headers

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._dataframe.index)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe.columns) + int(self._include_index)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if self._correlation_heatmap and index.column() > 0:
            try:
                value = float(self._dataframe.iat[index.row(), index.column()])
            except (TypeError, ValueError, IndexError):
                value = float("nan")
            if role == Qt.BackgroundRole and pd.notna(value):
                if index.column() == index.row() + 1:
                    return QBrush(QColor("#155EEF"))
                strength = min(abs(value), 1.0)
                if value > 0:
                    return QBrush(QColor(21, 94, 239, 24 + round(strength * 75)))
                if value < 0:
                    return QBrush(QColor(239, 68, 68, 22 + round(strength * 72)))
            if role == Qt.ForegroundRole and index.column() == index.row() + 1:
                return QBrush(QColor("#FFFFFF"))

        if role != Qt.DisplayRole:
            return None
        if self._include_index and index.column() == 0:
            return str(self._row_offset + index.row() + 1)
        column = index.column() - int(self._include_index)
        if not 0 <= column < len(self._dataframe.columns):
            return None
        value = self._dataframe.iat[index.row(), column]
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
        if orientation == Qt.Horizontal and 0 <= section < len(self._dataframe.columns) + int(self._include_index):
            if self._include_index:
                if section == 0:
                    return "#"
                section -= 1
            name = str(self._dataframe.columns[section])
            if not self._detailed_headers:
                return name
            series = self._dataframe.iloc[:, section]
            detail = "# numeric" if pd.api.types.is_numeric_dtype(series.dtype) else "Text"
            return f"{name}\n{detail}"
        if orientation == Qt.Vertical and 0 <= section < len(self._dataframe.index):
            return str(section + 1)
        return None

    def set_dataframe(self, dataframe: pd.DataFrame | None, *, row_offset: int = 0):
        self.beginResetModel()
        self._dataframe = dataframe if dataframe is not None else pd.DataFrame()
        self._row_offset = row_offset
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
        self._current_page = 0
        self._page_size = 10
        self._filtered_dataframe = pd.DataFrame()

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
        root.setContentsMargins(20, 9, 19, 20)
        root.setSpacing(0)

        header = QFrame()
        self.header = header
        header.setObjectName("datasetPreviewHeader")
        header_layout = QHBoxLayout(header)
        self.header_layout = header_layout
        header_layout.setContentsMargins(0, 0, 0, 2)
        title_block = QVBoxLayout()
        title_block.setSpacing(4)
        title = QLabel("Dataset Preview")
        title.setObjectName("pageTitle")
        title_block.addWidget(title)
        self.summary = QLabel(
            "Lihat dan jelajahi data Anda sebelum melakukan analisis lebih lanjut."
        )
        self.summary.setObjectName("pageSubtitle")
        self.summary.setWordWrap(True)
        self.summary.setMinimumWidth(0)
        self.summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        title_block.addWidget(self.summary)
        self.breadcrumb = QLabel("Dataset  ›  Pratinjau")
        self.breadcrumb.setObjectName("datasetPreviewBreadcrumb")
        title_block.addWidget(self.breadcrumb)
        header_layout.addLayout(title_block, 1)
        actions = QHBoxLayout()
        self.header_actions = actions
        actions.setContentsMargins(0, 1, 0, 0)
        actions.setSpacing(12)
        icon_root = Path(__file__).resolve().parent / "assets" / "icons"
        self.header_upload_button = QPushButton("Ganti Dataset")
        self.header_upload_button.setObjectName("datasetPreviewActionButton")
        self.header_upload_button.setIcon(QIcon(str(icon_root / "file-arrow-up.svg")))
        self.header_upload_button.setIconSize(QSize(16, 16))
        self.header_upload_button.setFixedSize(131, 40)
        self.header_upload_button.setAccessibleName("Ganti Dataset")
        self.header_upload_button.clicked.connect(self._replace_file)
        actions.addWidget(self.header_upload_button)
        self.clean_button = QPushButton("Bersihkan Data")
        self.clean_button.setObjectName("datasetPreviewActionButton")
        self.clean_button.setIcon(QIcon(str(icon_root / "about-dataset.svg")))
        self.clean_button.setIconSize(QSize(16, 16))
        self.clean_button.setFixedSize(139, 40)
        self.clean_button.clicked.connect(self._open_cleaning_actions)
        actions.addWidget(self.clean_button)
        self.continue_button = QPushButton("Lanjut ke Analisis")
        self.continue_button.setObjectName("datasetPreviewContinueButton")
        self.continue_button.setIcon(QIcon(str(icon_root / "arrow-right.svg")))
        self.continue_button.setIconSize(QSize(16, 16))
        self.continue_button.setFixedSize(178, 40)
        if self._main_window is not None:
            self.continue_button.clicked.connect(self._main_window.open_analysis_page)
        actions.addWidget(self.continue_button)
        header_layout.addLayout(actions)
        header_layout.setAlignment(actions, Qt.AlignTop)
        header.setFixedHeight(74)
        root.addWidget(header)

        overview = QWidget()
        self.overview = overview
        overview.setObjectName("datasetPreviewOverview")
        self.overview_layout = QGridLayout(overview)
        self.overview_layout.setContentsMargins(0, 0, 0, 0)
        self.overview_layout.setHorizontalSpacing(12)
        self.overview_layout.setVerticalSpacing(12)
        file_card = QFrame()
        self.file_card = file_card
        file_card.setObjectName("datasetPreviewFileCard")
        file_card.setMinimumHeight(109)
        file_card.setMaximumHeight(109)
        file_layout = QVBoxLayout(file_card)
        self.file_layout = file_layout
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(0)
        file_identity = QFrame()
        self.file_identity = file_identity
        file_identity.setObjectName("datasetPreviewFileIdentity")
        file_identity.setFixedHeight(109)
        identity_layout = QHBoxLayout(file_identity)
        identity_layout.setContentsMargins(30, 0, 12, 0)
        identity_layout.setSpacing(19)
        file_icon = QLabel()
        file_icon.setObjectName("datasetPreviewFileIcon")
        file_icon.setFixedSize(46, 56)
        file_icon.setAlignment(Qt.AlignCenter)
        icon_path = Path(__file__).resolve().parent / "assets" / "icons" / "file-text.svg"
        file_icon.setPixmap(
            render_icon_pixmap(icon_path, 29, file_icon.devicePixelRatioF())
        )
        file_icon.setToolTip("CSV")
        identity_layout.addWidget(file_icon, 0, Qt.AlignVCenter)
        file_text_layout = QVBoxLayout()
        file_text_layout.setSpacing(2)
        self.file_name = QLabel("Belum ada dataset yang dimuat")
        self.file_name.setObjectName("datasetPreviewFileName")
        self.file_name.setWordWrap(False)
        self.file_details = QLabel("Pilih file CSV untuk melihat pratinjau dataset.")
        self.file_details.setObjectName("datasetPreviewFileDetails")
        self.file_details.setWordWrap(True)
        file_text_layout.addWidget(self.file_name)
        file_text_layout.addWidget(self.file_details)
        identity_layout.addLayout(file_text_layout, 1)
        self.loaded_badge = QLabel("✓ Tersedia")
        self.loaded_badge.setObjectName("datasetPreviewLoadedBadge")
        identity_layout.addWidget(self.loaded_badge, 0, Qt.AlignVCenter)
        file_layout.addWidget(file_identity)
        file_actions = QHBoxLayout()
        self.file_actions = file_actions
        file_actions.setContentsMargins(0, 0, 0, 0)
        self.replace_button = QPushButton("Ganti file")
        self.replace_button.setObjectName("secondaryButton")
        self.replace_button.setFixedSize(162, 46)
        self.replace_button.setIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "icons" / "reload.svg")))
        self.replace_button.setIconSize(QSize(17, 17))
        self.replace_button.clicked.connect(self._replace_file)
        file_actions.addWidget(self.replace_button)
        file_actions.addStretch(1)
        self.upload_new_button = QPushButton("Unggah baru")
        self.upload_new_button.setObjectName("secondaryButton")
        self.upload_new_button.setFixedSize(162, 46)
        self.upload_new_button.setIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "icons" / "upload-small.svg")))
        self.upload_new_button.setIconSize(QSize(17, 17))
        self.upload_new_button.clicked.connect(self._open_upload)
        file_actions.addWidget(self.upload_new_button)
        file_layout.addLayout(file_actions)
        file_layout.removeItem(file_actions)
        self.replace_button.hide()
        self.upload_new_button.hide()
        apply_card_shadow(file_card)
        self.overview_layout.addWidget(file_card, 0, 0)

        self.metric_cards = []
        self.metric_descriptions = {}
        self.rows_value = self._create_metric_card(self.overview_layout, "ROWS", "—", "Baris")
        self.columns_value = self._create_metric_card(self.overview_layout, "COLUMNS", "—", "Kolom")
        self.missing_value = self._create_metric_card(self.overview_layout, "MISSING CELLS", "—", "")
        self.duplicates_value = self._create_metric_card(self.overview_layout, "DUPLICATES", "—", "Baris identik")
        self.file_size_value = self._create_metric_card(self.overview_layout, "FILE SIZE", "—", "Ukuran file")
        root.addWidget(overview)
        overview.setMinimumHeight(109)
        overview.setMaximumHeight(109)
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
        tab_icons = (
            "file-text.svg", "overview-quality.svg", "about-dataset.svg",
            "overview-quality-folder.svg", "overview-quality.svg", "analysis.svg",
        )
        icon_root = Path(__file__).resolve().parent / "assets" / "icons"
        for index, text in enumerate(("Data Preview", "Kualitas Data", "Informasi Kolom", "Missing Values", "Deteksi Outlier", "Korelasi")):
            button = QPushButton(text)
            button.setObjectName("datasetPreviewTab")
            button.setIcon(QIcon(str(icon_root / tab_icons[index])))
            button.setIconSize(QSize(13, 13))
            button.setProperty("active", index == 0)
            button.setCheckable(False)
            self.tab_buttons.append(button)
            self.tab_strip_layout.addWidget(button)
            button.clicked.connect(lambda checked=False, tab=index: self._select_tab(tab))
        self.tab_strip_layout.addStretch(1)
        self.tab_strip.adjustSize()
        self.tabs_scroll.setWidget(self.tab_strip)
        tabs_layout.addWidget(self.tabs_scroll)
        tabs.setFixedHeight(45)
        root.addWidget(tabs)
        body = QHBoxLayout()
        self.body_layout = body
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(14)
        preview_card = QFrame()
        self.preview_card = preview_card
        preview_card.setObjectName("datasetPreviewTableCard")
        preview_card.setMinimumWidth(0)
        preview_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(12, 12, 12, 10)
        preview_layout.setSpacing(7)
        table_heading_widget = QWidget()
        self.table_heading_widget = table_heading_widget
        table_heading_widget.setFixedHeight(48)
        table_heading = QHBoxLayout(table_heading_widget)
        table_heading.setContentsMargins(2, 0, 2, 0)
        table_heading.setSpacing(8)
        preview_title = QLabel("Pratinjau Data")
        self.preview_title = preview_title
        preview_title.setObjectName("datasetPreviewSectionTitle")
        preview_title.setFixedHeight(22)
        title_block = QVBoxLayout()
        title_block.setContentsMargins(0, 0, 0, 0)
        title_block.setSpacing(1)
        title_block.addWidget(preview_title)
        self.row_count_label = QLabel("Memuat dataset aktif")
        self.row_count_label.setObjectName("datasetPreviewCount")
        title_block.addWidget(self.row_count_label)
        table_heading.addLayout(title_block)
        table_heading.addStretch(1)
        self.table_search = QLineEdit()
        self.table_search.setObjectName("datasetPreviewTableSearch")
        self.table_search.setPlaceholderText("Cari dalam data...")
        self.table_search.setClearButtonEnabled(True)
        self.table_search.setFixedWidth(178)
        self.table_search.addAction(
            QIcon(str(icon_root / "search.svg")), QLineEdit.LeadingPosition
        )
        self.table_search.textChanged.connect(self._filter_table)
        table_heading.addWidget(self.table_search)
        rows_label = QLabel("Tampilkan")
        rows_label.setObjectName("datasetPreviewControlLabel")
        table_heading.addWidget(rows_label)
        self.page_size_combo = QComboBox()
        self.page_size_combo.setObjectName("datasetPreviewPageSize")
        self.page_size_combo.addItems(("10", "25", "50"))
        self.page_size_combo.setFixedWidth(64)
        self.page_size_combo.currentTextChanged.connect(self._set_page_size)
        table_heading.addWidget(self.page_size_combo)
        table_heading.addWidget(QLabel("baris"))
        preview_layout.addWidget(table_heading_widget)

        self.table = DatasetTableView()
        self.table.setObjectName("datasetPreviewTable")
        self.model = DataFrameTableModel(parent=self.table, include_index=True)
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.setCornerButtonEnabled(False)
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(27)
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
        self.empty_state = QLabel("Belum ada dataset yang dimuat")
        self.empty_state.setObjectName("emptyState")
        self.empty_state.setAlignment(Qt.AlignCenter)
        self.empty_state.setMinimumHeight(100)
        self.empty_state.hide()
        preview_layout.addWidget(self.empty_state, 0, Qt.AlignTop)
        pagination = QHBoxLayout()
        pagination.setContentsMargins(4, 0, 2, 0)
        pagination.setSpacing(6)
        self.pagination_label = QLabel("Menampilkan 0 data")
        self.pagination_label.setObjectName("datasetPreviewPaginationLabel")
        pagination.addWidget(self.pagination_label, 1)
        self.previous_page_button = QPushButton("‹")
        self.previous_page_button.setObjectName("datasetPreviewPageButton")
        self.previous_page_button.setProperty("arrow", True)
        self.previous_page_button.setFixedSize(30, 30)
        self.previous_page_button.clicked.connect(lambda: self._change_page(-1))
        pagination.addWidget(self.previous_page_button)
        self.pagination_pages_layout = QHBoxLayout()
        self.pagination_pages_layout.setContentsMargins(0, 0, 0, 0)
        self.pagination_pages_layout.setSpacing(6)
        self.pagination_page_buttons = []
        pagination.addLayout(self.pagination_pages_layout)
        self.next_page_button = QPushButton("›")
        self.next_page_button.setObjectName("datasetPreviewPageButton")
        self.next_page_button.setProperty("arrow", True)
        self.next_page_button.setFixedSize(30, 30)
        self.next_page_button.clicked.connect(lambda: self._change_page(1))
        pagination.addWidget(self.next_page_button)
        preview_layout.addLayout(pagination)
        apply_card_shadow(preview_card)
        preview_left_column = QVBoxLayout()
        preview_left_column.setContentsMargins(0, 0, 0, 0)
        preview_left_column.setSpacing(15)
        preview_left_column.addWidget(preview_card, 1)
        self.tips_card = self._create_exploration_tips()
        preview_left_column.addWidget(self.tips_card)
        preview_card.setFixedHeight(469)
        body.addLayout(preview_left_column, 8)
        body.setAlignment(preview_left_column, Qt.AlignTop)

        columns_card = QFrame()
        self.columns_card = columns_card
        columns_card.setObjectName("datasetPreviewColumnsCard")
        columns_card.setMinimumWidth(280)
        columns_card.setMaximumWidth(350)
        columns_card.setMinimumWidth(350)
        columns_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        columns_layout = QVBoxLayout(columns_card)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(0)
        columns_heading = QHBoxLayout()
        columns_title = QLabel("Informasi Kolom")
        columns_title.setObjectName("datasetPreviewSectionTitle")
        columns_heading.addWidget(columns_title)
        columns_heading.addStretch(1)
        self.show_all_columns_button = QPushButton("Lihat Semua")
        self.show_all_columns_button.setObjectName("datasetPreviewShowAll")
        self.show_all_columns_button.clicked.connect(lambda: self._select_tab(2))
        columns_heading.addWidget(self.show_all_columns_button)
        self.column_count_label = QLabel("0")
        self.column_count_label.setObjectName("datasetPreviewCount")
        columns_heading.addWidget(self.column_count_label)
        columns_heading.setContentsMargins(16, 12, 16, 8)
        columns_layout.addLayout(columns_heading)
        column_info_header = QHBoxLayout()
        column_info_header.setContentsMargins(16, 0, 16, 7)
        column_info_header.setSpacing(8)
        for label_text, stretch in (("Kolom", 4), ("Tipe", 3), ("Informasi", 6)):
            label = QLabel(label_text)
            label.setObjectName("datasetPreviewColumnInfoHeader")
            column_info_header.addWidget(label, stretch)
        columns_layout.addLayout(column_info_header)
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
        self.footer = None
        self.update_dataset(None)
        self.page_scroll.setWidget(container)
        outer.addWidget(self.page_scroll)
        self._layout_columns = 0
        self._apply_responsive_layout(self.width())

    def _select_tab(self, index: int):
        self.content_stack.setCurrentIndex(index)
        self.tips_card.setVisible(index == 0)
        self.columns_card.setVisible(index == 0)
        if index != 0:
            self.content_stack.currentWidget().setProperty("activePreviewState", True)
        for tab_index, button in enumerate(self.tab_buttons):
            button.setProperty("active", tab_index == index)
            button.style().unpolish(button)
            button.style().polish(button)

    def _create_exploration_tips(self):
        icon_root = Path(__file__).resolve().parent / "assets" / "icons"
        card = QFrame()
        card.setObjectName("datasetPreviewTipsCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(10)
        heading = QHBoxLayout()
        title = QLabel("Tips Eksplorasi Data")
        title.setObjectName("datasetPreviewTipsTitle")
        heading.addWidget(title)
        heading.addStretch(1)
        guide = QPushButton("Lihat Panduan Lengkap  →")
        guide.setObjectName("datasetPreviewGuideButton")
        guide.clicked.connect(
            getattr(self._main_window, "open_help", lambda: None)
            if self._main_window is not None else lambda: None
        )
        heading.addWidget(guide)
        layout.addLayout(heading)
        steps = QHBoxLayout()
        steps.setContentsMargins(0, 0, 0, 0)
        steps.setSpacing(8)
        for number, title_text, body_text, icon_name in (
            ("1.", "Periksa struktur data", "Lihat tipe data, nilai unik, dan Missing Values.", "file-text.svg"),
            ("2.", "Gunakan tab Analisis", "Lakukan analisis statistik, korelasi, dan deteksi outlier.", "analysis.svg"),
            ("3.", "Bersihkan data (opsional)", "Tangani Missing Values dan duplikat sebelum membuat model.", "about-dataset.svg"),
            ("4.", "Lanjut ke ML Intelligence", "Pilih target variabel dan bangun model machine learning.", "ml-intelligence.svg"),
        ):
            item = QFrame()
            item.setObjectName("datasetPreviewTipItem")
            item_layout = QHBoxLayout(item)
            item_layout.setContentsMargins(8, 8, 7, 8)
            item_layout.setSpacing(8)
            icon = QLabel()
            icon.setObjectName("datasetPreviewTipIcon")
            icon.setFixedSize(28, 28)
            icon.setAlignment(Qt.AlignCenter)
            icon.setPixmap(
                render_icon_pixmap(
                    icon_root / icon_name, 16, icon.devicePixelRatioF()
                )
            )
            item_layout.addWidget(icon, 0, Qt.AlignTop)
            text_layout = QVBoxLayout()
            text_layout.setContentsMargins(0, 0, 0, 0)
            text_layout.setSpacing(3)
            heading_label = QLabel(f"{number} {title_text}")
            heading_label.setObjectName("datasetPreviewTipHeading")
            heading_label.setWordWrap(True)
            body_label = QLabel(body_text)
            body_label.setObjectName("datasetPreviewTipBody")
            body_label.setWordWrap(True)
            text_layout.addWidget(heading_label)
            text_layout.addWidget(body_label)
            item_layout.addLayout(text_layout, 1)
            steps.addWidget(item, 1)
        layout.addLayout(steps, 1)
        card.setFixedHeight(173)
        return card

    def _open_cleaning_actions(self):
        if self._main_window is None or self._dataframe is None:
            return
        analysis_page = getattr(self._main_window, "analysis_page", None)
        if analysis_page is None:
            return
        menu = QMenu(self)
        actions = (
            ("Imputasi Missing Values", "_handle_impute_missing"),
            ("Hapus baris dengan Missing Values", "_handle_drop_missing"),
            ("Hapus baris duplikat", "_handle_remove_duplicates"),
        )
        can_version = bool(
            getattr(self._main_window, "version_manager", None)
            and getattr(self._main_window, "current_project_id", None)
        )
        for label, handler_name in actions:
            action = menu.addAction(label)
            handler = getattr(analysis_page, handler_name, None)
            action.setEnabled(can_version and callable(handler))
            if callable(handler):
                action.triggered.connect(handler)
        if not can_version:
            menu.addSection("Simpan dataset sebagai project untuk membuat versi bersih")
        menu.exec(self.clean_button.mapToGlobal(self.clean_button.rect().bottomLeft()))

    def _filter_table(self, query: str):
        dataframe = self._dataframe
        if dataframe is None:
            self._filtered_dataframe = pd.DataFrame()
        elif not query.strip():
            self._filtered_dataframe = dataframe
        else:
            normalized_query = query.strip().casefold()
            matches = dataframe.astype(str).apply(
                lambda column: column.str.casefold().str.contains(
                    normalized_query, regex=False, na=False
                )
            )
            self._filtered_dataframe = dataframe.loc[matches.any(axis=1)]
        self._current_page = 0
        self._apply_table_page()

    def _set_page_size(self, value: str):
        try:
            self._page_size = max(1, int(value))
        except (TypeError, ValueError):
            self._page_size = 10
        self._current_page = 0
        self._apply_table_page()

    def _change_page(self, offset: int):
        total_pages = max(
            1,
            (len(self._filtered_dataframe) + self._page_size - 1) // self._page_size,
        )
        self._current_page = min(max(self._current_page + offset, 0), total_pages - 1)
        self._apply_table_page()

    def _apply_table_page(self):
        dataframe = self._filtered_dataframe
        row_start = self._current_page * self._page_size
        row_end = min(row_start + self._page_size, len(dataframe))
        self.model.set_dataframe(
            dataframe.iloc[row_start:row_end],
            row_offset=row_start,
        )
        total_pages = max(1, (len(dataframe) + self._page_size - 1) // self._page_size)
        self._render_pagination(total_pages)
        self.previous_page_button.setEnabled(self._current_page > 0)
        self.next_page_button.setEnabled(self._current_page + 1 < total_pages)
        if len(dataframe):
            self.pagination_label.setText(
                f"Menampilkan {row_start + 1}–{row_end} dari {len(dataframe):,} baris"
            )
            total_columns = self._dataframe.shape[1] if self._dataframe is not None else 0
            self.row_count_label.setText(
                f"Menampilkan {row_start + 1}–{row_end} dari total "
                f"{len(dataframe):,} baris dan {total_columns:,} kolom."
            )
        else:
            self.pagination_label.setText("Tidak ada baris yang cocok")
            self.row_count_label.setText("Tidak ada baris yang cocok dengan pencarian.")
        self._configure_preview_columns(dataframe)

    def _render_pagination(self, total_pages: int):
        while self.pagination_pages_layout.count():
            item = self.pagination_pages_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        current = self._current_page
        if total_pages <= 7:
            visible_pages: list[int | None] = list(range(total_pages))
        elif current < 4:
            visible_pages = [0, 1, 2, 3, 4, None, total_pages - 1]
        elif current >= total_pages - 4:
            visible_pages = [0, None, *range(total_pages - 5, total_pages)]
        else:
            visible_pages = [0, None, current - 1, current, current + 1, None, total_pages - 1]
        self.pagination_page_buttons = []
        for page in visible_pages:
            if page is None:
                label = QLabel("…")
                label.setObjectName("datasetPreviewPaginationEllipsis")
                label.setAlignment(Qt.AlignCenter)
                label.setFixedWidth(20)
                self.pagination_pages_layout.addWidget(label)
                continue
            button = QPushButton(str(page + 1))
            button.setObjectName("datasetPreviewPageButton")
            button.setProperty("active", page == current)
            button.setFixedSize(32, 32)
            button.clicked.connect(
                lambda checked=False, target_page=page: self._go_to_page(target_page)
            )
            self.pagination_page_buttons.append(button)
            self.pagination_pages_layout.addWidget(button)

    def _go_to_page(self, page: int):
        total_pages = max(
            1,
            (len(self._filtered_dataframe) + self._page_size - 1) // self._page_size,
        )
        self._current_page = min(max(int(page), 0), total_pages - 1)
        self._apply_table_page()

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
        display_title = {
            "Data Quality": "Data Quality",
            "Columns": "Informasi Kolom",
            "Missing Values": "Missing Values",
            "Outliers": "Deteksi Outlier",
            "Correlation": "Korelasi",
        }.get(title, title)
        heading = QLabel(display_title)
        heading.setObjectName("datasetPreviewSectionTitle")
        card_layout.addWidget(heading)
        display_descriptions = {
            "Data Quality": "Ringkasan kelengkapan dan konsistensi dataset.",
            "Columns": "Tipe, jumlah nilai unik, dan contoh isi setiap kolom.",
            "Missing Values": "Kolom yang memiliki nilai kosong.",
            "Outliers": "Perkiraan outlier numerik berdasarkan rentang IQR.",
            "Correlation": "Korelasi antar kolom numerik (Pearson).",
        }
        description_label = QLabel(display_descriptions.get(title, description))
        description_label.setObjectName("datasetPreviewAnalysisDescription")
        description_label.setWordWrap(True)
        card_layout.addWidget(description_label)
        summary = QLabel("") if title == "Data Quality" else None
        if summary is not None:
            summary.setObjectName("datasetPreviewAnalysisSummary")
            summary.setWordWrap(True)
            card_layout.addWidget(summary)
        if title == "Data Quality":
            card.setMaximumWidth(820)
            card_layout.setContentsMargins(15, 12, 15, 12)
            card_layout.setSpacing(6)
        table = DatasetTableView()
        table.setObjectName("datasetPreviewAnalysisTable")
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(27)
        table.horizontalHeader().setFixedHeight(34)
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
        compact = logical_width(width) < 760
        self.root_layout.setSpacing(0)
        self.root_layout.setContentsMargins(
            12 if compact else 20,
            14 if compact else 1,
            12 if compact else 19,
            14 if compact else 20,
        )
        narrow = logical_width(width) < 620
        self.header_layout.setDirection(
            QBoxLayout.TopToBottom if narrow else QBoxLayout.LeftToRight
        )
        self.header_actions.setDirection(
            QBoxLayout.TopToBottom if narrow else QBoxLayout.LeftToRight
        )
        self.header.setFixedHeight(210 if narrow else 74)
        self.header_upload_button.setFixedSize(190 if narrow else 131, 40)
        self.clean_button.setFixedSize(190 if narrow else 139, 40)
        self.continue_button.setFixedSize(190 if narrow else 178, 40)
        self.file_card.setMinimumHeight(109)
        self.file_card.setMaximumHeight(109)
        columns = 6 if logical_width(width) >= 1100 else 3 if logical_width(width) >= 700 else 2
        if columns != self._layout_columns:
            for column in range(self.overview_layout.columnCount()):
                self.overview_layout.setColumnMinimumWidth(column, 0)
                self.overview_layout.setColumnStretch(column, 0)
            self.overview_layout.removeWidget(self.file_card)
            for card in self.metric_cards:
                self.overview_layout.removeWidget(card)
            if columns == 6:
                self.overview_layout.addWidget(self.file_card, 0, 0)
                for index, card in enumerate(self.metric_cards, start=1):
                    self.overview_layout.addWidget(card, 0, index)
                self.overview_layout.setColumnStretch(0, 380)
                for column, stretch in enumerate((152, 138, 154, 154, 169), start=1):
                    self.overview_layout.setColumnStretch(column, stretch)
                self.overview_layout.setHorizontalSpacing(0)
            else:
                self.overview_layout.addWidget(self.file_card, 0, 0, 1, columns)
                for index, card in enumerate(self.metric_cards):
                    self.overview_layout.addWidget(
                        card, 1 + index // columns, index % columns
                    )
                for column in range(columns):
                    self.overview_layout.setColumnStretch(column, 1)
                self.overview_layout.setHorizontalSpacing(8)
            self.overview_layout.invalidate()
            self._layout_columns = columns

        overview_rows = 1 if columns == 6 else 3
        self.overview.setMinimumHeight(109 if overview_rows == 1 else 109 * overview_rows + 12 * (overview_rows - 1))
        self.overview.setMaximumHeight(self.overview.minimumHeight())

        self.body_layout.setDirection(
            QBoxLayout.TopToBottom if logical_width(width) < 1000 else QBoxLayout.LeftToRight
        )
        if logical_width(width) < 1000:
            self.preview_card.setMinimumHeight(360)
            self.preview_card.setMaximumHeight(16777215)
            self.columns_card.setMinimumHeight(220)
            self.columns_card.setMaximumHeight(16777215)
            self.columns_card.setMaximumWidth(16777215)
            self.columns_card.setMinimumWidth(0)
            self.tips_card.setMinimumHeight(173)
            self.content_stack.setMinimumHeight(520)
            self.content_stack.setMaximumHeight(16777215)
        else:
            self.preview_card.setFixedHeight(469)
            self.columns_card.setMinimumWidth(250)
            self.columns_card.setMaximumWidth(420)
            self.columns_card.setFixedHeight(674)
            self.columns_card.setFixedWidth(350)
            self.tips_card.setMinimumHeight(173)
            self.content_stack.setFixedHeight(674)
        self.tabs_scroll.setMinimumHeight(45)
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
        card.setMinimumHeight(83)
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(13, 12, 9, 12)
        card_layout.setSpacing(10)
        icon = QLabel()
        icon.setObjectName("datasetPreviewMetricIcon")
        icon.setFixedSize(49, 50)
        icon.setAlignment(Qt.AlignCenter)
        icon_names = {
            "ROWS": "file-text.svg",
            "COLUMNS": "dataset.svg",
            "MISSING CELLS": "overview-quality.svg",
            "DUPLICATES": "overview-quality-folder.svg",
            "FILE SIZE": "file-text.svg",
        }
        icon_path = Path(__file__).resolve().parent / "assets" / "icons" / icon_names.get(title, "file-text.svg")
        icon.setPixmap(
            render_icon_pixmap(icon_path, 23, icon.devicePixelRatioF())
        )
        card_layout.addWidget(icon, 0, Qt.AlignVCenter)
        details = QVBoxLayout()
        details.setContentsMargins(0, 0, 0, 0)
        details.setSpacing(1)
        display_titles = {
            "ROWS": "BARIS",
            "COLUMNS": "KOLOM",
            "MISSING CELLS": "MISSING VALUES",
            "DUPLICATES": "DUPLIKAT",
            "FILE SIZE": "UKURAN FILE",
        }
        heading = QLabel(display_titles.get(title, title))
        heading.setObjectName("datasetPreviewMetricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("datasetPreviewMetricValue")
        details.addWidget(heading)
        details.addWidget(value_label)
        description_label = QLabel(description)
        description_label.setObjectName("datasetPreviewMetricDescription")
        details.addWidget(description_label)
        card_layout.addLayout(details, 1)
        self.metric_descriptions[title] = description_label
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
        if name in {"target", "survived", "label", "class"}:
            kind = "Numeric" if pd.api.types.is_numeric_dtype(series.dtype) else "Categorical"
            return f"{kind} (Target)"
        if name in {"name", "ticket", "cabin", "description", "text"}:
            return "Text"
        if pd.api.types.is_bool_dtype(series.dtype):
            return "Categorical"
        if pd.api.types.is_string_dtype(series.dtype):
            return "Categorical"
        if pd.api.types.is_numeric_dtype(series.dtype):
            return "Numeric"
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
        self.column_count_label.setText(f"{len(dataframe.columns)} kolom")
        for row_index, column in enumerate(dataframe.columns[:6]):
            name = str(column)
            series = dataframe[column]
            missing = int(series.isna().sum())
            non_null = series.dropna()
            unique_count = int(non_null.nunique())
            row = QFrame()
            row.setObjectName("datasetPreviewColumnRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(16, 6, 16, 6)
            row_layout.setSpacing(8)
            column_name = QLabel(name)
            column_name.setObjectName("datasetPreviewColumnName")
            column_name.setMinimumWidth(0)
            row_layout.addWidget(column_name, 4)
            kind = QLabel(self._column_type(name, series))
            kind.setObjectName("datasetPreviewColumnType")
            kind.setProperty(
                "columnType",
                "target" if "(Target)" in kind.text() else kind.text().lower(),
            )
            kind.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            kind.setMaximumHeight(22)
            row_layout.addWidget(kind, 3)
            details = QLabel()
            details.setObjectName("datasetPreviewColumnInformation")
            details.setWordWrap(True)
            if pd.api.types.is_numeric_dtype(series.dtype):
                numeric_values = pd.to_numeric(non_null, errors="coerce").dropna()
                if numeric_values.empty:
                    detail_text = f"{unique_count:,} unik · {missing:,} Missing Values"
                else:
                    detail_text = (
                        f"{unique_count:,} unik · {missing:,} Missing Values\n"
                        f"Rentang: {numeric_values.min():g} – {numeric_values.max():g}"
                    )
            elif unique_count <= 6:
                value_counts = non_null.astype(str).value_counts().head(2)
                distribution = ", ".join(
                    f"{value}: {count:,} ({count / row_count * 100:.1f}%)"
                    for value, count in value_counts.items()
                )
                detail_text = f"{unique_count:,} unik · {missing:,} Missing Values"
                if distribution:
                    detail_text += f"\n{distribution}"
            else:
                sample = ", ".join(non_null.astype(str).head(3).tolist())
                detail_text = f"{unique_count:,} unik · {missing:,} Missing Values\n{sample}"
            details.setText(detail_text)
            row_layout.addWidget(details, 6)
            row.setMinimumHeight(55)
            self.columns_content_layout.addWidget(row)
        self.columns_content_layout.addStretch(1)

    def _configure_preview_columns(self, dataframe: pd.DataFrame):
        names = [str(name) for name in dataframe.columns]
        normalized = {name.casefold(): index for index, name in enumerate(names)}
        is_titanic = {
            "passengerid", "survived", "pclass"
        }.issubset(normalized) and len(names) >= 10
        widths = {
            "passengerid": 72,
            "survived": 58,
            "pclass": 45,
            "name": 130,
            "sex": 50,
            "gender": 50,
            "age": 44,
            "sibsp": 40,
            "parch": 42,
            "ticket": 94,
            "fare": 58,
            "cabin": 55,
            "embarked": 54,
        }
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.table.setColumnHidden(0, False)
        self.table.setColumnWidth(0, 24)
        for index, name in enumerate(names):
            lowered = name.casefold()
            self.table.setColumnHidden(index + 1, False)
            if lowered in widths:
                self.table.setColumnWidth(index + 1, widths[lowered])
            elif not is_titanic:
                self.table.setColumnWidth(index + 1, 90)

    def _set_analysis_table(self, title: str, dataframe: pd.DataFrame):
        table = self.analysis_tables[title]
        table.setModel(DataFrameTableModel(
            dataframe,
            table,
            correlation_heatmap=title == "Correlation",
        ))
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
            ("Baris", f"{row_count:,}"),
            ("Kolom", f"{column_count:,}"),
            ("Sel kosong", f"{int(missing_by_column.sum()):,}"),
            ("Baris lengkap", f"{int((~dataframe.isna().any(axis=1)).sum()):,}"),
            ("Baris duplikat", f"{duplicate_count:,}"),
            ("Kolom dengan Missing Values", f"{int((missing_by_column > 0).sum()):,}"),
        ]
        quality_table = pd.DataFrame(quality_rows, columns=["Ukuran", "Nilai"])
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
            percentage = count / max(len(values), 1) * 100
            outlier_rows.append((
                str(column),
                count,
                f"{percentage:.1f}%",
                "Perlu ditinjau" if count else "Aman",
            ))

        # Bound correlation work for wide/large datasets while preserving a useful preview.
        correlation_columns = list(numeric.columns[:20])
        correlation_data = numeric.loc[:, correlation_columns]
        if len(correlation_data) > 5000:
            correlation_data = correlation_data.sample(n=5000, random_state=0)
        correlation = correlation_data.corr().round(3) if len(correlation_columns) >= 2 else pd.DataFrame()
        if not correlation.empty:
            correlation.insert(0, "Kolom", correlation.index.astype(str))
            correlation = correlation.reset_index(drop=True)

        self._set_analysis_table("Data Quality", quality_table)
        self._set_analysis_table(
            "Columns",
            pd.DataFrame(column_rows, columns=["Kolom", "Tipe", "Terisi", "Unik", "Contoh nilai"]),
        )
        self._set_analysis_table(
            "Missing Values",
            pd.DataFrame(missing_rows, columns=["Kolom", "Missing Values", "Persentase"]),
        )
        self._set_analysis_table(
            "Outliers",
            pd.DataFrame(outlier_rows, columns=["Kolom numerik", "Baris Outliers", "Persentase", "Status"]),
        )
        self._set_analysis_table("Correlation", correlation)

    def update_dataset(self, dataframe: pd.DataFrame | None, filename: str = ""):
        self._dataframe = dataframe
        self._dataset_name = filename or "Dataset"
        self._filtered_dataframe = dataframe if dataframe is not None else pd.DataFrame()
        self._current_page = 0
        if hasattr(self, "table_search"):
            self.table_search.blockSignals(True)
            self.table_search.clear()
            self.table_search.blockSignals(False)
        if hasattr(self, "page_size_combo"):
            self.page_size_combo.blockSignals(True)
            self.page_size_combo.setCurrentText(str(self._page_size))
            self.page_size_combo.blockSignals(False)
        if hasattr(self, "model"):
            self._apply_table_page()
        if hasattr(self, "clean_button"):
            self.clean_button.setEnabled(dataframe is not None)
        if dataframe is None:
            self.summary.setText(
                "Lihat dan jelajahi data Anda sebelum melakukan analisis lebih lanjut."
            )
            self.file_name.setText("Belum ada dataset yang dimuat")
            self.file_details.setText("Pilih file CSV untuk melihat pratinjau dataset.")
            self.rows_value.setText("—")
            self.columns_value.setText("—")
            self.missing_value.setText("—")
            self.duplicates_value.setText("—")
            self.file_size_value.setText("—")
            self.row_count_label.setText("0 baris")
            self.column_count_label.setText("0 kolom")
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
        self.breadcrumb.setText(f"Dataset  ›  {self._dataset_name}  ›  Pratinjau")
        file_size = None
        if self._main_window is not None and self._main_window.current_file_path:
            try:
                file_size = Path(self._main_window.current_file_path).stat().st_size
            except OSError:
                file_size = None
        file_size_text = (
            f"{file_size / 1024:.1f} KB" if file_size is not None else "Ukuran file tidak tersedia"
        )
        file_details = f"{file_size_text}  ·  {rows:,} baris  ·  {columns:,} kolom"
        self.file_details.setText(file_details)
        self.rows_value.setText(f"{rows:,}")
        self.columns_value.setText(f"{columns:,}")
        self.missing_value.setText(f"{missing_cells:,}")
        self.duplicates_value.setText(f"{duplicates:,}")
        self.file_size_value.setText(
            f"{file_size / 1024:.1f} KB" if file_size is not None else "—"
        )
        missing_percent = missing_cells / max(rows * columns, 1) * 100
        self.metric_descriptions["ROWS"].setText("Baris Data")
        self.metric_descriptions["COLUMNS"].setText("Kolom")
        self.metric_descriptions["MISSING CELLS"].setText(f"({missing_percent:.1f}%)")
        self.metric_descriptions["DUPLICATES"].setText("Duplikat Baris")
        self.metric_descriptions["FILE SIZE"].setText("Ukuran File")
        self.loaded_badge.show()
        self._populate_column_rows(dataframe)
        self._update_analysis_views(dataframe, duplicates)
        self._configure_preview_columns(dataframe)
