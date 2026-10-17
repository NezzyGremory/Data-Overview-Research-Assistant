"""Loaded-dataset upload workflow shown by the Page 4 Figma frame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from app.ui.dataset_preview_page import DataFrameTableModel, DatasetTableView
from app.ui.style_utils import render_icon_pixmap


def _px(value: float) -> int:
    """Return a design-space pixel size; MainWindow applies the active scale."""
    return round(float(value))


class LoadedUploadPage(QWidget):
    """Figma's upload wizard state after a real dataset has been selected."""

    def __init__(self, upload_page):
        super().__init__()
        self.upload_page = upload_page
        self.main_window = upload_page.main_window
        self._dataframe: pd.DataFrame | None = None
        self._filename = ""
        self._icons = Path(__file__).resolve().parent / "assets" / "icons"
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("uploadLoadedScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        container = QWidget()
        container.setObjectName("uploadLoadedContainer")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(_px(24), _px(16), _px(22), _px(16))
        layout.setSpacing(0)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(_px(12))
        title_block = QVBoxLayout()
        title_block.setContentsMargins(0, 0, 0, 0)
        title_block.setSpacing(_px(2))
        title = QLabel("Dataset")
        title.setObjectName("uploadLoadedTitle")
        subtitle = QLabel(
            "Unggah, pratinjau, dan bersihkan dataset sebelum melakukan analisis."
        )
        subtitle.setObjectName("uploadLoadedSubtitle")
        subtitle.setWordWrap(True)
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block, 1)

        guide_button = QPushButton("Lihat Panduan")
        guide_button.setObjectName("uploadLoadedGuideButton")
        self._set_button_icon(guide_button, "help.svg", 16)
        guide_button.clicked.connect(self.main_window.open_help)
        header.addWidget(guide_button, 0, Qt.AlignTop)
        layout.addLayout(header)
        layout.addSpacing(_px(8))

        self.file_summary = self._build_file_summary()
        self.file_summary.setFixedHeight(_px(60))
        layout.addWidget(self.file_summary)
        layout.addSpacing(_px(8))

        self.preview_panel = self._build_preview_panel()
        self.preview_panel.setMinimumHeight(_px(460))
        layout.addWidget(self.preview_panel, 1)

        self.scroll.setWidget(container)
        outer.addWidget(self.scroll)

    def _set_button_icon(self, button: QPushButton, icon_name: str, size: int):
        path = self._icons / icon_name
        if path.exists():
            button.setIcon(QIcon(str(path)))
            button.setIconSize(QSize(_px(size), _px(size)))

    def _set_label_icon(self, label: QLabel, icon_name: str, size: int):
        path = self._icons / icon_name
        if path.exists():
            logical_size = _px(size)
            label.setPixmap(
                render_icon_pixmap(path, logical_size, label.devicePixelRatioF())
            )

    def _build_file_summary(self) -> QFrame:
        card = QFrame()
        card.setObjectName("uploadLoadedFileSummary")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(_px(22), 0, _px(22), 0)
        layout.setSpacing(_px(14))
        icon = QLabel()
        icon.setObjectName("uploadLoadedFileIcon")
        icon.setFixedSize(_px(34), _px(36))
        icon.setAlignment(Qt.AlignCenter)
        self._set_label_icon(icon, "figma-upload-file-spreadsheet.svg", 26)
        layout.addWidget(icon)

        identity = QVBoxLayout()
        identity.setContentsMargins(0, 0, 0, 0)
        identity.setSpacing(_px(2))
        self.filename_label = QLabel("—")
        self.filename_label.setObjectName("uploadLoadedFilename")
        self.file_details_label = QLabel("Ukuran file tidak tersedia")
        self.file_details_label.setObjectName("uploadLoadedFileDetails")
        identity.addWidget(self.filename_label)
        identity.addWidget(self.file_details_label)
        layout.addLayout(identity, 1)

        self.success_badge = QFrame()
        self.success_badge.setObjectName("uploadLoadedSuccessBadge")
        success_layout = QHBoxLayout(self.success_badge)
        success_layout.setContentsMargins(_px(9), _px(5), _px(9), _px(5))
        success_layout.setSpacing(_px(5))
        success_icon = QLabel()
        success_icon.setObjectName("uploadLoadedSuccessIcon")
        success_icon.setFixedSize(_px(14), _px(14))
        self._set_label_icon(success_icon, "figma-upload-check.svg", 14)
        success_layout.addWidget(success_icon)
        success_label = QLabel("Berhasil diunggah")
        success_label.setObjectName("uploadLoadedSuccessText")
        success_layout.addWidget(success_label)
        layout.addWidget(self.success_badge)

        replace = QPushButton("Ganti File")
        replace.setObjectName("uploadLoadedReplaceButton")
        replace.setIcon(QIcon(str(self._icons / "figma-upload-reload.svg")))
        replace.clicked.connect(self.upload_page.select_dataset)
        layout.addWidget(replace)
        self.delete_button = QPushButton("Hapus File")
        self.delete_button.setObjectName("uploadLoadedDeleteButton")
        self.delete_button.setIcon(QIcon(str(self._icons / "figma-upload-trash.svg")))
        self.delete_button.setEnabled(False)
        self.delete_button.setToolTip(
            "Penghapusan dataset aktif belum tersedia pada alur aplikasi saat ini."
        )
        layout.addWidget(self.delete_button)
        return card

    def _build_preview_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("uploadLoadedPreviewPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(_px(10), 0, _px(10), _px(8))
        layout.setSpacing(0)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(_px(16))
        self.tab_buttons: list[QPushButton] = []
        for index, name in enumerate((
            "Data Preview", "Informasi Dataset", "Kualitas Data", "Deteksi Tipe Kolom"
        )):
            button = QPushButton(name)
            button.setObjectName("uploadLoadedTab")
            button.setProperty("active", index == 0)
            button.clicked.connect(lambda checked=False, tab=index: self._open_tab(tab))
            self.tab_buttons.append(button)
            top.addWidget(button)
        top.addStretch(1)
        self.rows_summary = QLabel("0 baris")
        self.rows_summary.setObjectName("uploadLoadedRowsSummary")
        self.rows_summary.setAlignment(Qt.AlignCenter)
        top.addWidget(self.rows_summary)
        layout.addLayout(top)
        layout.addSpacing(_px(8))

        self.table = DatasetTableView()
        self.table.setObjectName("uploadLoadedTable")
        self.model = DataFrameTableModel(
            pd.DataFrame(), include_index=False, detailed_headers=False, parent=self.table
        )
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.setCornerButtonEnabled(False)
        self.table.setAlternatingRowColors(True)
        self.table.setTabKeyNavigation(True)
        self.table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.verticalHeader().setVisible(True)
        self.table.verticalHeader().setFixedWidth(_px(54))
        self.table.verticalHeader().setMinimumSectionSize(_px(22))
        self.table.verticalHeader().setDefaultSectionSize(_px(27))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setDefaultSectionSize(_px(120))
        self.table.horizontalHeader().setMinimumSectionSize(_px(48))
        self.table.horizontalHeader().setFixedHeight(_px(34))
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setMinimumHeight(_px(300))
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.table, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, _px(8), 0, 0)
        grid_hint = QLabel("Pilih sel lalu tekan Ctrl+C untuk menyalin")
        grid_hint.setObjectName("uploadLoadedGridHint")
        footer.addWidget(grid_hint)
        footer.addStretch(1)
        self.continue_button = QPushButton("Lanjut ke Informasi Dataset")
        self.continue_button.setObjectName("uploadLoadedContinueButton")
        self._set_button_icon(self.continue_button, "figma-upload-continue.svg", 16)
        self.continue_button.setLayoutDirection(Qt.RightToLeft)
        self.continue_button.clicked.connect(lambda: self._open_tab(1))
        footer.addWidget(self.continue_button)
        layout.addLayout(footer)
        return panel

    def update_dataset(self, dataframe: pd.DataFrame | None, filename: str = ""):
        self._dataframe = dataframe
        self._filename = filename or ""
        self._refresh_file_summary()
        self._refresh_table()

    def _refresh_file_summary(self):
        if self._dataframe is None:
            self.filename_label.setText("—")
            self.file_details_label.setText("Ukuran file tidak tersedia")
            self.success_badge.setVisible(False)
            return
        self.success_badge.setVisible(True)
        self.filename_label.setText(self._filename or "Dataset aktif")
        path_value = getattr(self.main_window, "current_file_path", None)
        file_size = None
        try:
            path = Path(path_value) if path_value else None
            if path is not None and path.is_file():
                file_size = path.stat().st_size / 1024
        except (OSError, TypeError, ValueError):
            file_size = None
        size_text = f"{file_size:.1f} KB" if file_size is not None else "Ukuran file tidak tersedia"
        self.file_details_label.setText(
            f"{len(self._dataframe):,} baris  •  {len(self._dataframe.columns)} kolom  •  {size_text}"
        )

    def _refresh_table(self):
        dataframe = self._dataframe
        visible = dataframe if dataframe is not None else pd.DataFrame()
        self.model.set_dataframe(visible, row_offset=0)
        self.rows_summary.setText(
            f"{len(visible):,} baris  |  scroll untuk menjelajahi"
        )
        self._resize_columns()

    def _resize_columns(self):
        if self._dataframe is None:
            return
        preferred_widths = {
            "PassengerId": 104, "Survived": 92, "Pclass": 86,
            "Name": 220, "Sex": 90, "Gender": 100, "Age": 84,
            "SibSp": 84, "Parch": 84, "Ticket": 156, "Fare": 104,
            "Cabin": 110, "Embarked": 110,
        }
        for column in range(self.model.columnCount()):
            name = str(self._dataframe.columns[column])
            series = self._dataframe.iloc[:, column]
            fallback = 105 if pd.api.types.is_numeric_dtype(series.dtype) else 168
            width = preferred_widths.get(name, fallback)
            self.table.setColumnWidth(column, _px(width))

    def _open_tab(self, tab_index: int):
        for index, button in enumerate(self.tab_buttons):
            active = index == tab_index
            button.setProperty("active", active)
            button.style().unpolish(button)
            button.style().polish(button)
        if tab_index == 0:
            return
        preview = self.main_window.dataset_preview_page
        preview.update_dataset(self._dataframe, self._filename)
        preview_tabs = {1: 2, 2: 1, 3: 2}
        self.main_window.datasets_tabs.setCurrentWidget(preview)
        self.main_window._show_page("dataset")
        preview._select_tab(preview_tabs[tab_index])

