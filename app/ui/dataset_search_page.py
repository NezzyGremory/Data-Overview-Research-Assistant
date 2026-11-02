from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from app.ui.scale import logical_width

from PySide6.QtCore import QObject, QThread, Qt, Signal, QUrl, QStandardPaths, QSettings, QSize
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QBoxLayout,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from app.ui.style_utils import apply_card_shadow, render_icon_pixmap

from app.dataset_search.huggingface import (
    DatasetFile,
    DatasetSearchResult,
    HuggingFaceDatasetClient,
)
from app.dataset_search.kaggle import KaggleDatasetClient


class DatasetSearchWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        hf_client: HuggingFaceDatasetClient,
        kaggle_client: KaggleDatasetClient,
        query: str,
        source: str = "all",
    ):
        super().__init__()
        self.hf_client = hf_client
        self.kaggle_client = kaggle_client
        self.query = query
        self.source = source

    def run(self):
        try:
            results: list[DatasetSearchResult] = []
            errors: list[str] = []

            # 1. Search Kaggle
            if self.source in {"all", "kaggle"}:
                try:
                    kaggle_results = self.kaggle_client.search(self.query, limit=15)
                    results.extend(kaggle_results)
                except Exception as exc:
                    errors.append(f"Kaggle: {exc}")

            # 2. Search Hugging Face
            if self.source in {"all", "huggingface"}:
                try:
                    hf_results = self.hf_client.search(self.query, limit=15)
                    results.extend(hf_results)
                except Exception as exc:
                    errors.append(f"Hugging Face: {exc}")

            if not results and errors:
                raise RuntimeError("; ".join(errors))

            # Deduplicate by dataset_id and sort by relevance score
            seen_ids = set()
            unique_results = []
            for r in results:
                if r.dataset_id not in seen_ids:
                    seen_ids.add(r.dataset_id)
                    unique_results.append(r)

            unique_results.sort(key=lambda r: r.relevance_score, reverse=True)
            self.finished.emit(unique_results)
        except Exception as exc:
            self.failed.emit(str(exc))


class DatasetDownloadWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        hf_client: HuggingFaceDatasetClient,
        kaggle_client: KaggleDatasetClient,
        dataset_id: str,
        filename: str,
        destination: Path,
    ):
        super().__init__()
        self.hf_client = hf_client
        self.kaggle_client = kaggle_client
        self.dataset_id = dataset_id
        self.filename = filename
        self.destination = destination

    def run(self):
        try:
            if self.dataset_id.startswith("kaggle:"):
                paths = self.kaggle_client.download_dataset_files(
                    self.dataset_id,
                    self.destination,
                    filename=self.filename,
                )
                self.finished.emit([str(path) for path in paths])
            else:
                path = self.hf_client.download_file(
                    self.dataset_id,
                    self.filename,
                    self.destination,
                )
                self.finished.emit(str(path))
        except Exception as exc:
            self.failed.emit(str(exc))


class DatasetSearchPage(QWidget):
    """Real-time online dataset discovery powered by Hugging Face Hub & Kaggle."""

    DOWNLOAD_DIRECTORY_SETTING = "dataset_search/download_directory"

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.hf_client = HuggingFaceDatasetClient()
        self.kaggle_client = KaggleDatasetClient()
        self.client = self.hf_client  # Backward compatibility
        self.results: list[DatasetSearchResult] = []
        self._search_thread: Optional[QThread] = None
        self._search_worker: Optional[DatasetSearchWorker] = None
        self._download_thread: Optional[QThread] = None
        self._download_worker: Optional[DatasetDownloadWorker] = None
        self._pending_analyze = False
        self._settings = QSettings()
        self._build_ui()

    def _build_ui(self):
        self.setProperty("page4DatasetSearch", True)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("datasetSearchScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(self.scroll)

        self.container = QWidget()
        self.container.setObjectName("datasetSearchContainer")
        body = QVBoxLayout(self.container)
        body.setContentsMargins(24, 24, 24, 16)
        body.setSpacing(0)

        title = QLabel("Cari Dataset Publik")
        title.setObjectName("datasetSearchTitle")

        subtitle = QLabel(
            "Temukan dataset publik dari Hugging Face Hub dan Kaggle, periksa format filenya, "
            "lalu unduh atau impor dataset tersebut secara langsung ke ruang kerja DORA."
        )
        subtitle.setObjectName("datasetSearchSubtitle")
        subtitle.setWordWrap(True)
        subtitle.setMinimumWidth(0)
        subtitle.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        subtitle.setMaximumWidth(1155)

        body.addWidget(title)
        body.addSpacing(4)
        body.addWidget(subtitle)
        body.addSpacing(19)

        search_card = QFrame()
        self.search_card = search_card
        search_card.setObjectName("datasetSearchCard")
        search_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        search_layout = QVBoxLayout(search_card)
        self.search_layout = search_layout
        search_layout.setContentsMargins(23, 21, 23, 13)
        search_layout.setSpacing(8)

        row = QHBoxLayout()
        self.search_controls_layout = row
        row.setSpacing(14)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("datasetSearchInput")
        self.search_input.setPlaceholderText("Masukkan topik atau kata kunci dataset")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setMinimumHeight(56)
        self.search_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.search_input.returnPressed.connect(self.search_datasets)

        self.source_combo = QComboBox()
        self.source_combo.setMinimumHeight(56)
        self.source_combo.setMinimumWidth(0)
        self.source_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.source_combo.setObjectName("datasetProviderCombo")
        self.source_combo.addItem("Dataset Kaggle", "kaggle")
        self.source_combo.addItem("Dataset Hugging Face", "huggingface")
        database_icon = Path(__file__).resolve().parent / "assets" / "icons" / "figma-search-database.svg"
        self.source_combo.addItem(QIcon(str(database_icon)), "Semua sumber", "all")
        self.source_combo.setCurrentIndex(self.source_combo.findData("all"))

        self.search_button = QPushButton("Cari")
        self.search_button.setObjectName("primaryButton")
        self.search_button.setProperty("datasetSearch", True)
        self.search_button.setMinimumHeight(57)
        self.search_button.setMinimumWidth(120)
        search_icon = Path(__file__).resolve().parent / "assets" / "icons" / "search-white.svg"
        if search_icon.exists():
            self.search_button.setIcon(QIcon(str(search_icon)))
            self.search_button.setIconSize(QSize(24, 24))
        self.search_button.clicked.connect(self.search_datasets)

        row.addWidget(self.search_input, 1)
        row.addWidget(self.source_combo)
        row.addWidget(self.search_button)
        search_layout.addLayout(row)

        sources_row = QHBoxLayout()
        self.sources_row_layout = sources_row
        sources_row.setObjectName("datasetSupportedSources")
        sources_row.setSpacing(12)
        sources_label = QLabel("Cari berdasarkan topik atau kata kunci. Sumber yang didukung:")
        sources_label.setObjectName("datasetSupportedSourcesLabel")
        sources_row.addWidget(sources_label)
        for provider in ("Hugging Face", "Kaggle"):
            badge = QLabel(provider)
            badge.setObjectName("datasetSourceBadge")
            badge.setAlignment(Qt.AlignCenter)
            sources_row.addWidget(badge)
        sources_row.addStretch(1)
        search_layout.addLayout(sources_row)
        apply_card_shadow(search_card)
        body.addWidget(search_card)
        body.addSpacing(12)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setFixedHeight(4)
        body.addWidget(self.progress)

        toolbar = QWidget()
        toolbar.setObjectName("datasetSearchResultsToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(12)
        self.status_label = QLabel("Cari dataset berdasarkan topik atau kata kunci.")
        self.status_label.setObjectName("datasetSearchStatus")
        toolbar_layout.addWidget(self.status_label, 1)
        self.sort_combo = QComboBox()
        self.sort_combo.setObjectName("datasetSearchSort")
        self.sort_combo.addItem("Paling relevan")
        self.sort_combo.setFixedSize(208, 40)
        self.sort_combo.setFocusPolicy(Qt.NoFocus)
        toolbar_layout.addWidget(self.sort_combo)
        body.addWidget(toolbar)
        body.addSpacing(8)

        self.download_location_panel = QFrame()
        self.download_location_panel.setObjectName("datasetSearchDownloadLocation")
        self.download_location_layout = QBoxLayout(
            QBoxLayout.LeftToRight, self.download_location_panel
        )
        self.download_location_layout.setContentsMargins(14, 10, 14, 10)
        self.download_location_layout.setSpacing(12)

        location_text_layout = QVBoxLayout()
        location_text_layout.setContentsMargins(0, 0, 0, 0)
        location_text_layout.setSpacing(3)
        location_title = QLabel("Lokasi unduhan dataset")
        location_title.setObjectName("datasetSearchDownloadLocationTitle")
        self.download_directory_label = QLabel()
        self.download_directory_label.setObjectName("datasetSearchDownloadLocationPath")
        self.download_directory_label.setWordWrap(True)
        self.download_directory_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.download_directory_label.setMinimumWidth(0)
        self.download_directory_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        location_text_layout.addWidget(location_title)
        location_text_layout.addWidget(self.download_directory_label)
        self.download_location_layout.addLayout(location_text_layout, 1)

        location_actions = QHBoxLayout()
        location_actions.setContentsMargins(0, 0, 0, 0)
        location_actions.setSpacing(8)
        self.choose_download_directory_button = QPushButton()
        self.choose_download_directory_button.setObjectName(
            "datasetSearchLocationAction"
        )
        self.choose_download_directory_button.setAccessibleName(
            "Pilih folder unduhan dataset"
        )
        self.choose_download_directory_button.setToolTip(
            "Ubah lokasi unduhan dataset"
        )
        self.choose_download_directory_button.setFixedSize(36, 36)
        choose_location_icon = (
            Path(__file__).resolve().parent
            / "assets"
            / "icons"
            / "folder-location.svg"
        )
        self.choose_download_directory_button.setIcon(
            QIcon(str(choose_location_icon))
        )
        self.choose_download_directory_button.setIconSize(QSize(18, 18))
        self.choose_download_directory_button.clicked.connect(
            lambda _checked=False: self._choose_download_directory()
        )
        self.open_download_directory_button = QPushButton()
        self.open_download_directory_button.setObjectName(
            "datasetSearchLocationAction"
        )
        self.open_download_directory_button.setAccessibleName(
            "Buka folder unduhan dataset"
        )
        self.open_download_directory_button.setToolTip(
            "Buka lokasi unduhan dataset"
        )
        self.open_download_directory_button.setFixedSize(36, 36)
        open_location_icon = (
            Path(__file__).resolve().parent
            / "assets"
            / "icons"
            / "folder-open-location.svg"
        )
        self.open_download_directory_button.setIcon(QIcon(str(open_location_icon)))
        self.open_download_directory_button.setIconSize(QSize(18, 18))
        self.open_download_directory_button.clicked.connect(
            lambda _checked=False: self._open_download_directory()
        )
        location_actions.addWidget(self.choose_download_directory_button)
        location_actions.addWidget(self.open_download_directory_button)
        self.download_location_layout.addLayout(location_actions)
        body.addWidget(self.download_location_panel)
        body.addSpacing(8)
        self._refresh_download_directory_label()

        self.results_container = QWidget()
        self.results_container.setObjectName("datasetSearchResults")
        self.results_container.setAttribute(Qt.WA_StyledBackground, True)
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(10)

        self.empty_label = QLabel(
            "Hasil pencarian akan muncul di sini setelah pencarian dijalankan."
        )
        self.empty_label.setObjectName("datasetSearchEmpty")
        self.empty_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.empty_label.setWordWrap(True)
        self.results_layout.addWidget(self.empty_label)

        body.addWidget(self.results_container, 1)
        self.scroll.setWidget(self.container)
        self._result_rows = []

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = self.scroll.viewport().width()
        compact = logical_width(width) < 760
        stacked_controls = logical_width(width) < 860
        self.container.layout().setContentsMargins(
            16 if compact else 24,
            16 if compact else 24,
            16 if compact else 24,
            12 if compact else 16,
        )
        self.search_controls_layout.setDirection(
            QBoxLayout.TopToBottom if stacked_controls else QBoxLayout.LeftToRight
        )
        self.sources_row_layout.setDirection(
            QBoxLayout.TopToBottom if stacked_controls else QBoxLayout.LeftToRight
        )
        self.download_location_layout.setDirection(
            QBoxLayout.TopToBottom if stacked_controls else QBoxLayout.LeftToRight
        )
        if stacked_controls:
            self.search_card.setMinimumHeight(218)
            self.search_card.setMaximumHeight(16777215)
            self.search_layout.setContentsMargins(16, 16, 16, 14)
            self.search_input.setMinimumSize(0, 48)
            self.search_input.setMaximumSize(16777215, 16777215)
            self.source_combo.setMinimumSize(0, 48)
            self.source_combo.setMaximumSize(16777215, 16777215)
            self.search_button.setMinimumSize(120, 48)
            self.search_button.setMaximumSize(16777215, 16777215)
            self.sort_combo.setFixedSize(168 if compact else 208, 40)
        else:
            self.search_card.setFixedHeight(135)
            self.search_layout.setContentsMargins(23, 21, 23, 13)
            self.search_input.setMinimumSize(0, 56)
            self.search_input.setMaximumSize(16777215, 56)
            self.source_combo.setMinimumSize(271, 56)
            self.source_combo.setMaximumSize(271, 56)
            self.search_button.setMinimumSize(236, 57)
            self.search_button.setMaximumSize(236, 57)
            self.sort_combo.setFixedSize(208, 40)

    def search_datasets(self):
        query = self.search_input.text().strip()
        if not query:
            self.status_label.setText("Masukkan kata kunci terlebih dahulu.")
            self.search_input.setFocus()
            return

        source = self.source_combo.currentData() or "all"
        source_label = self.source_combo.currentText()

        self._clear_results()
        self.status_label.setText(
            f'Mencari dataset untuk "{query}" di {source_label}...'
        )
        self.search_button.setEnabled(False)
        self.progress.setVisible(True)

        self._search_thread = QThread(self)
        self._search_worker = DatasetSearchWorker(
            self.hf_client,
            self.kaggle_client,
            query,
            source=source,
        )
        self._search_worker.moveToThread(self._search_thread)
        self._search_thread.started.connect(self._search_worker.run)
        self._search_worker.finished.connect(self._on_search_finished)
        self._search_worker.failed.connect(self._on_search_failed)
        self._search_worker.finished.connect(self._finish_search_thread)
        self._search_worker.failed.connect(self._finish_search_thread)
        self._search_thread.start()

    def _on_search_finished(self, results):
        self.results = list(results or [])
        self.progress.setVisible(False)
        self.search_button.setEnabled(True)

        if not self.results:
            self.status_label.setText(
                "Tidak ada dataset yang cocok. Coba kata kunci yang lebih umum atau ganti sumber pencarian."
            )
            self._show_empty_message("Tidak ada dataset ditemukan.")
            return

        self.status_label.setText(f"Ditemukan {len(self.results)} dataset")
        self._render_results()

    def _on_search_failed(self, error: str):
        self.progress.setVisible(False)
        self.search_button.setEnabled(True)
        self.status_label.setText(
            "Pencarian dataset gagal. Periksa koneksi internet lalu coba lagi."
        )
        self._show_empty_message(
            f"Gagal mencari dataset.\n\n{error}"
        )

    def _finish_search_thread(self, *_args):
        if self._search_thread is not None:
            self._search_thread.quit()
            self._search_thread.wait(1500)
        self._search_worker = None
        self._search_thread = None

    def _render_results(self):
        self._clear_results()
        for result in self.results:
            self.results_layout.addWidget(self._build_result_card(result))
        self.results_layout.addStretch(1)

    def _build_result_card(self, result: DatasetSearchResult) -> QFrame:
        card = QFrame()
        card.setObjectName("datasetResultCard")
        card.setMinimumHeight(0)
        card.setMinimumWidth(0)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        apply_card_shadow(card)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        # The source badge and all visible metadata use the selected provider's
        # actual search result; no presentation-only result values are added.
        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        heading = QWidget()
        heading_layout = QVBoxLayout(heading)
        heading_layout.setContentsMargins(0, 0, 0, 0)
        heading_layout.setSpacing(2)

        title = QLabel(result.title)
        title.setObjectName("datasetResultTitle")
        title.setWordWrap(False)
        title.setMinimumWidth(0)
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        heading_layout.addWidget(title)

        clean_repo = result.dataset_id.replace("kaggle:", "")
        repo_label = QLabel(f"ID: {clean_repo}")
        repo_label.setObjectName("datasetResultId")
        repo_label.setWordWrap(False)
        repo_label.setMinimumWidth(0)
        repo_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        heading_layout.addWidget(repo_label)
        header_row.addWidget(heading, 1)

        source_badge = QLabel(f" {result.source_label} ")
        source_badge.setObjectName("scoreBadge")
        source_badge.setMinimumSize(0, 30)
        source_badge.setAlignment(Qt.AlignCenter)
        header_row.addWidget(source_badge)
        layout.addLayout(header_row)

        description_text = result.description or "Deskripsi tidak tersedia dari sumber dataset."
        if result.match_reason:
            description_text = f"{description_text} Alasan kecocokan: {result.match_reason}"
        description = QLabel(description_text)
        description.setObjectName("datasetResultDescription")
        description.setWordWrap(True)
        description.setMinimumWidth(0)
        description.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(description)

        stats_row = QHBoxLayout()
        stats_row.setObjectName("datasetResultStats")
        stats_row.setContentsMargins(0, 0, 0, 0)
        stats_row.setSpacing(8)
        stats = []
        formats = list(result.formats)
        if not formats:
            formats = list(dict.fromkeys(
                Path(item.filename.split(" ", 1)[0]).suffix.lstrip(".").lower()
                for item in result.files
                if Path(item.filename.split(" ", 1)[0]).suffix
            ))
        if formats:
            stats.append((", ".join(item.upper() for item in formats[:3]), "figma-search-file.svg"))
        known_sizes = [item.display_size for item in result.files if item.size_bytes is not None]
        if known_sizes:
            stats.append((known_sizes[0], "figma-search-database.svg"))
        if result.downloads > 0:
            stats.append((f"{result.downloads:,} downloads", "figma-search-download.svg"))
        if result.likes > 0:
            stats.append((f"{result.likes:,} likes", "figma-search-heart.svg"))

        icon_dir = Path(__file__).resolve().parent / "assets" / "icons"
        for text, icon_name in stats:
            chip = QFrame()
            chip.setObjectName("datasetSearchMetadataChip")
            chip_layout = QHBoxLayout(chip)
            chip_layout.setContentsMargins(10, 5, 10, 5)
            chip_layout.setSpacing(6)
            icon_label = QLabel()
            icon_label.setObjectName("datasetSearchMetadataIcon")
            icon_path = icon_dir / icon_name
            if icon_path.exists():
                icon_label.setPixmap(
                    render_icon_pixmap(
                        icon_path, 16, icon_label.devicePixelRatioF()
                    )
                )
            chip_layout.addWidget(icon_label)
            chip_text = QLabel(text)
            chip_text.setObjectName("datasetSearchMetadataText")
            chip_layout.addWidget(chip_text)
            stats_row.addWidget(chip)
        stats_row.addStretch(1)
        layout.addLayout(stats_row)

        divider = QFrame()
        divider.setObjectName("searchResultDivider")
        divider.setFixedHeight(1)
        layout.addWidget(divider)

        # Determine file choices
        files = result.files
        can_download = bool(files) or result.dataset_id.startswith("kaggle:")
        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(10)
        file_row = QHBoxLayout()
        file_row.setSpacing(8)
        files_label = QLabel(f"Files ({len(files)})")
        files_label.setObjectName("datasetFilesCount")
        file_row.addWidget(files_label)
        if files:
            folder_label = QLabel()
            folder_icon = icon_dir / "figma-search-folder.svg"
            if folder_icon.exists():
                folder_label.setPixmap(
                    render_icon_pixmap(
                        folder_icon, 20, folder_label.devicePixelRatioF()
                    )
                )
            folder_label.setObjectName("datasetFilesFolderIcon")
            file_row.addWidget(folder_label)
            combo = QComboBox()
            combo.setObjectName("datasetFileSelect")
            combo.setFixedHeight(35)
            combo.setMinimumWidth(0)
            for file in files[:12]:
                combo.addItem(file.filename, file.filename)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            file_row.addWidget(combo, 1)
            size_label = QLabel()
            size_label.setObjectName("datasetSelectedFileSize")
            size_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            size_label.setText(files[0].display_size)
            combo.currentIndexChanged.connect(
                lambda index, values=files, label=size_label: label.setText(
                    values[index].display_size if 0 <= index < len(values) else ""
                )
            )
            file_row.addWidget(size_label)
        else:
            unavailable = QLabel("File tabular tidak tersedia dari sumber ini")
            unavailable.setObjectName("datasetFilesUnavailable")
            file_row.addWidget(unavailable, 1)

        download_button = QPushButton()
        download_button.setObjectName("datasetSearchDownloadButton")
        download_button.setAccessibleName("Unduh dataset")
        download_button.setToolTip("Unduh dataset")
        download_button.setFixedSize(32, 32)
        download_icon = icon_dir / "figma-search-download-action.svg"
        if download_icon.exists():
            download_button.setIcon(QIcon(str(download_icon)))
            download_button.setIconSize(QSize(22, 22))
        download_button.setEnabled(can_download)
        download_button.clicked.connect(
            lambda _=False, r=result, c=combo if files else None: self._download_result(
                r,
                c.currentData() if c is not None else None,
                analyze=False,
            )
        )
        file_row.addWidget(download_button)
        actions_row.addLayout(file_row, 1)

        open_button = QPushButton("Buka sumber")
        open_button.setObjectName("secondaryButton")
        open_button.setFixedSize(160, 36)
        open_icon = icon_dir / "open-source.svg"
        if open_icon.exists():
            open_button.setIcon(QIcon(str(open_icon)))
            open_button.setIconSize(QSize(18, 18))
        open_button.clicked.connect(
            lambda _=False, url=result.url: QDesktopServices.openUrl(QUrl(url))
        )

        analyze_button = QPushButton("Impor && Analisis")
        analyze_button.setObjectName("datasetAnalyzeButton")
        analyze_button.setFixedSize(180, 36)
        analyze_button.setAccessibleName("Impor dan analisis dataset")
        analyze_icon = icon_dir / "figma-search-analyze.svg"
        if analyze_icon.exists():
            analyze_button.setIcon(QIcon(str(analyze_icon)))
            analyze_button.setIconSize(QSize(18, 18))
        analyze_button.setEnabled(can_download)
        analyze_button.setToolTip(
            "" if can_download else "Tidak ada format tabular yang dapat diunduh langsung."
        )
        analyze_button.clicked.connect(
            lambda _=False, r=result, c=combo if files else None: self._download_result(
                r,
                c.currentData() if c is not None else None,
                analyze=True,
            )
        )
        actions_row.addWidget(open_button)
        actions_row.addWidget(analyze_button)
        layout.addLayout(actions_row)

        return card

    def _download_result(
        self,
        result: DatasetSearchResult,
        filename: Optional[str],
        analyze: bool,
    ):
        if not filename and not result.dataset_id.startswith("kaggle:"):
            QMessageBox.warning(
                self,
                "Unduh dataset",
                "Pilih file dataset terlebih dahulu.",
            )
            return

        try:
            destination = self._download_directory()
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Lokasi unduhan tidak dapat digunakan",
                f"Pilih folder unduhan lain lalu coba kembali.\n\n{exc}",
            )
            return

        self._pending_analyze = bool(analyze)
        self.search_button.setEnabled(False)
        self.choose_download_directory_button.setEnabled(False)
        self.progress.setVisible(True)

        clean_name = filename or result.dataset_id.split("/")[-1]
        self.status_label.setText(
            f"Mengunduh {clean_name} dari {result.source_label}..."
        )

        self._download_thread = QThread(self)
        self._download_worker = DatasetDownloadWorker(
            self.hf_client,
            self.kaggle_client,
            result.dataset_id,
            filename or "",
            destination,
        )
        self._download_worker.moveToThread(self._download_thread)
        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.failed.connect(self._on_download_failed)
        self._download_worker.finished.connect(self._finish_download_thread)
        self._download_worker.failed.connect(self._finish_download_thread)
        self._download_thread.start()

    def _on_download_finished(self, download_result):
        self.search_button.setEnabled(True)
        self.choose_download_directory_button.setEnabled(True)
        self.progress.setVisible(False)
        raw_paths = (
            download_result
            if isinstance(download_result, (list, tuple))
            else [download_result]
        )
        downloaded_paths = [Path(path) for path in raw_paths]
        if not downloaded_paths:
            self._on_download_failed("Tidak ada file tabular yang berhasil diunduh.")
            return

        if not self._pending_analyze:
            file_list = "\n".join(str(path) for path in downloaded_paths)
            self.status_label.setText(
                f"{len(downloaded_paths)} file dataset berhasil diunduh."
            )
            QMessageBox.information(
                self,
                "Unduhan berhasil",
                "File dataset disimpan di folder unduhan yang dipilih:\n\n"
                f"{file_list}",
            )
            return

        if len(downloaded_paths) == 1:
            selected_path = downloaded_paths[0]
        else:
            labels = [
                f"{path.name} ({path.stat().st_size / 1024:.1f} KB)"
                for path in downloaded_paths
            ]
            selected_label, accepted = QInputDialog.getItem(
                self,
                "Pilih File untuk Analisis",
                "Arsip berisi beberapa file. Pilih satu file untuk dianalisis:",
                labels,
                0,
                False,
            )
            if not accepted:
                self.status_label.setText(
                    f"{len(downloaded_paths)} file berhasil diunduh; analisis dibatalkan."
                )
                return
            selected_path = downloaded_paths[labels.index(selected_label)]

        try:
            dataframe = self._load_downloaded_dataset(selected_path)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Dataset Berhasil Diunduh",
                "File berhasil diunduh, tetapi tidak dapat dibaca sebagai dataset.\n\n"
                f"{exc}",
            )
            return

        try:
            self.main_window.current_dataset = dataframe
            self.main_window.current_file_path = str(selected_path)
            self.main_window.update_dataset_state()
            self.status_label.setText(
                "Dataset tersimpan di folder unduhan dan dimuat untuk analisis:\n"
                f"{selected_path}"
            )
            self.main_window.open_analysis_page(on_ready=self._run_imported_analysis)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Dataset Berhasil Diunduh",
                "Dataset sudah dimuat, tetapi analisis belum dapat dimulai.\n\n"
                f"{exc}",
            )

    def _run_imported_analysis(self, page):
        try:
            page.run_analysis()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Analisis Belum Dimulai",
                "Dataset sudah dimuat, tetapi analisis belum dapat dimulai.\n\n"
                f"{exc}",
            )

    def _on_download_failed(self, error: str):
        self.search_button.setEnabled(True)
        self.choose_download_directory_button.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText(
            "Unduhan gagal. Coba file lain atau buka sumber dataset."
        )
        QMessageBox.warning(
            self,
                "Unduhan gagal",
            f"Dataset gagal diunduh.\n\n{error}",
        )

    def _finish_download_thread(self, *_args):
        if self._download_thread is not None:
            self._download_thread.quit()
            self._download_thread.wait(1500)
        self._download_worker = None
        self._download_thread = None

    def _load_downloaded_dataset(self, path: Path):
        return self.main_window.loader.load(str(path))

    def _default_download_directory(self) -> Path:
        downloads_location = QStandardPaths.writableLocation(
            QStandardPaths.DownloadLocation
        )
        destination = (
            Path(downloads_location)
            if downloads_location
            else Path.home() / "Downloads"
        )
        return destination / "Dataset Research"

    def _current_download_directory(self) -> Path:
        configured = str(
            self._settings.value(self.DOWNLOAD_DIRECTORY_SETTING, "") or ""
        ).strip()
        if configured:
            return Path(configured).expanduser()
        return self._default_download_directory()

    def _refresh_download_directory_label(self):
        destination = self._current_download_directory()
        self.download_directory_label.setText(str(destination))
        self.download_directory_label.setToolTip(str(destination))

    def _choose_download_directory(self):
        current = self._current_download_directory()
        if not current.is_dir():
            current = current.parent if current.parent.is_dir() else Path.home()
        selected = QFileDialog.getExistingDirectory(
            self,
            "Pilih folder unduhan dataset",
            str(current),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected:
            return

        destination = Path(selected).expanduser()
        if not destination.is_dir():
            QMessageBox.warning(
                self,
                "Folder tidak ditemukan",
                "Pilih folder yang masih tersedia.",
            )
            return

        self._settings.setValue(
            self.DOWNLOAD_DIRECTORY_SETTING, str(destination)
        )
        self._settings.sync()
        self._refresh_download_directory_label()
        self.status_label.setText(
            f"Unduhan dataset akan disimpan di: {destination}"
        )

    def _download_directory(self) -> Path:
        destination = self._current_download_directory()
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def _open_download_directory(self):
        try:
            destination = self._download_directory()
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Lokasi unduhan tidak dapat digunakan",
                f"Pilih folder unduhan lain lalu coba kembali.\n\n{exc}",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(destination)))

    def _clear_results(self):
        self._result_rows = []
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _show_empty_message(self, message: str):
        self._clear_results()
        label = QLabel(message)
        label.setObjectName("datasetSearchEmpty")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        self.results_layout.addWidget(label)
        self.results_layout.addStretch()
