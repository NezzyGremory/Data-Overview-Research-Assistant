from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PySide6.QtCore import QObject, QThread, Qt, Signal, QUrl, QTimer, QStandardPaths, QSize
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
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
from app.ui.style_utils import apply_card_shadow

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
    finished = Signal(str)
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
                path = self.kaggle_client.download_dataset(
                    self.dataset_id,
                    self.destination,
                    filename=self.filename,
                )
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
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 29, 28, 16)
        root.setSpacing(18)

        title = QLabel("Dataset Search")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Search for public datasets on Hugging Face Hub and Kaggle, check their "
            "file formats, and then download them directly to DORA workspace. "
            "Examples: Titanic, Iris, agriculture, students, healthcare, stocks, and home prices."
        )
        subtitle.setObjectName("datasetSearchSubtitle")
        subtitle.setWordWrap(True)

        root.addWidget(title)
        subtitle.setTextFormat(Qt.RichText)
        subtitle.setText(
            "Search for public datasets on Hugging Face Hub and Kaggle, check their file formats, "
            "and then download them directly to DORA workspace. <b>Examples: Titanic, Iris, agriculture, "
            "students, healthcare, stocks, and home prices.</b>"
        )
        root.addWidget(subtitle)

        search_card = QFrame()
        search_card.setObjectName("datasetSearchCard")
        search_card.setMinimumHeight(163)
        search_layout = QVBoxLayout(search_card)
        search_layout.setContentsMargins(22, 26, 22, 22)
        search_layout.setSpacing(14)

        row = QHBoxLayout()
        row.setSpacing(10)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("datasetSearchInput")
        self.search_input.setText("student")
        self.search_input.setPlaceholderText("Search datasets")
        self.search_input.setMinimumHeight(66)
        self.search_input.returnPressed.connect(self.search_datasets)

        self.source_combo = QComboBox()
        self.source_combo.setMinimumHeight(66)
        self.source_combo.setMinimumWidth(260)
        self.source_combo.setObjectName("datasetProviderCombo")
        self.source_combo.addItem("Kaggle dataset", "kaggle")
        self.source_combo.addItem("Hugging Face dataset", "huggingface")
        self.source_combo.addItem("All providers", "all")

        self.search_button = QPushButton("Search")
        self.search_button.setObjectName("primaryButton")
        self.search_button.setProperty("datasetSearch", True)
        self.search_button.setMinimumHeight(66)
        self.search_button.setMinimumWidth(150)
        search_icon = Path(__file__).resolve().parent / "assets" / "icons" / "search-white.svg"
        if search_icon.exists():
            self.search_button.setIcon(QIcon(str(search_icon)))
            self.search_button.setIconSize(QSize(24, 24))
        self.search_button.clicked.connect(self.search_datasets)

        row.addWidget(self.search_input, 1)
        row.addWidget(self.source_combo)
        row.addWidget(self.search_button)
        search_layout.addLayout(row)

        self.status_label = QLabel("Search for public datasets, then choose a file format to download.")
        self.status_label.setObjectName("cardDescription")
        self.status_label.setWordWrap(True)
        search_layout.addWidget(self.status_label)
        apply_card_shadow(search_card)
        root.addWidget(search_card)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setFixedHeight(6)
        root.addWidget(self.progress)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.results_container = QWidget()
        self.results_container.setObjectName("datasetSearchResults")
        self.results_container.setAttribute(Qt.WA_StyledBackground, True)
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 8, 0)
        self.results_layout.setSpacing(12)

        self.empty_label = QLabel(
            "Search results will appear here."
        )
        self.empty_label.setObjectName("cardDescription")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(self.empty_label)
        self.results_layout.addStretch()

        self.scroll.setWidget(self.results_container)
        root.addWidget(self.scroll, 1)

    def search_datasets(self):
        query = self.search_input.text().strip()
        if not query:
            self.status_label.setText("Masukkan keyword terlebih dahulu.")
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
                "Tidak ada dataset yang cocok. Coba keyword yang lebih umum atau ganti sumber pencarian."
            )
            self._show_empty_message("Tidak ada dataset ditemukan.")
            return

        self.status_label.setText(
            f"{len(self.results)} datasets found. Results are ranked by topic match and metadata. "
            "Choose a file format, then Download or Download & Analyze."
        )
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

        for start in range(0, len(self.results), 2):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(12)
            row_layout.addWidget(self._build_result_card(self.results[start]), 1)
            if start + 1 < len(self.results):
                row_layout.addWidget(self._build_result_card(self.results[start + 1]), 1)
            else:
                row_layout.addStretch(1)
            self.results_layout.addWidget(row)

        self.results_layout.addStretch()

    def _build_result_card(self, result: DatasetSearchResult) -> QFrame:
        card = QFrame()
        card.setObjectName("datasetResultCard")
        card.setMinimumHeight(370)
        apply_card_shadow(card)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(9)

        # Header: Title + Source badge
        header_row = QHBoxLayout()
        header_row.setSpacing(10)

        title = QLabel(result.title)
        title.setObjectName("datasetResultTitle")
        title.setWordWrap(True)
        header_row.addWidget(title, 1)

        source_badge = QLabel(f" {result.source_label} ")
        source_badge.setObjectName("scoreBadge")
        header_row.addWidget(source_badge)

        layout.addLayout(header_row)

        clean_repo = result.dataset_id.replace("kaggle:", "")
        repo_label = QLabel(f"ID: {clean_repo}")
        repo_label.setObjectName("cardDescription")

        description = QLabel(result.description or "No dataset description available.")
        description.setObjectName("cardDescription")
        description.setWordWrap(True)

        explanation = QLabel(
            "Why it matches: " + (
                result.match_reason
                or "Matched from the dataset search keyword."
            )
        )
        explanation.setObjectName("cardDescription")
        explanation.setWordWrap(True)

        meta_parts = [
            f"Source: {result.source_label}",
            f"Downloads: {result.downloads:,}",
            f"Votes/Likes: {result.likes:,}",
        ]
        if result.author:
            meta_parts.insert(1, f"Author: {result.author}")

        meta = QLabel("  |  ".join(meta_parts))
        meta.setObjectName("cardDescription")
        meta.setWordWrap(True)

        layout.addWidget(repo_label)
        layout.addWidget(description)
        layout.addWidget(explanation)
        layout.addWidget(meta)

        if result.tags:
            tag_text = ", ".join(result.tags[:1])
            tags = QLabel(tag_text)
            tags.setObjectName("datasetResultTag")
            tags.setWordWrap(True)
            tags.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            layout.addWidget(tags)

        # Determine file choices
        files = result.files
        if not files and result.dataset_id.startswith("kaggle:"):
            files = [DatasetFile(filename="Dataset Archive (.csv Auto-Extract)")]

        if files:
            file_row = QHBoxLayout()
            file_row.setSpacing(8)

            combo = QComboBox()
            combo.setObjectName("datasetFileSelect")
            combo.setMinimumHeight(54)
            for file in files[:12]:
                combo.addItem(
                    f"{file.filename}  ({file.display_size})",
                    file.filename,
                )
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            file_row.addWidget(combo, 1)

            layout.addLayout(file_row)

            buttons = QHBoxLayout()
            buttons.setSpacing(8)

            open_button = QPushButton("Open Source")
            open_button.setObjectName("secondaryButton")
            open_icon = Path(__file__).resolve().parent / "assets" / "icons" / "open-source.svg"
            if open_icon.exists():
                open_button.setIcon(QIcon(str(open_icon)))
                open_button.setIconSize(QSize(18, 18))
            open_button.clicked.connect(
                lambda _=False, url=result.url: QDesktopServices.openUrl(
                    QUrl(url)
                )
            )

            analyze_button = QPushButton("Download && Analyze")
            analyze_button.setObjectName("datasetAnalyzeButton")
            analyze_button.setAccessibleName("Download & Analyze")
            analyze_icon = Path(__file__).resolve().parent / "assets" / "icons" / "download.svg"
            if analyze_icon.exists():
                analyze_button.setIcon(QIcon(str(analyze_icon)))
                analyze_button.setIconSize(QSize(18, 18))
            analyze_button.clicked.connect(
                lambda _=False, r=result, c=combo: self._download_result(
                    r,
                    c.currentData(),
                    analyze=True,
                )
            )

            buttons.addStretch()
            buttons.addWidget(open_button)
            buttons.addWidget(analyze_button)
            layout.addLayout(buttons)
        else:
            note = QLabel(
                "Tidak ditemukan file tabular yang didukung untuk diunduh langsung. "
                "Buka halaman sumber untuk melihat isi dataset."
            )
            note.setObjectName("cardDescription")
            note.setWordWrap(True)
            layout.addWidget(note)

            open_button = QPushButton("Open Source")
            open_button.setObjectName("secondaryButton")
            open_button.clicked.connect(
                lambda _=False, url=result.url: QDesktopServices.openUrl(
                    QUrl(url)
                )
            )
            layout.addWidget(open_button, alignment=Qt.AlignLeft)

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
                "Download Dataset",
                "Pilih file dataset terlebih dahulu.",
            )
            return

        self._pending_analyze = bool(analyze)
        self.search_button.setEnabled(False)
        self.progress.setVisible(True)

        clean_name = filename or result.dataset_id.split("/")[-1]
        self.status_label.setText(
            f"Mengunduh {clean_name} dari {result.source_label}..."
        )

        destination = self._download_directory()

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

    def _on_download_finished(self, file_path: str):
        self.search_button.setEnabled(True)
        self.progress.setVisible(False)

        if not self._pending_analyze:
            self.status_label.setText(
                f"Dataset berhasil diunduh: {Path(file_path).name}"
            )
            QMessageBox.information(
                self,
                "Download Berhasil",
                "Dataset berhasil disimpan di folder Downloads:\n\n"
                f"{file_path}",
            )
            return

        try:
            dataframe = self._load_downloaded_dataset(Path(file_path))
            self.main_window.current_dataset = dataframe
            self.main_window.current_file_path = str(file_path)
            self.main_window.update_dataset_state()
            self.status_label.setText(
                f"Dataset tersimpan di Downloads dan dimuat untuk analisis:\n{file_path}"
            )
            self.main_window.open_analysis_page()
            QTimer.singleShot(150, self.main_window.analysis_page.run_analysis)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Dataset Berhasil Diunduh",
                "File berhasil diunduh, tetapi belum bisa dimuat otomatis.\n\n"
                f"{exc}",
            )

    def _on_download_failed(self, error: str):
        self.search_button.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText(
            "Download gagal. Coba file lain atau buka sumber dataset."
        )
        QMessageBox.warning(
            self,
            "Download Gagal",
            f"Dataset gagal diunduh.\n\n{error}",
        )

    def _finish_download_thread(self, *_args):
        if self._download_thread is not None:
            self._download_thread.quit()
            self._download_thread.wait(1500)
        self._download_worker = None
        self._download_thread = None

    def _load_downloaded_dataset(self, path: Path):
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return self.main_window.loader.load_csv(str(path))

        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")

        if suffix == ".json":
            return pd.read_json(path)

        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)

        if suffix == ".parquet":
            return pd.read_parquet(path)

        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)

        raise ValueError(
            f"Format {suffix} belum didukung untuk analisis otomatis."
        )

    def _download_directory(self) -> Path:
        downloads_location = QStandardPaths.writableLocation(
            QStandardPaths.DownloadLocation
        )
        destination = (
            Path(downloads_location)
            if downloads_location
            else Path.home() / "Downloads"
        )
        destination = destination / "Dataset Research"
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def _open_download_directory(self):
        destination = self._download_directory()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(destination)))

    def _clear_results(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _show_empty_message(self, message: str):
        self._clear_results()
        label = QLabel(message)
        label.setObjectName("cardDescription")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        self.results_layout.addWidget(label)
        self.results_layout.addStretch()
