"""Single Vision Intelligence workspace with image and video submodules."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.vision.analyzer import (
    AnalysisCancelledError,
    analyze_image_dataset,
    analyze_video_dataset,
    create_preview_thumbnails,
)
from app.vision.recommendations import IMAGE_TASKS, VIDEO_TASKS, recommend_approaches
from app.vision.storage import VisionAnalysisRepository


def _format_bytes(value: int | float | None) -> str:
    size = max(0, float(value or 0))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB".replace(".", ",")


def _format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}".replace(".", ",")
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


class _VisionWorker(QObject):
    progress = Signal(int, str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, operation: str, payload: dict[str, Any], cancel_event):
        super().__init__()
        self.operation = operation
        self.payload = payload
        self.cancel_event = cancel_event

    @Slot()
    def run(self):
        try:
            if self.operation == "analysis":
                dataset_type = self.payload["dataset_type"]
                analyzer = analyze_image_dataset if dataset_type == "image" else analyze_video_dataset
                options = {}
                if dataset_type == "video":
                    options["frame_sample_count"] = self.payload.get("frame_sample_count", 4)
                analysis = analyzer(
                    self.payload["path"],
                    progress=self.progress.emit,
                    cancel_event=self.cancel_event,
                    **options,
                )
                recommendations = recommend_approaches(
                    dataset_type,
                    analysis,
                    self.payload["task"],
                )
                self.completed.emit(
                    {
                        "operation": "analysis",
                        "dataset_type": dataset_type,
                        "path": self.payload["path"],
                        "task": self.payload["task"],
                        "analysis": analysis,
                        "recommendations": recommendations,
                    }
                )
            elif self.operation == "preview":
                previews = create_preview_thumbnails(
                    self.payload["path"],
                    self.payload["dataset_type"],
                    self.payload["references"],
                    cancel_event=self.cancel_event,
                )
                self.completed.emit({"operation": "preview", "previews": previews})
            else:
                raise ValueError(f"Operasi vision tidak dikenal: {self.operation}")
        except AnalysisCancelledError:
            self.cancelled.emit()
        except Exception as error:
            self.failed.emit(str(error))


class _VisionStatCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("visionStatCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(5)
        self.title = QLabel(title)
        self.title.setObjectName("visionMetricTitle")
        self.value = QLabel("—")
        self.value.setObjectName("visionMetricValue")
        self.detail = QLabel("")
        self.detail.setObjectName("visionMetricDetail")
        self.detail.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.value)
        layout.addWidget(self.detail)


class _VisionTab(QWidget):
    PAGE_SIZE = 12

    def __init__(self, page: "VisionIntelligencePage", dataset_type: str):
        super().__init__()
        self.page = page
        self.dataset_type = dataset_type
        self.analysis: dict[str, Any] | None = None
        self.recommendations: dict[str, Any] | None = None
        self.analysis_id: int | None = None
        self.sample_refs: list[dict[str, Any]] = []
        self._preview_page = 0
        self._thumbnail_cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._preview_items: dict[str, dict[str, Any]] = {}
        self._failed_previews: set[str] = set()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 18, 0, 0)
        root.setSpacing(16)

        input_card, input_layout = self._card()
        input_header = QHBoxLayout()
        input_title = QLabel("Dataset Input")
        input_title.setObjectName("visionSectionTitle")
        input_header.addWidget(input_title)
        input_header.addStretch(1)
        input_header.addWidget(self._small_note("Folder atau ZIP; media tidak disalin ke database."))
        input_layout.addLayout(input_header)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setObjectName("visionDatasetPath")
        self.path_edit.setPlaceholderText("Pilih folder dataset atau file ZIP")
        self.path_edit.setToolTip("Path lokal ke folder dataset atau ZIP")
        self.folder_button = QPushButton("Import Folder")
        self.folder_button.setObjectName("secondaryButton")
        self.zip_button = QPushButton("Import ZIP")
        self.zip_button.setObjectName("secondaryButton")
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(self.folder_button)
        path_row.addWidget(self.zip_button)
        input_layout.addLayout(path_row)

        actions = QHBoxLayout()
        task_label = QLabel("Jenis tugas")
        task_label.setObjectName("visionFieldLabel")
        self.task_combo = QComboBox()
        self.task_combo.setObjectName("visionTaskSelector")
        for key, label in (IMAGE_TASKS if self.dataset_type == "image" else VIDEO_TASKS):
            self.task_combo.addItem(label, key)
        self.sample_frames_combo = None
        if self.dataset_type == "video":
            sample_label = QLabel("Frame/video")
            sample_label.setObjectName("visionFieldLabel")
            self.sample_frames_combo = QComboBox()
            self.sample_frames_combo.setObjectName("visionTaskSelector")
            for count in (1, 2, 4, 8):
                self.sample_frames_combo.addItem(str(count), count)
            self.sample_frames_combo.setCurrentIndex(2)
        self.run_button = QPushButton("Analisis Dataset")
        self.run_button.setObjectName("primaryButton")
        self.cancel_button = QPushButton("Batalkan")
        self.cancel_button.setObjectName("secondaryButton")
        self.cancel_button.setEnabled(False)
        actions.addWidget(task_label)
        actions.addWidget(self.task_combo, 1)
        if self.sample_frames_combo is not None:
            actions.addWidget(sample_label)
            actions.addWidget(self.sample_frames_combo)
        actions.addWidget(self.run_button)
        actions.addWidget(self.cancel_button)
        input_layout.addLayout(actions)
        root.addWidget(input_card)

        self.status_label = QLabel("Pilih dataset untuk memulai analisis lokal.")
        self.status_label.setObjectName("visionStatusLabel")
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("visionProgressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        root.addWidget(self.status_label)
        root.addWidget(self.progress_bar)

        metric_grid = QGridLayout()
        metric_grid.setContentsMargins(0, 0, 0, 0)
        metric_grid.setHorizontalSpacing(12)
        metric_grid.setVerticalSpacing(12)
        self.metrics = [
            _VisionStatCard("TOTAL SAMPLES"),
            _VisionStatCard("NUMBER OF CLASSES"),
            _VisionStatCard("DATASET SIZE"),
            _VisionStatCard("INVALID FILES"),
        ]
        for index, card in enumerate(self.metrics):
            metric_grid.addWidget(card, index // 4, index % 4)
        root.addLayout(metric_grid)

        overview_row = QHBoxLayout()
        overview_row.setSpacing(14)
        self.distribution_card, self.distribution_layout = self._card("Distribusi Dataset")
        self.distribution_body = QVBoxLayout()
        self.distribution_body.setSpacing(8)
        self.distribution_empty = self._empty_label("Distribusi kelas tampil setelah analisis.")
        self.distribution_body.addWidget(self.distribution_empty)
        self.distribution_layout.addLayout(self.distribution_body)
        self.details_card, self.details_layout = self._card("Dataset Information")
        self.details_body = QVBoxLayout()
        self.details_body.setSpacing(7)
        self.details_layout.addLayout(self.details_body)
        overview_row.addWidget(self.distribution_card, 3)
        overview_row.addWidget(self.details_card, 2)
        root.addLayout(overview_row)

        gallery_card, gallery_layout = self._card("Sample Preview")
        gallery_heading = QHBoxLayout()
        gallery_title = QLabel("Gallery")
        gallery_title.setObjectName("visionSectionTitle")
        self.gallery_page_label = QLabel("0 sample")
        self.gallery_page_label.setObjectName("visionMutedLabel")
        gallery_heading.addWidget(gallery_title)
        gallery_heading.addStretch(1)
        self.previous_button = QPushButton("Sebelumnya")
        self.previous_button.setObjectName("secondaryButton")
        self.next_button = QPushButton("Berikutnya")
        self.next_button.setObjectName("secondaryButton")
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        gallery_heading.addWidget(self.previous_button)
        gallery_heading.addWidget(self.next_button)
        gallery_layout.addLayout(gallery_heading)
        gallery_row = QHBoxLayout()
        self.gallery = QListWidget()
        self.gallery.setObjectName("visionGallery")
        self.gallery.setViewMode(QListWidget.IconMode)
        self.gallery.setFlow(QListWidget.LeftToRight)
        self.gallery.setWrapping(True)
        self.gallery.setResizeMode(QListWidget.Adjust)
        self.gallery.setMovement(QListWidget.Static)
        self.gallery.setIconSize(QSize(108, 78))
        self.gallery.setGridSize(QSize(126, 122))
        self.gallery.setMaximumHeight(300)
        self.gallery.setMinimumHeight(170)
        self.gallery.setSelectionMode(QListWidget.SingleSelection)
        self.gallery_empty = self._empty_label("Contoh media akan dimuat bertahap.")
        self.viewer = QLabel("Pilih sample untuk melihat pratinjau")
        self.viewer.setObjectName("visionSampleViewer")
        self.viewer.setAlignment(Qt.AlignCenter)
        self.viewer.setMinimumSize(180, 145)
        self.viewer.setMaximumSize(330, 250)
        self.viewer.setWordWrap(True)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gallery_row.addWidget(self.gallery, 3)
        gallery_row.addWidget(self.viewer, 2)
        gallery_layout.addWidget(self.gallery_empty)
        gallery_layout.addLayout(gallery_row)
        root.addWidget(gallery_card)

        quality_card, quality_layout = self._card("Quality Assessment")
        self.quality_body = QVBoxLayout()
        self.quality_body.setSpacing(6)
        self.quality_body.addWidget(self._empty_label("Ringkasan kualitas tampil setelah analisis."))
        quality_layout.addLayout(self.quality_body)
        root.addWidget(quality_card)

        recommendation_card, recommendation_layout = self._card("Recommendation")
        self.recommendation_body = QVBoxLayout()
        self.recommendation_body.setSpacing(10)
        self.recommendation_body.addWidget(self._empty_label("Rekomendasi akan mengikuti jenis tugas dan karakteristik dataset."))
        recommendation_layout.addLayout(self.recommendation_body)
        root.addWidget(recommendation_card)

        root.addStretch(1)
        self.folder_button.clicked.connect(self._choose_folder)
        self.zip_button.clicked.connect(self._choose_zip)
        self.run_button.clicked.connect(lambda: self.page.start_analysis(self))
        self.cancel_button.clicked.connect(self.page.cancel_active_operation)
        self.task_combo.currentIndexChanged.connect(lambda _index: self.page.update_recommendations(self))
        self.previous_button.clicked.connect(lambda: self._change_preview_page(-1))
        self.next_button.clicked.connect(lambda: self._change_preview_page(1))
        self.gallery.currentItemChanged.connect(self._show_selected_preview)

    @staticmethod
    def _card(title: str | None = None):
        card = QFrame()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 15)
        layout.setSpacing(11)
        if title:
            heading = QLabel(title)
            heading.setObjectName("visionSectionTitle")
            layout.addWidget(heading)
        return card, layout

    @staticmethod
    def _small_note(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("visionMutedLabel")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _empty_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("visionEmptyLabel")
        label.setWordWrap(True)
        return label

    def _choose_folder(self):
        selected = QFileDialog.getExistingDirectory(self, "Pilih folder dataset")
        if selected:
            self.set_dataset_path(selected, restore=True)

    def _choose_zip(self):
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih ZIP dataset",
            "",
            "ZIP dataset (*.zip)",
        )
        if selected:
            self.set_dataset_path(selected, restore=True)

    def set_dataset_path(self, path: str, *, restore: bool = False):
        self.path_edit.setText(str(path))
        self.status_label.setText(f"Dataset dipilih: {Path(path).name}")
        self.progress_bar.setValue(0)
        if restore:
            self.page.restore_by_path(self, path)

    def task(self) -> str:
        return str(self.task_combo.currentData() or ("classification" if self.dataset_type == "image" else "video_classification"))

    def frame_sample_count(self) -> int:
        return int(self.sample_frames_combo.currentData() or 4) if self.sample_frames_combo else 4

    def selected_references(self) -> list[dict[str, Any]]:
        start = self._preview_page * self.PAGE_SIZE
        return self.sample_refs[start : start + self.PAGE_SIZE]

    def reset_previews(self):
        self._preview_page = 0
        self._preview_items.clear()
        self._failed_previews.clear()
        self._thumbnail_cache.clear()
        self.gallery.clear()
        self.viewer.setPixmap(QPixmap())
        self.viewer.setText("Pilih sample untuk melihat pratinjau")

    @staticmethod
    def _reference_key(reference: dict[str, Any]) -> str:
        return "|".join(
            str(reference.get(key, ""))
            for key in ("path", "member", "frame_index", "name")
        )

    def render_preview_page(self):
        refs = self.selected_references()
        self.gallery.clear()
        self._preview_items.clear()
        if not refs:
            self.gallery_empty.show()
            self.gallery_page_label.setText("0 sample")
            self.previous_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return
        self.gallery_empty.hide()
        first = self._preview_page * self.PAGE_SIZE + 1
        last = first + len(refs) - 1
        self.gallery_page_label.setText(f"{first}–{last} dari {len(self.sample_refs)} contoh")
        self.previous_button.setEnabled(self._preview_page > 0)
        self.next_button.setEnabled(last < len(self.sample_refs))
        missing = []
        for reference in refs:
            key = self._reference_key(reference)
            if key in self._thumbnail_cache or key in self._failed_previews:
                self._thumbnail_cache.move_to_end(key)
            else:
                missing.append(reference)
        if missing:
            self.status_label.setText("Memuat pratinjau sample terpilih…")
            self.page.load_previews(self, missing)
        else:
            self._populate_gallery(refs)

    def _populate_gallery(self, refs: list[dict[str, Any]]):
        self.gallery.clear()
        for reference in refs:
            key = self._reference_key(reference)
            item = QListWidgetItem()
            name = Path(reference.get("name") or "sample").name
            label = reference.get("label") or "Tanpa label"
            if self.dataset_type == "video":
                text = f"{name}\nFrame {int(reference.get('frame_index', 0)):,} · {label}"
            else:
                text = f"{name}\n{label}"
            item.setText(text)
            item.setToolTip(reference.get("name") or name)
            item.setData(Qt.UserRole, key)
            pixmap = self._thumbnail_cache.get(key)
            if pixmap is not None:
                item.setIcon(QIcon(pixmap))
            else:
                item.setText(f"{name}\nPratinjau gagal dimuat")
            self.gallery.addItem(item)
            self._preview_items[key] = reference
        if self.gallery.count():
            self.gallery.setCurrentRow(0)

    def add_previews(self, previews: list[dict[str, Any]]):
        for preview in previews:
            reference = preview["reference"]
            key = self._reference_key(reference)
            if preview.get("data"):
                pixmap = QPixmap()
                if pixmap.loadFromData(preview["data"], "JPG"):
                    self._thumbnail_cache[key] = pixmap
                    self._thumbnail_cache.move_to_end(key)
                    while len(self._thumbnail_cache) > 24:
                        self._thumbnail_cache.popitem(last=False)
            else:
                self._failed_previews.add(key)
                self.status_label.setText(
                    f"Pratinjau {Path(reference.get('name') or 'sample').name} tidak dapat dimuat."
                )
        self.render_preview_page()

    def _change_preview_page(self, change: int):
        maximum = max(0, (len(self.sample_refs) - 1) // self.PAGE_SIZE)
        self._preview_page = max(0, min(maximum, self._preview_page + change))
        self.render_preview_page()

    def _show_selected_preview(self, current: QListWidgetItem | None, _previous=None):
        if current is None:
            return
        key = current.data(Qt.UserRole)
        pixmap = self._thumbnail_cache.get(key)
        if pixmap is not None:
            scaled = pixmap.scaled(self.viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.viewer.setPixmap(scaled)
            self.viewer.setText("")
            self._thumbnail_cache.move_to_end(key)
        else:
            self.viewer.setPixmap(QPixmap())
            self.viewer.setText("Pratinjau tidak tersedia")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "gallery"):
            self._show_selected_preview(self.gallery.currentItem())

    def show_analysis(self, analysis: dict[str, Any], recommendations: dict[str, Any], analysis_id: int | None = None):
        self.analysis = analysis
        self.recommendations = recommendations
        self.analysis_id = analysis_id
        self.sample_refs = list(analysis.get("sample_references") or [])
        self.reset_previews()
        self.render_overview()
        self.render_recommendations()
        self.render_preview_page()
        self.progress_bar.setValue(100)
        self.status_label.setText("Analisis selesai. Rekomendasi adalah panduan, bukan hasil eksperimen model.")

    def show_empty_state(self):
        self.analysis = None
        self.recommendations = None
        self.analysis_id = None
        self.sample_refs = []
        self.reset_previews()
        for card in self.metrics:
            card.value.setText("—")
            card.detail.setText("")
        for layout, message in (
            (self.distribution_body, "Distribusi kelas tampil setelah analisis."),
            (self.details_body, "Informasi dataset tampil setelah analisis."),
            (self.quality_body, "Ringkasan kualitas tampil setelah analisis."),
            (self.recommendation_body, "Rekomendasi akan mengikuti jenis tugas dan karakteristik dataset."),
        ):
            self._clear_layout(layout)
            layout.addWidget(self._empty_label(message))
        self.render_preview_page()
        self.progress_bar.setValue(0)

    def render_overview(self):
        if self.analysis is None:
            return
        result = self.analysis
        valid_count = int(result.get("total_samples") or 0)
        class_count = int(result.get("class_count") or 0)
        self.metrics[0].value.setText(_format_number(valid_count))
        self.metrics[0].detail.setText(f"{_format_number(result.get('total_files'))} file terdeteksi")
        self.metrics[1].value.setText(_format_number(class_count))
        self.metrics[1].detail.setText("Label ditemukan" if result.get("has_labels") else "Label belum terdeteksi")
        self.metrics[2].value.setText(_format_bytes(result.get("source_size_bytes")))
        self.metrics[2].detail.setText("Ukuran ZIP" if result.get("source_kind") == "zip" else "Media dan metadata yang didukung")
        self.metrics[3].value.setText(_format_number(result.get("invalid_files")))
        self.metrics[3].detail.setText("File rusak / tidak dapat dibaca")
        self._render_distribution()
        self._render_details()
        self._render_quality()

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _render_distribution(self):
        self._clear_layout(self.distribution_body)
        if not self.analysis:
            self.distribution_body.addWidget(self._empty_label("Distribusi kelas tampil setelah analisis."))
            return
        result = self.analysis
        distribution = result.get("class_distribution") or {}
        if not distribution:
            self.distribution_body.addWidget(self._empty_label("Label tidak terdeteksi. Distribusi kelas tidak dapat dihitung."))
            return
        total = max(distribution.values())
        self.distribution_body.addWidget(self._small_note(f"{len(distribution)} kelas · menampilkan maksimal 12 kelas"))
        for name, count in sorted(distribution.items(), key=lambda item: (-item[1], str(item[0]).casefold()))[:12]:
            row = QHBoxLayout()
            label = QLabel(str(name))
            label.setObjectName("visionDistributionLabel")
            label.setMinimumWidth(110)
            label.setMaximumWidth(180)
            label.setToolTip(str(name))
            bar = QProgressBar()
            bar.setObjectName("visionDistributionBar")
            bar.setRange(0, max(1, total))
            bar.setValue(int(count))
            bar.setTextVisible(False)
            bar.setMinimumHeight(6)
            bar.setMaximumHeight(9)
            value = QLabel(f"{count:,}")
            value.setObjectName("visionDistributionValue")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(label)
            row.addWidget(bar, 1)
            row.addWidget(value)
            self.distribution_body.addLayout(row)
        if result.get("class_imbalance_ratio"):
            imbalance = float(result["class_imbalance_ratio"])
            self.distribution_body.addWidget(
                self._small_note(f"Rasio kelas terbesar/terkecil: {_format_number(imbalance, 1)}:1" + (" · indikasi ketimpangan" if imbalance >= 3 else ""))
            )

    def _add_detail(self, label: str, value: Any):
        row = QHBoxLayout()
        title = QLabel(label)
        title.setObjectName("visionFieldLabel")
        value_label = QLabel(str(value if value not in (None, "") else "—"))
        value_label.setObjectName("visionDetailValue")
        value_label.setWordWrap(True)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(title, 1)
        row.addWidget(value_label, 2)
        self.details_body.addLayout(row)

    def _render_details(self):
        self._clear_layout(self.details_body)
        if not self.analysis:
            self.details_body.addWidget(self._empty_label("Informasi dataset tampil setelah analisis."))
            return
        result = self.analysis
        self._add_detail("Dataset", Path(result.get("source_path") or "").name)
        self._add_detail("Sumber", "ZIP" if result.get("source_kind") == "zip" else "Folder")
        self._add_detail("Format", ", ".join(f"{key} ({value:,})" for key, value in result.get("formats", {}).items()) if result.get("formats") else "Video" )
        self._add_detail("Label", result.get("label_source") or "Tidak terdeteksi")
        if result.get("metadata_files"):
            self._add_detail("Metadata CSV", ", ".join(result["metadata_files"]))
        dimensions = result.get("dimensions") or {}
        if self.dataset_type == "image":
            self._add_detail(
                "Dimensi min–maks",
                f"{dimensions.get('width_min')}×{dimensions.get('height_min')} – {dimensions.get('width_max')}×{dimensions.get('height_max')} px",
            )
            ratio = result.get("aspect_ratio") or {}
            self._add_detail("Aspect ratio min–maks", f"{_format_number(ratio.get('min'), 3)} – {_format_number(ratio.get('max'), 3)} · rata-rata {_format_number(ratio.get('mean'), 3)}")
        else:
            self._add_detail(
                "Resolusi min–maks",
                f"{dimensions.get('width_min')}×{dimensions.get('height_min')} – {dimensions.get('width_max')}×{dimensions.get('height_max')} px",
            )
            fps = result.get("fps") or {}
            duration = result.get("duration_seconds") or {}
            self._add_detail("FPS rata-rata", _format_number(fps.get("mean"), 2))
            self._add_detail("Durasi rata-rata", f"{_format_number(duration.get('mean'))} detik")
            self._add_detail("Frame yang disampling", _format_number(result.get("frame_samples_read")))
            frame_buckets = result.get("frame_count_buckets") or {}
            self._add_detail(
                "Distribusi jumlah frame",
                ", ".join(f"{key.replace('_', ' ')}: {value:,}" for key, value in frame_buckets.items()) or "Tidak tersedia",
            )

    def _add_quality_note(self, text: str):
        label = QLabel("•  " + text)
        label.setObjectName("visionQualityNote")
        label.setWordWrap(True)
        self.quality_body.addWidget(label)

    def _render_quality(self):
        self._clear_layout(self.quality_body)
        if not self.analysis:
            self.quality_body.addWidget(self._empty_label("Ringkasan kualitas tampil setelah analisis."))
            return
        result = self.analysis
        self._add_quality_note(f"File tidak terbaca: {_format_number(result.get('invalid_files'))}.")
        if self.dataset_type == "image":
            self._add_quality_note(f"Duplikat indikatif via perceptual hash: {_format_number(result.get('duplicate_count'))}; tinjau manual sebelum penghapusan.")
            self._add_quality_note(f"Gambar hampir seragam pada thumbnail kecil: {_format_number(result.get('near_uniform_count'))}; ini bukan deteksi blur atau kualitas visual menyeluruh.")
            self._add_quality_note("Variasi resolusi: " + ", ".join(f"{key.replace('_', ' ')}: {value:,}" for key, value in result.get("resolution_buckets", {}).items()))
            self._add_quality_note("Aspect ratio: " + ", ".join(f"{key.replace('_', ' ')}: {value:,}" for key, value in result.get("aspect_ratio_buckets", {}).items()))
            characteristics = result.get("visual_characteristics") or {}
            self._add_quality_note(
                f"Brightness rata-rata {characteristics.get('mean_brightness_0_255', '—')} dari 255; dihitung dari thumbnail 24×24."
            )
        else:
            self._add_quality_note(f"Metadata video tidak lengkap: {_format_number(result.get('invalid_metadata_count'))}.")
            self._add_quality_note(f"Kemungkinan duplikat dari signature maksimal 4 frame: {_format_number(result.get('possible_duplicate_count'))}; bukan pemeriksaan byte-identik.")
            leakage = int(result.get("possible_split_leakage_signatures") or 0)
            if leakage:
                self._add_quality_note(
                    f"Kemungkinan kebocoran split: {leakage} signature frame yang sama muncul di folder train/val/test; indikasi ini perlu ditinjau manual."
                )
            else:
                self._add_quality_note("Tidak ada signature duplikat antar-folder train/val/test yang terdeteksi; ini bukan bukti bebas kebocoran.")
            self._add_quality_note("Frame sampling dibatasi per video; pemeriksaan ini dapat melewatkan kejadian singkat di antara frame sampel.")
            self._add_quality_note("Distribusi resolusi: " + ", ".join(f"{key.replace('_', ' ')}: {value:,}" for key, value in result.get("resolution_buckets", {}).items()))
        for note in result.get("analysis_notes", [])[:3]:
            self._add_quality_note(str(note))
        for error in result.get("error_log", [])[:3]:
            self._add_quality_note(f"File bermasalah — {error.get('file')}: {error.get('error')}")

    def render_recommendations(self):
        self._clear_layout(self.recommendation_body)
        if not self.recommendations:
            self.recommendation_body.addWidget(self._empty_label("Rekomendasi akan mengikuti jenis tugas dan karakteristik dataset."))
            return
        recommendations = self.recommendations
        primary = recommendations.get("primary") or {}
        self.recommendation_body.addWidget(self._recommendation_card("RECOMMENDED APPROACH", primary, True))
        alternatives = recommendations.get("alternatives") or []
        if alternatives:
            heading = QLabel("Alternative Methods")
            heading.setObjectName("visionSubsectionTitle")
            self.recommendation_body.addWidget(heading)
            for recommendation in alternatives:
                self.recommendation_body.addWidget(self._recommendation_card("ALTERNATIVE", recommendation, False))
        for warning in recommendations.get("warnings", []):
            note = QLabel("Catatan: " + str(warning))
            note.setObjectName("visionWarningLabel")
            note.setWordWrap(True)
            self.recommendation_body.addWidget(note)
        disclaimer = QLabel(str(recommendations.get("note") or "Rekomendasi tidak mewakili hasil eksperimen model."))
        disclaimer.setObjectName("visionMutedLabel")
        disclaimer.setWordWrap(True)
        self.recommendation_body.addWidget(disclaimer)

    def _recommendation_card(self, eyebrow: str, recommendation: dict[str, Any], primary: bool):
        card = QFrame()
        card.setObjectName("visionPrimaryRecommendation" if primary else "visionAlternativeRecommendation")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 13)
        layout.setSpacing(6)
        eyebrow_label = QLabel(eyebrow)
        eyebrow_label.setObjectName("visionRecommendationEyebrow")
        title = QLabel(str(recommendation.get("name") or "Metode"))
        title.setObjectName("visionRecommendationTitle")
        task = QLabel(str(recommendation.get("task") or ""))
        task.setObjectName("visionRecommendationTask")
        layout.addWidget(eyebrow_label)
        layout.addWidget(title)
        layout.addWidget(task)
        self._recommendation_field(layout, "Alasan", recommendation.get("reason"))
        self._recommendation_field(layout, "Kelebihan", " · ".join(recommendation.get("advantages") or []))
        self._recommendation_field(layout, "Kekurangan", " · ".join(recommendation.get("disadvantages") or []))
        self._recommendation_field(layout, "Kebutuhan komputasi", recommendation.get("computational_requirements"))
        self._recommendation_field(layout, "Kompleksitas", recommendation.get("complexity"))
        self._recommendation_field(layout, "Cocok untuk", recommendation.get("suitable_dataset_conditions"))
        self._recommendation_field(layout, "Batasan", recommendation.get("limitations"))
        return card

    @staticmethod
    def _recommendation_field(layout, name: str, value: Any):
        row = QHBoxLayout()
        label = QLabel(name)
        label.setObjectName("visionRecommendationFieldName")
        label.setMinimumWidth(120)
        label.setMaximumWidth(150)
        value_label = QLabel(str(value or "—"))
        value_label.setObjectName("visionRecommendationFieldValue")
        value_label.setWordWrap(True)
        row.addWidget(label)
        row.addWidget(value_label, 1)
        layout.addLayout(row)


class VisionIntelligencePage(QWidget):
    """One routed page; image and video live as two workspace tabs."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.repository = VisionAnalysisRepository(main_window.database)
        self._thread: QThread | None = None
        self._worker: _VisionWorker | None = None
        self._cancel_event = None
        self._operation: str | None = None
        self._active_tab: _VisionTab | None = None
        self._pending_preview: tuple[_VisionTab, list[dict[str, Any]]] | None = None
        self._build_ui()

    @property
    def active_thread(self):
        return self._thread

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 24, 30, 28)
        root.setSpacing(12)
        header = QVBoxLayout()
        header.setSpacing(5)
        title = QLabel("Vision Intelligence")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Visual Dataset Understanding")
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        header.addWidget(title)
        header.addWidget(subtitle)
        root.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("visionAnalysisTabs")
        self.image_tab = _VisionTab(self, "image")
        self.video_tab = _VisionTab(self, "video")
        self.tabs.addTab(self.image_tab, "Image Analysis")
        self.tabs.addTab(self.video_tab, "Video Analysis")
        self.tabs.currentChanged.connect(self._tab_changed)
        root.addWidget(self.tabs, 1)

    def _tab_changed(self, _index: int):
        tab = self.current_tab()
        if tab.analysis is not None:
            tab.render_preview_page()

    def current_tab(self) -> _VisionTab:
        return self.tabs.currentWidget()

    def restore_project(self, project_id: int | None):
        if project_id is None:
            return
        first_restored = None
        for dataset_type in ("image", "video"):
            tab = self.image_tab if dataset_type == "image" else self.video_tab
            if tab.analysis is not None or tab.path_edit.text().strip():
                continue
            try:
                record = self.repository.get_latest_for_project(project_id, dataset_type)
            except Exception:
                continue
            if record:
                tab.path_edit.setText(record["dataset_path"])
                tab.show_analysis(
                    record.get("results") or {},
                    record.get("recommendations") or {},
                    record.get("id"),
                )
                tab.status_label.setText(
                    f"Hasil sebelumnya dipulihkan · {record.get('created_at', 'waktu tidak tersedia')}"
                )
                first_restored = first_restored or tab
        if first_restored is not None and self.current_tab().analysis is None:
            self.tabs.setCurrentWidget(first_restored)

    def restore_by_path(self, tab: _VisionTab, path: str):
        try:
            record = self.repository.get_latest_by_path(path, tab.dataset_type)
        except Exception as error:
            tab.status_label.setText(f"Hasil tersimpan tidak dapat dibaca: {error}")
            return
        if not record:
            tab.show_empty_state()
            tab.status_label.setText("Dataset dipilih. Jalankan analisis untuk membaca metadata terbaru.")
            return
        tab.show_analysis(record.get("results") or {}, record.get("recommendations") or {}, record.get("id"))
        tab.status_label.setText(
            f"Hasil tersimpan dimuat · {record.get('created_at', 'waktu tidak tersedia')}. Jalankan ulang untuk memperbarui."
        )

    def start_analysis(self, tab: _VisionTab):
        if self._thread is not None:
            tab.status_label.setText("Tunggu operasi saat ini selesai atau batalkan terlebih dahulu.")
            return
        path = tab.path_edit.text().strip()
        if not path:
            tab.status_label.setText("Pilih folder dataset atau file ZIP terlebih dahulu.")
            return
        if not Path(path).exists():
            tab.status_label.setText("Path dataset tidak ditemukan.")
            return
        tab.show_empty_state()
        tab.progress_bar.setValue(0)
        tab.status_label.setText("Menyiapkan analisis lokal…")
        self._start_worker(
            "analysis",
            {
                "path": path,
                "dataset_type": tab.dataset_type,
                "task": tab.task(),
                "frame_sample_count": tab.frame_sample_count(),
            },
            tab,
        )

    def update_recommendations(self, tab: _VisionTab):
        if not tab.analysis:
            return
        try:
            task = tab.task()
            hardware = (tab.recommendations or {}).get("hardware")
            tab.recommendations = recommend_approaches(
                tab.dataset_type, tab.analysis, task, hardware=hardware
            )
            tab.render_recommendations()
            if tab.analysis_id is not None:
                self.repository.update_recommendations(
                    tab.analysis_id,
                    {
                        "task": task,
                        "preview_page_size": tab.PAGE_SIZE,
                        "sample_frame_limit": tab.frame_sample_count(),
                    },
                    tab.recommendations,
                )
        except Exception as error:
            tab.status_label.setText(f"Rekomendasi tidak dapat diperbarui: {error}")

    def load_previews(self, tab: _VisionTab, references: list[dict[str, Any]]):
        if self._thread is not None:
            self._pending_preview = (tab, references)
            return
        self._start_worker(
            "preview",
            {"path": tab.path_edit.text().strip(), "dataset_type": tab.dataset_type, "references": references},
            tab,
        )

    def _start_worker(self, operation: str, payload: dict[str, Any], tab: _VisionTab):
        import threading

        self._operation = operation
        self._active_tab = tab
        self._cancel_event = threading.Event()
        thread = QThread(self)
        worker = _VisionWorker(operation, payload, self._cancel_event)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.completed.connect(self._on_completed)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        self._set_busy(True)
        thread.start()

    def _set_busy(self, busy: bool):
        for tab in (self.image_tab, self.video_tab):
            tab.run_button.setEnabled(not busy)
            tab.folder_button.setEnabled(not busy)
            tab.zip_button.setEnabled(not busy)
            tab.task_combo.setEnabled(not busy)
            if tab.sample_frames_combo is not None:
                tab.sample_frames_combo.setEnabled(not busy)
            tab.cancel_button.setEnabled(busy)

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str):
        tab = self._active_tab
        if tab is None:
            return
        tab.progress_bar.setValue(percent)
        tab.status_label.setText(message)

    @Slot(object)
    def _on_completed(self, value: dict[str, Any]):
        if value.get("operation") == "preview":
            tab = self._active_tab
            if tab is not None:
                tab.add_previews(value.get("previews") or [])
                tab.status_label.setText("Pratinjau sample diperbarui.")
            return
        tab = self._active_tab
        if tab is None:
            return
        analysis = value["analysis"]
        recommendations = value["recommendations"]
        analysis_id = None
        project_id = getattr(self.main_window, "current_project_id", None)
        try:
            repository = getattr(self.main_window, "repository", None)
            if project_id is not None and repository is not None and not repository.project_exists(project_id):
                project_id = None
        except Exception:
            project_id = None
        storage_error = None
        try:
            analysis_id = self.repository.save(
                project_id=project_id,
                dataset_path=value["path"],
                dataset_type=value["dataset_type"],
                dataset_metadata={
                    "source_kind": analysis.get("source_kind"),
                    "total_files": analysis.get("total_files"),
                    "total_samples": analysis.get("total_samples"),
                    "class_count": analysis.get("class_count"),
                    "source_size_bytes": analysis.get("source_size_bytes"),
                    "label_source": analysis.get("label_source"),
                },
                analysis_config={
                    "task": value["task"],
                    "preview_page_size": tab.PAGE_SIZE,
                    "sample_frame_limit": tab.frame_sample_count(),
                },
                results=analysis,
                recommendations=recommendations,
                error_log=analysis.get("error_log") or [],
            )
        except Exception as error:
            storage_error = error
        tab.show_analysis(analysis, recommendations, analysis_id)
        if analysis_id is not None:
            tab.status_label.setText("Analisis selesai dan hasil tersimpan di SQLite lokal.")
        elif storage_error is not None:
            tab.status_label.setText(f"Analisis selesai, tetapi penyimpanan SQLite gagal: {storage_error}")

    @Slot(str)
    def _on_failed(self, error: str):
        tab = self._active_tab
        if tab is not None:
            prefix = "Pratinjau gagal" if self._operation == "preview" else "Analisis gagal"
            tab.status_label.setText(f"{prefix}: {error}")
            if self._operation != "preview":
                tab.progress_bar.setValue(0)

    @Slot()
    def _on_cancelled(self):
        tab = self._active_tab
        if tab is not None:
            tab.status_label.setText("Operasi dibatalkan. Hasil parsial tidak disimpan.")

    def cancel_active_operation(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
            if self._active_tab is not None:
                self._active_tab.cancel_button.setEnabled(False)
                self._active_tab.status_label.setText("Menghentikan operasi dengan aman…")
            self._pending_preview = None

    @Slot()
    def _on_thread_finished(self):
        operation = self._operation
        self._thread = None
        self._worker = None
        self._cancel_event = None
        self._operation = None
        active_tab = self._active_tab
        self._active_tab = None
        self._set_busy(False)
        if operation == "preview" and active_tab is not None and active_tab.analysis is not None:
            # Keep a completed analysis status when no more preview work is pending.
            if not self._pending_preview:
                active_tab.status_label.setText("Pratinjau sample siap.")
        pending = self._pending_preview
        self._pending_preview = None
        if pending is not None:
            tab, references = pending
            if references and tab.analysis is not None:
                QTimer.singleShot(0, lambda: self.load_previews(tab, references))

