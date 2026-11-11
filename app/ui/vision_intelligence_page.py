"""Single Vision Intelligence workspace with image and video submodules."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
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
from app.vision.readiness import assess_image_readiness, assess_image_tasks
from app.vision.image_classifier import ensure_model_bundle, recognize_image_dataset


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


class _RecognitionBarChart(QWidget):
    """Readable horizontal chart for the dataset-level model predictions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("visionRecognitionChart")
        self.setAccessibleName("Grafik distribusi prediksi kelas ImageNet")
        self.setAccessibleDescription(
            "Grafik batang horizontal menampilkan jumlah gambar dan persentase pada kelas prediksi teratas."
        )
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._rows: list[tuple[str, int]] = []
        self._classified_count = 0
        self.setFixedHeight(184)

    def _show_empty(self) -> None:
        self._rows = []
        self._classified_count = 0
        self.setFixedHeight(184)
        self.update()

    def set_distribution(self, distribution: dict[str, int], classified_count: int) -> None:
        rows = sorted(
            ((str(label), max(0, int(count or 0))) for label, count in distribution.items()),
            key=lambda item: (-item[1], item[0].casefold()),
        )[:8]
        rows = [(label, count) for label, count in rows if count > 0]
        if not rows or classified_count <= 0:
            self._show_empty()
            return

        self._rows = rows
        self._classified_count = classified_count
        self.setFixedHeight(78 + 40 * len(rows))
        chart_summary = "; ".join(
            f"{label}: {count:,} gambar ({_format_number(count * 100.0 / classified_count, 1)}%)"
            for label, count in rows
        )
        self.setAccessibleDescription(
            f"Delapan prediksi kelas terbanyak dari {classified_count:,} gambar yang berhasil diklasifikasi. {chart_summary}"
        )
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))

        label_font = QFont(self.font())
        label_font.setPointSizeF(10)
        painter.setFont(label_font)
        metrics = QFontMetrics(label_font)

        if not self._rows:
            painter.setPen(QColor("#65758B"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Grafik muncul setelah klasifikasi dijalankan.")
            painter.end()
            return

        width = self.width()
        height = self.height()
        row_height = 40
        plot_top = 24
        plot_bottom = plot_top + row_height * len(self._rows) - 8
        label_width = min(280, max(150, int(width * 0.28)))
        plot_left = label_width + 18
        value_column_width = 132
        plot_right = max(plot_left + 90, width - value_column_width - 16)
        plot_width = plot_right - plot_left
        max_count = max(count for _label, count in self._rows)
        max_scale = float(max(1, max_count))

        # Draw count ticks and grid behind the bars for easier comparison.
        grid_pen = QPen(QColor("#E6EDF7"))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        tick_values = sorted({round(max_scale * fraction) for fraction in (0, 0.25, 0.5, 0.75, 1)})
        tick_font = QFont(label_font)
        tick_font.setPointSizeF(9)
        painter.setFont(tick_font)
        tick_metrics = QFontMetrics(tick_font)
        for tick_value in tick_values:
            x = round(plot_left + plot_width * tick_value / max_scale)
            painter.drawLine(x, plot_top - 7, x, plot_bottom + 4)
            tick_label = f"{tick_value:,}".replace(",", ".")
            tick_rect = tick_metrics.boundingRect(tick_label)
            painter.setPen(QColor("#52657D"))
            label_x = max(
                plot_left,
                min(x - tick_rect.width() // 2, plot_right - tick_rect.width()),
            )
            painter.drawText(label_x, plot_bottom + 22, tick_label)
            painter.setPen(grid_pen)

        for index, (label, count) in enumerate(self._rows, 1):
            center_y = plot_top + (index - 1) * row_height + row_height // 2
            label_text = metrics.elidedText(
                f"{index}. {label}", Qt.TextElideMode.ElideRight, label_width - 18
            )
            painter.setFont(label_font)
            painter.setPen(QColor("#344158"))
            painter.drawText(10, center_y + metrics.ascent() // 2 - 1, label_text)

            bar_width = max(2, round(plot_width * count / max_scale))
            bar_rect = self.rect().adjusted(0, 0, 0, 0)
            bar_rect.setRect(plot_left, center_y - 9, bar_width, 18)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#4F8CFF"))
            painter.drawRoundedRect(bar_rect, 4, 4)

            percentage = count * 100.0 / self._classified_count
            value_font = QFont(label_font)
            value_font.setBold(True)
            painter.setFont(value_font)
            painter.setPen(QColor("#24364D"))
            value_text = f"{count:,}  ({_format_number(percentage, 1)}%)"
            painter.drawText(plot_right + 10, center_y + metrics.ascent() // 2 - 1, value_text)

        painter.setPen(QColor("#344158"))
        painter.setFont(tick_font)
        painter.drawText(
            plot_left,
            min(height - 3, plot_bottom + 42),
            max(0, plot_width),
            18,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            "Jumlah gambar",
        )
        painter.end()


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
            elif self.operation == "recognition":
                model_path, labels_path = ensure_model_bundle(
                    self.payload["model_dir"],
                    progress=self.progress.emit,
                    cancel_event=self.cancel_event,
                )
                recognition = recognize_image_dataset(
                    self.payload["path"],
                    model_path=model_path,
                    labels_path=labels_path,
                    model_directory=self.payload["model_dir"],
                    sample_references=self.payload.get("sample_references"),
                    known_label_counts=self.payload.get("class_distribution"),
                    label_source=self.payload.get("label_source"),
                    progress=self.progress.emit,
                    cancel_event=self.cancel_event,
                )
                self.completed.emit({"operation": "recognition", "recognition": recognition})
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

    @staticmethod
    def _configure_result_table(
        table: QTableWidget,
        headers: tuple[str, ...],
        *,
        max_height: int,
    ) -> None:
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().hide()
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setMinimumHeight(116)
        table.setMaximumHeight(max_height)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

    def _render_recognition_ml_tables(self, evaluation: dict[str, Any]) -> None:
        model_rows = evaluation.get("candidate_models") or []
        self.recognition_model_table.setRowCount(len(model_rows))
        for row, result in enumerate(model_rows):
            values = (
                str(result.get("model") or "Model"),
                f"{_format_number(float(result.get('mean_macro_f1') or 0) * 100, 1)}%",
                f"{_format_number(float(result.get('std_macro_f1') or 0) * 100, 1)}%",
                "Terpilih" if result.get("model") == evaluation.get("selected_model") else "Dibandingkan",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.recognition_model_table.setItem(row, column, item)

        class_rows = evaluation.get("class_metrics") or []
        self.recognition_class_table.setRowCount(len(class_rows))
        for row, result in enumerate(class_rows):
            values = (
                str(result.get("class") or "Tidak diketahui"),
                _format_number(result.get("support") or 0),
                f"{_format_number(float(result.get('precision') or 0) * 100, 1)}%",
                f"{_format_number(float(result.get('recall') or 0) * 100, 1)}%",
                f"{_format_number(float(result.get('f1') or 0) * 100, 1)}%",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.recognition_class_table.setItem(row, column, item)

        classes = [str(label) for label in (evaluation.get("classes") or [])]
        confusion = evaluation.get("confusion_matrix") or []
        has_confusion = (
            len(classes) == len(confusion)
            and bool(classes)
            and all(isinstance(row_values, (list, tuple)) and len(row_values) == len(classes) for row_values in confusion)
        )
        if has_confusion:
            self.recognition_confusion_table.setColumnCount(len(classes) + 1)
            self.recognition_confusion_table.setRowCount(len(classes))
            headers = ["Aktual ↓ / Prediksi →", *classes]
            self.recognition_confusion_table.setHorizontalHeaderLabels(headers)
            header = self.recognition_confusion_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
            self.recognition_confusion_table.setColumnWidth(0, 160)
            for column in range(1, len(classes) + 1):
                self.recognition_confusion_table.setColumnWidth(column, 78)
            max_value = max(
                (int(value) for row_values in confusion for value in row_values),
                default=0,
            )
            for row, true_label in enumerate(classes):
                row_header = QTableWidgetItem(true_label)
                row_header.setToolTip(true_label)
                self.recognition_confusion_table.setItem(row, 0, row_header)
                for column, predicted_label in enumerate(classes):
                    value = int(confusion[row][column])
                    item = QTableWidgetItem(_format_number(value))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item.setToolTip(f"Aktual: {true_label}\nPrediksi: {predicted_label}\nJumlah: {value}")
                    intensity = value / max_value if max_value else 0.0
                    if row == column:
                        pale, strong = (232, 246, 238), (47, 139, 91)
                    else:
                        pale, strong = (255, 240, 238), (215, 74, 74)
                    color = QColor(
                        round(pale[0] + (strong[0] - pale[0]) * intensity),
                        round(pale[1] + (strong[1] - pale[1]) * intensity),
                        round(pale[2] + (strong[2] - pale[2]) * intensity),
                    )
                    item.setBackground(color)
                    if intensity > 0.65:
                        item.setForeground(QColor("#FFFFFF"))
                    self.recognition_confusion_table.setItem(row, column + 1, item)

        has_results = evaluation.get("status") == "evaluated"
        self.recognition_model_table.setVisible(has_results and bool(model_rows))
        self.recognition_class_table.setVisible(has_results and bool(class_rows))
        self.recognition_confusion_table.setVisible(has_results and has_confusion)
        self.recognition_model_title.setVisible(has_results and bool(model_rows))
        self.recognition_class_title.setVisible(has_results and bool(class_rows))
        self.recognition_confusion_title.setVisible(has_results and has_confusion)

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 18, 0, 0)
        outer.setSpacing(16)
        if self.dataset_type == "image":
            self.view_stack = QStackedWidget()
            self.upload_page = QWidget()
            self.results_page = QWidget()
            self._build_upload_page()
            self.view_stack.addWidget(self.upload_page)
            self.view_stack.addWidget(self.results_page)
            outer.addWidget(self.view_stack)
            root = QVBoxLayout(self.results_page)
        else:
            self.view_stack = None
            self.upload_page = None
            self.results_page = None
            root = outer
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        input_card, input_layout = self._card()
        input_header = QHBoxLayout()
        input_title = QLabel("Dataset yang dianalisis" if self.dataset_type == "image" else "Dataset Input")
        input_title.setObjectName("visionSectionTitle")
        input_header.addWidget(input_title)
        input_header.addStretch(1)
        self.new_dataset_button = None
        if self.dataset_type == "image":
            self.new_dataset_button = QPushButton("Dataset baru")
            self.new_dataset_button.setObjectName("secondaryButton")
            self.new_dataset_button.clicked.connect(self.show_upload_page)
            input_header.addWidget(self.new_dataset_button)
        input_header.addWidget(self._small_note("Folder atau ZIP; media tidak disalin ke database."))
        input_layout.addLayout(input_header)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setObjectName("visionDatasetPath")
        self.path_edit.setPlaceholderText("Pilih folder dataset atau file ZIP")
        self.path_edit.setToolTip("Path lokal ke folder dataset atau ZIP")
        self.folder_button = QPushButton("Pilih Folder")
        self.folder_button.setObjectName("secondaryButton")
        self.zip_button = QPushButton("Pilih ZIP")
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

        self.result_scope_label = None
        if self.dataset_type == "image":
            self.result_scope_label = QLabel(
                "HASIL PEMINDAIAN DATASET  ·  Label dipetakan dari folder/CSV jika tersedia  ·  Inferensi model belum dijalankan"
            )
            self.result_scope_label.setObjectName("visionAnalysisScope")
            self.result_scope_label.setWordWrap(True)
            root.addWidget(self.result_scope_label)

        self.status_label = QLabel("Pilih dataset untuk memulai analisis lokal.")
        self.status_label.setObjectName("visionStatusLabel")
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("visionProgressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimumHeight(8)
        self.progress_bar.setMaximumHeight(8)
        root.addWidget(self.status_label)
        root.addWidget(self.progress_bar)

        metric_grid = QGridLayout()
        metric_grid.setContentsMargins(0, 0, 0, 0)
        metric_grid.setHorizontalSpacing(12)
        metric_grid.setVerticalSpacing(12)
        media_name = "GAMBAR" if self.dataset_type == "image" else "VIDEO"
        self.metrics = [
            _VisionStatCard("TOTAL FILE GAMBAR" if self.dataset_type == "image" else f"TOTAL {media_name} FILES"),
            _VisionStatCard("GAMBAR TERBACA" if self.dataset_type == "image" else f"VALID {media_name}S"),
            _VisionStatCard("GAMBAR BERLABEL" if self.dataset_type == "image" else f"LABELED {media_name}S"),
            _VisionStatCard("GAMBAR TANPA LABEL" if self.dataset_type == "image" else f"UNLABELED {media_name}S"),
            _VisionStatCard("KELAS TERBACA" if self.dataset_type == "image" else "DETECTED CLASSES"),
            _VisionStatCard("FORMAT FILE" if self.dataset_type == "image" else "FILE FORMATS"),
            _VisionStatCard("UKURAN DATASET" if self.dataset_type == "image" else "DATASET SIZE"),
            _VisionStatCard("FILE RUSAK" if self.dataset_type == "image" else "CORRUPTED FILES"),
        ]
        for index, card in enumerate(self.metrics):
            metric_grid.addWidget(card, index // 4, index % 4)
        root.addLayout(metric_grid)

        overview_row = QHBoxLayout()
        overview_row.setSpacing(14)
        self.distribution_card, self.distribution_layout = self._card(
            "Distribusi Label Kelas" if self.dataset_type == "image" else "Distribusi Dataset"
        )
        self.distribution_body = QVBoxLayout()
        self.distribution_body.setContentsMargins(0, 0, 0, 0)
        self.distribution_body.setSpacing(8)
        self.distribution_empty = self._empty_label("Distribusi kelas tampil setelah analisis.")
        self.distribution_body.addWidget(self.distribution_empty)
        self.distribution_layout.addLayout(self.distribution_body)
        self.details_card, self.details_layout = self._card(
            "Ringkasan Label & Dataset" if self.dataset_type == "image" else "Dataset Information"
        )
        self.details_body = QVBoxLayout()
        self.details_body.setSpacing(7)
        self.details_layout.addLayout(self.details_body)
        overview_row.addWidget(self.distribution_card, 3)
        overview_row.addWidget(self.details_card, 2)
        root.addLayout(overview_row)

        statistics_title = "Statistik & Visualisasi Gambar" if self.dataset_type == "image" else "Video Statistics"
        self.statistics_card, self.statistics_layout = self._card(statistics_title)
        self.statistics_body = QVBoxLayout()
        self.statistics_body.setSpacing(7)
        self.statistics_body.addWidget(self._empty_label("Resolusi, channels, dan distribusi ukuran file tampil setelah analisis."))
        self.statistics_layout.addLayout(self.statistics_body)
        root.addWidget(self.statistics_card)

        gallery_card, gallery_layout = self._card("Preview Gambar" if self.dataset_type == "image" else "Sample Preview")
        gallery_heading = QHBoxLayout()
        gallery_title = QLabel("Sampel" if self.dataset_type == "image" else "Gallery")
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
        if self.dataset_type == "image":
            # Preview is the first result users see after an upload.
            root.insertWidget(4, gallery_card)

        self.recognition = None
        self.recognition_card = None
        self.recognition_button = None
        self.recognition_status = None
        self.recognition_ml_summary = None
        self.recognition_model_table = None
        self.recognition_class_table = None
        self.recognition_confusion_table = None
        self.recognition_model_title = None
        self.recognition_class_title = None
        self.recognition_confusion_title = None
        self.recognition_chart = None
        self.recognition_samples = None
        if self.dataset_type == "image":
            self.recognition_card, recognition_layout = self._card(
                "Deep Learning + Training ML Berlabel"
            )
            recognition_action = QHBoxLayout()
            self.recognition_button = QPushButton("Jalankan Pengenalan + Evaluasi ML")
            self.recognition_button.setObjectName("primaryButton")
            self.recognition_button.setEnabled(False)
            self.recognition_status = QLabel(
                "MobileNetV2 pralatih mengenali kelas ImageNet dan menghasilkan fitur. Jika label kelas valid tersedia, DORA melatih serta membandingkan classifier ML pada fitur tersebut."
            )
            self.recognition_status.setObjectName("visionRecognitionStatus")
            self.recognition_status.setWordWrap(True)
            recognition_action.addWidget(self.recognition_status, 1)
            recognition_action.addWidget(self.recognition_button)
            recognition_layout.addLayout(recognition_action)
            recognition_layout.addWidget(self._small_note(
                "DORA membandingkan SGD Logistic, Linear SVM, dan Extra Trees dengan macro-F1 3-fold pada data latih, lalu melaporkan metrik pada holdout 20% yang terpisah. Bobot MobileNetV2 tetap pralatih dan tidak di-fine-tune. Label folder/CSV perlu divalidasi sebagai ground truth; akurasi 100% universal tidak dapat dijanjikan. Object detection/segmentation memerlukan anotasi dan model khusus."
            ))
            recognition_distribution_title = QLabel("Prediksi kelas pada seluruh gambar")
            recognition_distribution_title.setObjectName("visionSubsectionTitle")
            recognition_layout.addWidget(recognition_distribution_title)
            self.recognition_chart = _RecognitionBarChart(self.recognition_card)
            recognition_layout.addWidget(self.recognition_chart)
            recognition_chart_note = self._small_note(
                "8 kelas terbanyak. Batang menunjukkan jumlah gambar per kelas; label di ujung batang menunjukkan jumlah serta persentase dari semua gambar yang berhasil diklasifikasi."
            )
            recognition_chart_note.setObjectName("visionRecognitionNote")
            recognition_layout.addWidget(recognition_chart_note)
            self.recognition_ml_summary = QLabel("")
            self.recognition_ml_summary.setObjectName("visionRecognitionNote")
            self.recognition_ml_summary.setWordWrap(True)
            recognition_layout.addWidget(self.recognition_ml_summary)

            self.recognition_model_title = QLabel("Perbandingan classifier pada validasi silang")
            self.recognition_model_title.setObjectName("visionSubsectionTitle")
            recognition_layout.addWidget(self.recognition_model_title)
            self.recognition_model_table = QTableWidget(0, 4)
            self.recognition_model_table.setObjectName("visionRecognitionModelTable")
            self._configure_result_table(
                self.recognition_model_table,
                ("Model", "Macro-F1 mean", "Simpangan", "Status"),
                max_height=174,
            )
            recognition_layout.addWidget(self.recognition_model_table)

            self.recognition_class_title = QLabel("Metrik per kelas pada holdout")
            self.recognition_class_title.setObjectName("visionSubsectionTitle")
            recognition_layout.addWidget(self.recognition_class_title)
            self.recognition_class_table = QTableWidget(0, 5)
            self.recognition_class_table.setObjectName("visionRecognitionClassTable")
            self._configure_result_table(
                self.recognition_class_table,
                ("Kelas", "Support", "Precision", "Recall", "F1"),
                max_height=280,
            )
            recognition_layout.addWidget(self.recognition_class_table)
            self.recognition_model_table.hide()
            self.recognition_class_table.hide()

            self.recognition_confusion_title = QLabel("Confusion matrix · aktual menurut baris, prediksi menurut kolom")
            self.recognition_confusion_title.setObjectName("visionSubsectionTitle")
            recognition_layout.addWidget(self.recognition_confusion_title)
            self.recognition_confusion_table = QTableWidget(0, 0)
            self.recognition_confusion_table.setObjectName("visionRecognitionConfusionTable")
            self.recognition_confusion_table.verticalHeader().hide()
            self.recognition_confusion_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.recognition_confusion_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            self.recognition_confusion_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.recognition_confusion_table.setAlternatingRowColors(False)
            self.recognition_confusion_table.setMinimumHeight(132)
            self.recognition_confusion_table.setMaximumHeight(280)
            self.recognition_confusion_table.setVisible(False)
            self.recognition_model_title.hide()
            self.recognition_class_title.hide()
            self.recognition_confusion_title.hide()
            recognition_layout.addWidget(self.recognition_confusion_table)
            samples_title = QLabel("Prediksi sampel pada halaman ini")
            samples_title.setObjectName("visionSubsectionTitle")
            recognition_layout.addWidget(samples_title)
            self.recognition_samples = QListWidget()
            self.recognition_samples.setObjectName("visionRecognitionSamples")
            self.recognition_samples.setMaximumHeight(240)
            self.recognition_samples.setSelectionMode(QListWidget.NoSelection)
            recognition_layout.addWidget(self.recognition_samples)
            root.insertWidget(5, self.recognition_card)

        quality_card, quality_layout = self._card(
            "Kualitas Gambar & Metode Ukur" if self.dataset_type == "image" else "Quality Assessment"
        )
        self.quality_body = QVBoxLayout()
        self.quality_body.setSpacing(6)
        self.quality_body.addWidget(self._empty_label("Ringkasan kualitas tampil setelah analisis."))
        quality_layout.addLayout(self.quality_body)
        root.addWidget(quality_card)

        self.task_analysis_card, self.task_analysis_layout = self._card(
            "Kesiapan Tugas Computer Vision" if self.dataset_type == "image" else "Computer Vision Task Analysis"
        )
        self.task_analysis_body = QVBoxLayout()
        self.task_analysis_body.setSpacing(8)
        self.task_analysis_body.addWidget(self._empty_label("Kesiapan setiap task tampil setelah analisis."))
        self.task_analysis_layout.addLayout(self.task_analysis_body)
        self.task_analysis_card.setVisible(self.dataset_type == "image")
        root.addWidget(self.task_analysis_card)

        self.readiness_card, self.readiness_layout = self._card(
            "Kesiapan Dataset" if self.dataset_type == "image" else "Dataset Readiness Assessment"
        )
        self.readiness_body = QVBoxLayout()
        self.readiness_body.setSpacing(8)
        self.readiness_body.addWidget(self._empty_label("Dasar kesiapan eksperimen tampil setelah analisis."))
        self.readiness_layout.addLayout(self.readiness_body)
        self.readiness_card.setVisible(self.dataset_type == "image")
        root.addWidget(self.readiness_card)

        recommendation_card, recommendation_layout = self._card(
            "Kandidat Model (Belum Dijalankan)" if self.dataset_type == "image" else "Recommendation"
        )
        self.recommendation_body = QVBoxLayout()
        self.recommendation_body.setSpacing(10)
        self.recommendation_body.addWidget(self._empty_label("Rekomendasi akan mengikuti jenis tugas dan karakteristik dataset."))
        recommendation_layout.addLayout(self.recommendation_body)
        root.addWidget(recommendation_card)

        root.addStretch(1)
        self.folder_button.clicked.connect(self._choose_folder)
        self.zip_button.clicked.connect(self._choose_zip)
        self.run_button.clicked.connect(lambda: self.page.start_analysis(self))
        if self.recognition_button is not None:
            self.recognition_button.clicked.connect(lambda: self.page.start_recognition(self))
        self.cancel_button.clicked.connect(self.page.cancel_active_operation)
        self.task_combo.currentIndexChanged.connect(lambda _index: self.page.update_recommendations(self))
        self.previous_button.clicked.connect(lambda: self._change_preview_page(-1))
        self.next_button.clicked.connect(lambda: self._change_preview_page(1))
        self.gallery.currentItemChanged.connect(self._show_selected_preview)

    def _build_upload_page(self):
        page_layout = QVBoxLayout(self.upload_page)
        page_layout.setContentsMargins(0, 0, 0, 8)
        page_layout.setSpacing(14)

        eyebrow = QLabel("VISION INTELLIGENCE  /  IMAGE ANALYSIS")
        eyebrow.setObjectName("visionUploadEyebrow")
        page_layout.addWidget(eyebrow)

        card = QFrame()
        card.setObjectName("visionUploadCard")
        card.setMinimumHeight(420)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(36, 34, 36, 32)
        card_layout.setSpacing(14)
        card_layout.addStretch(1)

        icon_badge = QFrame()
        icon_badge.setObjectName("visionUploadIconBadge")
        icon_badge.setFixedSize(76, 76)
        icon_layout = QVBoxLayout(icon_badge)
        icon_layout.setContentsMargins(14, 14, 14, 14)
        icon = QLabel()
        icon.setAlignment(Qt.AlignCenter)
        upload_icon_path = Path(__file__).parent / "assets" / "icons" / "upload.svg"
        icon.setPixmap(QIcon(str(upload_icon_path)).pixmap(QSize(40, 32)))
        icon_layout.addWidget(icon)
        card_layout.addWidget(icon_badge, 0, Qt.AlignHCenter)

        title = QLabel("Mulai analisis dataset gambar")
        title.setObjectName("visionUploadTitle")
        title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title)
        description = QLabel(
            "Pilih folder atau ZIP. DORA akan memindai gambar, label, distribusi, resolusi, dan indikator kualitas, lalu menampilkan sampel serta ringkasan hasil."
        )
        description.setObjectName("visionUploadDescription")
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignCenter)
        description.setMaximumWidth(760)
        card_layout.addWidget(description, 0, Qt.AlignHCenter)

        choices = QHBoxLayout()
        choices.setSpacing(12)
        self.upload_folder_button = QPushButton("Pilih folder gambar")
        self.upload_folder_button.setObjectName("visionUploadPrimary")
        self.upload_folder_button.setIcon(QIcon(str(upload_icon_path)))
        self.upload_folder_button.setIconSize(QSize(18, 18))
        self.upload_zip_button = QPushButton("Pilih file ZIP")
        self.upload_zip_button.setObjectName("visionUploadSecondary")
        choices.addStretch(1)
        choices.addWidget(self.upload_folder_button)
        choices.addWidget(self.upload_zip_button)
        choices.addStretch(1)
        card_layout.addLayout(choices)

        formats = QLabel(
            "Format gambar: JPG, PNG, BMP, WEBP, TIFF  ·  Analisis dimulai otomatis setelah dipilih"
        )
        formats.setObjectName("visionUploadHint")
        formats.setAlignment(Qt.AlignCenter)
        formats.setWordWrap(True)
        card_layout.addWidget(formats)
        inference_note = QLabel(
            "Hasil berisi temuan pemindaian dataset dan kandidat metode; model belum melakukan prediksi atau training."
        )
        inference_note.setObjectName("visionUploadDisclaimer")
        inference_note.setWordWrap(True)
        inference_note.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(inference_note)
        card_layout.addStretch(1)
        page_layout.addWidget(card)
        page_layout.addStretch(1)

        self.upload_folder_button.clicked.connect(self._choose_folder)
        self.upload_zip_button.clicked.connect(self._choose_zip)

    def show_upload_page(self):
        if self.view_stack is not None:
            self.view_stack.setCurrentWidget(self.upload_page)

    def show_results_page(self):
        if self.view_stack is not None:
            self.view_stack.setCurrentWidget(self.results_page)

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
            self.show_results_page()
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
            self.render_recognition_samples()
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
        self.render_recognition_samples()

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
        self.show_results_page()
        self.analysis = analysis
        self.recommendations = recommendations
        self.recognition = analysis.get("recognition") if self.dataset_type == "image" else None
        self.analysis_id = analysis_id
        self.sample_refs = list(analysis.get("sample_references") or [])
        self.reset_previews()
        self.render_overview()
        self.render_recommendations()
        self.render_recognition()
        self.render_preview_page()
        if self.recognition_button is not None:
            self.recognition_button.setEnabled(bool(analysis.get("total_samples")))
        self.progress_bar.setValue(100)
        self.status_label.setText("Analisis selesai. Rekomendasi adalah panduan, bukan hasil eksperimen model.")

    def show_empty_state(self):
        self.show_results_page()
        self.analysis = None
        self.recommendations = None
        self.recognition = None
        self.analysis_id = None
        self.sample_refs = []
        self.reset_previews()
        for card in self.metrics:
            card.value.setText("—")
            card.detail.setText("")
        for layout, message in (
            (self.distribution_body, "Distribusi kelas tampil setelah analisis."),
            (self.details_body, "Informasi dataset tampil setelah analisis."),
            (self.statistics_body, "Statistik gambar tampil setelah analisis."),
            (self.quality_body, "Ringkasan kualitas tampil setelah analisis."),
            (self.task_analysis_body, "Kesiapan setiap task tampil setelah analisis."),
            (self.readiness_body, "Dasar kesiapan eksperimen tampil setelah analisis."),
            (self.recommendation_body, "Rekomendasi akan mengikuti jenis tugas dan karakteristik dataset."),
        ):
            self._clear_layout(layout)
            layout.addWidget(self._empty_label(message))
        self.render_preview_page()
        self.render_recognition()
        self.progress_bar.setValue(0)

    def render_recognition(self):
        if self.recognition_card is None:
            return
        self.recognition_samples.clear()
        if not self.recognition:
            self.recognition_status.setText(
                "MobileNetV2 mengenali satu kelas umum ImageNet per gambar. Model sekitar 13,3 MiB diunduh sekali; setelah itu inferensi berjalan lokal."
            )
            if self.result_scope_label is not None:
                self.result_scope_label.setText(
                    "HASIL PEMINDAIAN DATASET  ·  Label dipetakan dari folder/CSV jika tersedia  ·  Inferensi model belum dijalankan"
                )
            self.recognition_ml_summary.setText("")
            self._render_recognition_ml_tables({})
            self.recognition_chart.set_distribution({}, 0)
            self.render_recognition_samples()
            return

        result = self.recognition
        if self.result_scope_label is not None:
            self.result_scope_label.setText(
                "ANALISIS DATASET + PREDIKSI MODEL  ·  Prediksi MobileNetV2 ImageNet-1K dijalankan lokal  ·  Label folder/CSV tetap sumber label dataset, bukan prediksi model"
            )
        self.recognition_status.setText(
            f"{result.get('model')} · {result.get('classified_count', 0):,} gambar berhasil dikenali · "
            f"{result.get('unreadable_count', 0):,} tidak terbaca · rata-rata skor top-1 "
            f"{_format_number(float(result.get('mean_top1_score') or 0) * 100, 1)}%."
        )
        distribution = result.get("predicted_class_distribution") or {}
        self.recognition_chart.set_distribution(
            distribution,
            int(result.get("classified_count") or 0),
        )

        custom_ml = result.get("custom_ml") or {}
        if custom_ml.get("status") == "evaluated":
            selected_model = str(custom_ml.get("selected_model") or "Classifier ML")
            cv_result = next(
                (
                    candidate
                    for candidate in (custom_ml.get("candidate_models") or [])
                    if candidate.get("model") == selected_model
                ),
                {},
            )
            metric_labels = (
                ("accuracy", "accuracy"),
                ("balanced_accuracy", "balanced accuracy"),
                ("macro_precision", "macro precision"),
                ("macro_recall", "macro recall"),
                ("macro_f1", "macro F1"),
                ("weighted_f1", "weighted F1"),
            )
            metric_text = " · ".join(
                f"{label} {_format_number(float(custom_ml[key]) * 100, 1)}%"
                for key, label in metric_labels
                if custom_ml.get(key) is not None
            )
            evaluation_text = (
                f"Model ML terpilih: {selected_model} · "
                f"validasi silang macro-F1 {_format_number(float(cv_result.get('mean_macro_f1') or 0) * 100, 1)}% "
                f"± {_format_number(float(cv_result.get('std_macro_f1') or 0) * 100, 1)}%.\n"
                f"Holdout berstrata 20%: {custom_ml.get('validation_samples', 0)} gambar uji; "
                f"{custom_ml.get('training_samples', 0)} gambar latih · {metric_text}.\n"
                + str(custom_ml.get("note") or "")
            )
            if custom_ml.get("model_path"):
                evaluation_text += f"\nModel hasil training disimpan di: {custom_ml['model_path']}"
            if custom_ml.get("artifact_note"):
                evaluation_text += f"\n{custom_ml['artifact_note']}"
            self.recognition_ml_summary.setText(evaluation_text)
            self._render_recognition_ml_tables(custom_ml)
        else:
            self.recognition_ml_summary.setText(
                str(custom_ml.get("note") or "Classifier ML khusus tidak dijalankan.")
            )
            self._render_recognition_ml_tables(custom_ml)
        self.render_recognition_samples()

    def render_recognition_samples(self):
        if self.recognition_samples is None:
            return
        self.recognition_samples.clear()
        if not self.recognition:
            self.recognition_samples.hide()
            return
        self.recognition_samples.show()
        by_reference = {
            self._reference_key(item): item
            for item in (self.recognition.get("sample_predictions") or [])
        }
        for reference in self.selected_references():
            prediction = by_reference.get(self._reference_key(reference))
            if not prediction:
                continue
            predictions = prediction.get("predictions") or []
            if not predictions:
                continue
            top = predictions[0]
            name = Path(prediction.get("name") or "sample").name
            score = float(top.get("score") or 0) * 100
            item = QListWidgetItem(
                f"{name}  →  {top.get('label', 'Tidak diketahui')}  ·  {_format_number(score, 1)}%"
            )
            item.setToolTip(
                "Top-5 model ImageNet:\n"
                + "\n".join(
                    f"{entry.get('label')}: {_format_number(float(entry.get('score') or 0) * 100, 1)}%"
                    for entry in predictions
                )
            )
            self.recognition_samples.addItem(item)
        if self.recognition_samples.count() == 0:
            self.recognition_samples.addItem("Belum ada prediksi untuk sampel pada halaman ini.")

    def render_overview(self):
        if self.analysis is None:
            return
        result = self.analysis
        valid_count = int(result.get("total_samples") or 0)
        total_files = int(result.get("total_files") or valid_count)
        class_distribution = result.get("class_distribution") or {}
        labeled_count = int(result.get("labeled_count") if result.get("labeled_count") is not None else sum(class_distribution.values()))
        unlabeled_count = int(result.get("unlabeled_count") or max(0, valid_count - labeled_count))
        class_count = int(result.get("class_count") or 0)
        formats = result.get("formats") or {}
        self.metrics[0].value.setText(_format_number(total_files))
        media_label = "gambar" if self.dataset_type == "image" else "video"
        self.metrics[0].detail.setText(f"File {media_label} berekstensi didukung")
        self.metrics[1].value.setText(_format_number(valid_count))
        self.metrics[1].detail.setText("Berhasil dibaca")
        self.metrics[2].value.setText(_format_number(labeled_count))
        self.metrics[2].detail.setText("Label kelas terpetakan dari CSV/folder")
        self.metrics[3].value.setText(_format_number(unlabeled_count))
        self.metrics[3].detail.setText(f"{media_label.title()} valid tanpa label")
        self.metrics[4].value.setText(_format_number(class_count))
        self.metrics[4].detail.setText(str(result.get("label_source") or "Sumber label tidak terdeteksi"))
        self.metrics[5].value.setText(_format_number(len(formats)))
        self.metrics[5].detail.setText(", ".join(f"{name} {count:,}" for name, count in formats.items()) or "Tidak tersedia")
        self.metrics[6].value.setText(_format_bytes(result.get("media_size_bytes", result.get("source_size_bytes"))))
        size_note = "Total bytes media; ZIP dihitung dari ukuran file anggota" if self.dataset_type == "image" else "Ukuran sumber/arsip dataset"
        self.metrics[6].detail.setText(size_note)
        self.metrics[7].value.setText(_format_number(result.get("invalid_files")))
        self.metrics[7].detail.setText("File rusak / tidak dapat dibaca")
        self._render_distribution()
        self._render_details()
        self._render_statistics()
        self._render_quality()
        if self.dataset_type == "image":
            self._render_task_analysis()
            self._render_readiness()

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
        self.distribution_body.addWidget(self._small_note(f"{len(distribution)} kelas · semua kelas ditampilkan"))
        self.distribution_body.addWidget(self._small_note(str(result.get("label_source_description") or "Validasi bahwa label benar-benar mewakili target penelitian.")))
        for name, count in sorted(distribution.items(), key=lambda item: (-item[1], str(item[0]).casefold())):
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
            coefficient = result.get("class_count_coefficient_of_variation")
            self.distribution_body.addWidget(
                self._small_note(
                    f"Rasio maksimum/minimum: {_format_number(imbalance, 2)}:1 · CV populasi: "
                    f"{_format_number(coefficient, 3) if coefficient is not None else '—'} · "
                    "rasio >3:1 adalah ambang triase, bukan aturan universal."
                )
            )

    def _add_detail(self, label: str, value: Any, layout=None):
        row = QHBoxLayout()
        title = QLabel(label)
        title.setObjectName("visionFieldLabel")
        value_label = QLabel(str(value if value not in (None, "") else "—"))
        value_label.setObjectName("visionDetailValue")
        value_label.setWordWrap(True)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(title, 1)
        row.addWidget(value_label, 2)
        (layout or self.details_body).addLayout(row)

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
        if self.dataset_type == "image":
            self._add_detail("Arti label", result.get("label_source_description"))
            channels = result.get("channel_counts") or {}
            self._add_detail("Channels", ", ".join(f"{name}: {count:,}" for name, count in channels.items()) or "Tidak tersedia")
            self._add_detail("Konten", f"Grayscale {result.get('grayscale_content_count', 0):,} · warna {result.get('color_content_count', 0):,}")
        if result.get("metadata_files"):
            self._add_detail("Metadata CSV", ", ".join(result["metadata_files"]))
        dimensions = result.get("dimensions") or {}
        if self.dataset_type == "image":
            self._add_detail(
                "Dimensi min–maks",
                f"{dimensions.get('width_min')}×{dimensions.get('height_min')} – {dimensions.get('width_max')}×{dimensions.get('height_max')} px",
            )
            ratio = result.get("aspect_ratio") or {}
            self._add_detail("Aspect ratio min–maks", f"{_format_number(ratio.get('min'), 3)} – {_format_number(ratio.get('max'), 3)} · median {_format_number(ratio.get('median'), 3)} · rata-rata {_format_number(ratio.get('mean'), 3)}")
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

    @staticmethod
    def _distribution_text(distribution: dict[str, int] | None) -> str:
        return ", ".join(
            f"{key.replace('_', ' ')}: {value:,}"
            for key, value in (distribution or {}).items()
        ) or "Tidak tersedia"

    def _render_statistics(self):
        self._clear_layout(self.statistics_body)
        if not self.analysis:
            self.statistics_body.addWidget(self._empty_label("Statistik dataset tampil setelah analisis."))
            return
        if self.dataset_type != "image":
            self.statistics_body.addWidget(self._empty_label("Ringkasan dimensi, FPS, durasi, dan frame video tersedia di Dataset Information."))
            return
        result = self.analysis
        dimensions = result.get("dimensions") or {}
        file_sizes = result.get("file_size_bytes") or {}
        sample_note = "Median dihitung dari seluruh gambar valid." if not result.get("statistics_method", {}).get("median_is_sampled") else (
            f"Median memakai reservoir {result.get('statistics_method', {}).get('median_sample_size', 0):,} gambar; mean, min, max, dan histogram dihitung dari semua file valid."
        )
        self._add_detail(
            "Width px",
            f"min {_format_number(dimensions.get('width_min'))} · median {_format_number(dimensions.get('width_median'))} · mean {_format_number(dimensions.get('width_mean'))} · max {_format_number(dimensions.get('width_max'))}",
            self.statistics_body,
        )
        self._add_detail(
            "Height px",
            f"min {_format_number(dimensions.get('height_min'))} · median {_format_number(dimensions.get('height_median'))} · mean {_format_number(dimensions.get('height_mean'))} · max {_format_number(dimensions.get('height_max'))}",
            self.statistics_body,
        )
        self._add_detail(
            "Resolusi (piksel)",
            f"min {_format_number(dimensions.get('resolution_pixels_min'))} · median {_format_number(dimensions.get('resolution_pixels_median'))} · mean {_format_number(dimensions.get('resolution_pixels_mean'))} · max {_format_number(dimensions.get('resolution_pixels_max'))}",
            self.statistics_body,
        )
        self._add_detail("Distribusi width", self._distribution_text(result.get("width_distribution")), self.statistics_body)
        self._add_detail("Distribusi height", self._distribution_text(result.get("height_distribution")), self.statistics_body)
        self._add_detail("Aspect ratio", self._distribution_text(result.get("aspect_ratio_buckets")), self.statistics_body)
        self._add_detail(
            "Ukuran file",
            f"min {_format_bytes(file_sizes.get('min'))} · median {_format_bytes(file_sizes.get('median'))} · mean {_format_bytes(file_sizes.get('mean'))} · max {_format_bytes(file_sizes.get('max'))}",
            self.statistics_body,
        )
        self._add_detail("Distribusi ukuran", self._distribution_text(result.get("file_size_distribution")), self.statistics_body)
        self._add_detail("Format", self._distribution_text(result.get("formats")), self.statistics_body)
        self._add_detail("Mode gambar", self._distribution_text(result.get("image_modes")), self.statistics_body)
        self._add_detail("Jumlah channel", self._distribution_text(result.get("channel_counts")), self.statistics_body)
        self._add_histogram_rows("Sebaran resolusi", result.get("resolution_buckets"), self.statistics_body)
        self._add_histogram_rows("Sebaran aspect ratio", result.get("aspect_ratio_buckets"), self.statistics_body)
        self._add_histogram_rows("Sebaran ukuran file", result.get("file_size_distribution"), self.statistics_body)
        self.statistics_body.addWidget(
            self._small_note(f"Grafik memakai seluruh file gambar valid. {sample_note}")
        )

    def _add_histogram_rows(self, title: str, distribution: dict[str, int] | None, layout):
        if not distribution:
            return
        heading = QLabel(title)
        heading.setObjectName("visionSubsectionTitle")
        layout.addWidget(heading)
        maximum = max((int(value or 0) for value in distribution.values()), default=0)
        for bucket, count in distribution.items():
            row = QHBoxLayout()
            label = QLabel(str(bucket).replace("_", " "))
            label.setObjectName("visionDistributionLabel")
            label.setMinimumWidth(120)
            label.setMaximumWidth(180)
            bar = QProgressBar()
            bar.setObjectName("visionDistributionBar")
            bar.setRange(0, max(1, maximum))
            bar.setValue(max(0, int(count or 0)))
            bar.setTextVisible(False)
            bar.setMinimumHeight(6)
            bar.setMaximumHeight(9)
            value = QLabel(_format_number(count))
            value.setObjectName("visionDistributionValue")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(label)
            row.addWidget(bar, 1)
            row.addWidget(value)
            layout.addLayout(row)

    def _render_assessment_items(self, layout, items: list[dict[str, Any]]):
        for item in items:
            name = QLabel(str(item.get("name") or item.get("task") or "Assessment"))
            name.setObjectName("visionSubsectionTitle")
            status = QLabel(str(item.get("status") or "Status tidak tersedia"))
            status.setObjectName("visionRecommendationTask")
            evidence = QLabel(str(item.get("evidence") or "Bukti tidak tersedia."))
            evidence.setObjectName("visionMutedLabel")
            evidence.setWordWrap(True)
            layout.addWidget(name)
            layout.addWidget(status)
            layout.addWidget(evidence)

    def _render_task_analysis(self):
        self._clear_layout(self.task_analysis_body)
        if not self.analysis:
            self.task_analysis_body.addWidget(self._empty_label("Kesiapan setiap task tampil setelah analisis."))
            return
        assessments = assess_image_tasks(self.analysis)
        self._render_assessment_items(self.task_analysis_body, assessments)
        self.task_analysis_body.addWidget(self._small_note(
            "Detection dan segmentation berstatus belum diverifikasi karena pemindai ini belum membaca file YOLO/COCO atau mask."
        ))

    def _render_readiness(self):
        self._clear_layout(self.readiness_body)
        if not self.analysis:
            self.readiness_body.addWidget(self._empty_label("Dasar kesiapan eksperimen tampil setelah analisis."))
            return
        assessment = assess_image_readiness(self.analysis, self.task())
        self.readiness_body.addWidget(self._small_note(assessment.get("method_note", "")))
        self._render_assessment_items(self.readiness_body, assessment.get("items") or [])

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
                f"Brightness luminans (0–255): mean {_format_number(characteristics.get('mean_brightness_0_255'))}, median {_format_number(characteristics.get('brightness_median_0_255'))}; thumbnail maksimal 64×64."
            )
            self._add_quality_note(
                f"Contrast (simpangan baku luminans 0–255): mean {_format_number(characteristics.get('mean_contrast_luminance_std_0_255'))}, median {_format_number(characteristics.get('contrast_median_luminance_std_0_255'))}."
            )
            self._add_quality_note(
                f"Blur proxy (varians Laplacian 4-neighbor): mean {_format_number(characteristics.get('mean_laplacian_variance_64px'))}, median {_format_number(characteristics.get('laplacian_variance_median_64px'))}; nilai rendah bukan diagnosis blur tanpa kalibrasi domain."
            )
            self._add_quality_note(
                f"Kandungan grayscale: {characteristics.get('grayscale_content_count', 0):,} gambar; aturan ≥99% piksel thumbnail dengan rentang RGB ≤2. Mode file: {self._distribution_text(result.get('image_modes'))}."
            )
            self._add_quality_note("Estimasi noise tidak dilakukan karena metode umum memerlukan asumsi/flat-region tervalidasi; metrik visual dihitung pada thumbnail, bukan inspeksi kualitas menyeluruh.")
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
            self.start_analysis(tab)
            return
        if not record:
            self.start_analysis(tab)
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

    def start_recognition(self, tab: _VisionTab):
        if tab.dataset_type != "image" or not tab.analysis:
            return
        if self._thread is not None:
            tab.status_label.setText("Tunggu operasi saat ini selesai atau batalkan terlebih dahulu.")
            return
        path = tab.path_edit.text().strip()
        if not path or not Path(path).exists():
            tab.recognition_status.setText("Path dataset gambar tidak tersedia. Pilih dataset kembali.")
            return
        from app.core.config import get_data_dir

        tab.recognition_status.setText(
            "Memeriksa model. Unduhan pertama memerlukan koneksi internet; inferensi berikutnya berjalan lokal."
        )
        tab.progress_bar.setValue(0)
        self._start_worker(
            "recognition",
            {
                "path": path,
                "model_dir": str(get_data_dir() / "vision_models"),
                "sample_references": tab.sample_refs,
                "class_distribution": tab.analysis.get("class_distribution") or {},
                "label_source": tab.analysis.get("label_source"),
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
            if tab.dataset_type == "image":
                tab._render_task_analysis()
                tab._render_readiness()
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
            if tab.dataset_type == "image":
                tab.upload_folder_button.setEnabled(not busy)
                tab.upload_zip_button.setEnabled(not busy)
                if tab.new_dataset_button is not None:
                    tab.new_dataset_button.setEnabled(not busy)
            tab.task_combo.setEnabled(not busy)
            if tab.sample_frames_combo is not None:
                tab.sample_frames_combo.setEnabled(not busy)
            if tab.recognition_button is not None:
                tab.recognition_button.setEnabled(not busy and bool(tab.analysis))
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
        if value.get("operation") == "recognition":
            recognition = value.get("recognition") or {}
            tab.recognition = recognition
            tab.analysis = {**(tab.analysis or {}), "recognition": recognition}
            tab.render_recognition()
            tab.progress_bar.setValue(100)
            tab.status_label.setText("Pengenalan MobileNetV2 selesai; skor model bukan label ground truth.")
            if tab.analysis_id is not None:
                try:
                    self.repository.update_results(tab.analysis_id, tab.analysis)
                except Exception as error:
                    tab.status_label.setText(f"Pengenalan selesai; hasil SQLite tidak diperbarui: {error}")
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
            if self._operation == "recognition":
                prefix = "Pengenalan model gagal"
                if tab.recognition_status is not None:
                    tab.recognition_status.setText(f"{prefix}: {error}")
            tab.status_label.setText(f"{prefix}: {error}")
            if self._operation != "preview":
                tab.progress_bar.setValue(0)

    @Slot()
    def _on_cancelled(self):
        tab = self._active_tab
        if tab is not None:
            tab.status_label.setText("Operasi dibatalkan. Hasil parsial tidak disimpan.")
            if self._operation == "recognition" and tab.recognition_status is not None:
                tab.recognition_status.setText("Pengenalan model dibatalkan; hasil parsial tidak ditampilkan.")

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

