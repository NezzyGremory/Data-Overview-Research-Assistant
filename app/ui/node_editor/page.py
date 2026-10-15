from __future__ import annotations

import ast
import codecs
import importlib.util
import json
import math
import os
import pickle
import struct
import subprocess
import sys
from pathlib import Path

import pandas as pd

from PySide6.QtCore import QMimeData, QObject, QPointF, QRectF, QStandardPaths, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QDrag, QPainter, QPainterPath, QPen, QPolygonF, QTextCursor
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGraphicsEllipseItem,
    QGraphicsItem, QGraphicsPathItem, QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsScene, QGraphicsTextItem,
    QGraphicsView, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QSpinBox, QCheckBox, QScrollArea, QProgressBar,
    QPlainTextEdit,
    QListWidget, QListWidgetItem, QSizePolicy, QTabWidget,
)

from app.ml.graph import GraphEngine, GraphError


PORT_COLORS = {"table": "#0071C2", "estimator": "#8B5CF6", "model": "#098E00", "metrics": "#E09A00"}
RESULT_PREVIEW_ROWS = 200
RESULT_PREVIEW_COLUMNS = 50
PACKAGE_DISTRIBUTIONS = {"imblearn": "imbalanced-learn", "xgboost": "xgboost", "lightgbm": "lightgbm"}
TASKS = [("Semua", "all"), ("Klasifikasi", "classification"), ("Regresi", "regression"), ("Pengelompokan", "clustering"), ("Anomali", "anomaly"), ("Rangkaian Waktu", "time_series")]
CATEGORY_TRANSLATIONS = {"Data": "Data", "Classification": "Klasifikasi", "Regression": "Regresi", "Clustering": "Clustering", "Anomaly Detection": "Deteksi Anomali", "Validation": "Validasi", "Preprocessing": "Pra-pemrosesan", "Model": "Model", "Ensemble": "Ensemble", "Evaluation": "Evaluasi", "Tuning": "Penyetelan", "Forecasting": "Peramalan", "Data Tidak Seimbang": "Data Tidak Seimbang", "Transformer": "Transformer"}


def _read_graph_record(stream):
    def read_exact(size):
        chunks = bytearray()
        while len(chunks) < size:
            chunk = stream.read(size - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        return bytes(chunks)

    header = read_exact(8)
    if not header:
        return None
    if len(header) != 8:
        raise EOFError("Header pesan graf terpotong.")
    size = struct.unpack(">Q", header)[0]
    payload = read_exact(size)
    if len(payload) != size:
        raise EOFError("Isi pesan graf terpotong.")
    return pickle.loads(payload)


class GraphNodeItem(QGraphicsRectItem):
    W, H = 212, 118

    def __init__(self, page, node, x, y):
        super().__init__(0, 0, self.W, self.H)
        self.page, self.node_id = page, node.id
        self.setPos(x, y)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setCursor(Qt.OpenHandCursor)
        self.setBrush(QColor("white"))
        self.setPen(QColor("#B7D5F3"))
        self.setZValue(1)
        self.title = QGraphicsTextItem(page.engine.specs[node.spec_id].name, self)
        self.title.setDefaultTextColor("#172033")
        title_font = self.title.font()
        title_font.setBold(True)
        title_font.setPointSize(9)
        self.title.setFont(title_font)
        self.title.setPos(12, 8)
        self.title.setTextWidth(self.W - 24)
        self.title.setAcceptedMouseButtons(Qt.NoButton)
        self.port_items = {}
        spec = page.engine.specs[node.spec_id]
        self._accent = (
            QColor(PORT_COLORS.get(spec.outputs[0].type, "#0878D1"))
            if spec.outputs
            else QColor("#0878D1")
        )
        for side, ports, xx in (("in", spec.inputs, 0), ("out", spec.outputs, self.W)):
            for index, port in enumerate(ports):
                yy = 44 + index * 23
                dot = QGraphicsEllipseItem(-5, -5, 10, 10, self)
                dot.setPos(xx, yy)
                dot.setBrush(QColor(PORT_COLORS.get(port.type, "#75849A")))
                dot.setPen(QColor("white"))
                dot.setData(0, (self.node_id, side, port.name))
                dot.setToolTip(f"{port.name} ({port.type}) - seret ke port yang sesuai")
                dot.setCursor(Qt.CrossCursor)
                self.port_items[(side, port.name)] = dot
                port_label = {"table": "data", "estimator": "estimator", "model": "model terlatih", "metrics": "metrik", "train": "latih", "test": "uji", "estimators": "estimator"}.get(port.name, port.name)
                text = QGraphicsTextItem(port_label, self)
                text.setDefaultTextColor("#526579")
                text.setFont(page.small_font)
                text.setTextWidth(88)
                text.setPos(10 if side == "in" else 112, yy - 10)
                text.setAcceptedMouseButtons(Qt.NoButton)
        self.status = QGraphicsTextItem(node.status, self)
        self.status.setDefaultTextColor("#75849A")
        self.status.setPos(12, self.H - 25)
        self.status.setTextWidth(self.W - 24)
        self.status.setAcceptedMouseButtons(Qt.NoButton)

    def paint(self, painter, option, widget=None):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawRoundedRect(self.rect(), 9, 9)
        painter.setPen(QPen(QColor("#EAF1F9"), 1))
        painter.drawLine(QPointF(10, 34), QPointF(self.W - 10, 34))
        painter.setPen(QPen(Qt.NoPen))
        painter.setBrush(self._accent)
        painter.drawRoundedRect(QRectF(1, 10, 3, 18), 1.5, 1.5)
        painter.restore()

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.page.update_node_cables(self.node_id)
            self.page.engine.nodes[self.node_id].position = (self.pos().x(), self.pos().y())
        elif change == QGraphicsItem.ItemSelectedHasChanged:
            self.setPen(QPen(QColor("#0878D1" if value else "#B7D5F3"), 2 if value else 1))
        return result

    def set_status(self, text):
        self.status.setPlainText(text)
        color = {"berjalan": "#0D99FF", "selesai": "#098E00", "error": "#E00000"}.get(text, "#75849A")
        self.status.setDefaultTextColor(color)
        border = color if text in {"berjalan", "error"} else "#B7D5F3"
        self.setPen(QPen(QColor(border), 2 if text == "berjalan" else 1))


class ForecastMetricsChart(QWidget):
    """Inline chart for the error metrics returned by a forecasting node."""

    METRICS = (
        ("mae", "MAE", "#0878D1"),
        ("rmse", "RMSE", "#8B5CF6"),
        ("mape", "MAPE", "#E09A00"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._values = {}
        self.setMinimumHeight(158)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_metrics(self, metrics):
        values = {}
        for key, _, _ in self.METRICS:
            try:
                value = float(metrics.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0:
                values[key] = value
        self._values = values
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))
        painter.setPen(QColor("#173B5C"))
        painter.drawText(16, 22, "Metrik Forecasting")
        painter.setPen(QColor("#64748B"))
        painter.drawText(16, 42, "MAE / RMSE satuan target; MAPE persen; error lebih kecil lebih baik")

        rows = [(key, label, color, self._values[key]) for key, label, color in self.METRICS if key in self._values]
        if not rows:
            painter.drawText(16, 78, "Belum ada metrik forecasting untuk ditampilkan.")
            return

        error_max = max((value for key, _, _, value in rows if key in {"mae", "rmse"}), default=0.0)
        left, right = 82, 100
        bar_width = max(60, self.width() - left - right)
        row_height = 30
        for index, (key, label, color, value) in enumerate(rows):
            y = 58 + index * row_height
            scale_max = error_max if key in {"mae", "rmse"} else max(100.0, value)
            fill_width = bar_width * value / scale_max if scale_max else 0
            painter.setPen(QColor("#526579"))
            painter.drawText(16, y + 16, label)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#E9EEF5"))
            painter.drawRoundedRect(QRectF(left, y + 3, bar_width, 14), 5, 5)
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(QRectF(left, y + 3, fill_width, 14), 5, 5)
            suffix = "%" if key == "mape" else ""
            painter.setPen(QColor("#173B5C"))
            painter.drawText(left + bar_width + 10, y + 16, f"{value:,.4g}{suffix}")


class NodePalette(QListWidget):
    MIME_TYPE = "application/x-dora-node"

    def startDrag(self, supported_actions):
        item = self.currentItem()
        if item is None:
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(self.MIME_TYPE, item.data(Qt.UserRole).encode("utf-8"))
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)


class GraphView(QGraphicsView):
    MIN_ZOOM = 0.35
    MAX_ZOOM = 1.8
    ZOOM_STEP = 1.15

    def __init__(self, scene, page):
        super().__init__(scene)
        self.page, self._pan = page, False
        self._pan_button = None
        self._space_held = False
        self._connecting = False
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setBackgroundBrush(QColor("#F1F7FF"))
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if not delta:
                delta = event.pixelDelta().y()
                steps = delta / 120 if delta else 0
            else:
                steps = delta / 120
            if steps:
                self.zoom_by(self.ZOOM_STEP ** steps)
                event.accept()
                return
        super().wheelEvent(event)

    def set_zoom(self, value):
        value = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(value)))
        current = self.transform().m11()
        if current > 0 and not math.isclose(current, value, rel_tol=1e-4):
            self.scale(value / current, value / current)
        self.sync_zoom_label()

    def sync_zoom_label(self):
        value = self.transform().m11()
        self.page.zoom_label.setText(f"{round(value * 100)}%")

    def zoom_by(self, factor):
        self.set_zoom(self.transform().m11() * factor)

    def _node_item_at(self, position):
        item = self.itemAt(position)
        while item is not None:
            if isinstance(item, GraphNodeItem):
                return item
            item = item.parentItem()
        return None

    def _begin_pan(self, event):
        self._pan = True
        self._pan_button = event.button()
        self._last = event.position().toPoint()
        self.setCursor(Qt.ClosedHandCursor)
        self.setFocus(Qt.MouseFocusReason)

    def _end_pan(self):
        self._pan = False
        self._pan_button = None
        self.setCursor(Qt.ArrowCursor)

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        grid = 32
        left = int(rect.left()) - (int(rect.left()) % grid)
        top = int(rect.top()) - (int(rect.top()) % grid)
        painter.save()
        painter.setPen(QPen(QColor("#E3ECF7"), 1))
        x = left
        while x < rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += grid
        y = top
        while y < rect.bottom():
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += grid
        painter.restore()

    def _port_at(self, position):
        item = self.itemAt(position)
        while item is not None:
            data = item.data(0)
            if isinstance(data, tuple) and len(data) == 3:
                return data
            item = item.parentItem()
        return None

    def _finish_or_cancel_connection(self, position):
        end = self._port_at(position)
        if end is not None and end != self.page.connection_start:
            self.page.finish_connection(end)
            self.page.cancel_connection()
        self._connecting = False
        self.page.remove_connection_preview()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton and self.page.connection_start is None:
            self._begin_pan(event)
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            port = self._port_at(event.position().toPoint())
            if port is not None:
                self.setFocus()
                if self.page.connection_start is None:
                    self.page.start_connection(port)
                    self._connecting = True
                elif port == self.page.connection_start:
                    self.page.cancel_connection()
                    self._connecting = False
                else:
                    self.page.finish_connection(port)
                    self.page.cancel_connection()
                    self._connecting = False
                event.accept()
                return
            if (
                self._space_held
                and self.page.connection_start is None
                and not self._node_item_at(event.position().toPoint())
            ):
                self._begin_pan(event)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan:
            delta = event.position().toPoint() - self._last
            self._last = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        if self._connecting and self.page.connection_start is not None:
            self.page.update_connection_preview(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pan and event.button() == self._pan_button:
            self._end_pan()
            event.accept()
            return
        if self._connecting:
            self._finish_or_cancel_connection(event.position().toPoint())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            event.accept()
            return
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.page.delete_selected_nodes()
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self.page.connection_start is not None:
            self.page.cancel_connection()
            self._connecting = False
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self._space_held = False
        if self._pan:
            self._end_pan()
        super().focusOutEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(NodePalette.MIME_TYPE):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(NodePalette.MIME_TYPE):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        if not event.mimeData().hasFormat(NodePalette.MIME_TYPE):
            super().dropEvent(event)
            return
        spec_id = bytes(event.mimeData().data(NodePalette.MIME_TYPE)).decode("utf-8")
        center = self.mapToScene(event.position().toPoint())
        position = (center.x() - GraphNodeItem.W / 2, center.y() - GraphNodeItem.H / 2)
        self.page.add_node(spec_id, position=position)
        event.acceptProposedAction()


class _RunWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str, str)
    output = Signal(str)
    phase = Signal(str)

    def __init__(self, graph_data, dataframe, packages=(), package_target=None):
        super().__init__()
        self.graph_data, self.dataframe = graph_data, dataframe
        self.packages = tuple(packages)
        self.package_target = Path(package_target) if package_target else None

    @Slot()
    def run(self):
        try:
            if self.packages:
                self._install_packages()
            if getattr(sys, "frozen", False):
                self._run_in_thread()
            else:
                self._run_in_subprocess()
        except Exception as exc:
            self.failed.emit(str(exc))

    def _install_packages(self):
        package_list = ", ".join(self.packages)
        self.phase.emit(f"Memasang library: {package_list}")
        self.output.emit(f"Memasang library yang dibutuhkan: {package_list}\n")
        frozen = getattr(sys, "frozen", False)
        if frozen:
            if self.package_target is None:
                raise RuntimeError("Lokasi pemasangan library aplikasi tidak tersedia.")
            self.package_target.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "--dora-editor-pip",
                "install",
                "--disable-pip-version-check",
                "--target",
                str(self.package_target),
                *self.packages,
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *self.packages,
            ]
        self.output.emit(f"Perintah: {' '.join(command)}\n\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=None if frozen else str(Path(__file__).resolve().parents[3]),
            env={**os.environ, "PIP_NO_INPUT": "1", "PYTHONUTF8": "1"},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while True:
                chunk = process.stdout.read1(4096)
                if not chunk:
                    break
                self.output.emit(decoder.decode(chunk))
            final_text = decoder.decode(b"", final=True)
            if final_text:
                self.output.emit(final_text)
            return_code = process.wait()
        except Exception:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
        if return_code != 0:
            raise RuntimeError(
                f"pip gagal memasang library (kode keluar {return_code}). "
                "Lihat rincian di tab Output."
            )
        if frozen and self.package_target is not None:
            target = str(self.package_target)
            if target not in sys.path:
                sys.path.insert(0, target)
            importlib.invalidate_caches()
        self.output.emit("\nInstalasi library berhasil. Menjalankan graf...\n\n")
        self.phase.emit("Library terpasang; menjalankan graf...")

    def _run_in_subprocess(self):
        process = None
        try:
            helper = Path(__file__).with_name("run_graph_process.py")
            process = subprocess.Popen(
                [sys.executable, str(helper)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=str(Path(__file__).resolve().parents[3]),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            pickle.dump(
                {"graph": self.graph_data, "dataframe": self.dataframe},
                process.stdin,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            process.stdin.close()
            terminal_record_received = False
            while True:
                record = _read_graph_record(process.stdout)
                if record is None:
                    break
                if not isinstance(record, tuple) or not record:
                    continue
                if record[0] == "progress" and len(record) == 3:
                    self.progress.emit(str(record[1]), str(record[2]))
                elif record[0] == "finished" and len(record) == 2:
                    terminal_record_received = True
                    self.finished.emit(record[1])
                elif record[0] == "failed" and len(record) == 2:
                    terminal_record_received = True
                    self.failed.emit(str(record[1]))
            return_code = process.wait()
            if not terminal_record_received:
                raise RuntimeError(
                    f"Proses graf berhenti tanpa hasil (kode {return_code})."
                )
        except Exception as exc:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            self.failed.emit(str(exc))

    def _run_in_thread(self):
        try:
            graph = _restore_graph(self.graph_data)
            self.finished.emit(graph.run(self.dataframe.copy(), self.progress.emit))
        except Exception as exc:
            self.failed.emit(str(exc))


def _restore_graph(graph_data):
    graph = GraphEngine()
    for raw in graph_data.get("nodes", []):
        graph.add_node(
            raw["spec_id"],
            raw["id"],
            raw.get("parameters"),
            tuple(raw.get("position", (0, 0))),
        )
    for edge in graph_data.get("connections", []):
        graph.connect(*edge)
    return graph


class NodeEditorPage(QWidget):
    """Editor visual untuk membangun alur analisis machine learning."""

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.engine = GraphEngine()
        self.items = {}
        self.cables = {}
        self._node_connections = {}
        self.connection_start = None
        self.connection_preview = None
        self._active_dataset_token = None
        self._suppressed_source_dataset_id = None
        self.thread = None
        self.worker = None
        self.small_font = self.font()
        self.small_font.setPointSize(8)
        self._build_ui()

    def _build_ui(self):
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        page_layout.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(22, 18, 22, 16)
        title = QLabel("Node Editor")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Dataset aktif masuk otomatis. Hubungkan port dengan garis, lalu geser kanvas memakai tombol tengah atau tahan Space sambil menyeret.")
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)
        toolbar = QHBoxLayout()
        self.run_button = QPushButton("▶  Jalankan")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self.run_graph)
        toolbar.addWidget(self.run_button)
        self.delete_button = QPushButton("Hapus Node")
        self.delete_button.setObjectName("secondaryButton")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self.delete_selected_nodes)
        toolbar.addWidget(self.delete_button)
        self.fit_button = QPushButton("Sesuaikan Tampilan")
        self.fit_button.setObjectName("secondaryButton")
        self.fit_button.clicked.connect(self.fit_graph)
        toolbar.addWidget(self.fit_button)
        self.dataset_status = QLabel("Dataset belum dimuat")
        self.dataset_status.setObjectName("nodeEditorDatasetStatus")
        self.dataset_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.dataset_status.setMinimumWidth(0)
        self.dataset_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        toolbar.addWidget(self.dataset_status, 1)
        root.addLayout(toolbar)

        file_toolbar = QHBoxLayout()
        self.save_button = QPushButton("Simpan Graf")
        self.save_button.setObjectName("secondaryButton")
        self.save_button.clicked.connect(self.save_graph)
        file_toolbar.addWidget(self.save_button)
        self.load_button = QPushButton("Muat Graf")
        self.load_button.setObjectName("secondaryButton")
        self.load_button.clicked.connect(self.load_graph)
        file_toolbar.addWidget(self.load_button)
        self.template_combo = QComboBox()
        self.template_combo.addItems(["Template…", "Klasifikasi cepat", "Regresi cepat", "Ensemble klasifikasi", "Imbalanced klasifikasi", "Forecasting cepat"])
        self.template_combo.activated.connect(self.load_template)
        self.template_combo.setObjectName("nodeEditorTemplateCombo")
        self.template_combo.setMinimumWidth(136)
        self.template_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        file_toolbar.addWidget(self.template_combo)
        file_toolbar.addStretch(1)
        root.addLayout(file_toolbar)

        run_progress = QHBoxLayout()
        self.run_spinner = QLabel("|")
        self.run_spinner.setObjectName("nodeEditorRunSpinner")
        self.run_spinner.setFixedWidth(18)
        self.run_spinner.setAlignment(Qt.AlignCenter)
        self.run_spinner.hide()
        run_progress.addWidget(self.run_spinner)
        self.run_progress_bar = QProgressBar()
        self.run_progress_bar.setRange(0, 100)
        self.run_progress_bar.setValue(0)
        self.run_progress_bar.setTextVisible(False)
        self.run_progress_bar.setMaximumHeight(9)
        self.run_progress_bar.hide()
        run_progress.addWidget(self.run_progress_bar, 1)
        self.run_status = QLabel("")
        self.run_status.setObjectName("nodeEditorRunStatus")
        self.run_status.setMinimumWidth(230)
        self.run_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.run_status.hide()
        run_progress.addWidget(self.run_status)
        root.addLayout(run_progress)
        self._spinner_frames = ("|", "/", "-", "\\")
        self._spinner_index = 0
        self._run_timer = QTimer(self)
        self._run_timer.setInterval(140)
        self._run_timer.timeout.connect(lambda: self._advance_run_spinner())
        self._completed_nodes = set()
        self._run_node_count = 0

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setObjectName("nodeEditorSplitter")
        self.splitter.setChildrenCollapsible(True)
        self.splitter.addWidget(self._build_palette())
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(-1000, -600, 3000, 2200)
        self.view = GraphView(self.scene, self)
        self.view.setFrameShape(QFrame.NoFrame)
        self.view.setToolTip("Geser dengan tombol tengah atau tahan Space sambil menyeret. Ctrl + roda mouse untuk zoom.")
        self.canvas = QFrame()
        self.canvas.setObjectName("nodeEditorCanvas")
        canvas_layout = QGridLayout(self.canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self.view, 0, 0)
        zoom_controls = QFrame(self.canvas)
        zoom_controls.setObjectName("nodeEditorZoomControls")
        zoom_layout = QHBoxLayout(zoom_controls)
        zoom_layout.setContentsMargins(6, 4, 6, 4)
        zoom_layout.setSpacing(4)
        zoom_out = QPushButton("−")
        zoom_out.setToolTip("Perkecil kanvas")
        zoom_out.setFixedSize(32, 30)
        zoom_out.clicked.connect(lambda: self.view.zoom_by(1 / GraphView.ZOOM_STEP))
        zoom_layout.addWidget(zoom_out)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(48)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        zoom_layout.addWidget(self.zoom_label)
        zoom_in = QPushButton("+")
        zoom_in.setToolTip("Perbesar kanvas")
        zoom_in.setFixedSize(32, 30)
        zoom_in.clicked.connect(lambda: self.view.zoom_by(GraphView.ZOOM_STEP))
        zoom_layout.addWidget(zoom_in)
        canvas_layout.addWidget(zoom_controls, 0, 0, Qt.AlignTop | Qt.AlignRight)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self._build_parameters())
        self.splitter.setCollapsible(0, True)
        self.splitter.setCollapsible(2, True)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([205, 1000, 250])
        self.splitter.handle(1).setToolTip("Seret untuk menyesuaikan atau tutup panel Palet Node")
        self.splitter.handle(2).setToolTip("Seret untuk menyesuaikan atau tutup panel Parameter")
        root.addWidget(self.splitter, 1)
        self.results = QTableWidget(0, 0)
        self.results.setMaximumHeight(185)
        self.results.setAlternatingRowColors(True)
        self.result_tabs = QTabWidget()
        self.result_tabs.addTab(self.results, "Tabel")
        self.forecast_chart = ForecastMetricsChart()
        self.result_tabs.addTab(self.forecast_chart, "Grafik Forecast")
        self.result_tabs.setTabVisible(1, False)
        self.output_log = QPlainTextEdit()
        self.output_log.setObjectName("nodeEditorOutput")
        self.output_log.setReadOnly(True)
        self.output_log.document().setMaximumBlockCount(2000)
        self.result_tabs.addTab(self.output_log, "Output")
        self.result_tabs.setTabVisible(2, False)
        self.result_tabs.setMaximumHeight(225)
        root.addWidget(QLabel("Hasil Evaluasi dan Compare"))
        root.addWidget(self.result_tabs)

    def _build_palette(self):
        panel = QFrame()
        panel.setObjectName("contentCard")
        panel.setMinimumWidth(190)
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Palet Node"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Cari node…")
        self.search.textChanged.connect(self.refresh_palette)
        layout.addWidget(self.search)
        self.task_filter = QComboBox()
        for label, task in TASKS:
            self.task_filter.addItem(label, task)
        self.task_filter.currentIndexChanged.connect(self.refresh_palette)
        layout.addWidget(self.task_filter)
        self.node_palette = NodePalette()
        self.node_palette.setDragEnabled(True)
        self.node_palette.setDragDropMode(QListWidget.DragOnly)
        self.node_palette.itemDoubleClicked.connect(self.add_palette_item)
        self.node_palette.setMinimumHeight(180)
        layout.addWidget(self.node_palette)
        button = QPushButton("Tambah Node")
        button.setObjectName("secondaryButton")
        button.clicked.connect(self.add_palette_node)
        self.add_node_button = button
        layout.addWidget(button)
        self.refresh_palette()
        layout.addStretch(1)
        return panel

    def _build_parameters(self):
        panel = QFrame()
        panel.setObjectName("contentCard")
        panel.setMinimumWidth(210)
        panel.setMaximumWidth(340)
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Parameter"))
        self.parameter_form = QFormLayout()
        self.parameter_box = QWidget()
        self.parameter_box.setLayout(self.parameter_form)
        layout.addWidget(self.parameter_box)
        self.scene.selectionChanged.connect(self.show_parameters)
        self.scene.selectionChanged.connect(self._update_delete_button)
        layout.addStretch(1)
        return panel

    def refresh_palette(self, *_):
        self.node_palette.clear()
        task = self.task_filter.currentData() if hasattr(self, "task_filter") else "all"
        query = self.search.text().casefold() if hasattr(self, "search") else ""
        for spec_id, spec in self.engine.specs.items():
            task_match = task in spec.task_types or "all" in spec.task_types
            if task == "classification":
                task_match = task_match or any(t.endswith("_classification") for t in spec.task_types)
            if task == "anomaly":
                task_match = task_match or "anomaly_detection" in spec.task_types
            if task != "all" and not task_match:
                continue
            if query and query not in spec.name.casefold():
                continue
            category = CATEGORY_TRANSLATIONS.get(spec.category, spec.category)
            label = f"{category} · {spec.name}"
            missing_package = None
            if spec.check_package:
                try:
                    missing_package = (
                        None
                        if importlib.util.find_spec(spec.check_package) is not None
                        else PACKAGE_DISTRIBUTIONS.get(spec.check_package, spec.check_package)
                    )
                except (ImportError, ModuleNotFoundError, ValueError):
                    missing_package = PACKAGE_DISTRIBUTIONS.get(
                        spec.check_package, spec.check_package
                    )
                if missing_package:
                    label += "  (belum terpasang; install otomatis saat Run)"
            if spec.max_recommended_rows < 100_000:
                label += f"  ⚠ {spec.max_recommended_rows:,} baris maks."
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, spec_id)
            if missing_package:
                item.setForeground(QBrush(QColor("#8A94A3")))
                item.setToolTip(
                    f"Library '{missing_package}' belum terpasang. "
                    "Library akan dicoba dipasang otomatis saat graf dijalankan."
                )
            self.node_palette.addItem(item)

    def add_palette_node(self, *_):
        item = self.node_palette.currentItem()
        spec_id = item.data(Qt.UserRole) if item is not None else None
        if spec_id:
            self.add_node(spec_id)

    def add_palette_item(self, item):
        spec_id = item.data(Qt.UserRole)
        if spec_id:
            self.add_node(spec_id)

    def add_node(self, spec_id, parameters=None, position=None):
        auto_center = position is None
        if position is None:
            selected = [item for item in self.scene.selectedItems() if isinstance(item, GraphNodeItem)]
            if selected:
                position = (selected[0].pos().x() + GraphNodeItem.W + 100, selected[0].pos().y())
            elif self.items:
                bounds = next(iter(self.items.values())).sceneBoundingRect()
                for item in self.items.values():
                    bounds = bounds.united(item.sceneBoundingRect())
                position = (bounds.right() + 100, bounds.top())
            else:
                center = self.view.mapToScene(self.view.viewport().rect().center())
                position = (center.x() - GraphNodeItem.W / 2, center.y() - GraphNodeItem.H / 2)
        node_id = self._next_node_id()
        node = self.engine.add_node(spec_id, node_id=node_id, parameters=parameters, position=position)
        item = GraphNodeItem(self, node, *node.position)
        self.items[node.id] = item
        self.scene.addItem(item)
        self.scene.clearSelection()
        item.setSelected(True)
        self.engine._cache.clear()
        self.view.setFocus()
        if auto_center:
            self.view.centerOn(item)
        return node

    def _next_node_id(self):
        index = 1
        while f"n{index}" in self.engine.nodes:
            index += 1
        return f"n{index}"

    def prepare_for_open(self):
        if self.thread is not None and self.thread.isRunning():
            self.dataset_status.setText("Graf sedang berjalan")
            return
        self.refresh_palette()
        self.sync_dataset()

    def sync_dataset(self):
        dataframe = getattr(self.main_window, "current_dataset", None)
        if dataframe is None:
            self._active_dataset_token = None
            self.dataset_status.setText("Dataset belum dimuat")
            self.dataset_status.setToolTip("Muat dataset dari halaman Dataset untuk mulai.")
            return

        rows, columns = dataframe.shape
        previous_token = self._active_dataset_token
        self._active_dataset_token = id(dataframe)
        if previous_token is not None and previous_token != self._active_dataset_token:
            self.engine._cache.clear()
            for node_id, node in self.engine.nodes.items():
                node.status = "menunggu"
                if node_id in self.items:
                    self.items[node_id].set_status(node.status)
        self.dataset_status.setText(f"Dataset aktif | {rows:,} rows x {columns:,} cols")
        filename = "Dataset"
        if self.main_window is not None:
            filename = self.main_window._current_filename()
        self.dataset_status.setToolTip(filename)

        source_nodes = [node for node in self.engine.nodes.values() if node.spec_id == "load_dataset"]
        if not source_nodes and self._suppressed_source_dataset_id != self._active_dataset_token:
            self.add_node("load_dataset", position=(0, 0))
            QTimer.singleShot(0, self.fit_graph)
            source_nodes = [node for node in self.engine.nodes.values() if node.spec_id == "load_dataset"]
        status = f"Dataset siap | {rows:,} x {columns:,}"
        for node in source_nodes:
            node.status = status
            if node.id in self.items:
                self.items[node.id].set_status(status)

    def fit_graph(self):
        if not self.items:
            return
        bounds = None
        for item in self.items.values():
            rect = item.sceneBoundingRect()
            bounds = rect if bounds is None else bounds.united(rect)
        if bounds is None or bounds.isEmpty():
            return
        self.view.fitInView(bounds.adjusted(-72, -64, 72, 64), Qt.KeepAspectRatio)
        scale = self.view.transform().m11()
        if scale > 1.25:
            self.view.scale(1.25 / scale, 1.25 / scale)
        self.view.sync_zoom_label()

    def _update_delete_button(self):
        self.delete_button.setEnabled(any(isinstance(item, GraphNodeItem) for item in self.scene.selectedItems()))

    def delete_selected_nodes(self):
        selected = [item for item in self.scene.selectedItems() if isinstance(item, GraphNodeItem)]
        if not selected:
            return
        removed_ids = {item.node_id for item in selected}
        if any(self.engine.nodes.get(node_id) and self.engine.nodes[node_id].spec_id == "load_dataset" for node_id in removed_ids):
            self._suppressed_source_dataset_id = self._active_dataset_token
        if self.connection_start and self.connection_start[0] in removed_ids:
            self.cancel_connection()
        for item in selected:
            self.items.pop(item.node_id, None)
            self.engine.nodes.pop(item.node_id, None)
            self.scene.removeItem(item)
        self.engine.connections = [
            edge for edge in self.engine.connections
            if edge[0] not in removed_ids and edge[2] not in removed_ids
        ]
        self.engine._cache.clear()
        self.redraw_cables()
        self.show_parameters()
        self._update_delete_button()
        self._clear_results()

    def start_connection(self, port):
        self.connection_start = port
        self.update_connection_preview(self.items[port[0]].port_items[(port[1], port[2])].mapToScene(QPointF(0, 0)))

    def cancel_connection(self):
        self.connection_start = None
        self.remove_connection_preview()

    def remove_connection_preview(self):
        if self.connection_preview is not None:
            self.scene.removeItem(self.connection_preview)
            self.connection_preview = None

    def update_connection_preview(self, end_point):
        if self.connection_start is None:
            return
        start_item = self.items.get(self.connection_start[0])
        if start_item is None:
            return
        port = start_item.port_items.get((self.connection_start[1], self.connection_start[2]))
        if port is None:
            return
        start_point = port.mapToScene(QPointF(0, 0))
        direction = 1 if end_point.x() >= start_point.x() else -1
        bend = max(48, abs(end_point.x() - start_point.x()) * 0.45)
        path = QPainterPath(start_point)
        path.cubicTo(
            QPointF(start_point.x() + direction * bend, start_point.y()),
            QPointF(end_point.x() - direction * bend, end_point.y()),
            end_point,
        )
        if self.connection_preview is None:
            self.connection_preview = QGraphicsPathItem()
            self.connection_preview.setPen(QPen(QColor("#4898E8"), 2, Qt.DashLine))
            self.connection_preview.setZValue(0.5)
            self.scene.addItem(self.connection_preview)
        self.connection_preview.setPath(path)

    def finish_connection(self, end):
        if self.connection_start is None:
            return
        source, target = self.connection_start, end
        if source[1] != "out" or target[1] != "in":
            source, target = target, source
        try:
            self.engine.connect(source[0], source[2], target[0], target[2])
            self.engine._cache.clear()
            self.redraw_cables()
        except GraphError as exc:
            QMessageBox.warning(self, "Koneksi tidak valid", str(exc))

    def redraw_cables(self):
        connections = list(self.engine.connections)
        active = set(connections)
        for edge, (cable, arrow) in list(self.cables.items()):
            if edge not in active:
                self.scene.removeItem(cable)
                self.scene.removeItem(arrow)
                del self.cables[edge]
        self._node_connections = {}
        for edge in connections:
            self._node_connections.setdefault(edge[0], []).append(edge)
            self._node_connections.setdefault(edge[2], []).append(edge)
            self._update_cable(edge)

    def update_node_cables(self, node_id):
        for edge in self._node_connections.get(node_id, ()):
            self._update_cable(edge)

    def _update_cable(self, edge):
        source, output, target, input_name = edge
        source_item, target_item = self.items.get(source), self.items.get(target)
        if source_item is None or target_item is None:
            return
        start_port = source_item.port_items.get(("out", output))
        end_port = target_item.port_items.get(("in", input_name))
        if start_port is None or end_port is None:
            return
        a = start_port.mapToScene(QPointF(0, 0))
        b = end_port.mapToScene(QPointF(0, 0))
        direction = 1 if b.x() >= a.x() else -1
        bend = max(48, abs(b.x() - a.x()) * 0.45)
        path = QPainterPath(a)
        path.cubicTo(
            QPointF(a.x() + direction * bend, a.y()),
            QPointF(b.x() - direction * bend, b.y()),
            b,
        )
        port_type = next(
            port.type
            for port in self.engine.specs[self.engine.nodes[source].spec_id].outputs
            if port.name == output
        )
        color = QColor(PORT_COLORS.get(port_type, "#0071C2"))
        cable, arrow = self.cables.get(edge, (None, None))
        if cable is None:
            cable = QGraphicsPathItem()
            cable.setZValue(0)
            arrow = QGraphicsPolygonItem()
            arrow.setPen(QPen(Qt.NoPen))
            arrow.setZValue(0.2)
            self.scene.addItem(cable)
            self.scene.addItem(arrow)
            self.cables[edge] = (cable, arrow)
        cable.setPath(path)
        pen = QPen(color, 2.6, Qt.SolidLine)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        cable.setPen(pen)
        arrow.setBrush(color)
        arrow.setPolygon(QPolygonF((
            QPointF(b.x() - direction * 3, b.y()),
            QPointF(b.x() - direction * 14, b.y() - 5),
            QPointF(b.x() - direction * 14, b.y() + 5),
        )))

    def show_parameters(self):
        while self.parameter_form.rowCount():
            self.parameter_form.removeRow(0)
        selected = self.scene.selectedItems()
        if not selected or not isinstance(selected[0], GraphNodeItem):
            return
        node = self.engine.nodes[selected[0].node_id]
        for key, value in node.parameters.items():
            if isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
                widget.toggled.connect(lambda v, k=key, n=node: self._set_parameter(n, k, v))
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(1, 1_000_000)
                widget.setValue(value)
                widget.valueChanged.connect(lambda v, k=key, n=node: self._set_parameter(n, k, v))
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setRange(0, 100000)
                widget.setDecimals(4)
                widget.setValue(value)
                widget.valueChanged.connect(lambda v, k=key, n=node: self._set_parameter(n, k, v))
            elif isinstance(value, (list, dict)):
                widget = QLineEdit(json.dumps(value, ensure_ascii=False))
                widget.textChanged.connect(lambda v, k=key, n=node: self._set_structured_parameter(n, k, v))
            else:
                widget = QLineEdit(str(value or ""))
                widget.textChanged.connect(lambda v, k=key, n=node: self._set_parameter(n, k, v))
            self.parameter_form.addRow(key.replace("_", " ").title(), widget)

    def _set_parameter(self, node, key, value):
        if node.parameters.get(key) == value:
            return
        node.parameters[key] = value
        self.engine._cache.clear()

    def _set_structured_parameter(self, node, key, value):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return
        self._set_parameter(node, key, parsed)

    def _package_target_path(self):
        if not getattr(sys, "frozen", False):
            return None
        app_data = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
        app_data_root = Path(app_data) if app_data else Path.home() / ".dora"
        return app_data_root / "node_editor" / "site-packages"

    def _missing_graph_packages(self):
        package_target = self._package_target_path()
        if package_target is not None and package_target.is_dir():
            target = str(package_target)
            if target not in sys.path:
                sys.path.insert(0, target)
            importlib.invalidate_caches()
        required = set()
        for node in self.engine.nodes.values():
            spec = self.engine.specs[node.spec_id]
            module_name = spec.check_package
            if spec.id == "forecast" and node.parameters.get("method", "naive") not in {
                "naive", "seasonal_naive",
            }:
                module_name = "statsmodels"
            if not module_name:
                continue
            try:
                installed = importlib.util.find_spec(module_name) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                installed = False
            if not installed:
                required.add(PACKAGE_DISTRIBUTIONS.get(module_name, module_name))
        return sorted(required)

    def run_graph(self):
        frame = getattr(self.main_window, "current_dataset", None)
        if frame is None:
            QMessageBox.information(self, "Dataset belum tersedia", "Muat dataset terlebih dahulu sebelum menjalankan graf.")
            return
        self.sync_dataset()
        packages = self._missing_graph_packages()
        package_target = self._package_target_path()
        self._clear_results()
        for node_id, node in self.engine.nodes.items():
            node.status = "menunggu"
            item = self.items.get(node_id)
            if item is not None:
                item.set_status("menunggu")
        if packages:
            self.output_log.clear()
            self.result_tabs.setTabVisible(2, True)
            self.result_tabs.setCurrentWidget(self.output_log)
        self._completed_nodes.clear()
        self._run_node_count = len(self.engine.nodes)
        self.run_progress_bar.setValue(0)
        self.run_progress_bar.show()
        self.run_spinner.show()
        if packages:
            self.run_status.setText(f"Memasang {len(packages)} library yang dibutuhkan...")
        else:
            self.run_status.setText(f"Menyiapkan graf ({self._run_node_count} node)...")
        self.run_status.show()
        self._spinner_index = 0
        self._advance_run_spinner()
        self._run_timer.start()
        self._set_running(True)
        self.thread = QThread(self)
        self.worker = _RunWorker(self.engine.to_dict(), frame, packages, package_target)
        thread, worker = self.thread, self.worker
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_status)
        self.worker.output.connect(self.append_output)
        self.worker.phase.connect(self.set_run_phase)
        self.worker.finished.connect(self.show_results)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    def _set_running(self, running):
        self.run_button.setEnabled(not running)
        self.delete_button.setEnabled(not running and any(isinstance(item, GraphNodeItem) for item in self.scene.selectedItems()))
        self.fit_button.setEnabled(not running)
        self.save_button.setEnabled(not running)
        self.load_button.setEnabled(not running)
        self.template_combo.setEnabled(not running)
        self.splitter.setEnabled(not running)

    @Slot()
    def _thread_finished(self):
        thread = self.thread
        if self.thread is thread:
            self.thread = None
            self.worker = None
        self._run_timer.stop()
        self.run_spinner.hide()
        self._set_running(False)
        if thread is not None:
            thread.deleteLater()
        self.refresh_palette()
        self.sync_dataset()

    @Slot(str, str)
    def update_status(self, node_id, status):
        node = self.engine.nodes.get(node_id)
        if node is not None:
            node.status = status
        item = self.items.get(node_id)
        if item is not None:
            item.set_status(status)
        if status == "selesai":
            self._completed_nodes.add(node_id)
            if self._run_node_count:
                self.run_progress_bar.setValue(
                    int(100 * len(self._completed_nodes) / self._run_node_count)
                )
            name = self.engine.specs[node.spec_id].name if node is not None else node_id
            self.run_status.setText(
                f"Selesai: {name} ({len(self._completed_nodes)}/{self._run_node_count})"
            )
        elif status == "berjalan":
            index = len(self._completed_nodes) + 1
            name = self.engine.specs[node.spec_id].name if node is not None else node_id
            self.run_status.setText(
                f"Menjalankan: {name} (node {index}/{self._run_node_count})"
            )
        elif status == "dilewati":
            name = self.engine.specs[node.spec_id].name if node is not None else node_id
            self.run_status.setText(f"Dilewati: {name} (graf dihentikan karena error)")

    def _advance_run_spinner(self):
        if not self._spinner_frames:
            return
        self.run_spinner.setText(self._spinner_frames[self._spinner_index])
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)

    @Slot(str)
    def set_run_phase(self, message):
        self.run_status.setText(message)

    @Slot(str)
    def append_output(self, message):
        if not message:
            return
        self.result_tabs.setTabVisible(2, True)
        cursor = self.output_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(message.replace("\r\n", "\n").replace("\r", "\n"))
        self.output_log.setTextCursor(cursor)
        self.output_log.ensureCursorVisible()

    def _finish_run_indicator(self, status):
        self._run_timer.stop()
        self.run_spinner.hide()
        if status == "success":
            self.run_progress_bar.setValue(100)
            self.run_status.setText("Graf selesai")
        else:
            self.run_status.setText("Graf gagal dijalankan")

    @Slot(object)
    def show_results(self, output):
        forecast_metrics = None
        for node_id, node in self.engine.nodes.items():
            if node_id in self.items:
                self.items[node_id].set_status(node.status)
        for node_id, result in output.items():
            if "table" in result and hasattr(result["table"], "columns"):
                self._display_frame(result["table"])
            if "metrics" in result:
                metrics = result["metrics"].get("metrics", {})
                node = self.engine.nodes.get(node_id)
                model_name = result["metrics"].get(
                    "model",
                    self.engine.specs[node.spec_id].name if node is not None else node_id,
                )
                self._display_frame(pd.DataFrame([{"model": model_name, **metrics}]))
                if node is not None and node.spec_id == "forecast":
                    forecast_metrics = metrics
        if forecast_metrics is not None:
            self.forecast_chart.set_metrics(forecast_metrics)
            self.result_tabs.setTabVisible(1, True)
            self.result_tabs.setCurrentWidget(self.forecast_chart)
        self._finish_run_indicator("success")

    def _clear_results(self):
        self.results.clear()
        self.results.setRowCount(0)
        self.results.setColumnCount(0)
        self.results.setToolTip("")
        self.forecast_chart.set_metrics({})
        self.result_tabs.setCurrentWidget(self.results)
        self.result_tabs.setTabVisible(1, False)

    def _display_frame(self, frame):
        preview = frame.iloc[:RESULT_PREVIEW_ROWS, :RESULT_PREVIEW_COLUMNS]
        self.results.setUpdatesEnabled(False)
        try:
            self.results.setColumnCount(len(preview.columns))
            self.results.setRowCount(len(preview))
            self.results.setHorizontalHeaderLabels([str(c) for c in preview.columns])
            for row, values in enumerate(preview.itertuples(index=False, name=None)):
                for column, value in enumerate(values):
                    self.results.setItem(row, column, QTableWidgetItem(str(value)))
        finally:
            self.results.setUpdatesEnabled(True)
        if len(frame) > RESULT_PREVIEW_ROWS or len(frame.columns) > RESULT_PREVIEW_COLUMNS:
            self.results.setToolTip(
                f"Pratinjau {len(preview):,} dari {len(frame):,} baris dan "
                f"{len(preview.columns):,} dari {len(frame.columns):,} kolom."
            )
        else:
            self.results.setToolTip("")

    @Slot(str)
    def show_error(self, message):
        for node_id, node in self.engine.nodes.items():
            if node_id in self.items:
                self.items[node_id].set_status(node.status)
        self.append_output(f"\nRun gagal: {message}\n")
        self.result_tabs.setCurrentWidget(self.output_log)
        self._finish_run_indicator("error")
        QMessageBox.critical(self, "Graf gagal dijalankan", message)

    def save_graph(self):
        path, _ = QFileDialog.getSaveFileName(self, "Simpan Graf", "graf-node.json", "JSON (*.json)")
        if path:
            self.engine.save(path)

    def load_graph(self):
        path, _ = QFileDialog.getOpenFileName(self, "Muat Graf", "", "JSON (*.json)")
        if path:
            try:
                self.engine = GraphEngine.load(path)
                self._clear_results()
                self.cancel_connection()
                self.scene.clear()
                self.items.clear()
                self.cables.clear()
                for node in self.engine.nodes.values():
                    item = GraphNodeItem(self, node, *node.position)
                    self.items[node.id] = item
                    self.scene.addItem(item)
                self.sync_dataset()
                self.redraw_cables()
                QTimer.singleShot(0, self.fit_graph)
            except Exception as exc:
                QMessageBox.critical(self, "Graf tidak dapat dimuat", str(exc))

    def load_template(self, index):
        names = ["", "Klasifikasi cepat", "Regresi cepat", "Ensemble klasifikasi", "Imbalanced klasifikasi", "Forecasting cepat"]
        if index <= 0:
            return
        name = names[index]
        graph = GraphEngine()
        active = getattr(self.main_window, "current_dataset", None)
        guessed_target = str(active.columns[-1]) if active is not None and len(active.columns) else ""
        if active is not None:
            numeric_columns = active.select_dtypes(include=["number"]).columns
            if len(numeric_columns):
                guessed_target = str(numeric_columns[-1])
        guessed_time = ""
        if active is not None:
            dates = [
                str(col)
                for col in active.columns
                if pd.api.types.is_datetime64_any_dtype(active[col])
                or any(
                    token in str(col).casefold()
                    for token in ("date", "time", "tanggal", "waktu", "timestamp")
                )
            ]
            guessed_time = dates[0] if dates else str(active.columns[0])
        load = graph.add_node("load_dataset", "dataset", position=(40, 170))
        if name == "Forecasting cepat":
            split = graph.add_node("time_split", "split", {"time_column": guessed_time, "test_size": .2}, (300, 170))
            forecast = graph.add_node("forecast", "forecast", {"target": guessed_target, "method": "naive"}, (600, 170))
            graph.connect(load.id, "table", split.id, "table")
            graph.connect(split.id, "train", forecast.id, "train")
            graph.connect(split.id, "test", forecast.id, "test")
        else:
            task = "regression" if name == "Regresi cepat" else "classification"
            target = graph.add_node("select_target", "target", {"target": guessed_target}, (250, 170))
            split = graph.add_node("split", "split", {"target": guessed_target, "task": task, "test_size": .2}, (480, 170))
            prep = graph.add_node("preprocess", "prep", {"target": guessed_target}, (700, 170))
            methods = ["model:random_forest_regressor" if task == "regression" else "model:logistic_regression"]
            if name == "Ensemble klasifikasi":
                methods = ["model:logistic_regression", "model:random_forest_classifier"]
            estimators = []
            x0 = 700
            for i, mid in enumerate(methods):
                m = graph.add_node(mid, f"model_{i+1}", position=(x0 + (i * 20), 390 + i * 130))
                estimators.append(m)
            if name == "Ensemble klasifikasi":
                model_id = "ensemble:voting"
                ensemble = graph.add_node(model_id, "ensemble", {"kind": "voting", "voting": "hard"}, (950, 400))
                for model in estimators:
                    graph.connect(model.id, "estimator", ensemble.id, "estimators")
                model_ref = ensemble
            else:
                model_ref = estimators[0]
            train = graph.add_node("train", "train", {"target": guessed_target, "task": task}, (1120, 170))
            if name == "Imbalanced klasifikasi":
                smote = graph.add_node("smote", "smote", {"target": guessed_target}, (950, 180))
            evaluate = graph.add_node("evaluate", "evaluate", {"target": guessed_target, "task": task}, (1370, 170))
            compare = graph.add_node("compare", "compare", position=(1620, 170))
            graph.connect(load.id, "table", target.id, "table")
            graph.connect(target.id, "table", split.id, "table")
            graph.connect(split.id, "train", prep.id, "train")
            graph.connect(split.id, "test", prep.id, "test")
            if name == "Imbalanced klasifikasi":
                graph.connect(prep.id, "train", smote.id, "train")
                graph.connect(smote.id, "train", train.id, "train")
            else:
                graph.connect(prep.id, "train", train.id, "train")
            graph.connect(model_ref.id, "estimator", train.id, "estimator")
            graph.connect(train.id, "model", evaluate.id, "model")
            graph.connect(prep.id, "test", evaluate.id, "test")
            graph.connect(evaluate.id, "metrics", compare.id, "metrics")
        self.engine = graph
        self._clear_results()
        self.cancel_connection()
        self.scene.clear()
        self.items.clear()
        self.cables.clear()
        for node in graph.nodes.values():
            item = GraphNodeItem(self, node, *node.position)
            self.items[node.id] = item
            self.scene.addItem(item)
        self.redraw_cables()
        self.template_combo.setCurrentIndex(0)
        self.refresh_palette()
        self.sync_dataset()
        QTimer.singleShot(0, self.fit_graph)
