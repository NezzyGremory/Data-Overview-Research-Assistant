from __future__ import annotations

import ast
import codecs
import hashlib
import importlib.util
import json
import math
import os
import pickle
import struct
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from PySide6.QtCore import QMimeData, QObject, QPointF, QRectF, QStandardPaths, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QDrag, QPainter, QPainterPath, QPen, QPolygonF, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGraphicsEllipseItem,
    QGraphicsItem, QGraphicsPathItem, QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsScene, QGraphicsTextItem,
    QGraphicsView, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QSpinBox, QCheckBox, QScrollArea, QProgressBar,
    QPlainTextEdit, QInputDialog,
    QListWidget, QListWidgetItem, QSizePolicy, QTabWidget,
)

from app.ml.graph import GraphEngine, GraphError
from app.ml.graph.exporter import (
    export_jupyter_notebook, export_model_pkl, export_python_script, export_runs_csv,
)
from app.ml.graph.project import GraphHistoryStack, compute_dataframe_hash


PORT_COLORS = {"table": "#0071C2", "estimator": "#8B5CF6", "model": "#098E00", "metrics": "#E09A00"}
RESULT_PREVIEW_ROWS = 200
RESULT_PREVIEW_COLUMNS = 50
PACKAGE_DISTRIBUTIONS = {"imblearn": "imbalanced-learn", "xgboost": "xgboost", "lightgbm": "lightgbm", "optuna": "optuna"}
TASKS = [("Semua", "all"), ("Klasifikasi", "classification"), ("Regresi", "regression"), ("Pengelompokan", "clustering"), ("Anomali", "anomaly"), ("Rangkaian Waktu", "time_series")]
CATEGORY_TRANSLATIONS = {"Data": "Data", "Classification": "Klasifikasi", "Regression": "Regresi", "Clustering": "Clustering", "Anomaly Detection": "Deteksi Anomali", "Validation": "Validasi", "Preprocessing": "Pra-pemrosesan", "Model": "Model", "Ensemble": "Ensemble", "Evaluation": "Evaluasi", "Tuning": "Penyetelan", "Forecasting": "Peramalan", "Data Tidak Seimbang": "Data Tidak Seimbang", "Transformer": "Transformer"}
WORKFLOW_FILTERS = [
    ("Semua tahapan", "all"),
    ("Data dan pemahaman", "data"),
    ("Pra-pemrosesan", "preprocessing"),
    ("Fitur dan transformer", "features"),
    ("Model", "models"),
    ("Validasi dan tuning", "validation"),
    ("Evaluasi", "evaluation"),
]
WORKFLOW_CATEGORIES = {
    "data": {"Data"},
    "preprocessing": {"Preprocessing", "Data Tidak Seimbang"},
    "features": {"Transformer"},
    "models": {"Model", "Classification", "Regression", "Clustering", "Anomaly Detection", "Ensemble", "Forecasting"},
    "validation": {"Validation", "Validasi", "Tuning"},
    "evaluation": {"Evaluation", "Evaluasi"},
}
METRIC_LABELS = {
    "accuracy": "Akurasi",
    "f1": "F1",
    "f1_macro": "F1 Macro",
    "f1_weighted": "F1 Weighted",
    "balanced_accuracy": "Balanced Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "roc_auc": "ROC AUC",
    "mse": "MSE",
    "median_ae": "Median AE",
    "explained_variance": "Explained Variance",
    "best_score": "Best CV Score",
    "best_params": "Best Parameters",
    "scoring": "Scoring",
    "search_method": "Metode Tuning",
    "candidate_count": "Jumlah Kandidat",
    "trials_completed": "Percobaan Selesai",
    "mean_score": "Rata-rata CV Score",
    "std_score": "Simpangan Baku CV",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "silhouette": "Silhouette",
    "calinski_harabasz": "Calinski-Harabasz",
    "davies_bouldin": "Davies-Bouldin",
    "cluster_count": "Jumlah Cluster",
    "noise_count": "Jumlah Noise",
}
PERCENT_METRICS = {"accuracy", "f1", "f1_macro", "f1_weighted", "balanced_accuracy", "precision", "recall", "r2", "roc_auc", "explained_variance"}
PERCENT_SCORING = {
    "accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_weighted",
    "precision_macro", "recall_macro", "roc_auc_ovr_weighted", "r2", "explained_variance",
}
NON_PERFORMANCE_METRICS = {"random_state", "is_imbalanced", "imbalance_ratio", "confusion_matrix"}
PARAMETER_CHOICES = {
    "task": [("Deteksi otomatis", "auto"), ("Klasifikasi", "classification"), ("Regresi", "regression"), ("Clustering", "clustering"), ("Anomali", "anomaly_detection")],
    "scaler": [("Standard", "standard"), ("Min-Max", "minmax"), ("Robust", "robust")],
    "imputer": [("Median", "median"), ("Rata-rata", "mean"), ("Paling sering", "most_frequent"), ("Konstan", "constant")],
    "encoder": [("One-hot", "onehot")],
    "kind": [("Voting", "voting"), ("Stacking", "stacking"), ("Bagging", "bagging")],
    "voting": [("Hard", "hard"), ("Soft", "soft")],
    "sampler": [("TPE Bayesian", "tpe"), ("Random", "random")],
    "resource": [("Jumlah sampel", "n_samples")],
    "min_resources": [("Otomatis: terkecil", "smallest"), ("Otomatis: efisien", "exhaust")],
    "max_resources": [("Otomatis", "auto")],
    "method": [("Naive", "naive"), ("Seasonal Naive", "seasonal_naive"), ("ARIMA", "arima"), ("Exponential Smoothing", "exponential_smoothing"), ("SARIMAX", "sarimax")],
    "scoring": [("Accuracy", "accuracy"), ("F1 Macro", "f1_macro"), ("F1 Weighted", "f1_weighted"), ("R²", "r2"), ("MAE", "neg_mean_absolute_error")],
}
PARAMETER_HELP = {
    "target": "Kolom yang ingin diprediksi. Dipilih dari kolom dataset aktif.",
    "task": "Jenis masalah yang menentukan cara model dilatih dan dinilai.",
    "time_column": "Kolom waktu untuk mengurutkan data sebelum pemisahan kronologis.",
    "test_size": "Proporsi data yang disimpan untuk pengujian; harus lebih dari 0 dan kurang dari 1.",
    "random_state": "Seed agar pemisahan dan hasil model acak dapat diulang.",
    "cv": "Jumlah fold untuk validasi silang; minimal 2.",
    "scoring": "Metrik yang dioptimalkan. Auto memilih F1 Macro untuk klasifikasi tidak seimbang, Accuracy untuk klasifikasi lain, dan R² untuk regresi.",
    "param_grid": "Ruang parameter dalam JSON. Contoh Grid: {\"C\": [0.1, 1, 10], \"max_iter\": [500, 1000]}. Random juga menerima rentang {\"n_estimators\": {\"type\": \"int\", \"low\": 50, \"high\": 400}}.",
    "search_space": "Ruang Optuna dalam JSON. Contoh: {\"C\": {\"type\": \"float\", \"low\": 0.001, \"high\": 100, \"log\": true}, \"max_depth\": {\"type\": \"int\", \"low\": 2, \"high\": 30}}. Kosongkan untuk memakai ruang otomatis dari estimator.",
    "n_iter": "Jumlah kombinasi acak atau jumlah kandidat awal pada Successive Halving Random.",
    "trials": "Jumlah percobaan Optuna. Setiap percobaan menilai satu kombinasi dengan CV.",
    "timeout": "Batas waktu optimasi Optuna dalam detik. Nilai 0 berarti tanpa batas waktu.",
    "sampler": "TPE memakai riwayat hasil untuk memilih kombinasi berikutnya; Random mengambil sampel acak.",
    "n_jobs": "Jumlah pekerjaan paralel untuk pencarian. Nilai -1 memakai seluruh core CPU.",
    "factor": "Rasio pengurangan kandidat pada Successive Halving. Nilai lebih besar memangkas kandidat lebih cepat.",
    "min_resources": "Jumlah sumber daya minimum per kandidat. Pilih otomatis atau isi angka.",
    "max_resources": "Jumlah sumber daya maksimum. Otomatis memakai seluruh data yang tersedia.",
    "resource": "Sumber daya yang ditambah pada tiap putaran Successive Halving.",
    "aggressive_elimination": "Jalankan putaran eliminasi tambahan jika jumlah kandidat tidak habis terbagi oleh factor.",
    "scaler": "Cara menyetarakan skala fitur numerik.",
    "imputer": "Cara mengisi nilai kosong pada fitur numerik.",
    "encoder": "Cara mengubah fitur kategori menjadi representasi numerik.",
    "method": "Metode yang digunakan oleh node ini.",
}
NODE_HELP = {
    "load_dataset": "Mengirim dataset aktif ke alur graf.",
    "select_target": "Memastikan kolom target yang dipilih tersedia pada tabel.",
    "split": "Membagi dataset menjadi data latih dan data uji. Letakkan transformasi setelah node ini agar data uji tidak ikut melatih transformasi.",
    "time_split": "Membagi data berdasarkan urutan waktu, bukan pengacakan.",
    "preprocess": "Mengisi nilai kosong, menyetarakan fitur numerik, dan mengodekan fitur kategori dengan parameter node.",
    "train": "Melatih estimator pada data latih. Untuk klasifikasi/regresi, target sebaiknya dipilih secara eksplisit.",
    "evaluate": "Mengukur performa model pada data uji yang tidak dipakai saat training.",
    "compare": "Menggabungkan metrik beberapa model untuk perbandingan.",
    "cross_validate": "Mengukur kestabilan estimator memakai beberapa fold.",
    "grid_search": "Menguji kombinasi parameter yang ditentukan secara menyeluruh.",
    "random_search": "Menguji sampel kombinasi parameter secara acak.",
    "halving_grid_search": "Successive Halving Grid mengevaluasi kombinasi bertahap dan mengalokasikan lebih banyak data untuk kandidat terbaik.",
    "halving_random_search": "Successive Halving Random menyaring sampel acak bertahap agar pencarian luas lebih hemat waktu.",
    "optuna": "Optimasi Bayesian TPE atau Random dengan percobaan yang tercatat. Optuna memasang paket opsional saat graf dijalankan.",
    "forecast": "Membuat dan mengukur prediksi deret waktu.",
}


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
    SOURCE_HEIGHT = 150

    def __init__(self, page, node, x, y):
        self.is_dataset_source = node.spec_id == "load_dataset"
        self.node_height = self.SOURCE_HEIGHT if self.is_dataset_source else self.H
        super().__init__(0, 0, self.W, self.node_height)
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
        self.status.setPos(12, self.node_height - 25)
        self.status.setTextWidth(self.W - 24)
        self.status.setAcceptedMouseButtons(Qt.NoButton)
        if self.is_dataset_source:
            self.dataset_action_rect = QRectF(12, 84, self.W - 24, 26)
            self._task_label = getattr(page, "_task_label", "Belum terdeteksi")
            self._has_dataset = getattr(page, "_has_dataset", False)
            self.setToolTip("Node sumber memakai dataset aktif. Klik tombol di dalam node untuk menambah atau mengganti dataset.")

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
        if self.is_dataset_source:
            task_rect = QRectF(12, 58, self.W - 24, 19)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#EAF4FF"))
            painter.drawRoundedRect(task_rect, 5, 5)
            painter.setPen(QColor("#245B8F"))
            task_font = painter.font()
            task_font.setPointSize(8)
            task_font.setBold(True)
            painter.setFont(task_font)
            painter.drawText(task_rect.adjusted(7, 0, -5, 0), Qt.AlignLeft | Qt.AlignVCenter, f"Task: {self._task_label}")

            painter.setPen(QPen(QColor("#AFCFF2"), 1))
            painter.setBrush(QColor("#F7FBFF"))
            painter.drawRoundedRect(self.dataset_action_rect, 6, 6)
            painter.setPen(QColor("#0878D1"))
            action_font = painter.font()
            action_font.setPointSize(8)
            action_font.setBold(True)
            painter.setFont(action_font)
            if self.page._dataset_scope == "graph":
                action = "Ganti Dataset Graf" if self._has_dataset else "Pilih Dataset Graf"
            else:
                action = "Ganti Dataset" if self._has_dataset else "Tambah Dataset"
            painter.drawText(self.dataset_action_rect, Qt.AlignCenter, action)
        painter.restore()

    def mousePressEvent(self, event):
        if (
            self.is_dataset_source
            and event.button() == Qt.LeftButton
            and self.dataset_action_rect.contains(event.pos())
        ):
            self.setSelected(True)
            event.accept()
            self.page.choose_dataset_for_node(self.node_id)
            return
        super().mousePressEvent(event)

    def set_dataset_context(self, task_label, has_dataset):
        if not self.is_dataset_source:
            return
        self._task_label = task_label
        self._has_dataset = has_dataset
        self.update()

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
        self._positions_before_drag = None
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
            if not self._space_held:
                self._positions_before_drag = {
                    node_id: tuple(node.position)
                    for node_id, node in self.page.engine.nodes.items()
                }
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
        if self._positions_before_drag is not None:
            changed = any(
                self.page.engine.nodes[node_id].position != position
                for node_id, position in self._positions_before_drag.items()
                if node_id in self.page.engine.nodes
            )
            self._positions_before_drag = None
            if changed:
                self.page._record_graph_history()
                self.page.validate_graph(show_tab=False)

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Z and event.modifiers() & Qt.ShiftModifier:
                self.page.redo_graph()
                event.accept()
                return
            if event.key() == Qt.Key_Z:
                self.page.undo_graph()
                event.accept()
                return
            if event.key() == Qt.Key_Y:
                self.page.redo_graph()
                event.accept()
                return
            if event.key() == Qt.Key_D:
                self.page.duplicate_selected_nodes()
                event.accept()
                return
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
    cancelled = Signal()
    progress = Signal(str, str)
    output = Signal(str)
    phase = Signal(str)

    def __init__(self, graph_data, dataframe, packages=(), package_target=None, export_model=None):
        super().__init__()
        self.graph_data, self.dataframe = graph_data, dataframe
        self.packages = tuple(packages)
        self.package_target = Path(package_target) if package_target else None
        self.export_model = export_model
        self._process = None
        self._cancel_requested = False
        self.cooperative_cancel = bool(getattr(sys, "frozen", False))
        self.can_cancel = True

    def cancel(self):
        self._cancel_requested = True
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    @Slot()
    def run(self):
        try:
            if self._cancel_requested:
                self.cancelled.emit()
                return
            if self.packages:
                self._install_packages()
            if getattr(sys, "frozen", False):
                self._run_in_thread()
            else:
                self._run_in_subprocess()
        except Exception as exc:
            if self._cancel_requested:
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))
        else:
            if self._cancel_requested:
                self.cancelled.emit()

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
        self._process = process
        if self._cancel_requested and process.poll() is None:
            process.terminate()
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
            self._process = None
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
            self._process = process
            if self._cancel_requested and process.poll() is None:
                process.terminate()
            pickle.dump(
                {
                    "graph": self.graph_data,
                    "dataframe": self.dataframe,
                    "export_model": self.export_model,
                },
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
            self._process = None
            if self._cancel_requested:
                return
            if not terminal_record_received:
                raise RuntimeError(
                    f"Proses graf berhenti tanpa hasil (kode {return_code})."
                )
        except Exception as exc:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            self._process = None
            if not self._cancel_requested:
                self.failed.emit(str(exc))

    def _run_in_thread(self):
        try:
            graph = _restore_graph(self.graph_data)
            def report_progress(node_id, status):
                self.progress.emit(node_id, status)
                if self._cancel_requested:
                    raise RuntimeError("Run dihentikan setelah node aktif selesai.")

            output = graph.run(self.dataframe.copy(), report_progress)
            export_result = None
            if self.export_model:
                node_output = output.get(self.export_model.get("node_id"), {})
                model = node_output.get("model", {}).get("estimator")
                if model is None:
                    raise ValueError("Model dari node Train tidak ditemukan untuk diekspor.")
                export_result = export_model_pkl(model, self.export_model["path"])
            display_output = {
                node_id: {
                    key: value for key, value in node_output.items()
                    if key in {"table", "metrics"}
                }
                for node_id, node_output in output.items()
                if isinstance(node_output, dict)
                and any(key in node_output for key in ("table", "metrics"))
            }
            for node_id, node_output in output.items():
                if not isinstance(node_output, dict):
                    continue
                model = node_output.get("model")
                if isinstance(model, dict) and model.get("estimator") is not None:
                    estimator = model["estimator"]
                    display_output.setdefault(node_id, {})["model_summary"] = {
                        "Estimator": type(estimator).__name__,
                        "Task": model.get("task", ""),
                        "Target": model.get("target", ""),
                    }
                estimator = node_output.get("estimator")
                if estimator is not None:
                    display_output.setdefault(node_id, {})["estimator_summary"] = {
                        "Estimator": type(estimator).__name__,
                    }
            if export_result is not None:
                display_output["__model_export__"] = {"path": export_result}
            self.finished.emit(display_output)
        except Exception:
            if not self._cancel_requested:
                raise


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
        self._suppressed_source_dataset_id = -1
        self._task_label = "Belum terdeteksi"
        self._has_dataset = False
        self._dataset_scope = "graph"
        self._dataset_scope_explicit = False
        self._graph_dataset = None
        self._graph_dataset_path = ""
        self._graph_dataset_error = ""
        self.thread = None
        self.worker = None
        self._is_running = False
        self._session_graph_restored = False
        self._session_graph_touched = False
        self._session_graph_timer = QTimer(self)
        self._session_graph_timer.setSingleShot(True)
        self._session_graph_timer.setInterval(700)
        self._session_graph_timer.timeout.connect(self.save_session_graph)
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
        content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        scroll.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(22, 18, 22, 16)
        title = QLabel("Node Editor")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Pilih dataset workspace atau dataset khusus graf. Hubungkan port dengan garis, lalu geser kanvas memakai tombol tengah atau tahan Space sambil menyeret.")
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)
        toolbar = QHBoxLayout()
        self.run_button = QPushButton("Jalankan")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self.run_graph)
        toolbar.addWidget(self.run_button)
        self.run_to_button = QPushButton("Jalankan s.d. Node")
        self.run_to_button.setObjectName("secondaryButton")
        self.run_to_button.setToolTip("Jalankan hanya node terpilih beserta seluruh input di hulunya.")
        self.run_to_button.setEnabled(False)
        self.run_to_button.clicked.connect(self.run_to_selected_node)
        toolbar.addWidget(self.run_to_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("secondaryButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.cancel_run)
        toolbar.addWidget(self.stop_button)
        self.validate_button = QPushButton("Validasi")
        self.validate_button.setObjectName("secondaryButton")
        self.validate_button.clicked.connect(lambda: self.validate_graph(show_tab=True))
        toolbar.addWidget(self.validate_button)
        self.validation_status = QLabel("Belum divalidasi")
        self.validation_status.setObjectName("nodeEditorRunStatus")
        toolbar.addWidget(self.validation_status)
        self.delete_button = QPushButton("Hapus Node")
        self.delete_button.setObjectName("secondaryButton")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self.delete_selected_nodes)
        toolbar.addWidget(self.delete_button)
        self.duplicate_button = QPushButton("Duplikat")
        self.duplicate_button.setObjectName("secondaryButton")
        self.duplicate_button.setEnabled(False)
        self.duplicate_button.clicked.connect(self.duplicate_selected_nodes)
        toolbar.addWidget(self.duplicate_button)
        self.fit_button = QPushButton("Sesuaikan Tampilan")
        self.fit_button.setObjectName("secondaryButton")
        self.fit_button.clicked.connect(self.fit_graph)
        toolbar.addWidget(self.fit_button)
        self.dataset_status = QLabel("Dataset belum dimuat")
        self.dataset_status.setObjectName("nodeEditorDatasetStatus")
        self.dataset_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.dataset_status.setMinimumWidth(0)
        self.dataset_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        root.addLayout(toolbar)

        dataset_context = QHBoxLayout()
        dataset_context.setContentsMargins(0, 0, 0, 0)
        dataset_context.setSpacing(10)
        self.dataset_scope_combo = QComboBox()
        self.dataset_scope_combo.setObjectName("nodeEditorDatasetScope")
        self.dataset_scope_combo.addItem("Dataset khusus graf", "graph")
        self.dataset_scope_combo.addItem("Ikuti Workspace", "workspace")
        self.dataset_scope_combo.setToolTip(
            "Ikuti Workspace memakai dataset aktif dari halaman Workspace. Dataset khusus graf disimpan bersama file graf dan tidak mengubah dataset Workspace."
        )
        self.dataset_scope_combo.currentIndexChanged.connect(self._on_dataset_scope_changed)
        dataset_context.addWidget(self.dataset_scope_combo, 0)
        dataset_context.addWidget(self.dataset_status, 1)
        self.dataset_task_status = QLabel("Task belum terdeteksi")
        self.dataset_task_status.setObjectName("nodeEditorTaskStatus")
        self.dataset_task_status.setAlignment(Qt.AlignCenter)
        dataset_context.addWidget(self.dataset_task_status, 0)
        root.addLayout(dataset_context)

        file_toolbar = QHBoxLayout()
        self.save_button = QPushButton("Simpan Graf")
        self.save_button.setObjectName("secondaryButton")
        self.save_button.clicked.connect(self.save_graph)
        file_toolbar.addWidget(self.save_button)
        self.load_button = QPushButton("Muat Graf")
        self.load_button.setObjectName("secondaryButton")
        self.load_button.clicked.connect(self.load_graph)
        file_toolbar.addWidget(self.load_button)
        self.undo_button = QPushButton("Undo")
        self.undo_button.setObjectName("secondaryButton")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undo_graph)
        file_toolbar.addWidget(self.undo_button)
        self.redo_button = QPushButton("Redo")
        self.redo_button.setObjectName("secondaryButton")
        self.redo_button.setEnabled(False)
        self.redo_button.clicked.connect(self.redo_graph)
        file_toolbar.addWidget(self.redo_button)
        self.template_combo = QComboBox()
        self.template_combo.addItems(["Pilih template alur kerja…", "Klasifikasi cepat", "Regresi cepat", "Ensemble klasifikasi", "Imbalanced klasifikasi", "Forecasting cepat"])
        self.template_combo.activated.connect(self.load_template)
        self.template_combo.setObjectName("nodeEditorTemplateCombo")
        self.template_combo.setMinimumWidth(220)
        self.template_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.template_combo.setMinimumContentsLength(25)
        self.template_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.template_combo.setToolTip("Pilih alur siap pakai untuk task yang terdeteksi. Setiap template dapat disunting setelah dimuat.")
        file_toolbar.addWidget(self.template_combo)
        self.export_python_button = QPushButton("Ekspor Python")
        self.export_python_button.setObjectName("secondaryButton")
        self.export_python_button.clicked.connect(self.export_python)
        file_toolbar.addWidget(self.export_python_button)
        self.export_notebook_button = QPushButton("Notebook")
        self.export_notebook_button.setObjectName("secondaryButton")
        self.export_notebook_button.clicked.connect(self.export_notebook)
        file_toolbar.addWidget(self.export_notebook_button)
        self.export_model_button = QPushButton("Model .pkl")
        self.export_model_button.setObjectName("secondaryButton")
        self.export_model_button.clicked.connect(self.export_model)
        file_toolbar.addWidget(self.export_model_button)
        file_toolbar.addStretch(1)
        self.graph_state_label = QLabel("Graf belum disimpan")
        self.graph_state_label.setObjectName("nodeEditorRunStatus")
        file_toolbar.addWidget(self.graph_state_label)
        root.addLayout(file_toolbar)

        snippet_toolbar = QHBoxLayout()
        snippet_label = QLabel("Potongan alur")
        snippet_label.setObjectName("nodeEditorRunStatus")
        snippet_toolbar.addWidget(snippet_label)
        self.save_snippet_button = QPushButton("Simpan pilihan sebagai potongan")
        self.save_snippet_button.setObjectName("secondaryButton")
        self.save_snippet_button.clicked.connect(self.save_selected_snippet)
        snippet_toolbar.addWidget(self.save_snippet_button)
        self.insert_snippet_button = QPushButton("Sisipkan potongan")
        self.insert_snippet_button.setObjectName("secondaryButton")
        self.insert_snippet_button.clicked.connect(self.insert_graph_snippet)
        snippet_toolbar.addWidget(self.insert_snippet_button)
        snippet_toolbar.addStretch(1)
        root.addLayout(snippet_toolbar)

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
        self._active_run_node_ids = set()
        self._run_started_at = None
        self._run_history = []
        self._result_outputs = {}
        self._history = GraphHistoryStack()
        self._history_state = ""
        self._history_suspended = False
        self._history_timer = QTimer(self)
        self._history_timer.setSingleShot(True)
        self._history_timer.setInterval(350)
        self._history_timer.timeout.connect(self._record_graph_history)
        self._pending_export_model = None

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setObjectName("nodeEditorSplitter")
        self.splitter.setChildrenCollapsible(True)
        self.splitter.addWidget(self._build_palette())
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(-1000, -600, 3000, 2200)
        self.view = GraphView(self.scene, self)
        self.view.setFrameShape(QFrame.NoFrame)
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
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
        self.workspace_splitter = QSplitter(Qt.Vertical)
        self.workspace_splitter.setObjectName("nodeEditorWorkspaceSplitter")
        self.workspace_splitter.addWidget(self.splitter)
        self.results_panel = QFrame()
        self.results_panel.setObjectName("contentCard")
        self.results_panel.setMinimumHeight(280)
        results_layout = QVBoxLayout(self.results_panel)
        results_layout.setContentsMargins(8, 6, 8, 8)
        self.results = QTableWidget(0, 0)
        self.results.setObjectName("nodeEditorTable")
        self.results.setAlternatingRowColors(True)
        self.results.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview_node_combo = QComboBox()
        self.preview_node_combo.setObjectName("nodeEditorPreviewCombo")
        self.preview_node_combo.addItem("Pilih output node…", None)
        self.preview_node_combo.currentIndexChanged.connect(self._show_selected_result)
        preview_page = QWidget()
        preview_layout = QVBoxLayout(preview_page)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self.preview_node_combo)
        preview_layout.addWidget(self.results, 1)
        self.result_tabs = QTabWidget()
        self.result_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.result_tabs.addTab(preview_page, "Pratinjau Node")
        metrics_page = QWidget()
        metrics_layout = QVBoxLayout(metrics_page)
        metrics_layout.setContentsMargins(4, 4, 4, 4)
        metrics_header = QHBoxLayout()
        metrics_title = QLabel("Ringkasan metrik")
        metrics_title.setObjectName("nodeEditorSectionTitle")
        metrics_header.addWidget(metrics_title)
        metrics_header.addStretch(1)
        self.metric_summary_model = QComboBox()
        self.metric_summary_model.setObjectName("nodeEditorMetricModel")
        self.metric_summary_model.setMinimumWidth(180)
        self.metric_summary_model.currentIndexChanged.connect(self._render_metric_summary)
        self.metric_summary_model_label = QLabel("Model")
        metrics_header.addWidget(self.metric_summary_model_label)
        metrics_header.addWidget(self.metric_summary_model)
        metrics_layout.addLayout(metrics_header)
        self.metric_summary_host = QWidget()
        self.metric_summary_layout = QHBoxLayout(self.metric_summary_host)
        self.metric_summary_layout.setContentsMargins(0, 0, 0, 0)
        self.metric_summary_layout.setSpacing(8)
        metrics_layout.addWidget(self.metric_summary_host)
        self._metric_rows = []
        self.metrics_results = QTableWidget(0, 0)
        self.metrics_results.setObjectName("nodeEditorTable")
        self.metrics_results.setAlternatingRowColors(True)
        self.metrics_results.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_results.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.metrics_results.horizontalHeader().setStretchLastSection(True)
        metrics_layout.addWidget(self.metrics_results, 1)
        self.result_tabs.addTab(metrics_page, "Metrik")
        self.confusion_matrix_table = QTableWidget(0, 0)
        self.confusion_matrix_table.setObjectName("nodeEditorTable")
        self.confusion_matrix_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_tabs.addTab(self.confusion_matrix_table, "Confusion Matrix")
        self.forecast_chart = ForecastMetricsChart()
        self.result_tabs.addTab(self.forecast_chart, "Grafik Forecast")
        self.result_tabs.setTabVisible(3, False)
        self.output_log = QPlainTextEdit()
        self.output_log.setObjectName("nodeEditorOutput")
        self.output_log.setReadOnly(True)
        self.output_log.document().setMaximumBlockCount(2000)
        self.result_tabs.addTab(self.output_log, "Output")
        self.result_tabs.setTabVisible(4, False)
        self.problems_list = QListWidget()
        self.problems_list.setObjectName("nodeEditorProblems")
        self.problems_list.itemDoubleClicked.connect(self._focus_problem_node)
        self.result_tabs.addTab(self.problems_list, "Masalah")
        self.result_tabs.setTabVisible(5, False)
        history_page = QWidget()
        history_layout = QVBoxLayout(history_page)
        history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_table = QTableWidget(0, 5)
        self.history_table.setObjectName("nodeEditorTable")
        self.history_table.setHorizontalHeaderLabels(["Run", "Waktu", "Task", "Target", "Dataset"])
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        history_layout.addWidget(self.history_table, 2)
        history_actions = QHBoxLayout()
        compare_button = QPushButton("Bandingkan dua run terpilih")
        compare_button.setObjectName("secondaryButton")
        compare_button.clicked.connect(self.compare_selected_runs)
        history_actions.addWidget(compare_button)
        export_runs_button = QPushButton("Ekspor riwayat CSV")
        export_runs_button.setObjectName("secondaryButton")
        export_runs_button.clicked.connect(self.export_run_history)
        history_actions.addWidget(export_runs_button)
        history_actions.addStretch(1)
        history_layout.addLayout(history_actions)
        self.history_comparison = QTableWidget(0, 4)
        self.history_comparison.setObjectName("nodeEditorTable")
        self.history_comparison.setHorizontalHeaderLabels(["Metrik", "Run A", "Run B", "Delta B − A"])
        self.history_comparison.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        history_layout.addWidget(self.history_comparison, 1)
        self.result_tabs.addTab(history_page, "Riwayat")

        profile_page = QWidget()
        profile_layout = QVBoxLayout(profile_page)
        profile_layout.setContentsMargins(8, 8, 8, 8)
        profile_intro = QLabel("Ringkasan struktur dan kualitas dataset aktif. Gunakan hasil ini untuk memahami kolom sebelum menyusun preprocessing dan model.")
        profile_intro.setObjectName("pageSubtitle")
        profile_intro.setWordWrap(True)
        profile_layout.addWidget(profile_intro)
        self.profile_summary_layout = QHBoxLayout()
        self.profile_summary_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.addLayout(self.profile_summary_layout)
        self.profile_table = QTableWidget(0, 4)
        self.profile_table.setObjectName("nodeEditorTable")
        self.profile_table.setHorizontalHeaderLabels(["Kolom", "Tipe", "Nilai unik", "Kosong"])
        self.profile_table.setAlternatingRowColors(True)
        self.profile_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.profile_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.profile_table.horizontalHeader().setStretchLastSection(True)
        profile_layout.addWidget(self.profile_table, 1)
        self.profile_tab_index = self.result_tabs.addTab(profile_page, "Profil Data")
        self.result_tabs.setTabVisible(self.profile_tab_index, False)
        results_layout.addWidget(self.result_tabs, 1)
        self.workspace_splitter.addWidget(self.results_panel)
        self.workspace_splitter.setStretchFactor(0, 2)
        self.workspace_splitter.setStretchFactor(1, 3)
        self.workspace_splitter.setSizes([430, 340])
        self.workspace_splitter.handle(1).setToolTip("Seret untuk menambah ruang kanvas atau area output.")
        root.addWidget(self.workspace_splitter, 1)
        self._history.push(self.engine.to_dict())
        self._history_state = self._graph_state_key()
        self._saved_graph_state = self._graph_document_key()
        self._update_history_buttons()

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
        self.workflow_filter = QComboBox()
        self.workflow_filter.setObjectName("nodeEditorWorkflowFilter")
        for label, workflow in WORKFLOW_FILTERS:
            self.workflow_filter.addItem(label, workflow)
        self.workflow_filter.currentIndexChanged.connect(self.refresh_palette)
        self.workflow_filter.setToolTip("Filter tahap alur kerja: pemahaman data, preprocessing, model, validasi, atau evaluasi.")
        layout.addWidget(self.workflow_filter)
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
        self.parameter_tabs = QTabWidget()
        parameter_page = QWidget()
        parameter_layout = QVBoxLayout(parameter_page)
        parameter_layout.setContentsMargins(6, 6, 6, 6)
        parameter_layout.addWidget(QLabel("Parameter"))
        self.parameter_form = QFormLayout()
        self.parameter_box = QWidget()
        self.parameter_box.setLayout(self.parameter_form)
        self.parameter_scroll = QScrollArea(parameter_page)
        self.parameter_scroll.setWidgetResizable(True)
        self.parameter_scroll.setFrameShape(QFrame.NoFrame)
        self.parameter_scroll.setWidget(self.parameter_box)
        parameter_layout.addWidget(self.parameter_scroll, 1)
        self.parameter_tabs.addTab(parameter_page, "Parameter")
        help_page = QWidget()
        help_layout = QVBoxLayout(help_page)
        help_layout.setContentsMargins(8, 8, 8, 8)
        self.node_help_label = QLabel("Pilih node untuk melihat fungsi, port, dan saran alurnya.")
        self.node_help_label.setWordWrap(True)
        self.node_help_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.node_help_label.setObjectName("nodeEditorHelp")
        help_layout.addWidget(self.node_help_label)
        help_layout.addStretch(1)
        self.parameter_tabs.addTab(help_page, "Bantuan")
        layout.addWidget(self.parameter_tabs, 1)
        self.scene.selectionChanged.connect(self.show_parameters)
        self.scene.selectionChanged.connect(self._update_delete_button)
        return panel

    def refresh_palette(self, *_):
        self.node_palette.clear()
        task = self.task_filter.currentData() if hasattr(self, "task_filter") else "all"
        workflow = self.workflow_filter.currentData() if hasattr(self, "workflow_filter") else "all"
        query = self.search.text().casefold() if hasattr(self, "search") else ""
        for spec_id, spec in self.engine.specs.items():
            if workflow != "all" and spec.category not in WORKFLOW_CATEGORIES.get(workflow, set()):
                continue
            if task != "all":
                if task == "classification":
                    is_match = (
                        spec.category in ("Classification", "Data Tidak Seimbang", "Ensemble")
                        or any(t.endswith("classification") for t in spec.task_types)
                        or spec_id in ("load_dataset", "select_target", "split", "preprocess", "train", "evaluate", "compare", "cross_validate")
                    )
                elif task == "regression":
                    is_match = (
                        spec.category in ("Regression", "Ensemble")
                        or "regression" in spec.task_types
                        or spec_id in ("load_dataset", "select_target", "split", "preprocess", "train", "evaluate", "compare", "cross_validate")
                    )
                elif task == "clustering":
                    is_match = (
                        spec.category == "Clustering"
                        or "clustering" in spec.task_types
                        or spec_id in ("load_dataset", "preprocess", "train", "evaluate", "compare")
                    )
                elif task == "time_series":
                    is_match = (
                        spec.category in ("Forecasting", "Regression")
                        or any(t in ("time_series", "forecasting") for t in spec.task_types)
                        or spec_id in ("load_dataset", "time_split", "forecast", "train", "evaluate", "compare")
                    )
                elif task == "anomaly":
                    is_match = (
                        spec.category == "Anomaly Detection"
                        or any("anomaly" in t for t in spec.task_types)
                        or spec_id in ("load_dataset", "preprocess", "train", "evaluate", "compare")
                    )
                else:
                    is_match = task in spec.task_types or "all" in spec.task_types
                if not is_match:
                    continue

            if query and query not in spec.name.casefold() and query not in spec.category.casefold():
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
                label += f"  (batas saran {spec.max_recommended_rows:,} baris)"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, spec_id)
            input_names = ", ".join(port.name for port in spec.inputs) or "tidak ada"
            output_names = ", ".join(port.name for port in spec.outputs) or "tidak ada"
            item.setToolTip(
                f"Tahap: {CATEGORY_TRANSLATIONS.get(spec.category, spec.category)}\n"
                f"Input: {input_names}\nOutput: {output_names}"
            )
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
        parameters = dict(parameters or {})
        dataframe = self._current_dataset()
        if dataframe is not None and "target" in self.engine.specs[spec_id].parameters and not parameters.get("target"):
            detected = getattr(self, "detected_task", {}) or {}
            target = detected.get("target_column")
            if target in dataframe.columns:
                parameters["target"] = target
        if spec_id in {
            "train", "evaluate", "cross_validate", "grid_search", "random_search",
            "halving_grid_search", "halving_random_search", "optuna",
        }:
            detected_type = (getattr(self, "detected_task", {}) or {}).get("task_type", "")
            if detected_type in {"classification", "regression"} and parameters.get("task", "auto") == "auto":
                parameters["task"] = detected_type
            elif detected_type == "regression" and parameters.get("task") == "classification":
                parameters["task"] = "regression"
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
        self._schedule_history_snapshot()
        self.validate_graph(show_tab=False)
        return node

    def _next_node_id(self):
        index = 1
        while f"n{index}" in self.engine.nodes:
            index += 1
        return f"n{index}"

    def _graph_state_key(self):
        return json.dumps(self.engine.to_dict(), ensure_ascii=False, sort_keys=True, default=str)

    def _dataset_source_document(self):
        return {
            "mode": self._dataset_scope,
            "scope_selected": self._dataset_scope_explicit,
            "path": self._graph_dataset_path,
        }

    def _graph_document_key(self):
        return json.dumps(
            {
                "graph": self.engine.to_dict(),
                "dataset_source": self._dataset_source_document(),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def _current_dataset(self):
        if self._dataset_scope != "graph":
            return getattr(self.main_window, "current_dataset", None)
        if self._graph_dataset is None and self._graph_dataset_path and not self._graph_dataset_error:
            path = Path(self._graph_dataset_path)
            loader = getattr(self.main_window, "loader", None)
            if not path.is_file():
                self._graph_dataset_error = f"File dataset tidak ditemukan: {path}"
            elif loader is None:
                self._graph_dataset_error = "Loader dataset aplikasi tidak tersedia."
            else:
                try:
                    self._graph_dataset = loader.load(str(path))
                    self._graph_dataset_error = ""
                except Exception as exc:
                    self._graph_dataset_error = str(exc)
        return self._graph_dataset

    def _current_dataset_name(self):
        if self._dataset_scope == "graph":
            return Path(self._graph_dataset_path).name if self._graph_dataset_path else "Dataset khusus graf"
        if self.main_window is not None and hasattr(self.main_window, "_current_filename"):
            try:
                return self.main_window._current_filename() or "Dataset Workspace"
            except Exception:
                pass
        return "Dataset Workspace"

    def _restore_dataset_source(self, source):
        source = source if isinstance(source, dict) else {}
        mode = source.get("mode", "graph")
        scope_selected = source.get("scope_selected") is True
        if mode not in {"workspace", "graph"}:
            mode = "graph"
        if mode == "workspace" and not scope_selected:
            mode = "graph"
        self._dataset_scope = mode
        self._dataset_scope_explicit = scope_selected
        self._graph_dataset_path = str(source.get("path") or "")
        self._graph_dataset = None
        self._graph_dataset_error = ""
        self.dataset_scope_combo.blockSignals(True)
        index = self.dataset_scope_combo.findData(self._dataset_scope)
        self.dataset_scope_combo.setCurrentIndex(max(0, index))
        self.dataset_scope_combo.blockSignals(False)
        if self._dataset_scope == "graph":
            self._current_dataset()

    def _schedule_history_snapshot(self):
        if self._history_suspended:
            return
        self._mark_graph_dirty()
        self._history_timer.start()

    def _record_graph_history(self):
        if self._history_suspended:
            return
        self._history_timer.stop()
        state_key = self._graph_state_key()
        if state_key == self._history_state:
            self._mark_graph_dirty()
            return
        self._history.push(self.engine.to_dict())
        self._history_state = state_key
        self._update_history_buttons()
        self._mark_graph_dirty()

    def _reset_graph_history(self, saved=False):
        self._history_timer.stop()
        self._history.clear()
        self._history.push(self.engine.to_dict())
        self._history_state = self._graph_state_key()
        if saved:
            self._saved_graph_state = self._graph_document_key()
        self._update_history_buttons()
        self._mark_graph_dirty()

    def _mark_graph_dirty(self):
        if not hasattr(self, "graph_state_label"):
            return
        self._session_graph_touched = True
        self._session_graph_timer.start()
        current = self._graph_document_key()
        saved = getattr(self, "_saved_graph_state", "")
        dirty = bool(saved) and current != saved
        self.graph_state_label.setText("Perubahan belum disimpan" if dirty else "Graf tersimpan")
        self.graph_state_label.setToolTip("Simpan graf untuk menyimpan perubahan terakhir." if dirty else "Graf sesuai dengan snapshot tersimpan.")

    def _update_history_buttons(self):
        if hasattr(self, "undo_button"):
            self.undo_button.setEnabled(self._history.can_undo() and not getattr(self, "_is_running", False))
            self.redo_button.setEnabled(self._history.can_redo() and not getattr(self, "_is_running", False))

    def _restore_graph_state(self, state):
        self._history_suspended = True
        try:
            self.cancel_connection()
            self.engine = _restore_graph(state)
            self.scene.clear()
            self.items.clear()
            self.cables.clear()
            for node in self.engine.nodes.values():
                item = GraphNodeItem(self, node, *node.position)
                self.items[node.id] = item
                self.scene.addItem(item)
            self.redraw_cables()
            self._clear_results()
            self._history_state = self._graph_state_key()
            self.refresh_palette()
            self.show_parameters()
            self._update_delete_button()
        finally:
            self._history_suspended = False
        self.validate_graph(show_tab=False)
        self._mark_graph_dirty()

    def undo_graph(self):
        state = self._history.undo()
        if state is not None:
            self._restore_graph_state(state)
            self._update_history_buttons()

    def redo_graph(self):
        state = self._history.redo()
        if state is not None:
            self._restore_graph_state(state)
            self._update_history_buttons()

    def prepare_for_open(self):
        if self.thread is not None and self.thread.isRunning():
            self.dataset_status.setText("Graf sedang berjalan")
            return
        self._restore_session_graph()
        self.refresh_palette()
        self.sync_dataset()

    def _session_graph_path(self) -> Path:
        app_data = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
        app_data_root = Path(app_data) if app_data else Path.home() / ".dora"
        return app_data_root / "node_editor" / "autosave.json"

    def _schedule_session_graph_save(self):
        self._session_graph_touched = True
        self._session_graph_timer.start()

    @staticmethod
    def _snapshot_frame(frame):
        if not isinstance(frame, pd.DataFrame):
            return None
        preview = frame.iloc[:RESULT_PREVIEW_ROWS, :RESULT_PREVIEW_COLUMNS].astype(str)
        return {
            "columns": [str(column) for column in preview.columns],
            "rows": preview.values.tolist(),
        }

    @staticmethod
    def _snapshot_table(table):
        columns = table.columnCount()
        rows = table.rowCount()
        return {
            "columns": [
                table.horizontalHeaderItem(column).text()
                if table.horizontalHeaderItem(column) is not None else ""
                for column in range(columns)
            ],
            "row_labels": [
                table.verticalHeaderItem(row).text()
                if table.verticalHeaderItem(row) is not None else str(row)
                for row in range(rows)
            ],
            "rows": [
                [
                    table.item(row, column).text()
                    if table.item(row, column) is not None else ""
                    for column in range(columns)
                ]
                for row in range(rows)
            ],
        }

    @staticmethod
    def _restore_table_snapshot(table, snapshot):
        if not isinstance(snapshot, dict):
            return False
        columns = snapshot.get("columns")
        rows = snapshot.get("rows")
        if not isinstance(columns, list) or not isinstance(rows, list):
            return False
        table.setColumnCount(len(columns))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels([str(column) for column in columns])
        row_labels = snapshot.get("row_labels")
        if isinstance(row_labels, list) and len(row_labels) == len(rows):
            table.setVerticalHeaderLabels([str(label) for label in row_labels])
        for row_index, values in enumerate(rows):
            if not isinstance(values, list):
                continue
            for column_index, value in enumerate(values[:len(columns)]):
                table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        return bool(columns and rows)

    def save_session_graph(self):
        if not self._session_graph_touched and not self.engine.nodes:
            return True
        self._session_graph_timer.stop()
        selected_key = self.preview_node_combo.currentData()
        selected_frame = self._result_outputs.get(selected_key)
        state = {
            "version": 1,
            "graph": self.engine.to_dict(),
            "dataset_source": self._dataset_source_document(),
            "workspace_project_id": getattr(self.main_window, "current_project_id", None),
            "run_history": self._run_history,
            "output_log": self.output_log.toPlainText(),
            "selected_result": {
                "key": selected_key,
                "label": self.preview_node_combo.currentText(),
                "frame": self._snapshot_frame(selected_frame),
            } if selected_key is not None and selected_frame is not None else None,
            "metric_rows": self._metric_rows,
            "confusion_matrix": self._snapshot_table(self.confusion_matrix_table),
            "forecast_metrics": getattr(self.forecast_chart, "_values", {}),
            "result_tab_index": self.result_tabs.currentIndex(),
        }
        path = self._session_graph_path()
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        try:
            serialized = json.dumps(state, ensure_ascii=False, indent=2, default=str)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(serialized, encoding="utf-8")
            temporary_path.replace(path)
        except (OSError, TypeError, ValueError) as error:
            self.graph_state_label.setText(f"Autosave graf gagal: {error}")
            return False
        return True

    def _restore_session_graph(self):
        if self._session_graph_restored:
            return
        self._session_graph_restored = True
        path = self._session_graph_path()
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or state.get("version") != 1:
                return
            graph = state.get("graph")
            if not isinstance(graph, dict):
                return
            self._restore_graph_state(graph)
            self._restore_dataset_source(state.get("dataset_source"))
            self._reset_graph_history(saved=True)
            run_history = state.get("run_history")
            self._run_history = []
            if isinstance(run_history, list):
                for index, run in enumerate(run_history[:100]):
                    if not isinstance(run, dict):
                        continue
                    self._run_history.append({
                        "id": str(run.get("id") or f"restored-{index + 1}"),
                        "created_at": str(run.get("created_at") or ""),
                        "task": str(run.get("task") or ""),
                        "target": str(run.get("target") or ""),
                        "dataset": str(run.get("dataset") or ""),
                        "duration_ms": run.get("duration_ms", 0),
                        "status": str(run.get("status") or "unknown"),
                        "metrics": run.get("metrics") if isinstance(run.get("metrics"), dict) else {},
                        "graph_version_id": str(run.get("graph_version_id") or ""),
                        "random_state": run.get("random_state", 42),
                        "notes": str(run.get("notes") or ""),
                    })
            self._refresh_run_history()
            self.output_log.setPlainText(str(state.get("output_log") or ""))

            same_workspace = (
                state.get("workspace_project_id")
                == getattr(self.main_window, "current_project_id", None)
            )
            keep_results = self._dataset_scope == "graph" or same_workspace
            if keep_results:
                metric_rows = state.get("metric_rows")
                if isinstance(metric_rows, list):
                    self._metric_rows = metric_rows
                    self._display_table(self.metrics_results, pd.DataFrame(metric_rows))
                    self.metric_summary_model.blockSignals(True)
                    self.metric_summary_model.clear()
                    for row in metric_rows:
                        if isinstance(row, dict):
                            self.metric_summary_model.addItem(str(row.get("Model", "Model")))
                    self.metric_summary_model.blockSignals(False)
                    self.metric_summary_model.setVisible(len(metric_rows) > 1)
                    self.metric_summary_model_label.setVisible(len(metric_rows) > 1)
                    if metric_rows:
                        self.metric_summary_model.setCurrentIndex(0)
                    self._render_metric_summary()
                    self.result_tabs.setTabVisible(1, bool(metric_rows))
                if self._restore_table_snapshot(
                    self.confusion_matrix_table, state.get("confusion_matrix")
                ):
                    self.result_tabs.setTabVisible(2, True)
                forecast_metrics = state.get("forecast_metrics")
                if isinstance(forecast_metrics, dict) and forecast_metrics:
                    self.forecast_chart.set_metrics(forecast_metrics)
                    self.result_tabs.setTabVisible(3, True)
                selected_result = state.get("selected_result")
                if isinstance(selected_result, dict) and isinstance(selected_result.get("frame"), dict):
                    frame_data = selected_result["frame"]
                    columns = frame_data.get("columns")
                    rows = frame_data.get("rows")
                    if isinstance(columns, list) and isinstance(rows, list):
                        key = str(selected_result.get("key") or "restored:table")
                        frame = pd.DataFrame(rows, columns=columns)
                        self._result_outputs = {key: frame}
                        self.preview_node_combo.blockSignals(True)
                        self.preview_node_combo.clear()
                        self.preview_node_combo.addItem("Pilih output node…", None)
                        self.preview_node_combo.addItem(
                            str(selected_result.get("label") or "Output dipulihkan"), key
                        )
                        self.preview_node_combo.blockSignals(False)
                        self.preview_node_combo.setCurrentIndex(1)
                        self.result_tabs.setTabVisible(0, True)
                saved_tab = state.get("result_tab_index", 0)
                if (
                    isinstance(saved_tab, int)
                    and 0 <= saved_tab < self.result_tabs.count()
                    and self.result_tabs.isTabVisible(saved_tab)
                ):
                    self.result_tabs.setCurrentIndex(saved_tab)
            self._session_graph_touched = True
            self._mark_graph_dirty()
            self._reset_graph_history(saved=True)
            self.graph_state_label.setText("Graf dan progres dipulihkan")
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError, TypeError, ValueError, GraphError) as error:
            self.graph_state_label.setText(f"Autosave graf tidak dapat dipulihkan: {error}")

    def _on_dataset_scope_changed(self, index):
        scope = self.dataset_scope_combo.itemData(index)
        if scope not in {"workspace", "graph"} or scope == self._dataset_scope:
            return
        self._dataset_scope = scope
        self._dataset_scope_explicit = True
        self._active_dataset_token = None
        if scope == "graph":
            self._current_dataset()
        self.sync_dataset()
        self._mark_graph_dirty()

    def _choose_graph_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih Dataset Khusus Graf",
            self._graph_dataset_path,
            "Datasets (*.csv *.tsv *.xlsx *.xls *.json *.parquet);;All Files (*)",
        )
        if not path:
            return
        loader = getattr(self.main_window, "loader", None)
        if loader is None:
            QMessageBox.critical(self, "Dataset tidak dapat dimuat", "Loader dataset aplikasi tidak tersedia.")
            return
        try:
            dataframe = loader.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Dataset tidak dapat dimuat", str(exc))
            return
        self._graph_dataset = dataframe
        self._graph_dataset_path = str(Path(path).resolve())
        self._graph_dataset_error = ""
        if self._dataset_scope != "graph":
            self._dataset_scope = "graph"
            self.dataset_scope_combo.setCurrentIndex(self.dataset_scope_combo.findData("graph"))
        self._active_dataset_token = None
        self.sync_dataset()
        self._mark_graph_dirty()

    def choose_dataset_for_node(self, node_id):
        if self._is_running:
            QMessageBox.information(self, "Graf sedang berjalan", "Tunggu hingga proses graf selesai sebelum mengganti dataset.")
            return
        if self._dataset_scope == "graph":
            self._choose_graph_dataset()
            return
        upload_page = getattr(self.main_window, "upload_page", None)
        if upload_page is None:
            QMessageBox.information(self, "Upload dataset", "Halaman pemuatan dataset tidak tersedia.")
            return

        self.main_window.open_upload_page()
        try:
            upload_page.select_dataset()
        finally:
            self.main_window.open_node_editor_page()
            graph_item = self.items.get(node_id)
            if graph_item is not None:
                self.view.centerOn(graph_item)

    def sync_dataset(self):
        dataframe = self._current_dataset()
        if dataframe is None:
            self._active_dataset_token = None
            self._template_key = None
            if self._dataset_scope == "graph":
                message = "Dataset khusus graf belum dipilih"
                tooltip = self._graph_dataset_error or "Pilih file dataset dari tombol pada node Muat Dataset."
            else:
                message = "Workspace belum memiliki dataset aktif"
                tooltip = "Muat dataset pada Workspace untuk mulai."
            self.dataset_status.setText(message)
            self.dataset_status.setToolTip(tooltip)
            self._task_label = "Belum terdeteksi"
            self._has_dataset = False
            self.dataset_task_status.setText("Task belum terdeteksi")
            for node in self.engine.nodes.values():
                if node.spec_id == "load_dataset":
                    node.status = "Dataset graf belum dipilih" if self._dataset_scope == "graph" else "Dataset workspace belum dimuat"
                    item = self.items.get(node.id)
                    if item is not None:
                        item.set_status(node.status)
                        item.set_dataset_context(self._task_label, False)
            source_nodes = [node for node in self.engine.nodes.values() if node.spec_id == "load_dataset"]
            if (
                self._dataset_scope == "graph"
                and not source_nodes
                and self._suppressed_source_dataset_id != self._active_dataset_token
            ):
                self.add_node("load_dataset", position=(0, 0))
                QTimer.singleShot(0, self.fit_graph)
            self.template_combo.blockSignals(True)
            self.template_combo.clear()
            self.template_combo.addItem("Pilih template alur kerja…", None)
            self.template_combo.blockSignals(False)
            self.result_tabs.setTabVisible(self.profile_tab_index, False)
            self._refresh_profile_view(None)
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
        # 1. Identify target from existing nodes if configured
        target_col = None
        for node in self.engine.nodes.values():
            if node.parameters.get("target"):
                target_col = node.parameters["target"]
                break

        # 2. Automated Task Detection
        from app.ml.graph.task_detector import detect_task, generate_template_graphs
        detection = detect_task(dataframe, target_column=target_col, auto_guess_target=True)
        self.detected_task = detection
        self._refresh_profile_view(dataframe)
        self.result_tabs.setTabVisible(self.profile_tab_index, True)
        task_type = detection["task_type"]
        detected_target = detection.get("target_column")

        task_label = {
            "binary_classification": "Klasifikasi Biner",
            "multiclass_classification": "Klasifikasi Multikelas",
            "regression": "Regresi",
            "clustering": "Pengelompokan (Clustering)",
            "time_series": "Rangkaian Waktu",
        }.get(task_type, task_type.capitalize())
        target_desc = f" ({detected_target})" if detected_target else ""
        source_name = "Dataset graf" if self._dataset_scope == "graph" else "Workspace"
        self.dataset_status.setText(f"{source_name} | {rows:,} baris x {columns:,} kolom")
        self.dataset_task_status.setText(f"Task: {task_label}")
        self._task_label = task_label
        self._has_dataset = True

        filename = self._current_dataset_name()
        self.dataset_status.setToolTip(
            f"{filename}\nTarget: {detected_target or 'Tidak ada'}\nAlasan: {detection['reason']}"
        )
        self.dataset_task_status.setToolTip(
            f"Task terdeteksi: {task_label}\nTarget: {detected_target or 'Tidak ada'}\n{detection['reason']}"
        )

        # 3. Automatically set task filter to match detected task
        filter_key = (
            "classification" if "classification" in task_type
            else ("regression" if task_type == "regression"
            else ("clustering" if task_type == "clustering"
            else ("time_series" if task_type == "time_series" else "all")))
        )
        if hasattr(self, "task_filter"):
            idx = self.task_filter.findData(filter_key)
            if idx >= 0 and self.task_filter.currentIndex() != idx:
                self.task_filter.blockSignals(True)
                self.task_filter.setCurrentIndex(idx)
                self.task_filter.blockSignals(False)
                self.refresh_palette()

        # 4. Dynamically adapt template dropdown to task
        template_key = (self._active_dataset_token, detected_target, task_type)
        if hasattr(self, "template_combo") and template_key != getattr(self, "_template_key", None):
            self._template_key = template_key
            self.current_templates = self._build_dataset_templates(
                dataframe, detection, generate_template_graphs(dataframe, detected_target)
            )
            self.template_combo.blockSignals(True)
            self.template_combo.clear()
            self.template_combo.addItem(f"Pilih template ({len(self.current_templates)})…", None)
            for tmpl in self.current_templates:
                self.template_combo.addItem(tmpl["name"], tmpl)
                self.template_combo.setItemData(
                    self.template_combo.count() - 1,
                    tmpl.get("description", "Template alur kerja untuk dataset aktif."),
                    Qt.ToolTipRole,
                )
            self.template_combo.blockSignals(False)

        source_nodes = [node for node in self.engine.nodes.values() if node.spec_id == "load_dataset"]
        if not source_nodes and self._suppressed_source_dataset_id != self._active_dataset_token:
            self.add_node("load_dataset", position=(0, 0))
            QTimer.singleShot(0, self.fit_graph)
            source_nodes = [node for node in self.engine.nodes.values() if node.spec_id == "load_dataset"]
        status = f"Dataset graf siap | {rows:,} x {columns:,}" if self._dataset_scope == "graph" else f"Dataset workspace siap | {rows:,} x {columns:,}"
        for node in source_nodes:
            node.status = status
            if node.id in self.items:
                self.items[node.id].set_status(status)
                self.items[node.id].set_dataset_context(task_label, True)

    def _build_dataset_templates(self, dataframe, detection, generated):
        task_type = detection.get("task_type", "")
        target = detection.get("target_column")
        profile = detection.get("profile", {})
        templates = []
        if "classification" in task_type:
            templates.extend(
                template for template in generated
                if "classification" in str(template.get("task_type", ""))
            )
            templates.extend(self._classification_templates(dataframe, target, profile))
        elif task_type == "regression":
            templates.extend(
                template for template in generated
                if template.get("task_type") == "regression"
            )
            templates.extend(self._regression_templates(target))
        elif task_type == "clustering":
            templates.extend(self._clustering_templates(dataframe))
        elif task_type == "time_series":
            templates.extend(self._time_series_templates(dataframe, target, profile))
        if "classification" in task_type or task_type == "regression":
            normalized_task = "classification" if "classification" in task_type else "regression"
            templates.extend(self._tuning_templates(target, normalized_task))
        return [
            template for template in templates
            if all(node.get("spec_id") in self.engine.specs for node in template.get("graph", {}).get("nodes", []))
        ]

    def _template_record(self, template_id, name, description, task_type, nodes, connections):
        return {
            "id": template_id,
            "name": name,
            "tag": "workflow",
            "description": description,
            "task_type": task_type,
            "graph": {"nodes": nodes, "connections": connections},
        }

    def _tuning_templates(self, target, task):
        if not target:
            return []
        model_id = "random_forest_classifier" if task == "classification" else "random_forest_regressor"
        if f"model:{model_id}" not in self.engine.specs:
            return []
        definitions = [
            ("grid_search", "Grid Search", "Uji kombinasi parameter secara menyeluruh dengan CV."),
            ("random_search", "Random Search", "Uji sampel kombinasi parameter secara acak."),
            ("halving_grid_search", "Successive Halving Grid", "Saring kombinasi bertahap memakai sumber daya data yang meningkat."),
            ("halving_random_search", "Successive Halving Random", "Saring kandidat acak secara bertahap untuk mempercepat pencarian luas."),
            ("optuna", "Optuna TPE", "Cari parameter dengan sampler Bayesian TPE dan catat semua percobaan."),
        ]
        templates = []
        for spec_id, label, description in definitions:
            if spec_id not in self.engine.specs:
                continue
            nodes = [
                {"id": "dataset", "spec_id": "load_dataset", "position": (30, 210)},
                {"id": "split", "spec_id": "split", "parameters": {
                    "target": target, "task": task, "test_size": 0.2, "random_state": 42,
                }, "position": (250, 210)},
                {"id": "estimator", "spec_id": f"model:{model_id}", "position": (250, 430)},
                {"id": "tuning", "spec_id": spec_id, "parameters": {
                    "target": target, "task": task, "cv": 5, "scoring": "auto",
                }, "position": (500, 210)},
                {"id": "train", "spec_id": "train", "parameters": {
                    "target": target, "task": task,
                }, "position": (760, 210)},
                {"id": "evaluate", "spec_id": "evaluate", "parameters": {
                    "target": target, "task": task,
                }, "position": (1010, 210)},
                {"id": "compare", "spec_id": "compare", "position": (1260, 210)},
            ]
            connections = [
                ["dataset", "table", "split", "table"],
                ["split", "train", "tuning", "table"],
                ["estimator", "estimator", "tuning", "estimator"],
                ["split", "train", "train", "train"],
                ["tuning", "estimator", "train", "estimator"],
                ["train", "model", "evaluate", "model"],
                ["split", "test", "evaluate", "test"],
                ["evaluate", "metrics", "compare", "metrics"],
            ]
            if spec_id == "random_search":
                nodes[3]["parameters"].update({"n_iter": 10})
            elif spec_id == "halving_random_search":
                nodes[3]["parameters"].update({"n_iter": 20})
            elif spec_id == "optuna":
                nodes[3]["parameters"].update({"sampler": "tpe", "trials": 30, "search_space": {}})
            templates.append(self._template_record(
                f"ui_{task}_tuning_{spec_id}",
                f"Tuning · {label}",
                f"{description} Pencarian hanya memakai data latih; data uji disimpan untuk evaluasi akhir.",
                task,
                nodes,
                connections,
            ))
        return templates

    def _supervised_template(
        self, template_id, name, description, target, task, model_ids,
        ensemble_kind=None, voting="hard", sampler=None, model_parameters=None,
    ):
        model_ids = [model_id for model_id in model_ids if f"model:{model_id}" in self.engine.specs]
        if not target or not model_ids or (ensemble_kind and len(model_ids) < 2):
            return None
        positions = {
            "dataset": (30, 180), "target": (220, 180), "split": (410, 180),
            "prep": (600, 180), "sampler": (830, 180), "ensemble": (850, 430),
            "train": (1060, 180), "evaluate": (1270, 180), "compare": (1480, 180),
        }
        nodes = [
            {"id": "dataset", "spec_id": "load_dataset", "position": positions["dataset"]},
            {"id": "target", "spec_id": "select_target", "parameters": {"target": target}, "position": positions["target"]},
            {"id": "split", "spec_id": "split", "parameters": {"target": target, "task": task, "test_size": 0.2, "random_state": 42}, "position": positions["split"]},
            {"id": "prep", "spec_id": "preprocess", "parameters": {"target": target, "scaler": "standard", "imputer": "median", "encoder": "onehot"}, "position": positions["prep"]},
        ]
        connections = [
            ["dataset", "table", "target", "table"],
            ["target", "table", "split", "table"],
            ["split", "train", "prep", "train"],
            ["split", "test", "prep", "test"],
        ]
        estimator_nodes = []
        model_parameters = model_parameters or {}
        for index, model_id in enumerate(model_ids):
            node_id = f"estimator_{index + 1}"
            nodes.append({
                "id": node_id,
                "spec_id": f"model:{model_id}",
                "parameters": dict(model_parameters.get(model_id, {})),
                "position": (610, 370 + index * 145),
            })
            estimator_nodes.append(node_id)
        if sampler and sampler in self.engine.specs:
            nodes.append({
                "id": "sampler",
                "spec_id": sampler,
                "parameters": {"target": target, "random_state": 42},
                "position": positions["sampler"],
            })
            connections.extend([
                ["prep", "train", "sampler", "train"],
                ["sampler", "train", "train", "train"],
            ])
        else:
            connections.append(["prep", "train", "train", "train"])

        estimator_source = estimator_nodes[0]
        if ensemble_kind:
            ensemble_spec = f"ensemble:{ensemble_kind}"
            if ensemble_spec not in self.engine.specs:
                return None
            ensemble_parameters = {"kind": ensemble_kind}
            if ensemble_kind == "voting":
                ensemble_parameters["voting"] = voting
            nodes.append({"id": "ensemble", "spec_id": ensemble_spec, "parameters": ensemble_parameters, "position": positions["ensemble"]})
            for node_id in estimator_nodes:
                connections.append([node_id, "estimator", "ensemble", "estimators"])
            estimator_source = "ensemble"

        nodes.extend([
            {"id": "train", "spec_id": "train", "parameters": {"target": target, "task": task}, "position": positions["train"]},
            {"id": "evaluate", "spec_id": "evaluate", "parameters": {"target": target, "task": task}, "position": positions["evaluate"]},
            {"id": "compare", "spec_id": "compare", "position": positions["compare"]},
        ])
        connections.extend([
            [estimator_source, "estimator", "train", "estimator"],
            ["train", "model", "evaluate", "model"],
            ["prep", "test", "evaluate", "test"],
            ["evaluate", "metrics", "compare", "metrics"],
        ])
        return self._template_record(template_id, name, description, task, nodes, connections)

    def _classification_templates(self, dataframe, target, profile):
        templates = []
        ensemble_models = ["logistic_regression", "random_forest_classifier", "gradient_boosting_classifier"]
        definitions = [
            (
                "ui_classification_hybrid_voting",
                "Ensemble · Hybrid Voting (3 model)",
                "Menggabungkan Logistic Regression, Random Forest, dan Gradient Boosting dalam pipeline preprocessing dan evaluasi.",
                ensemble_models, "voting", "hard",
            ),
            (
                "ui_classification_soft_voting",
                "Ensemble · Hybrid Soft Voting",
                "Menggabungkan probabilitas tiga classifier untuk menentukan prediksi akhir.",
                ensemble_models, "voting", "soft",
            ),
        ]
        if target in dataframe.columns:
            min_class_size = int(dataframe[target].value_counts().min()) if not dataframe[target].empty else 0
        else:
            min_class_size = 0
        stacking_models = ["logistic_regression", "random_forest_classifier", "extra_trees_classifier"]
        if min_class_size >= 7 and all(f"model:{model_id}" in self.engine.specs for model_id in stacking_models):
            definitions.append((
                "ui_classification_stacking",
                "Ensemble · Stacking (3 model)",
                "Menggabungkan Logistic Regression, Random Forest, dan Extra Trees melalui meta-classifier. Ditawarkan saat tiap kelas cukup besar untuk validasi internal.",
                stacking_models, "stacking", "hard",
            ))
        for template_id, name, description, model_ids, kind, voting in definitions:
            template = self._supervised_template(
                template_id, name, description, target, "classification", model_ids,
                ensemble_kind=kind, voting=voting,
            )
            if template:
                templates.append(template)

        if profile.get("target_info", {}).get("is_imbalanced"):
            weighted = self._supervised_template(
                "ui_classification_weighted_forest",
                "Imbalanced · Class-weighted Random Forest",
                "Menggunakan bobot kelas pada Random Forest dan melaporkan F1 serta balanced accuracy.",
                target, "classification", ["random_forest_classifier"],
                model_parameters={"random_forest_classifier": {"class_weight": "balanced_subsample"}},
            )
            if weighted:
                templates.append(weighted)
            if "smote" in self.engine.specs:
                smote_forest = self._supervised_template(
                    "ui_classification_smote_forest",
                    "Imbalanced · SMOTE + Random Forest",
                    "Menyeimbangkan data latih dengan SMOTE sebelum Random Forest. Node SMOTE memakai paket imbalanced-learn.",
                    target, "classification", ["random_forest_classifier"], sampler="smote",
                )
                smote_hybrid = self._supervised_template(
                    "ui_classification_smote_hybrid",
                    "Imbalanced · SMOTE + Hybrid Voting",
                    "Menggabungkan SMOTE dengan voting Logistic Regression, Random Forest, dan Gradient Boosting.",
                    target, "classification", ensemble_models, ensemble_kind="voting", sampler="smote",
                )
                if smote_forest:
                    templates.append(smote_forest)
                if smote_hybrid:
                    templates.append(smote_hybrid)
        return templates

    def _regression_templates(self, target):
        definitions = [
            (
                "ui_regression_ridge", "Model · Ridge Regression",
                "Baseline linear dengan regularisasi L2 dan preprocessing fitur.",
                ["ridge_regression"], None,
            ),
            (
                "ui_regression_hist_gradient", "Model · Hist Gradient Boosting",
                "Pipeline boosting untuk target numerik dengan evaluasi MAE, RMSE, dan R².",
                ["hist_gradient_boosting_regressor"], None,
            ),
            (
                "ui_regression_hybrid_voting", "Ensemble · Voting Regressor (3 model)",
                "Menggabungkan Ridge, Random Forest, dan Gradient Boosting untuk prediksi numerik.",
                ["ridge_regression", "random_forest_regressor", "gradient_boosting_regressor"], "voting",
            ),
            (
                "ui_regression_tree_ensemble", "Ensemble · Extra Trees + Random Forest",
                "Menggabungkan dua ensemble pohon melalui Voting Regressor.",
                ["extra_trees_regressor", "random_forest_regressor"], "voting",
            ),
            (
                "ui_regression_elastic_net", "Model · Elastic Net",
                "Model linear dengan regularisasi L1 dan L2 setelah imputasi serta penskalaan fitur.",
                ["elasticnet_regression"], None,
            ),
        ]
        templates = []
        for template_id, name, description, model_ids, ensemble_kind in definitions:
            template = self._supervised_template(
                template_id, name, description, target, "regression", model_ids,
                ensemble_kind=ensemble_kind,
            )
            if template:
                templates.append(template)
        return templates

    def _clustering_templates(self, dataframe):
        if len(dataframe) < 3:
            return []
        templates = []
        cluster_count = max(2, min(5, len(dataframe) // 2))
        definitions = [
            ("kmeans", "Clustering · K-Means", {"n_clusters": min(cluster_count, 3), "random_state": 42}),
            ("minibatch_kmeans", "Clustering · Mini-Batch K-Means", {"n_clusters": min(cluster_count, 3), "random_state": 42}),
            ("agglomerative_clustering", "Clustering · Agglomerative", {"n_clusters": min(cluster_count, 3)}),
            ("dbscan", "Clustering · DBSCAN", {"eps": 0.5, "min_samples": 5}),
            ("gaussian_mixture", "Clustering · Gaussian Mixture", {"n_components": min(cluster_count, 3), "random_state": 42}),
            ("birch", "Clustering · BIRCH", {"n_clusters": min(cluster_count, 3)}),
        ]
        for model_id, name, parameters in definitions:
            spec = self.engine.specs.get(f"model:{model_id}")
            if spec is None or len(dataframe) > spec.max_recommended_rows:
                continue
            nodes = [
                {"id": "dataset", "spec_id": "load_dataset", "position": (30, 180)},
                {"id": "split", "spec_id": "split", "parameters": {"task": "clustering", "test_size": 0.2, "random_state": 42}, "position": (230, 180)},
                {"id": "prep", "spec_id": "preprocess", "parameters": {"scaler": "standard", "imputer": "median", "encoder": "onehot"}, "position": (440, 180)},
                {"id": "model", "spec_id": f"model:{model_id}", "parameters": parameters, "position": (460, 390)},
                {"id": "train", "spec_id": "train", "parameters": {"task": "clustering"}, "position": (690, 180)},
                {"id": "evaluate", "spec_id": "evaluate", "parameters": {"task": "clustering"}, "position": (920, 180)},
                {"id": "compare", "spec_id": "compare", "position": (1150, 180)},
            ]
            connections = [
                ["dataset", "table", "split", "table"],
                ["split", "train", "prep", "train"],
                ["split", "test", "prep", "test"],
                ["prep", "train", "train", "train"],
                ["model", "estimator", "train", "estimator"],
                ["train", "model", "evaluate", "model"],
                ["prep", "test", "evaluate", "test"],
                ["evaluate", "metrics", "compare", "metrics"],
            ]
            templates.append(self._template_record(
                f"ui_{model_id}_clustering", name,
                f"Membagi dataset, memproses fitur kategori, lalu membentuk cluster dengan {spec.name}.",
                "clustering", nodes, connections,
            ))
        return templates

    def _time_series_templates(self, dataframe, target, profile):
        if not target or target not in dataframe.columns:
            return []
        datetime_columns = profile.get("datetime_cols", [])
        time_column = datetime_columns[0] if datetime_columns else str(dataframe.columns[0])
        train_size = max(1, int(len(dataframe) * 0.8))
        methods = [("naive", "Forecast · Naive", {})]
        periods = sorted({period for period in (3, 6, 12) if period <= train_size})
        methods.extend(
            ("seasonal_naive", f"Forecast · Seasonal Naive ({period})", {"period": period})
            for period in periods
        )
        if len(dataframe) >= 12 and importlib.util.find_spec("statsmodels") is not None:
            methods.extend([
                ("exponential_smoothing", "Forecast · Exponential Smoothing", {}),
                ("arima", "Forecast · ARIMA (1, 1, 1)", {"order": [1, 1, 1]}),
                ("sarimax", "Forecast · SARIMAX", {"order": [1, 1, 1], "seasonal_order": [1, 0, 0, max(periods, default=1)]}),
            ])
        templates = []
        for index, (method, name, parameters) in enumerate(methods):
            nodes = [
                {"id": "dataset", "spec_id": "load_dataset", "position": (30, 170)},
                {"id": "split", "spec_id": "time_split", "parameters": {"time_column": time_column, "test_size": 0.2}, "position": (280, 170)},
                {"id": "forecast", "spec_id": "forecast", "parameters": {"target": target, "method": method, **parameters}, "position": (560, 170)},
            ]
            connections = [
                ["dataset", "table", "split", "table"],
                ["split", "train", "forecast", "train"],
                ["split", "test", "forecast", "test"],
            ]
            templates.append(self._template_record(
                f"ui_forecast_{method}_{index}", name,
                f"Menguji metode {method.replace('_', ' ')} pada pembagian kronologis. Hasil mencakup MAE, RMSE, dan MAPE.",
                "time_series", nodes, connections,
            ))
        return templates

    def _refresh_profile_view(self, dataframe):
        while self.profile_summary_layout.count():
            item = self.profile_summary_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.profile_table.setRowCount(0)
        if dataframe is None:
            self.profile_table.setColumnCount(0)
            return
        profile = getattr(self, "detected_task", {}).get("profile", {})
        row_count, column_count = dataframe.shape
        total_missing = int(profile.get("total_missing", int(dataframe.isna().sum().sum())))
        total_cells = max(row_count * column_count, 1)
        missing_percent = total_missing / total_cells * 100
        target = getattr(self, "detected_task", {}).get("target_column") or "Belum dipilih"
        summary = [
            ("Baris", f"{row_count:,}"),
            ("Kolom", f"{column_count:,}"),
            ("Nilai kosong", f"{total_missing:,} ({missing_percent:.1f}%)"),
            ("Outlier IQR", f"{int(profile.get('outlier_count', 0)):,}"),
            ("Target terdeteksi", str(target)),
        ]
        for label, value in summary:
            self.profile_summary_layout.addWidget(self._make_summary_card(label, value))
        self.profile_summary_layout.addStretch(1)

        column_types = profile.get("column_types", {})
        missing_by_column = profile.get("missing_percent", {})
        type_labels = {
            "numeric": "Numerik",
            "categorical": "Kategori",
            "datetime": "Tanggal/waktu",
            "id_like": "ID berurutan",
        }
        self.profile_table.setColumnCount(4)
        self.profile_table.setHorizontalHeaderLabels(["Kolom", "Jenis data", "Nilai unik", "Kosong"])
        self.profile_table.setRowCount(column_count)
        for row, column in enumerate(dataframe.columns):
            unique_count = int(dataframe[column].nunique(dropna=True))
            missing = float(missing_by_column.get(column, 0.0))
            values = (
                str(column),
                type_labels.get(column_types.get(column, ""), str(column_types.get(column, "Tidak diketahui"))),
                f"{unique_count:,}",
                f"{missing:.2f}%",
            )
            for index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.profile_table.setItem(row, index, item)
        self.profile_table.resizeColumnsToContents()

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
        selected = self.scene.selectedItems()
        has_node = any(isinstance(item, GraphNodeItem) for item in selected)
        has_connection = any(
            isinstance(item, QGraphicsPathItem) and item.data(1) is not None
            for item in selected
        )
        self.delete_button.setEnabled(not getattr(self, "_is_running", False) and (has_node or has_connection))
        self.duplicate_button.setEnabled(not getattr(self, "_is_running", False) and has_node)
        self.run_to_button.setEnabled(not getattr(self, "_is_running", False) and has_node)

    def delete_selected_nodes(self):
        selected = [item for item in self.scene.selectedItems() if isinstance(item, GraphNodeItem)]
        selected_cables = [
            item for item in self.scene.selectedItems()
            if isinstance(item, QGraphicsPathItem) and item.data(1) is not None
        ]
        if not selected and not selected_cables:
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
        for cable in selected_cables:
            edge = cable.data(1)
            if edge in self.engine.connections:
                self.engine.connections.remove(edge)
        self.engine.connections = [
            edge for edge in self.engine.connections
            if edge[0] not in removed_ids and edge[2] not in removed_ids
        ]
        self.engine._cache.clear()
        self.redraw_cables()
        self.show_parameters()
        self._update_delete_button()
        self._clear_results()
        self._schedule_history_snapshot()
        self.validate_graph(show_tab=False)

    def duplicate_selected_nodes(self):
        selected = [item for item in self.scene.selectedItems() if isinstance(item, GraphNodeItem)]
        if not selected:
            return
        old_ids = {item.node_id for item in selected}
        old_edges = [edge for edge in self.engine.connections if edge[0] in old_ids and edge[2] in old_ids]
        id_map = {}
        self._history_suspended = True
        try:
            for item in selected:
                node = self.engine.nodes[item.node_id]
                duplicate = self.add_node(
                    node.spec_id,
                    json.loads(json.dumps(node.parameters, default=str)),
                    (item.pos().x() + 44, item.pos().y() + 44),
                )
                id_map[node.id] = duplicate.id
            for source, output, target, input_name in old_edges:
                self.engine.connect(id_map[source], output, id_map[target], input_name)
        except GraphError as exc:
            QMessageBox.warning(self, "Node tidak dapat disalin", str(exc))
        finally:
            self._history_suspended = False
        self.redraw_cables()
        self._record_graph_history()
        self.validate_graph(show_tab=False)

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
            self._record_graph_history()
            self.validate_graph(show_tab=False)
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
            cable.setData(1, edge)
            cable.setFlag(QGraphicsItem.ItemIsSelectable, True)
            cable.setCursor(Qt.PointingHandCursor)
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
            self.node_help_label.setText("Pilih node untuk melihat fungsi, port, dan saran alurnya.")
            return
        node = self.engine.nodes[selected[0].node_id]
        spec = self.engine.specs[node.spec_id]
        input_text = ", ".join(f"{port.name}: {port.type}" for port in spec.inputs) or "tidak ada"
        output_text = ", ".join(f"{port.name}: {port.type}" for port in spec.outputs) or "tidak ada"
        description = NODE_HELP.get(node.spec_id, f"Node {spec.name} untuk kategori {CATEGORY_TRANSLATIONS.get(spec.category, spec.category)}.")
        self.node_help_label.setText(
            f"<b>{spec.name}</b><br><br>{description}<br><br>"
            f"<b>Input:</b> {input_text}<br><b>Output:</b> {output_text}<br><br>"
            f"<b>Saran berikutnya:</b> {self._next_node_suggestion(node.spec_id)}"
        )
        metadata = self._node_parameter_metadata(node.spec_id)
        for key, value in node.parameters.items():
            if key == "method" and node.spec_id in {"halving_grid_search", "halving_random_search"}:
                continue
            param_meta = metadata.get(key, {})
            help_text = param_meta.get("description") or PARAMETER_HELP.get(key, f"Parameter {key.replace('_', ' ')} untuk node ini.")
            choices = self._parameter_choices(node, key, param_meta)
            if key in {"target", "time_column"}:
                widget = QComboBox()
                widget.addItem("Pilih kolom…", "")
                dataframe = self._current_dataset()
                columns = [str(column) for column in dataframe.columns] if dataframe is not None else []
                for column in columns:
                    widget.addItem(column, column)
                if value and str(value) not in columns:
                    widget.addItem(f"{value} (tidak ada di dataset)", value)
                index = widget.findData(value or "")
                if index >= 0:
                    widget.setCurrentIndex(index)
                widget.currentIndexChanged.connect(
                    lambda index, k=key, n=node, w=widget: self._set_parameter(n, k, w.itemData(index))
                )
            elif key in {"min_resources", "max_resources"}:
                widget = QLineEdit(str(value))
                widget.textChanged.connect(lambda v, k=key, n=node: self._set_parameter(n, k, v))
            elif choices:
                widget = QComboBox()
                for label, option in choices:
                    widget.addItem(label, option)
                index = widget.findData(value)
                if index < 0 and value is not None:
                    widget.addItem(f"{value} (custom)", value)
                    index = widget.count() - 1
                if index >= 0:
                    widget.setCurrentIndex(index)
                widget.currentIndexChanged.connect(
                    lambda index, k=key, n=node, w=widget: self._set_parameter(n, k, w.itemData(index))
                )
            elif isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
                widget.toggled.connect(lambda v, k=key, n=node: self._set_parameter(n, k, v))
            elif isinstance(value, int):
                widget = QSpinBox()
                low, high = self._integer_parameter_range(key)
                widget.setRange(low, high)
                widget.setValue(value)
                widget.valueChanged.connect(lambda v, k=key, n=node: self._set_parameter(n, k, v))
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                low, high, decimals = self._float_parameter_range(key)
                widget.setRange(low, high)
                widget.setDecimals(decimals)
                widget.setValue(value)
                widget.valueChanged.connect(lambda v, k=key, n=node: self._set_parameter(n, k, v))
            elif isinstance(value, (list, dict)):
                widget = QPlainTextEdit()
                widget.setPlainText(json.dumps(value, ensure_ascii=False, indent=2))
                widget.setFixedHeight(82 if key in {"param_grid", "search_space"} else 64)
                widget.textChanged.connect(lambda k=key, n=node, w=widget: self._set_structured_parameter(n, k, w.toPlainText()))
            else:
                widget = QLineEdit(str(value or ""))
                widget.textChanged.connect(lambda v, k=key, n=node: self._set_parameter(n, k, v))
            widget.setToolTip(help_text)
            self.parameter_form.addRow(key.replace("_", " ").title(), widget)

    def _node_parameter_metadata(self, spec_id):
        try:
            from app.ml.graph.node_registry import get_node_registry
            entry = get_node_registry().get(spec_id)
            if entry is None:
                return {}
            return {
                name: value if isinstance(value, dict) else value.to_dict()
                for name, value in entry.params.items()
            }
        except Exception:
            return {}

    def _parameter_choices(self, node, key, metadata):
        if key == "task":
            return PARAMETER_CHOICES["task"]
        if key == "scoring":
            task = node.parameters.get("task", "auto")
            if task == "auto":
                task = (getattr(self, "detected_task", {}) or {}).get("task_type", "classification")
            options = [("Auto sesuai task/data", "auto")]
            if task == "regression":
                options.extend([
                    ("R²", "r2"),
                    ("Mean Absolute Error", "neg_mean_absolute_error"),
                    ("Root Mean Squared Error", "neg_root_mean_squared_error"),
                    ("Mean Squared Error", "neg_mean_squared_error"),
                    ("Median Absolute Error", "neg_median_absolute_error"),
                    ("Explained Variance", "explained_variance"),
                ])
            else:
                options.extend([
                    ("Accuracy", "accuracy"),
                    ("Balanced Accuracy", "balanced_accuracy"),
                    ("F1 Macro", "f1_macro"),
                    ("F1 Weighted", "f1_weighted"),
                    ("Precision Macro", "precision_macro"),
                    ("Recall Macro", "recall_macro"),
                    ("ROC AUC OVR Weighted", "roc_auc_ovr_weighted"),
                ])
            return options
        options = metadata.get("options") or []
        if options:
            return [(str(option), option) for option in options]
        if key == "method" and node.spec_id in {"grid_search", "random_search"}:
            return [("Grid Search", "grid"), ("Random Search", "random")]
        if key == "method" and node.spec_id == "forecast":
            return PARAMETER_CHOICES["method"]
        return PARAMETER_CHOICES.get(key, [])

    @staticmethod
    def _integer_parameter_range(key):
        return {
            "random_state": (0, 2_147_483_647),
            "cv": (2, 50),
            "n_iter": (1, 10_000),
            "trials": (1, 5_000),
            "timeout": (0, 86_400),
            "n_jobs": (-1, 128),
            "factor": (2, 100),
            "min_resources": (1, 2_147_483_647),
            "max_resources": (1, 2_147_483_647),
            "n_estimators": (1, 10_000),
            "n_clusters": (2, 10_000),
            "n_neighbors": (1, 100_000),
            "k_neighbors": (1, 100_000),
            "components": (1, 100_000),
            "k": (1, 100_000),
            "max_depth": (1, 10_000),
            "min_samples": (1, 100_000),
            "period": (1, 100_000),
        }.get(key, (-2_147_483_648, 2_147_483_647))

    @staticmethod
    def _float_parameter_range(key):
        if key in {"test_size", "contamination"}:
            return 0.001, 0.999, 3
        if key == "components":
            return 0.001, 1.0, 3
        return -1_000_000_000.0, 1_000_000_000.0, 6

    @staticmethod
    def _next_node_suggestion(spec_id):
        if spec_id in {"load_dataset", "select_target"}:
            return "Split Train/Test, lalu preprocessing dan Train."
        if spec_id in {"split", "time_split"}:
            return "Preprocessing pada cabang train; gunakan cabang test untuk Evaluasi."
        if spec_id in {"preprocess", "feature:scale", "feature:pca", "feature:select_k"}:
            return "Hubungkan hasil train ke Train dan hasil test ke Evaluasi."
        if spec_id.startswith("model:") or spec_id.startswith("extra_model:"):
            return "Hubungkan port estimator ke node Train atau Cross-Validation."
        if spec_id == "train":
            return "Tambahkan Evaluasi dan hubungkan data uji dari node Split."
        if spec_id in {"evaluate", "cross_validate"}:
            return "Bandingkan metrik dari beberapa model atau simpan graf eksperimen."
        return "Periksa port dan parameter node berikutnya pada panel Bantuan."

    def _set_parameter(self, node, key, value):
        if key in {"min_resources", "max_resources"} and isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if node.parameters.get(key) == value:
            return
        node.parameters[key] = value
        self.engine._cache.clear()
        self._schedule_history_snapshot()
        self.validate_graph(show_tab=False)
        if key == "task":
            self.show_parameters()

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

    def _missing_graph_packages(self, node_ids=None):
        package_target = self._package_target_path()
        if package_target is not None and package_target.is_dir():
            target = str(package_target)
            if target not in sys.path:
                sys.path.insert(0, target)
            importlib.invalidate_caches()
        required = set()
        selected_ids = set(node_ids) if node_ids is not None else set(self.engine.nodes)
        for node_id, node in self.engine.nodes.items():
            if node_id not in selected_ids:
                continue
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

    def validate_graph(self, show_tab=True, node_ids=None):
        active_ids = set(node_ids) if node_ids is not None else set(self.engine.nodes)
        connections = [
            edge for edge in self.engine.connections
            if edge[0] in active_ids and edge[2] in active_ids
        ]
        dataframe = self._current_dataset()
        issues = []

        def add_issue(severity, node_id, message, fix):
            issues.append({
                "severity": severity,
                "node_id": node_id,
                "message": message,
                "fix": fix,
            })

        if not active_ids:
            add_issue("error", None, "Graf belum memiliki node.", "Tambahkan node dari Palet Node atau pilih template.")
        else:
            try:
                _restore_graph(self._graph_subset_data(active_ids)).topological_order()
            except GraphError as exc:
                add_issue("error", None, str(exc), "Periksa arah sambungan dan hapus siklus pada graf.")
        if dataframe is None and any(
            self.engine.nodes[node_id].spec_id == "load_dataset"
            for node_id in active_ids if node_id in self.engine.nodes
        ):
            add_issue("error", None, "Dataset aktif belum dimuat.", "Muat dataset dari halaman Dataset sebelum menjalankan graf.")

        target_required = {
            "select_target", "train", "evaluate", "cross_validate", "grid_search", "random_search",
            "halving_grid_search", "halving_random_search", "optuna", "smote",
        }
        for node_id in active_ids:
            node = self.engine.nodes.get(node_id)
            if node is None:
                continue
            spec = self.engine.specs[node.spec_id]
            for port in spec.inputs:
                edges = [edge for edge in connections if edge[2] == node_id and edge[3] == port.name]
                if not edges:
                    add_issue(
                        "error", node_id,
                        f"Input '{port.name}' pada {spec.name} belum terhubung.",
                        "Hubungkan output node yang sesuai ke port ini.",
                    )
            target = node.parameters.get("target")
            task = node.parameters.get("task", "classification")
            if node.spec_id in target_required and task not in {"clustering", "anomaly", "anomaly_detection"} and not target:
                add_issue("error", node_id, f"Target untuk {spec.name} belum dipilih.", "Pilih kolom target pada panel Parameter.")
            if target and dataframe is not None and target not in dataframe.columns:
                add_issue("error", node_id, f"Kolom target '{target}' tidak ada pada dataset aktif.", "Pilih kembali kolom dari dataset yang sedang dimuat.")
            time_column = node.parameters.get("time_column")
            if node.spec_id == "time_split" and not time_column:
                add_issue("error", node_id, "Kolom waktu belum dipilih.", "Pilih kolom waktu pada panel Parameter.")
            if time_column and dataframe is not None and time_column not in dataframe.columns:
                add_issue("error", node_id, f"Kolom waktu '{time_column}' tidak ada pada dataset aktif.", "Pilih kembali kolom waktu dari dataset aktif.")

            if "test_size" in node.parameters:
                try:
                    test_size = float(node.parameters["test_size"])
                    if not 0 < test_size < 1:
                        raise ValueError
                except (TypeError, ValueError):
                    add_issue("error", node_id, "Test size harus berada di antara 0 dan 1.", "Gunakan nilai seperti 0.2 atau 0.25.")
            if "cv" in node.parameters:
                try:
                    if int(node.parameters["cv"]) < 2:
                        raise ValueError
                except (TypeError, ValueError):
                    add_issue("error", node_id, "Jumlah fold CV minimal 2.", "Naikkan nilai CV menjadi 2 atau lebih.")
            if "factor" in node.parameters:
                try:
                    if int(node.parameters["factor"]) < 2:
                        raise ValueError
                except (TypeError, ValueError):
                    add_issue("error", node_id, "Factor Successive Halving minimal 2.", "Gunakan factor 2 atau lebih.")
            if "timeout" in node.parameters:
                try:
                    if int(node.parameters["timeout"]) < 0:
                        raise ValueError
                except (TypeError, ValueError):
                    add_issue("error", node_id, "Timeout Optuna tidak boleh negatif.", "Gunakan 0 untuk tanpa batas waktu atau nilai positif dalam detik.")
            for resource_key, automatic in (("min_resources", {"smallest", "exhaust"}), ("max_resources", {"auto"})):
                if resource_key not in node.parameters:
                    continue
                resource_value = node.parameters[resource_key]
                if isinstance(resource_value, str) and resource_value.strip().lower() in automatic:
                    continue
                try:
                    if int(resource_value) < 1:
                        raise ValueError
                except (TypeError, ValueError):
                    add_issue("error", node_id, f"Nilai {resource_key.replace('_', ' ')} tidak valid.", "Gunakan pilihan otomatis atau bilangan bulat positif.")
            if node.spec_id in {"grid_search", "random_search", "halving_grid_search", "halving_random_search"}:
                grid = node.parameters.get("param_grid", {})
                if not isinstance(grid, dict):
                    add_issue("error", node_id, "Ruang param_grid harus berupa objek JSON.", "Masukkan pasangan nama parameter dan daftar nilainya.")
                elif grid:
                    for parameter, values in grid.items():
                        if not isinstance(values, (list, dict)) or (isinstance(values, list) and not values):
                            add_issue("error", node_id, f"Ruang parameter '{parameter}' tidak valid.", "Isi daftar nilai atau rentang distribusi yang didukung Random Search.")
                            break
            if node.spec_id == "optuna" and not isinstance(node.parameters.get("search_space", {}), dict):
                add_issue("error", node_id, "Ruang search_space harus berupa objek JSON.", "Masukkan tipe dan rentang parameter dalam objek JSON.")
            if node.spec_id in {"random_search", "halving_random_search", "optuna"}:
                try:
                    count = int(node.parameters.get("trials", node.parameters.get("n_iter", 1)) if node.spec_id == "optuna" else node.parameters.get("n_iter", 1))
                    if count < 1:
                        raise ValueError
                except (TypeError, ValueError):
                    add_issue("error", node_id, "Jumlah iterasi atau percobaan minimal 1.", "Naikkan n_iter atau trials menjadi 1 atau lebih.")

        for source_id, output_name, target_id, input_name in connections:
            source = self.engine.nodes.get(source_id)
            target = self.engine.nodes.get(target_id)
            if source is None or target is None:
                continue
            source_spec = self.engine.specs[source.spec_id]
            source_port_names = {port.name for port in source_spec.outputs}
            if target.spec_id == "train" and input_name == "train" and {"train", "test"}.issubset(source_port_names) and output_name != "train":
                add_issue("error", target_id, "Node Train menerima cabang data uji.", "Hubungkan port 'train' dari Split atau Preprocessing ke node Train.")
            if target.spec_id == "evaluate" and input_name == "test" and {"train", "test"}.issubset(source_port_names) and output_name != "test":
                add_issue("error", target_id, "Node Evaluasi menerima cabang data latih sebagai data uji.", "Hubungkan port 'test' dari Split atau Preprocessing ke node Evaluasi.")
            if target.spec_id == "train" and input_name == "estimator":
                task = self.engine.nodes[target_id].parameters.get("task", "classification")
                model_types = self.engine.specs[source.spec_id].task_types
                if task == "classification" and model_types and "all" not in model_types and not any("classification" in value for value in model_types):
                    add_issue("error", target_id, f"Estimator {self.engine.specs[source.spec_id].name} bukan model klasifikasi.", "Pilih estimator yang mendukung klasifikasi.")
                if task == "regression" and model_types and "all" not in model_types and "regression" not in model_types:
                    add_issue("error", target_id, f"Estimator {self.engine.specs[source.spec_id].name} bukan model regresi.", "Pilih estimator yang mendukung regresi.")

        for leakage in self.engine.detect_leakage():
            if leakage.get("node_id") in active_ids:
                add_issue("warning", leakage.get("node_id"), leakage["message"], leakage.get("suggested_fix", "Periksa urutan preprocessing."))

        self.problems_list.clear()
        for issue in issues:
            prefix = "Error" if issue["severity"] == "error" else "Peringatan"
            item = QListWidgetItem(f"{prefix}: {issue['message']}  —  {issue['fix']}")
            item.setData(Qt.UserRole, issue["node_id"])
            item.setToolTip(issue["message"] + "\n\n" + issue["fix"])
            item.setForeground(QColor("#B42318" if issue["severity"] == "error" else "#9A6700"))
            self.problems_list.addItem(item)
        error_count = sum(issue["severity"] == "error" for issue in issues)
        warning_count = len(issues) - error_count
        if error_count:
            self.validation_status.setText(f"{error_count} error · {warning_count} peringatan")
        elif warning_count:
            self.validation_status.setText(f"Valid · {warning_count} peringatan")
        else:
            self.validation_status.setText("Graf valid")
        self.validation_status.setToolTip("Double-click masalah untuk memusatkan kanvas pada node terkait.")
        self.result_tabs.setTabVisible(5, bool(issues))
        if show_tab and issues:
            self.result_tabs.setCurrentIndex(5)
        return {"valid": error_count == 0, "errors": error_count, "warnings": warning_count, "issues": issues}

    def _focus_problem_node(self, item):
        node_id = item.data(Qt.UserRole)
        graph_item = self.items.get(node_id)
        if graph_item is None:
            return
        self.scene.clearSelection()
        graph_item.setSelected(True)
        self.view.centerOn(graph_item)
        self.show_parameters()

    def _upstream_nodes(self, end_node_id):
        included = {end_node_id}
        pending = [end_node_id]
        while pending:
            current = pending.pop()
            for source_id, _output, target_id, _input in self.engine.connections:
                if target_id == current and source_id not in included:
                    included.add(source_id)
                    pending.append(source_id)
        return included

    def _graph_subset_data(self, node_ids):
        graph = self.engine.to_dict()
        graph["nodes"] = [node for node in graph["nodes"] if node["id"] in node_ids]
        graph["connections"] = [
            edge for edge in graph["connections"]
            if edge[0] in node_ids and edge[2] in node_ids
        ]
        return graph

    def run_to_selected_node(self):
        selected = [item for item in self.scene.selectedItems() if isinstance(item, GraphNodeItem)]
        if not selected:
            return
        self._start_graph_run(end_node_id=selected[0].node_id)

    def run_graph(self):
        self._start_graph_run()

    def _start_graph_run(self, end_node_id=None, export_model=None):
        if self._is_running:
            return
        self.sync_dataset()
        frame = self._current_dataset()
        if end_node_id is not None and end_node_id not in self.engine.nodes:
            QMessageBox.warning(self, "Node tidak ditemukan", "Pilih node yang masih ada pada graf.")
            return
        active_ids = (
            self._upstream_nodes(end_node_id)
            if end_node_id is not None else set(self.engine.nodes)
        )
        validation = self.validate_graph(show_tab=True, node_ids=active_ids)
        if not validation["valid"]:
            return

        leakages = [
            item for item in self.engine.detect_leakage()
            if item.get("node_id") in active_ids
        ]
        if leakages:
            reply = QMessageBox.question(
                self,
                "Data Leakage Terdeteksi",
                f"{leakages[0]['message']}\n\nApakah Anda ingin memperbaiki sambungan secara otomatis sebelum menjalankan?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.engine.auto_fix_leakage()
                self.redraw_cables()
                self._record_graph_history()
                active_ids = (
                    self._upstream_nodes(end_node_id)
                    if end_node_id is not None else set(self.engine.nodes)
                )
                validation = self.validate_graph(show_tab=True, node_ids=active_ids)
                if not validation["valid"]:
                    return
            else:
                return

        packages = self._missing_graph_packages(active_ids)
        package_target = self._package_target_path()
        self._clear_results()
        self.output_log.clear()
        self._active_run_node_ids = set(active_ids)
        for node_id, node in self.engine.nodes.items():
            node.status = "menunggu" if node_id in active_ids else "tidak dijalankan"
            item = self.items.get(node_id)
            if item is not None:
                item.set_status(node.status)
        if packages:
            self.result_tabs.setTabVisible(4, True)
            self.result_tabs.setCurrentWidget(self.output_log)
        self._completed_nodes.clear()
        self._run_node_count = len(active_ids)
        self._run_started_at = time.monotonic()
        self._pending_export_model = export_model
        self.run_progress_bar.setValue(0)
        self.run_progress_bar.show()
        self.run_spinner.show()
        if packages:
            self.run_status.setText(f"Memasang {len(packages)} library yang dibutuhkan...")
        elif end_node_id is not None:
            name = self.engine.specs[self.engine.nodes[end_node_id].spec_id].name
            self.run_status.setText(f"Menyiapkan alur sampai {name} ({self._run_node_count} node)...")
        else:
            self.run_status.setText(f"Menyiapkan graf ({self._run_node_count} node)...")
        self.run_status.show()
        self._spinner_index = 0
        self._advance_run_spinner()
        self._run_timer.start()
        self.thread = QThread(self)
        graph_data = self._graph_subset_data(active_ids)
        self.worker = _RunWorker(graph_data, frame, packages, package_target, export_model)
        self._set_running(True)
        thread, worker = self.thread, self.worker
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_status)
        self.worker.output.connect(self.append_output)
        self.worker.phase.connect(self.set_run_phase)
        self.worker.finished.connect(self.show_results)
        self.worker.failed.connect(self.show_error)
        self.worker.cancelled.connect(self.show_cancelled)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.cancelled.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    def cancel_run(self):
        if self.worker is None or not self.worker.can_cancel:
            return
        self.stop_button.setEnabled(False)
        self.run_status.setText(
            "Node aktif diselesaikan dulu…"
            if self.worker.cooperative_cancel else "Menghentikan proses graf…"
        )
        self.worker.cancel()

    def _set_running(self, running):
        self._is_running = bool(running)
        self.run_button.setEnabled(not running)
        self.run_to_button.setEnabled(not running and any(isinstance(item, GraphNodeItem) for item in self.scene.selectedItems()))
        self.stop_button.setEnabled(running and self.worker is not None and self.worker.can_cancel)
        self.validate_button.setEnabled(not running)
        self.delete_button.setEnabled(not running and any(isinstance(item, GraphNodeItem) or (isinstance(item, QGraphicsPathItem) and item.data(1) is not None) for item in self.scene.selectedItems()))
        self.duplicate_button.setEnabled(not running and any(isinstance(item, GraphNodeItem) for item in self.scene.selectedItems()))
        self.fit_button.setEnabled(not running)
        self.save_button.setEnabled(not running)
        self.load_button.setEnabled(not running)
        self.template_combo.setEnabled(not running)
        self.dataset_scope_combo.setEnabled(not running)
        self.save_snippet_button.setEnabled(not running)
        self.insert_snippet_button.setEnabled(not running)
        self.export_python_button.setEnabled(not running)
        self.export_notebook_button.setEnabled(not running)
        self.export_model_button.setEnabled(not running)
        self.splitter.setEnabled(not running)
        self._update_history_buttons()

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
        elif status == "cancelled":
            self.run_status.setText("Run dibatalkan")
        else:
            self.run_status.setText("Graf gagal dijalankan")

    @Slot(object)
    def show_results(self, output):
        output = output if isinstance(output, dict) else {}
        metric_rows = []
        comparison_metrics = {}
        forecast_metrics = None
        confusion_matrix = None
        self._result_outputs.clear()
        self.preview_node_combo.blockSignals(True)
        self.preview_node_combo.clear()
        self.preview_node_combo.addItem("Pilih output node…", None)
        for node_id, node in self.engine.nodes.items():
            if node_id in self.items:
                self.items[node_id].set_status(node.status)
        for node_id, result in output.items():
            if not isinstance(result, dict):
                continue
            if node_id == "__model_export__":
                export_path = result.get("path", "")
                self.append_output(f"Model berhasil diekspor: {export_path}\n")
                continue
            node = self.engine.nodes.get(node_id)
            node_name = self.engine.specs[node.spec_id].name if node is not None else node_id
            frame = result.get("table")
            if isinstance(frame, pd.DataFrame):
                if node is not None and node.spec_id == "compare":
                    frame = frame.drop(columns=list(NON_PERFORMANCE_METRICS), errors="ignore")
                output_key = f"{node_id}:table"
                self._result_outputs[output_key] = frame
                self.preview_node_combo.addItem(
                    f"{node_name} · {len(frame):,} × {len(frame.columns):,}", output_key
                )
            for summary_key, summary_label in (("model_summary", "Model terlatih"), ("estimator_summary", "Estimator")):
                summary = result.get(summary_key)
                if isinstance(summary, dict):
                    summary_frame = pd.DataFrame([summary])
                    output_key = f"{node_id}:{summary_key}"
                    self._result_outputs[output_key] = summary_frame
                    self.preview_node_combo.addItem(f"{node_name} · {summary_label}", output_key)
            metric_payload = result.get("metrics")
            if isinstance(metric_payload, dict):
                metrics = metric_payload.get("metrics", {})
                model_name = self._metric_model_name(
                    node_id, metric_payload.get("model", node_name)
                )
                if isinstance(metrics, dict):
                    clean_metrics = {
                        key: value for key, value in metrics.items()
                        if key not in NON_PERFORMANCE_METRICS
                    }
                    if clean_metrics:
                        metric_rows.append({"Node": node_name, "Model": model_name, **clean_metrics})
                        for key, value in clean_metrics.items():
                            try:
                                numeric_value = float(value)
                            except (TypeError, ValueError):
                                continue
                            if math.isfinite(numeric_value):
                                comparison_metrics[f"{model_name} · {key}"] = numeric_value
                    if metrics.get("confusion_matrix"):
                        confusion_matrix = (node_name, metrics["confusion_matrix"])
                    if node is not None and node.spec_id == "forecast":
                        forecast_metrics = clean_metrics
        self.preview_node_combo.blockSignals(False)
        if self._result_outputs:
            self.preview_node_combo.setCurrentIndex(1)
            self.result_tabs.setTabVisible(0, True)
        self._metric_rows = metric_rows
        self.metric_summary_model.blockSignals(True)
        self.metric_summary_model.clear()
        for row in metric_rows:
            self.metric_summary_model.addItem(str(row.get("Model", "Model")))
        self.metric_summary_model.blockSignals(False)
        self.metric_summary_model.setVisible(len(metric_rows) > 1)
        self.metric_summary_model_label.setVisible(len(metric_rows) > 1)
        if metric_rows:
            self.metric_summary_model.setCurrentIndex(0)
        self._render_metric_summary()
        self._display_table(self.metrics_results, pd.DataFrame(metric_rows))
        self.result_tabs.setTabVisible(1, bool(metric_rows))
        self.result_tabs.setTabVisible(2, confusion_matrix is not None)
        if confusion_matrix is not None:
            self._display_confusion_matrix(*confusion_matrix)
        if forecast_metrics is not None:
            self.forecast_chart.set_metrics(forecast_metrics)
            self.result_tabs.setTabVisible(1, True)
            self.result_tabs.setTabVisible(3, True)
        else:
            self.forecast_chart.set_metrics({})
            self.result_tabs.setTabVisible(3, False)
        self._record_run_history(comparison_metrics, "success")
        if metric_rows:
            self.result_tabs.setCurrentIndex(1)
        elif self._result_outputs:
            self.result_tabs.setCurrentIndex(0)
        self._finish_run_indicator("success")

    def _clear_results(self):
        self._result_outputs.clear()
        self.results.clear()
        self.results.setRowCount(0)
        self.results.setColumnCount(0)
        self.results.setToolTip("")
        self.metrics_results.clear()
        self.metrics_results.setRowCount(0)
        self.metrics_results.setColumnCount(0)
        self._metric_rows = []
        self.metric_summary_model.blockSignals(True)
        self.metric_summary_model.clear()
        self.metric_summary_model.blockSignals(False)
        self.metric_summary_model.setVisible(False)
        self.metric_summary_model_label.setVisible(False)
        self._render_metric_summary()
        self.confusion_matrix_table.clear()
        self.confusion_matrix_table.setRowCount(0)
        self.confusion_matrix_table.setColumnCount(0)
        self.preview_node_combo.blockSignals(True)
        self.preview_node_combo.clear()
        self.preview_node_combo.addItem("Pilih output node…", None)
        self.preview_node_combo.blockSignals(False)
        self.forecast_chart.set_metrics({})
        self.result_tabs.setCurrentIndex(0)
        self.result_tabs.setTabVisible(1, False)
        self.result_tabs.setTabVisible(2, False)
        self.result_tabs.setTabVisible(3, False)
        self.result_tabs.setTabVisible(4, False)

    def _display_frame(self, frame):
        self._display_table(self.results, frame)

    def _display_table(self, table, frame):
        if frame is None or not hasattr(frame, "columns"):
            table.clear()
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        preview = frame.iloc[:RESULT_PREVIEW_ROWS, :RESULT_PREVIEW_COLUMNS]
        table.setUpdatesEnabled(False)
        try:
            table.setColumnCount(len(preview.columns))
            table.setRowCount(len(preview))
            headers = [
                METRIC_LABELS.get(str(column), str(column))
                if table is self.metrics_results else str(column)
                for column in preview.columns
            ]
            table.setHorizontalHeaderLabels(headers)
            for row, values in enumerate(preview.itertuples(index=False, name=None)):
                scoring = preview.iloc[row].get("scoring")
                for column, value in enumerate(values):
                    raw_value = str(value)
                    if table is self.metrics_results:
                        value = self._format_metric_value(str(preview.columns[column]), value, scoring)
                    item = QTableWidgetItem(str(value))
                    item.setToolTip(raw_value)
                    table.setItem(row, column, item)
        finally:
            table.setUpdatesEnabled(True)
        if table is self.results and (len(frame) > RESULT_PREVIEW_ROWS or len(frame.columns) > RESULT_PREVIEW_COLUMNS):
            table.setToolTip(
                f"Pratinjau {len(preview):,} dari {len(frame):,} baris dan "
                f"{len(preview.columns):,} dari {len(frame.columns):,} kolom."
            )
        else:
            table.setToolTip("")

    def _format_metric_value(self, key, value, scoring=None):
        if isinstance(value, bool):
            return "Ya" if value else "Tidak"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value
        if not math.isfinite(number):
            return "—"
        if key in {"best_score", "mean_score"} and scoring in PERCENT_SCORING:
            return f"{number * 100:.1f}%"
        if key in {"best_score", "mean_score"} and isinstance(scoring, str) and scoring.startswith("neg_"):
            return f"{number:.4f} ({scoring})"
        if key in PERCENT_METRICS:
            return f"{number * 100:.1f}%"
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:.4f}".rstrip("0").rstrip(".")

    def _make_summary_card(self, label, value, tooltip=""):
        card = QFrame()
        card.setObjectName("nodeEditorMetricCard")
        card.setToolTip(tooltip)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)
        title = QLabel(str(label))
        title.setObjectName("nodeEditorMetricLabel")
        value_label = QLabel(str(value))
        value_label.setObjectName("nodeEditorMetricValue")
        value_label.setWordWrap(True)
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        card.setMinimumWidth(100)
        card.setMaximumWidth(230)
        layout.addWidget(title)
        layout.addWidget(value_label)
        return card

    def _render_metric_summary(self, *_):
        while self.metric_summary_layout.count():
            item = self.metric_summary_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        if not self._metric_rows:
            empty = QLabel("Jalankan alur untuk melihat metrik evaluasi.")
            empty.setObjectName("nodeEditorEmptyState")
            self.metric_summary_layout.addWidget(empty)
            return
        index = self.metric_summary_model.currentIndex()
        index = index if 0 <= index < len(self._metric_rows) else 0
        row = self._metric_rows[index]
        available = [key for key in row if key not in {"Node", "Model"}]
        if "accuracy" in available:
            preferred = ["accuracy", "f1", "precision", "recall", "balanced_accuracy", "f1_macro", "f1_weighted"]
        elif any(key in available for key in {"f1_macro", "f1_weighted", "balanced_accuracy", "roc_auc"}):
            preferred = ["f1_macro", "f1_weighted", "balanced_accuracy", "roc_auc", "precision", "recall"]
        elif "mae" in available or "rmse" in available:
            preferred = ["r2", "mae", "rmse"]
        elif "silhouette" in available or "cluster_count" in available:
            preferred = ["silhouette", "calinski_harabasz", "davies_bouldin", "cluster_count"]
        else:
            preferred = available
        keys = [key for key in preferred if key in available][:4]
        for key in keys:
            self.metric_summary_layout.addWidget(
                self._make_summary_card(
                    METRIC_LABELS.get(key, key.replace("_", " ").title()),
                    self._format_metric_value(key, row[key]),
                    f"{row.get('Model', 'Model')} · {METRIC_LABELS.get(key, key)}",
                ),
                1,
            )
        self.metric_summary_layout.addStretch(1)

    def _show_selected_result(self, index):
        key = self.preview_node_combo.itemData(index)
        frame = self._result_outputs.get(key)
        if frame is not None:
            self._display_frame(frame)

    def _metric_model_name(self, node_id, fallback):
        source_id = next((
            edge[0] for edge in self.engine.connections
            if edge[2] == node_id and edge[3] in {"model", "estimator"}
        ), None)
        source = self.engine.nodes.get(source_id)
        if source is not None and source.spec_id == "train":
            source_id = next((
                edge[0] for edge in self.engine.connections
                if edge[2] == source_id and edge[3] == "estimator"
            ), None)
            source = self.engine.nodes.get(source_id)
        if source is not None:
            return self.engine.specs[source.spec_id].name
        return fallback

    def _display_confusion_matrix(self, node_name, values):
        matrix = pd.DataFrame(values)
        matrix.index = [f"Aktual {index}" for index in matrix.index]
        matrix.columns = [f"Prediksi {index}" for index in matrix.columns]
        self._display_table(self.confusion_matrix_table, matrix)
        self.confusion_matrix_table.setToolTip(f"Confusion matrix dari {node_name}.")

    def _record_run_history(self, metrics, status):
        duration = 0.0
        if self._run_started_at is not None:
            duration = round((time.monotonic() - self._run_started_at) * 1000, 1)
        dataframe = self._current_dataset()
        try:
            dataset_hash = compute_dataframe_hash(dataframe) if dataframe is not None else ""
        except Exception:
            dataset_hash = ""
        task, target = self._export_context()
        try:
            filename = self._current_dataset_name()
        except Exception:
            filename = "Dataset aktif"
        graph_version = hashlib.sha256(self._graph_state_key().encode("utf-8")).hexdigest()[:12]
        entry = {
            "id": f"run-{uuid.uuid4().hex[:8]}",
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "task": task,
            "target": target,
            "dataset_hash": dataset_hash,
            "dataset": filename,
            "duration_ms": duration,
            "status": status,
            "metrics": dict(metrics or {}),
            "graph_version_id": graph_version,
            "random_state": next((node.parameters.get("random_state", 42) for node in self.engine.nodes.values() if "random_state" in node.parameters), 42),
            "notes": "",
        }
        self._run_history.insert(0, entry)
        del self._run_history[100:]
        self._refresh_run_history()
        self._schedule_session_graph_save()

    def _refresh_run_history(self):
        self.history_table.setRowCount(len(self._run_history))
        for row, run in enumerate(self._run_history):
            values = [run["id"], run["created_at"], run["task"], run["target"], run["dataset"]]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value or "—"))
                item.setData(Qt.UserRole, run["id"])
                if column == 0:
                    item.setToolTip(f"Status: {run['status']} · Durasi: {run['duration_ms']} ms")
                self.history_table.setItem(row, column, item)

    def compare_selected_runs(self):
        rows = sorted({index.row() for index in self.history_table.selectionModel().selectedRows()})
        if len(rows) != 2:
            QMessageBox.information(self, "Pilih dua run", "Pilih tepat dua baris riwayat untuk dibandingkan.")
            return
        first, second = (self._run_history[index] for index in rows)
        keys = sorted(set(first["metrics"]) | set(second["metrics"]))
        self.history_comparison.setRowCount(0)
        for key in keys:
            a, b = first["metrics"].get(key), second["metrics"].get(key)
            try:
                delta = float(b) - float(a)
                delta_text = f"{delta:+.6g}"
            except (TypeError, ValueError):
                delta_text = "—"
            values = [key, a if a is not None else "—", b if b is not None else "—", delta_text]
            row = self.history_comparison.rowCount()
            self.history_comparison.insertRow(row)
            for column, value in enumerate(values):
                self.history_comparison.setItem(row, column, QTableWidgetItem(str(value)))
        self.result_tabs.setCurrentIndex(6)

    def export_run_history(self):
        if not self._run_history:
            QMessageBox.information(self, "Riwayat kosong", "Jalankan graf terlebih dahulu untuk membuat riwayat.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Ekspor Riwayat Run", "riwayat-node.csv", "CSV (*.csv)")
        if path:
            export_runs_csv(self._run_history, path)

    def show_cancelled(self):
        for node_id in self._active_run_node_ids:
            node = self.engine.nodes.get(node_id)
            if node is not None and node.status in {"menunggu", "berjalan"}:
                node.status = "dilewati"
                if node_id in self.items:
                    self.items[node_id].set_status("dilewati")
        self.append_output("\nRun dibatalkan oleh pengguna.\n")
        self._record_run_history({}, "cancelled")
        self._finish_run_indicator("cancelled")

    @Slot(str)
    def show_error(self, message):
        for node_id, node in self.engine.nodes.items():
            if node_id in self.items:
                self.items[node_id].set_status(node.status)
        self.append_output(f"\nRun gagal: {message}\n")
        self.result_tabs.setCurrentWidget(self.output_log)
        self._record_run_history({}, "failed")
        self._finish_run_indicator("error")
        QMessageBox.critical(self, "Graf gagal dijalankan", message)

    def save_graph(self):
        path, _ = QFileDialog.getSaveFileName(self, "Simpan Graf", "graf-node.json", "JSON (*.json)")
        if path:
            self._history_timer.stop()
            self._record_graph_history()
            graph_document = self.engine.to_dict()
            graph_document["dataset_source"] = self._dataset_source_document()
            Path(path).write_text(
                json.dumps(graph_document, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._saved_graph_state = self._graph_document_key()
            self._mark_graph_dirty()

    def _export_context(self):
        target = ""
        task = ""
        for node in self.engine.nodes.values():
            target = target or str(node.parameters.get("target", ""))
            if node.spec_id == "forecast":
                task = "time_series"
            elif not task and node.parameters.get("task"):
                task = str(node.parameters["task"])
        if not task:
            detected = getattr(self, "detected_task", {}) or {}
            task_type = str(detected.get("task_type", ""))
            task = "regression" if task_type == "regression" else (
                "clustering" if task_type == "clustering" else "classification"
            )
        if not target:
            detected = getattr(self, "detected_task", {}) or {}
            target = str(detected.get("target_column") or "target")
        return task, target

    def export_python(self):
        path, _ = QFileDialog.getSaveFileName(self, "Ekspor Pipeline Python", "pipeline-node.py", "Python (*.py)")
        if not path:
            return
        task, target = self._export_context()
        Path(path).write_text(export_python_script(self.engine.to_dict(), target, task), encoding="utf-8")
        self.append_output(f"Script Python diekspor: {path}\n")

    def export_notebook(self):
        path, _ = QFileDialog.getSaveFileName(self, "Ekspor Notebook", "pipeline-node.ipynb", "Jupyter Notebook (*.ipynb)")
        if not path:
            return
        task, target = self._export_context()
        notebook = export_jupyter_notebook(self.engine.to_dict(), target, task)
        Path(path).write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
        self.append_output(f"Notebook diekspor: {path}\n")

    def export_model(self):
        train_nodes = [node for node in self.engine.nodes.values() if node.spec_id == "train"]
        if not train_nodes:
            QMessageBox.information(self, "Node Train belum ada", "Tambahkan dan hubungkan node Train sebelum mengekspor model.")
            return
        labels = [f"{node.id} · {node.parameters.get('target') or 'target belum dipilih'}" for node in train_nodes]
        selected, accepted = QInputDialog.getItem(self, "Pilih hasil model", "Node Train:", labels, 0, False)
        if not accepted:
            return
        train_node = train_nodes[labels.index(selected)]
        path, _ = QFileDialog.getSaveFileName(self, "Simpan Model Terlatih", f"{train_node.id}-model.pkl", "Model (*.pkl)")
        if path:
            self._start_graph_run(
                end_node_id=train_node.id,
                export_model={"node_id": train_node.id, "path": path},
            )

    def save_selected_snippet(self):
        node_ids = {
            item.node_id for item in self.scene.selectedItems()
            if isinstance(item, GraphNodeItem)
        }
        if not node_ids:
            QMessageBox.information(self, "Pilih node", "Pilih satu atau lebih node untuk menyimpan potongan alur.")
            return
        graph = self.engine.to_dict()
        snippet = {
            "version": graph.get("version", 1),
            "nodes": [node for node in graph["nodes"] if node["id"] in node_ids],
            "connections": [edge for edge in graph["connections"] if edge[0] in node_ids and edge[2] in node_ids],
        }
        path, _ = QFileDialog.getSaveFileName(self, "Simpan Potongan Graf", "potongan-node.json", "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(snippet, ensure_ascii=False, indent=2), encoding="utf-8")

    def insert_graph_snippet(self):
        path, _ = QFileDialog.getOpenFileName(self, "Sisipkan Potongan Graf", "", "JSON (*.json)")
        if not path:
            return
        try:
            snippet = json.loads(Path(path).read_text(encoding="utf-8"))
            raw_nodes = snippet.get("nodes", [])
            if not raw_nodes:
                raise ValueError("File potongan tidak memiliki node.")
            existing = self.engine.to_dict()
            used_ids = {node["id"] for node in existing["nodes"]}
            id_map = {}
            next_index = 1
            for raw in raw_nodes:
                while f"n{next_index}" in used_ids:
                    next_index += 1
                new_id = f"n{next_index}"
                next_index += 1
                used_ids.add(new_id)
                id_map[raw["id"]] = new_id
            positions = [tuple(raw.get("position", (0, 0))) for raw in raw_nodes]
            min_x = min(position[0] for position in positions)
            min_y = min(position[1] for position in positions)
            target = self.view.mapToScene(self.view.viewport().rect().center())
            offset_x, offset_y = target.x() - min_x, target.y() - min_y
            inserted = []
            for raw in raw_nodes:
                x, y = raw.get("position", (0, 0))
                inserted.append({
                    **raw,
                    "id": id_map[raw["id"]],
                    "position": (x + offset_x, y + offset_y),
                })
            combined = {
                **existing,
                "nodes": existing["nodes"] + inserted,
                "connections": existing["connections"] + [
                    [id_map[edge[0]], edge[1], id_map[edge[2]], edge[3]]
                    for edge in snippet.get("connections", [])
                    if edge[0] in id_map and edge[2] in id_map
                ],
            }
            self.engine = _restore_graph(combined)
            self.scene.clear()
            self.items.clear()
            self.cables.clear()
            for node in self.engine.nodes.values():
                item = GraphNodeItem(self, node, *node.position)
                self.items[node.id] = item
                self.scene.addItem(item)
            self.redraw_cables()
            self.refresh_palette()
            self._record_graph_history()
            self.validate_graph(show_tab=False)
        except Exception as exc:
            QMessageBox.critical(self, "Potongan tidak dapat disisipkan", str(exc))

    def load_graph(self):
        path, _ = QFileDialog.getOpenFileName(self, "Muat Graf", "", "JSON (*.json)")
        if path:
            try:
                graph_document = json.loads(Path(path).read_text(encoding="utf-8"))
                self.engine = GraphEngine.load(path)
                self._restore_dataset_source(graph_document.get("dataset_source", {}))
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
                self._reset_graph_history(saved=True)
                QTimer.singleShot(0, self.fit_graph)
            except Exception as exc:
                QMessageBox.critical(self, "Graf tidak dapat dimuat", str(exc))

    def load_template(self, index):
        if index <= 0:
            return
        tmpl_data = self.template_combo.itemData(index) if hasattr(self, "template_combo") else None
        if tmpl_data and isinstance(tmpl_data, dict) and "graph" in tmpl_data:
            graph_def = tmpl_data["graph"]
            graph = GraphEngine()
            for n in graph_def.get("nodes", []):
                graph.add_node(n["spec_id"], n["id"], n.get("parameters"), tuple(n.get("position", (0, 0))))
            for edge in graph_def.get("connections", []):
                graph.connect(*edge)
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
            self._reset_graph_history(saved=True)
            QTimer.singleShot(0, self.fit_graph)
            return

        names = ["", "Klasifikasi cepat", "Regresi cepat", "Ensemble klasifikasi", "Imbalanced klasifikasi", "Forecasting cepat"]
        if index >= len(names):
            return
        name = names[index]
        graph = GraphEngine()
        active = self._current_dataset()
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
        self._reset_graph_history(saved=True)
        QTimer.singleShot(0, self.fit_graph)
