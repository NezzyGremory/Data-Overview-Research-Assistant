from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QStandardPaths, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QApplication,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from app.ml.registry import ModelRegistry
FORECAST_METHODS = [
    ("Naive Forecast", "naive", None),
    ("Seasonal Naive", "seasonal_naive", None),
    ("ARIMA", "arima", "statsmodels"),
    ("Exponential Smoothing", "exponential_smoothing", "statsmodels"),
    ("SARIMAX", "sarimax", "statsmodels"),
]


class MLRunOptionsDialog(QDialog):
    TASKS = [
        ("Klasifikasi biner", "binary_classification"),
        ("Klasifikasi multikelas", "multiclass_classification"),
        ("Regresi", "regression"),
        ("Clustering", "clustering"),
        ("Deteksi anomali", "anomaly_detection"),
        ("Forecasting time-series", "time_series_forecasting"),
    ]

    def __init__(self, dataframe: pd.DataFrame, task_result: dict, parent=None):
        super().__init__(parent)
        if getattr(sys, "frozen", False):
            app_data = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
            app_data_root = Path(app_data) if app_data else Path.home() / ".dora"
            package_target = str(app_data_root / "node_editor" / "site-packages")
            if Path(package_target).is_dir() and package_target not in sys.path:
                sys.path.insert(0, package_target)
                importlib.invalidate_caches()
        self.dataframe = dataframe
        self.task_result = task_result or {}
        self._items: dict[str, QListWidgetItem] = {}
        self.setWindowTitle("Opsi Machine Learning")
        screen = parent.screen() if parent is not None else QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        max_width = max(360, available.width() - 32) if available is not None else 650
        max_height = max(420, available.height() - 32) if available is not None else 720
        self.setMinimumSize(min(570, max_width), min(520, max_height))
        self.setMaximumSize(max_width, max_height)
        self.resize(min(650, max_width), min(720, max_height))

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 20)
        root.setSpacing(14)

        title = QLabel("Atur task dan algoritma")
        title.setObjectName("sectionTitle")
        intro = QLabel(
            "Mode otomatis tetap paling cepat. Pilih mode manual untuk menentukan task "
            "dan menjalankan satu atau beberapa algoritma."
        )
        intro.setObjectName("cardDescription")
        intro.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(intro)

        self.auto_check = QCheckBox("Gunakan rekomendasi otomatis (lebih cepat)")
        self.auto_check.setChecked(True)
        root.addWidget(self.auto_check)

        fields = QGridLayout()
        fields.setHorizontalSpacing(12)
        fields.setVerticalSpacing(10)
        fields.addWidget(QLabel("Task"), 0, 0)
        self.task_combo = QComboBox()
        for label, task in self.TASKS:
            self.task_combo.addItem(label, task)
        detected = (self.task_result.get("primary_task") or {}).get("task")
        index = self.task_combo.findData(detected)
        if index >= 0:
            self.task_combo.setCurrentIndex(index)
        fields.addWidget(self.task_combo, 0, 1)

        fields.addWidget(QLabel("Target"), 1, 0)
        self.target_combo = QComboBox()
        fields.addWidget(self.target_combo, 1, 1)

        fields.addWidget(QLabel("Kolom waktu"), 2, 0)
        self.time_combo = QComboBox()
        fields.addWidget(self.time_combo, 2, 1)
        root.addLayout(fields)

        methods_heading = QHBoxLayout()
        methods_heading.addWidget(QLabel("Algoritma"), 1)
        self.select_all_button = QPushButton("Pilih semua")
        self.select_all_button.setObjectName("secondaryButton")
        self.clear_button = QPushButton("Kosongkan")
        self.clear_button.setObjectName("secondaryButton")
        methods_heading.addWidget(self.select_all_button)
        methods_heading.addWidget(self.clear_button)
        root.addLayout(methods_heading)

        self.model_list = QListWidget()
        self.model_list.setMinimumHeight(170)
        root.addWidget(self.model_list, 1)

        self.hint = QLabel("Library opsional akan diunduh jika belum tersedia. Beberapa model dapat memerlukan waktu lebih lama.")
        self.hint.setObjectName("cardDescription")
        self.hint.setWordWrap(True)
        root.addWidget(self.hint)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Batal")
        cancel.setObjectName("secondaryButton")
        run = QPushButton("Jalankan")
        run.setObjectName("primaryButton")
        buttons.addWidget(cancel)
        buttons.addWidget(run)
        root.addLayout(buttons)

        self.auto_check.toggled.connect(self._sync_mode)
        self.task_combo.currentIndexChanged.connect(self._populate_task_controls)
        self.select_all_button.clicked.connect(lambda: self._set_all(Qt.CheckState.Checked))
        self.clear_button.clicked.connect(lambda: self._set_all(Qt.CheckState.Unchecked))
        cancel.clicked.connect(self.reject)
        run.clicked.connect(self._accept_options)
        self._populate_task_controls()
        self._sync_mode()

    def options(self) -> dict:
        automatic = self.auto_check.isChecked()
        task = self.task_combo.currentData()
        return {
            "task_override": None if automatic else task,
            "target_override": None if automatic or task in {"clustering", "anomaly_detection"} else self.target_combo.currentData(),
            "time_column": None if automatic or task != "time_series_forecasting" else self.time_combo.currentData(),
            "requested_methods": None if automatic else [
                method_id for method_id, item in self._items.items()
                if item.checkState() == Qt.CheckState.Checked
            ],
        }

    def _populate_task_controls(self):
        task = self.task_combo.currentData()
        self.target_combo.clear()
        if task in {"binary_classification", "multiclass_classification", "regression", "time_series_forecasting"}:
            for column in self.dataframe.columns:
                if task in {"regression", "time_series_forecasting"} and not pd.api.types.is_numeric_dtype(self.dataframe[column]):
                    continue
                self.target_combo.addItem(str(column), str(column))
            detected_target = (self.task_result.get("primary_task") or {}).get("target")
            target_index = self.target_combo.findData(detected_target)
            if target_index >= 0:
                self.target_combo.setCurrentIndex(target_index)

        self.time_combo.clear()
        for column in self.dataframe.columns:
            series = self.dataframe[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                self.time_combo.addItem(str(column), str(column))
            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                sample = series.head(300).dropna()
                if not sample.empty:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() >= 0.8:
                        self.time_combo.addItem(str(column), str(column))
        self.model_list.clear()
        self._items.clear()
        if task == "time_series_forecasting":
            entries = [(name, method_id, package) for name, method_id, package in FORECAST_METHODS]
        else:
            entries = [
                (spec.name, spec.id, spec.check_package)
                for spec in ModelRegistry.get_models_for_task(task, available_only=False)
            ]
        for name, method_id, package in entries:
            available = self._package_available(package)
            suffix = "tersedia" if available else f"perlu {package}"
            item = QListWidgetItem(f"{name}  ·  {suffix}")
            item.setData(Qt.ItemDataRole.UserRole, method_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            if not available:
                item.setForeground(QColor("#73839B"))
            self.model_list.addItem(item)
            self._items[method_id] = item

        self._sync_mode()

    @staticmethod
    def _package_available(package: str | None) -> bool:
        if not package:
            return True
        try:
            return importlib.util.find_spec(package) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def _set_all(self, state):
        for item in self._items.values():
            item.setCheckState(state)

    def _sync_mode(self):
        manual = not self.auto_check.isChecked()
        self.task_combo.setEnabled(manual)
        task = self.task_combo.currentData()
        self.target_combo.setEnabled(manual and task in {"binary_classification", "multiclass_classification", "regression", "time_series_forecasting"})
        self.time_combo.setEnabled(manual and task == "time_series_forecasting")
        self.model_list.setEnabled(manual)
        self.select_all_button.setEnabled(manual)
        self.clear_button.setEnabled(manual)

    def _accept_options(self):
        if not self.auto_check.isChecked():
            task = self.task_combo.currentData()
            if task in {"binary_classification", "multiclass_classification", "regression", "time_series_forecasting"} and not self.target_combo.currentData():
                QMessageBox.warning(self, "Target diperlukan", "Pilih kolom target untuk task ini.")
                return
            if task == "time_series_forecasting" and not self.time_combo.currentData():
                QMessageBox.warning(self, "Kolom waktu diperlukan", "Pilih kolom tanggal atau waktu untuk forecasting.")
                return
            if not self.options()["requested_methods"]:
                QMessageBox.warning(self, "Algoritma belum dipilih", "Pilih minimal satu algoritma untuk dijalankan.")
                return
        self.accept()
