from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
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
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.ml.registry import ModelRegistry
FORECAST_METHODS = [
    ("Naive Forecast", "naive", None),
    ("Seasonal Naive", "seasonal_naive", None),
    ("ARIMA", "arima", "statsmodels"),
    ("Exponential Smoothing", "exponential_smoothing", "statsmodels"),
    ("SARIMAX", "sarimax", "statsmodels"),
]


class MLRunOptionsPanel(QWidget):
    TASKS = [
        ("Klasifikasi biner", "binary_classification"),
        ("Klasifikasi multikelas", "multiclass_classification"),
        ("Regresi", "regression"),
        ("Clustering", "clustering"),
        ("Deteksi anomali", "anomaly_detection"),
        ("Forecasting time-series", "time_series_forecasting"),
    ]

    def __init__(
        self,
        dataframe: pd.DataFrame,
        task_result: dict,
        parent=None,
        embedded=True,
        on_submit=None,
        on_cancel=None,
    ):
        super().__init__(parent)
        if getattr(sys, "frozen", False):
            from app.core.runtime_packages import runtime_package_directory

            package_target = str(runtime_package_directory())
            if Path(package_target).is_dir() and package_target not in sys.path:
                sys.path.insert(0, package_target)
                importlib.invalidate_caches()
        self.dataframe = dataframe
        self.task_result = task_result or {}
        self.embedded = embedded
        self.on_submit = on_submit
        self.on_cancel = on_cancel
        self._items: dict[str, QListWidgetItem] = {}
        self.setObjectName("mlRunOptionsPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        root = QVBoxLayout(self)
        root.setContentsMargins(
            20 if embedded else 24,
            18 if embedded else 22,
            20 if embedded else 24,
            18 if embedded else 20,
        )
        root.setSpacing(12 if embedded else 14)

        title = QLabel("Training Model" if embedded else "Atur task dan algoritma")
        title.setObjectName("sectionTitle")
        intro = QLabel(
            (
                "Pilih task, target, dan fitur. Sistem membandingkan algoritma "
                "yang cocok lalu menampilkan hasil evaluasinya."
            )
            if embedded
            else (
                "Tentukan tujuan analisis dan kolom target sebelum training. "
                "Dataset yang sama dapat dipakai untuk klasifikasi atau clustering."
            )
        )
        intro.setObjectName("cardDescription")
        intro.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(intro)

        primary = self.task_result.get("primary_task") or {}
        candidates = self.task_result.get("target_candidates") or []
        suggested = primary.get("label") or "belum dapat dipastikan"
        suggested_target = primary.get("target") or "belum ditentukan"
        self.suggestion_label = QLabel(
            f"Saran sistem: {suggested} | target: {suggested_target}. "
            "Ini perkiraan dari struktur data; pilihan Anda menentukan analisis yang dijalankan."
        )
        self.suggestion_label.setObjectName("cardDescription")
        self.suggestion_label.setWordWrap(True)
        root.addWidget(self.suggestion_label)
        if candidates:
            names = [str(item.get("name") or item.get("column") or "") for item in candidates[:3]]
            names = [name for name in names if name]
            if names:
                candidate_label = QLabel("Kandidat target: " + ", ".join(names))
                candidate_label.setObjectName("cardDescription")
                candidate_label.setWordWrap(True)
                root.addWidget(candidate_label)

        self.auto_check = QCheckBox(
            "Rekomendasikan algoritma otomatis" if embedded else "Pilih algoritma otomatis"
        )
        self.auto_check.setChecked(True)
        root.addWidget(self.auto_check)

        fields = QVBoxLayout() if embedded else QGridLayout()
        if embedded:
            fields.setSpacing(10)
        else:
            fields.setHorizontalSpacing(12)
            fields.setVerticalSpacing(10)
        self.task_combo = QComboBox()
        self.task_combo.addItem("Pilih task analisis...", None)
        for label, task in self.TASKS:
            self.task_combo.addItem(label, task)
        detected = (self.task_result.get("primary_task") or {}).get("task")
        index = self.task_combo.findData(detected)
        if index > 0 and not (detected == "clustering" and candidates):
            self.task_combo.setCurrentIndex(index)

        self.target_combo = QComboBox()
        self.time_combo = QComboBox()
        self.test_percent = QSpinBox()
        self.test_percent.setRange(10, 40)
        self.test_percent.setValue(20)
        self.test_percent.setSuffix(" %")
        self.test_percent.setToolTip(
            "Persentase data yang disisihkan untuk evaluasi akhir model."
        )
        self.cv_folds = QSpinBox()
        self.cv_folds.setRange(2, 10)
        self.cv_folds.setValue(3)
        self.cv_folds.setToolTip(
            "Jumlah lipatan validasi silang untuk membandingkan model."
        )

        self.time_label = QLabel("Kolom waktu")
        if embedded:
            task_target_row = QHBoxLayout()
            task_target_row.setSpacing(12)
            task_target_row.addWidget(QLabel("Task"))
            task_target_row.addWidget(self.task_combo, 1)
            task_target_row.addWidget(QLabel("Target"))
            task_target_row.addWidget(self.target_combo, 1)
            fields.addLayout(task_target_row)

            self.time_row_widget = QWidget()
            time_row = QHBoxLayout(self.time_row_widget)
            time_row.setContentsMargins(0, 0, 0, 0)
            time_row.setSpacing(12)
            time_row.addWidget(self.time_label)
            time_row.addWidget(self.time_combo, 1)
            time_row.addStretch(1)
            fields.addWidget(self.time_row_widget)

            validation_row = QHBoxLayout()
            validation_row.setSpacing(12)
            validation_row.addWidget(QLabel("Data uji akhir"))
            validation_row.addWidget(self.test_percent)
            validation_row.addStretch(1)
            validation_row.addWidget(QLabel("Fold validasi"))
            validation_row.addWidget(self.cv_folds)
            fields.addLayout(validation_row)
            self.test_percent.setFixedWidth(144)
            self.cv_folds.setFixedWidth(144)
        else:
            self.time_row_widget = None
            fields.addWidget(QLabel("Task"), 0, 0)
            fields.addWidget(self.task_combo, 0, 1)
            fields.addWidget(QLabel("Target"), 1, 0)
            fields.addWidget(self.target_combo, 1, 1)
            fields.addWidget(self.time_label, 2, 0)
            fields.addWidget(self.time_combo, 2, 1)
            fields.addWidget(QLabel("Data uji akhir"), 3, 0)
            fields.addWidget(self.test_percent, 3, 1)
            fields.addWidget(QLabel("Fold validasi"), 4, 0)
            fields.addWidget(self.cv_folds, 4, 1)
        root.addLayout(fields)

        tabs = QTabWidget()
        models_tab = QWidget()
        models_layout = QVBoxLayout(models_tab)
        methods_heading = QHBoxLayout()
        methods_heading.addWidget(QLabel("Algoritma"), 1)
        self.select_all_button = QPushButton("Pilih semua")
        self.select_all_button.setObjectName("secondaryButton")
        self.clear_button = QPushButton("Kosongkan")
        self.clear_button.setObjectName("secondaryButton")
        methods_heading.addWidget(self.select_all_button)
        methods_heading.addWidget(self.clear_button)
        models_layout.addLayout(methods_heading)

        self.model_list = QListWidget()
        self.model_list.setMinimumHeight(120)
        models_layout.addWidget(self.model_list, 1)
        tabs.addTab(models_tab, "Algoritma")

        features_tab = QWidget()
        features_layout = QVBoxLayout(features_tab)
        feature_hint = QLabel("Centang fitur yang dipakai. Target dan kandidat label clustering tidak dicentang otomatis.")
        feature_hint.setObjectName("cardDescription")
        features_layout.addWidget(feature_hint)
        self.feature_list = QListWidget()
        self.feature_list.setMinimumHeight(120)
        features_layout.addWidget(self.feature_list, 1)
        tabs.addTab(features_tab, "Fitur")
        root.addWidget(tabs, 1)

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
        if embedded:
            cancel.setText("Kembali")
            run.setText("Latih dan tampilkan rekomendasi")
        buttons.addWidget(cancel)
        buttons.addWidget(run)
        root.addLayout(buttons)

        self.auto_check.toggled.connect(lambda _checked=False: self._sync_mode())
        self.task_combo.currentIndexChanged.connect(lambda _index: self._populate_task_controls())
        self.target_combo.currentIndexChanged.connect(lambda _index: self._populate_features())
        self.select_all_button.clicked.connect(
            lambda _checked=False: self._set_all(Qt.CheckState.Checked)
        )
        self.clear_button.clicked.connect(
            lambda _checked=False: self._set_all(Qt.CheckState.Unchecked)
        )
        cancel.clicked.connect(lambda _checked=False: self._cancel_options())
        run.clicked.connect(lambda _checked=False: self._accept_options())
        self._populate_task_controls()
        self._sync_mode()

    def options(self) -> dict:
        automatic = self.auto_check.isChecked()
        task = self.task_combo.currentData()
        return {
            "task_override": task,
            "target_override": None if task in {"clustering", "anomaly_detection"} else self.target_combo.currentData(),
            "time_column": None if task != "time_series_forecasting" else self.time_combo.currentData(),
            "requested_methods": None if automatic else [
                method_id for method_id, item in self._items.items()
                if item.checkState() == Qt.CheckState.Checked
            ],
            "selected_features": None if task == "time_series_forecasting" else [
                item.data(Qt.ItemDataRole.UserRole)
                for index in range(self.feature_list.count())
                if (item := self.feature_list.item(index)).checkState() == Qt.CheckState.Checked
            ],
            "test_size": self.test_percent.value() / 100.0,
            "cv_folds": self.cv_folds.value(),
        }

    def _populate_task_controls(self):
        task = self.task_combo.currentData()
        self.target_combo.clear()
        self.target_combo.addItem("Pilih kolom target...", None)
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
        self._populate_features()

    def _populate_features(self):
        if not hasattr(self, "feature_list"):
            return
        previous = {
            self.feature_list.item(index).data(Qt.ItemDataRole.UserRole)
            for index in range(self.feature_list.count())
            if self.feature_list.item(index).checkState() == Qt.CheckState.Checked
        }
        initial = self.feature_list.count() == 0
        target = self.target_combo.currentData()
        task = self.task_combo.currentData()
        suggested_target = str((self.task_result.get("primary_task") or {}).get("target") or "")
        if not suggested_target:
            candidates = self.task_result.get("target_candidates") or []
            if candidates:
                first_candidate = candidates[0]
                suggested_target = str(
                    first_candidate.get("name") or first_candidate.get("column") or ""
                ) if isinstance(first_candidate, dict) else str(first_candidate)
        self.feature_list.clear()
        for column in self.dataframe.columns:
            if str(column) == target:
                continue
            item = QListWidgetItem(str(column))
            item.setData(Qt.ItemDataRole.UserRole, str(column))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if (initial or str(column) in previous)
                and not (task in {"clustering", "anomaly_detection"} and str(column) == suggested_target)
                else Qt.CheckState.Unchecked
            )
            self.feature_list.addItem(item)

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
        task = self.task_combo.currentData()
        self.target_combo.setEnabled(task in {"binary_classification", "multiclass_classification", "regression", "time_series_forecasting"})
        self.time_combo.setEnabled(task == "time_series_forecasting")
        show_time = task == "time_series_forecasting"
        if self.time_row_widget is not None:
            self.time_row_widget.setVisible(show_time)
        else:
            self.time_label.setVisible(show_time)
            self.time_combo.setVisible(show_time)
        self.test_percent.setEnabled(task != "clustering" and task != "anomaly_detection")
        self.cv_folds.setEnabled(task in {"binary_classification", "multiclass_classification", "regression"})
        self.feature_list.setEnabled(task != "time_series_forecasting")
        self.model_list.setEnabled(manual)
        self.select_all_button.setEnabled(manual)
        self.clear_button.setEnabled(manual)

    def _accept_options(self):
        task = self.task_combo.currentData()
        if not task:
            QMessageBox.warning(self, "Task diperlukan", "Pilih jenis analisis yang ingin dijalankan.")
            return
        target = self.target_combo.currentData()
        if task in {"binary_classification", "multiclass_classification", "regression", "time_series_forecasting"} and not target:
            QMessageBox.warning(self, "Target diperlukan", "Pilih kolom target untuk task ini.")
            return
        if task in {"binary_classification", "multiclass_classification"}:
            class_count = int(self.dataframe[target].nunique(dropna=True))
            if task == "binary_classification" and class_count != 2:
                QMessageBox.warning(self, "Target tidak sesuai", "Klasifikasi biner membutuhkan tepat 2 kelas pada target.")
                return
            if task == "multiclass_classification" and class_count < 3:
                QMessageBox.warning(self, "Target tidak sesuai", "Klasifikasi multikelas membutuhkan minimal 3 kelas pada target.")
                return
        if task == "time_series_forecasting" and not self.time_combo.currentData():
            QMessageBox.warning(self, "Kolom waktu diperlukan", "Pilih kolom tanggal atau waktu untuk forecasting.")
            return
        if task != "time_series_forecasting" and not self.options()["selected_features"]:
            QMessageBox.warning(self, "Fitur diperlukan", "Pilih minimal satu fitur untuk eksperimen.")
            return
        if not self.auto_check.isChecked() and not self.options()["requested_methods"]:
            QMessageBox.warning(self, "Algoritma belum dipilih", "Pilih minimal satu algoritma untuk dijalankan.")
            return
        if callable(self.on_submit):
            self.on_submit(self.options())

    def _cancel_options(self):
        if callable(self.on_cancel):
            self.on_cancel()


class MLRunOptionsDialog(QDialog):
    """Modal wrapper retained for existing callers and manual-option flows."""

    TASKS = MLRunOptionsPanel.TASKS

    def __init__(self, dataframe: pd.DataFrame, task_result: dict, parent=None):
        super().__init__(parent)
        self.dataframe = dataframe
        self.task_result = task_result or {}
        self.embedded = False
        self.setWindowTitle("Opsi Machine Learning")
        screen = parent.screen() if parent is not None else QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        max_width = max(360, available.width() - 32) if available is not None else 650
        max_height = max(420, available.height() - 32) if available is not None else 720
        self.setMinimumSize(min(570, max_width), min(520, max_height))
        self.setMaximumSize(max_width, max_height)
        self.resize(min(650, max_width), min(720, max_height))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.panel = MLRunOptionsPanel(
            dataframe,
            self.task_result,
            self,
            embedded=False,
            on_submit=self._accept_options,
            on_cancel=self.reject,
        )
        layout.addWidget(self.panel)

        for name in (
            "suggestion_label",
            "auto_check",
            "task_combo",
            "target_combo",
            "time_combo",
            "test_percent",
            "cv_folds",
            "hint",
            "model_list",
            "feature_list",
            "select_all_button",
            "clear_button",
        ):
            setattr(self, name, getattr(self.panel, name))
        self._items = self.panel._items

    def options(self) -> dict:
        return self.panel.options()

    def _accept_options(self, _options):
        self.accept()
