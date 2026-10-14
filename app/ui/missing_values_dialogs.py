"""Dialogs used by the missing-value drop workflow."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.analyzer.cleaner import CleanResult, normalize_missing


def _fit_dialog_to_available_screen(dialog: QDialog) -> None:
    """Keep drop dialogs within the active display's usable area."""
    parent = dialog.parentWidget()
    screen = parent.screen() if parent is not None else None
    screen = screen or QApplication.primaryScreen()
    if screen is None:
        return
    available = screen.availableGeometry()
    max_width = max(1, round(available.width() * 0.9))
    max_height = max(1, round(available.height() * 0.9))
    min_width = min(dialog.minimumWidth(), max_width)
    min_height = min(dialog.minimumHeight(), max_height)
    dialog.setMinimumSize(min_width, min_height)
    dialog.setMaximumSize(max_width, max_height)
    hint = dialog.sizeHint()
    dialog.resize(
        min(max_width, max(min_width, hint.width())),
        min(max_height, max(min_height, hint.height())),
    )


class DropMissingOptionsDialog(QDialog):
    """Choose columns, row-drop behavior, and zero handling."""

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Drop Missing Values")
        self.setMinimumSize(540, 620)
        self._column_checks: dict[str, QCheckBox] = {}
        self._zero_checks: dict[str, QCheckBox] = {}

        dialog_layout = QVBoxLayout(self)
        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setFrameShape(QFrame.NoFrame)
        content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        intro = QLabel("Pilih kolom yang diperiksa dan aturan penghapusan baris.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        strategy_row = QHBoxLayout()
        strategy_row.addWidget(QLabel("Strategi"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItem("Drop baris", "drop_rows")
        self.strategy_combo.addItem("Drop kolom", "drop_cols")
        self.strategy_combo.addItem("Imputasi", "impute")
        self.strategy_combo.addItem("Laporan saja", "report")
        strategy_row.addWidget(self.strategy_combo, 1)
        layout.addLayout(strategy_row)

        self.strategy_options = QWidget()
        strategy_options_layout = QVBoxLayout(self.strategy_options)
        strategy_options_layout.setContentsMargins(0, 0, 0, 0)
        self.col_threshold = QDoubleSpinBox()
        self.col_threshold.setRange(0.0, 1.0)
        self.col_threshold.setSingleStep(0.05)
        self.col_threshold.setValue(0.5)
        self.col_threshold.setSuffix(" missing")
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Hapus kolom jika missing melebihi"))
        threshold_row.addWidget(self.col_threshold)
        strategy_options_layout.addLayout(threshold_row)
        self.impute_method = QComboBox()
        for method in ("mean", "median", "mode", "ffill", "bfill", "interpolate", "constant"):
            self.impute_method.addItem(method, method)
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Metode imputasi"))
        method_row.addWidget(self.impute_method, 1)
        strategy_options_layout.addLayout(method_row)
        self.fill_value = QLineEdit()
        self.fill_value.setPlaceholderText("Nilai konstanta (untuk method constant)")
        strategy_options_layout.addWidget(self.fill_value)
        self.group_by = QLineEdit()
        self.group_by.setPlaceholderText("Kolom grup, pisahkan dengan koma (opsional)")
        strategy_options_layout.addWidget(self.group_by)
        layout.addWidget(self.strategy_options)

        layout.addWidget(QLabel("Kolom yang diperiksa"))
        self.column_scroll = self._make_scroll_area()
        column_content = QWidget()
        column_layout = QVBoxLayout(column_content)
        column_layout.setContentsMargins(8, 6, 8, 6)
        column_layout.setSpacing(4)
        for column in dataframe.columns:
            check = QCheckBox(str(column))
            check.setChecked(True)
            check.setProperty("columnName", column)
            self._column_checks[str(column)] = check
            column_layout.addWidget(check)
        column_layout.addStretch(1)
        self.column_scroll.setWidget(column_content)
        self.column_scroll.setMinimumHeight(155)
        layout.addWidget(self.column_scroll, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Hapus baris jika"))
        self.how_combo = QComboBox()
        self.how_combo.addItem("any - minimal satu kolom kosong", "any")
        self.how_combo.addItem("all - semua kolom kosong", "all")
        row.addWidget(self.how_combo, 1)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Perlakuan nilai 0"))
        self.zero_mode_combo = QComboBox()
        self.zero_mode_combo.addItem("Auto (deteksi 0 mencurigakan)", "auto")
        self.zero_mode_combo.addItem("Jangan anggap 0 missing", "off")
        self.zero_mode_combo.addItem("Selective (pilih kolom)", "selective")
        self.zero_mode_combo.addItem("Strict (semua kolom)", "strict")
        row.addWidget(self.zero_mode_combo, 1)
        layout.addLayout(row)

        self.zero_panel = QFrame()
        self.zero_panel.setObjectName("dropMissingZeroPanel")
        zero_layout = QVBoxLayout(self.zero_panel)
        zero_layout.setContentsMargins(8, 6, 8, 6)
        zero_layout.setSpacing(3)
        zero_layout.addWidget(QLabel("Kolom numerik yang nilai 0-nya dianggap missing"))
        zero_scroll = self._make_scroll_area()
        zero_content = QWidget()
        zero_checks_layout = QVBoxLayout(zero_content)
        zero_checks_layout.setContentsMargins(4, 4, 4, 4)
        zero_checks_layout.setSpacing(3)
        normalized = normalize_missing(dataframe, zero_as_missing=False)
        for column in dataframe.columns:
            series = normalized[column]
            if not self._is_numeric_column(series):
                continue
            numeric_values = pd.to_numeric(series, errors="coerce")
            zero_count = int(numeric_values.eq(0).fillna(False).sum())
            check = QCheckBox(f"{column}  ({zero_count:,} nilai 0)")
            check.setProperty("columnName", column)
            self._zero_checks[str(column)] = check
            zero_checks_layout.addWidget(check)
        if not self._zero_checks:
            empty_label = QLabel("Tidak ada kolom numerik yang dapat dipilih.")
            empty_label.setWordWrap(True)
            zero_checks_layout.addWidget(empty_label)
        zero_checks_layout.addStretch(1)
        zero_scroll.setWidget(zero_content)
        zero_scroll.setMinimumHeight(115)
        zero_layout.addWidget(zero_scroll)
        layout.addWidget(self.zero_panel)

        self.strict_warning = QLabel(
            "PERINGATAN: nilai 0 pada semua kolom akan dianggap missing. "
            "Kolom biner dan kolom target dapat kehilangan satu kelas atau rusak."
        )
        self.strict_warning.setObjectName("dropMissingWarningBanner")
        self.strict_warning.setWordWrap(True)
        layout.addWidget(self.strict_warning)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Lanjutkan")
        buttons.button(QDialogButtonBox.Cancel).setText("Batal")
        buttons.accepted.connect(
            lambda: self.done(QDialog.DialogCode.Accepted)
        )
        buttons.rejected.connect(
            lambda: self.done(QDialog.DialogCode.Rejected)
        )
        content_scroll.setWidget(content)
        dialog_layout.addWidget(content_scroll, 1)
        dialog_layout.addWidget(buttons)

        self.zero_mode_combo.currentIndexChanged.connect(self._update_zero_visibility)
        self.strategy_combo.currentIndexChanged.connect(self._update_strategy_visibility)
        self._update_zero_visibility()
        self._update_strategy_visibility()
        _fit_dialog_to_available_screen(self)

    @staticmethod
    def _make_scroll_area() -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    @staticmethod
    def _is_numeric_column(series: pd.Series) -> bool:
        if pd.api.types.is_bool_dtype(series.dtype):
            return False
        if pd.api.types.is_numeric_dtype(series.dtype):
            return True
        present = series.dropna()
        if present.empty:
            return False
        return bool(pd.to_numeric(present, errors="coerce").notna().all())

    @Slot(int)
    def _update_zero_visibility(self, _index: int = -1) -> None:
        mode = self.zero_mode_combo.currentData()
        self.zero_panel.setVisible(mode == "selective")
        self.strict_warning.setVisible(mode == "strict")

    @Slot(int)
    def _update_strategy_visibility(self, _index: int = -1) -> None:
        strategy = self.strategy_combo.currentData()
        self.strategy_options.setVisible(strategy in {"drop_cols", "impute"})
        self.col_threshold.setVisible(strategy == "drop_cols")
        for widget in (self.impute_method, self.fill_value, self.group_by):
            widget.setVisible(strategy == "impute")

    def options(self) -> dict:
        selected_columns = [
            str(check.property("columnName"))
            for check in self._column_checks.values()
            if check.isChecked()
        ]
        zero_mode = str(self.zero_mode_combo.currentData())
        strategy = str(self.strategy_combo.currentData())
        return {
            "columns": selected_columns,
            "how": str(self.how_combo.currentData()),
            "strategy": strategy,
            "col_threshold": float(self.col_threshold.value()),
            "method": str(self.impute_method.currentData()),
            "fill_value": self.fill_value.text() or None,
            "group_by": [name.strip() for name in self.group_by.text().split(",") if name.strip()],
            "zero_mode": None if zero_mode == "auto" else zero_mode,
            "zero_as_missing": "auto" if zero_mode == "auto" else False,
            "zero_columns": [
                str(check.property("columnName"))
                for check in self._zero_checks.values()
                if check.isChecked()
            ],
        }


class DropMissingPreviewDialog(QDialog):
    """Show row impact and warnings before the user confirms a drop."""

    def __init__(self, result: CleanResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pratinjau Drop Missing Values")
        self.setMinimumSize(510, 440)
        dialog_layout = QVBoxLayout(self)
        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setFrameShape(QFrame.NoFrame)
        content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)

        strategy = result.report.get("strategy", "drop_rows")
        if strategy == "drop_rows":
            summary_text = (
                f"Akan menghapus {result.rows_dropped:,} dari {result.rows_before:,} "
                f"baris ({result.percent_dropped:.1f}%)."
            )
        elif strategy == "drop_cols":
            summary_text = f"Akan menghapus {result.report.get('columns_dropped', 0):,} kolom."
        elif strategy == "impute":
            summary_text = f"Missing value akan diisi pada {len(result.report.get('columns_imputed', [])):,} kolom."
        else:
            summary_text = "Laporan dibuat tanpa mengubah DataFrame."
        summary = QLabel(summary_text)
        summary.setObjectName("sectionTitle")
        summary.setWordWrap(True)
        layout.addWidget(summary)

        layout.addWidget(QLabel("Rincian kolom yang diperiksa"))
        details = QLabel(result.formatted_report or "Tidak ada kolom dipilih.")
        details.setWordWrap(True)
        details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(details)

        if result.warnings:
            warning_label = QLabel("Peringatan")
            warning_label.setObjectName("dropMissingWarningBanner")
            warning_label.setWordWrap(True)
            layout.addWidget(warning_label)
            warnings = QLabel("\n".join(f"- {warning}" for warning in result.warnings))
            warnings.setObjectName("dropMissingWarningBanner")
            warnings.setWordWrap(True)
            warnings.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(warnings)

        if not result.dropped_df.empty:
            audit_note = QLabel(
                f"{len(result.dropped_df):,} baris yang dibuang akan disimpan terpisah "
                "untuk audit saat versi bersih dibuat."
            )
            audit_note.setWordWrap(True)
            layout.addWidget(audit_note)

        layout.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Lanjutkan")
        buttons.button(QDialogButtonBox.Cancel).setText("Batal")
        buttons.accepted.connect(
            lambda: self.done(QDialog.DialogCode.Accepted)
        )
        buttons.rejected.connect(
            lambda: self.done(QDialog.DialogCode.Rejected)
        )
        content_scroll.setWidget(content)
        dialog_layout.addWidget(content_scroll, 1)
        dialog_layout.addWidget(buttons)
        _fit_dialog_to_available_screen(self)
