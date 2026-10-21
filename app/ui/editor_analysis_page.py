"""Guided, editable analysis templates for the active dataset."""

from __future__ import annotations

import base64
import copy
import hashlib
import html
import json
import os
import re
import shlex
import sys
import sysconfig
import uuid
from pathlib import Path

from app.ui.scale import logical_width, ui_px

from PySide6.QtCore import (
    QEasingCurve,
    QProcess,
    QProcessEnvironment,
    QPropertyAnimation,
    QRect,
    QRegularExpression,
    Signal,
    Slot,
    QStandardPaths,
    QTimer,
    Qt,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPixmap,
    QPainter,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
    QTextFormat,
)
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QComboBox,
    QFrame,
    QFileDialog,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QPlainTextEdit,
    QTextBrowser,
    QTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from app.ui.style_utils import apply_card_shadow


class PythonCodeHighlighter(QSyntaxHighlighter):
    """Lightweight Python syntax colors for the notebook code editor."""

    def __init__(self, document):
        super().__init__(document)
        self.rules = []
        self._add_rule(
            r"\b(?:False|None|True|and|as|assert|async|await|break|class|continue|def|del|"
            r"elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|"
            r"or|pass|raise|return|try|while|with|yield|match|case)\b",
            "#c792ea",
            bold=True,
        )
        self._add_rule(
            r"\b(?:abs|all|any|bool|dict|enumerate|filter|float|int|len|list|map|max|min|"
            r"print|range|set|sorted|str|sum|tuple|type|zip|display|help)\b",
            "#82aaff",
        )
        self._add_rule(r"@[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", "#ffcb6b")
        self._add_rule(r"(?<=\bdef )\w+|(?<=\bclass )\w+", "#82aaff", bold=True)
        self._add_rule(r"\b(?:0[xX][\da-fA-F]+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\b", "#f78c6c")
        self._add_rule(
            r"(?i)(?:[rubf]{0,2})?(?:\"\"\".*?\"\"\"|'''.*?'''|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')",
            "#c3e88d",
        )
        self._add_rule(r"#.*$", "#7f8c9a", italic=True)

    def _add_rule(self, pattern: str, color: str, *, bold: bool = False, italic: bool = False):
        text_format = QTextCharFormat()
        text_format.setForeground(QColor(color))
        if bold:
            text_format.setFontWeight(QFont.Bold)
        if italic:
            text_format.setFontItalic(True)
        self.rules.append((QRegularExpression(pattern), text_format))

    def highlightBlock(self, text: str):
        for expression, text_format in self.rules:
            matches = expression.globalMatch(text)
            while matches.hasNext():
                match = matches.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), text_format)


class PythonCodeEditor(QPlainTextEdit):
    runRequested = Signal()
    runAndAdvanceRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.line_number_area = CodeLineNumberArea(self)
        self.line_number_area.setObjectName("editorAnalysisLineNumberArea")
        self.blockCountChanged.connect(self._update_line_number_width)
        self.updateRequest.connect(self._update_line_number_area)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self._update_line_number_width()
        self._highlight_current_line()

    def line_number_area_width(self) -> int:
        digits = max(2, len(str(max(1, self.blockCount()))))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    @Slot(int)
    def _update_line_number_width(self, *_args):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    @Slot(QRect, int)
    def _update_line_number_area(self, rect, dy: int):
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(), self.line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_number_width()

    def paint_line_number_area(self, event):
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor("#111b2d"))
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()
        current_block = self.textCursor().blockNumber()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                color = "#82aaff" if block_number == current_block else "#718096"
                painter.setPen(QColor(color))
                painter.drawText(
                    0,
                    int(top),
                    self.line_number_area.width() - 8,
                    self.fontMetrics().height(),
                    Qt.AlignRight,
                    str(block_number + 1),
                )
            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            block_number += 1

    @Slot()
    def _highlight_current_line(self):
        selections = []
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            selection.format.setBackground(QColor("#203047"))
            selection.format.setProperty(QTextFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            selections.append(selection)
        self.setExtraSelections(selections)
        self.line_number_area.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        contents = self.contentsRect()
        self.line_number_area.setGeometry(
            contents.left(), contents.top(), self.line_number_area_width(), contents.height()
        )

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and event.modifiers() & Qt.ShiftModifier:
            self.runAndAdvanceRequested.emit()
            event.accept()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and event.modifiers() & Qt.ControlModifier:
            self.runRequested.emit()
            event.accept()
            return
        if event.key() == Qt.Key_Tab and not event.modifiers() & (Qt.ControlModifier | Qt.AltModifier):
            self.textCursor().insertText("    ")
            event.accept()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            current_line = self.textCursor().block().text()
            indentation = re.match(r"\s*", current_line).group(0)
            if current_line.rstrip().endswith(":"):
                indentation += "    "
            super().keyPressEvent(event)
            self.textCursor().insertText(indentation)
            return
        super().keyPressEvent(event)


class CodeLineNumberArea(QWidget):
    def __init__(self, editor: PythonCodeEditor):
        super().__init__(editor)
        self.editor = editor

    def paintEvent(self, event):
        self.editor.paint_line_number_area(event)


class EditorAnalysisPage(QWidget):
    """Guided notebook-style editor with editable templates and local run output."""

    TARGET_ASSIGNMENT = re.compile(r"(?m)^target_column\s*=.*$")
    MISSING_MODULE_PACKAGES = {
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "sklearn": "scikit-learn",
        "imblearn": "imbalanced-learn",
        "seaborn": "seaborn",
        "statsmodels": "statsmodels",
        "optuna": "optuna",
        "yaml": "PyYAML",
        "pil": "Pillow",
        "cv2": "opencv-python-headless",
        "bs4": "beautifulsoup4",
        "openpyxl": "openpyxl",
    }

    MODEL_METHODS = {
        "Klasifikasi": [
            ("Logistic Regression", "from sklearn.linear_model import LogisticRegression", "LogisticRegression(max_iter=1000)"),
            ("SGD Classifier", "from sklearn.linear_model import SGDClassifier", "SGDClassifier(max_iter=2000, random_state=42)"),
            ("Decision Tree Classifier", "from sklearn.tree import DecisionTreeClassifier", "DecisionTreeClassifier(max_depth=12, random_state=42)"),
            ("Random Forest Classifier", "from sklearn.ensemble import RandomForestClassifier", "RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')"),
            ("Extra Trees Classifier", "from sklearn.ensemble import ExtraTreesClassifier", "ExtraTreesClassifier(n_estimators=200, random_state=42, class_weight='balanced')"),
            ("Gradient Boosting Classifier", "from sklearn.ensemble import GradientBoostingClassifier", "GradientBoostingClassifier(n_estimators=150, random_state=42)"),
            ("HistGradientBoosting Classifier", "from sklearn.ensemble import HistGradientBoostingClassifier", "HistGradientBoostingClassifier(max_iter=150, random_state=42)"),
            ("AdaBoost Classifier", "from sklearn.ensemble import AdaBoostClassifier", "AdaBoostClassifier(n_estimators=100, random_state=42)"),
            ("Support Vector Classifier (SVC)", "from sklearn.svm import SVC", "SVC(kernel='rbf', probability=True, random_state=42)"),
            ("Linear SVC", "from sklearn.svm import LinearSVC", "LinearSVC(max_iter=3000, random_state=42)"),
            ("K-Nearest Neighbors Classifier", "from sklearn.neighbors import KNeighborsClassifier", "KNeighborsClassifier(n_neighbors=5)"),
            ("Gaussian Naive Bayes", "from sklearn.naive_bayes import GaussianNB", "GaussianNB()"),
            ("Linear Discriminant Analysis (LDA)", "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis", "LinearDiscriminantAnalysis()"),
            ("Multi-Layer Perceptron (MLP)", "from sklearn.neural_network import MLPClassifier", "MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)"),
            ("XGBoost Classifier", "from xgboost import XGBClassifier", "XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, eval_metric='logloss', verbosity=0)"),
            ("LightGBM Classifier", "from lightgbm import LGBMClassifier", "LGBMClassifier(n_estimators=200, learning_rate=0.05, verbosity=-1)"),
            ("CatBoost Classifier", "from catboost import CatBoostClassifier", "CatBoostClassifier(iterations=200, learning_rate=0.05, verbose=False)"),
        ],
        "Regresi": [
            ("Linear Regression", "from sklearn.linear_model import LinearRegression", "LinearRegression()"),
            ("Ridge Regression", "from sklearn.linear_model import Ridge", "Ridge(alpha=1.0)"),
            ("Lasso Regression", "from sklearn.linear_model import Lasso", "Lasso(alpha=0.01, max_iter=5000)"),
            ("ElasticNet Regression", "from sklearn.linear_model import ElasticNet", "ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)"),
            ("SGD Regressor", "from sklearn.linear_model import SGDRegressor", "SGDRegressor(max_iter=2000, random_state=42)"),
            ("Huber Regressor", "from sklearn.linear_model import HuberRegressor", "HuberRegressor(max_iter=500)"),
            ("Decision Tree Regressor", "from sklearn.tree import DecisionTreeRegressor", "DecisionTreeRegressor(max_depth=12, random_state=42)"),
            ("Random Forest Regressor", "from sklearn.ensemble import RandomForestRegressor", "RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)"),
            ("Extra Trees Regressor", "from sklearn.ensemble import ExtraTreesRegressor", "ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)"),
            ("Gradient Boosting Regressor", "from sklearn.ensemble import GradientBoostingRegressor", "GradientBoostingRegressor(n_estimators=150, random_state=42)"),
            ("HistGradientBoosting Regressor", "from sklearn.ensemble import HistGradientBoostingRegressor", "HistGradientBoostingRegressor(max_iter=150, random_state=42)"),
            ("AdaBoost Regressor", "from sklearn.ensemble import AdaBoostRegressor", "AdaBoostRegressor(n_estimators=100, random_state=42)"),
            ("Support Vector Regressor (SVR)", "from sklearn.svm import SVR", "SVR(kernel='rbf', C=1.0)"),
            ("K-Nearest Neighbors Regressor", "from sklearn.neighbors import KNeighborsRegressor", "KNeighborsRegressor(n_neighbors=5)"),
            ("Multi-Layer Perceptron Regressor", "from sklearn.neural_network import MLPRegressor", "MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)"),
            ("XGBoost Regressor", "from xgboost import XGBRegressor", "XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, verbosity=0)"),
            ("LightGBM Regressor", "from lightgbm import LGBMRegressor", "LGBMRegressor(n_estimators=200, learning_rate=0.05, verbosity=-1)"),
            ("CatBoost Regressor", "from catboost import CatBoostRegressor", "CatBoostRegressor(iterations=200, learning_rate=0.05, verbose=False)"),
        ],
        "Clustering": [
            ("K-Means", "from sklearn.cluster import KMeans", "KMeans(n_clusters=3, random_state=42, n_init='auto')"),
            ("Mini-Batch K-Means", "from sklearn.cluster import MiniBatchKMeans", "MiniBatchKMeans(n_clusters=3, random_state=42, n_init='auto')"),
            ("Agglomerative Clustering", "from sklearn.cluster import AgglomerativeClustering", "AgglomerativeClustering(n_clusters=3)"),
            ("DBSCAN", "from sklearn.cluster import DBSCAN", "DBSCAN(eps=0.8, min_samples=5)"),
            ("Gaussian Mixture Model", "from sklearn.mixture import GaussianMixture", "GaussianMixture(n_components=3, random_state=42)"),
            ("BIRCH", "from sklearn.cluster import Birch", "Birch(n_clusters=3)"),
            ("Spectral Clustering", "from sklearn.cluster import SpectralClustering", "SpectralClustering(n_clusters=3, random_state=42, assign_labels='kmeans')"),
        ],
        "Anomaly Detection": [
            ("Isolation Forest", "from sklearn.ensemble import IsolationForest", "IsolationForest(contamination='auto', random_state=42)"),
            ("Local Outlier Factor (LOF)", "from sklearn.neighbors import LocalOutlierFactor", "LocalOutlierFactor(n_neighbors=20, contamination='auto')"),
            ("One-Class SVM", "from sklearn.svm import OneClassSVM", "OneClassSVM(nu=0.05, kernel='rbf')"),
            ("Elliptic Envelope", "from sklearn.covariance import EllipticEnvelope", "EllipticEnvelope(contamination=0.05, random_state=42)"),
        ],
    }

    STEPS = (
        ("Data Understanding", "Kenali ukuran, tipe kolom, dan contoh isi dataset."),
        ("Data Cleaning", "Periksa nilai kosong, duplikasi, dan data yang perlu dibersihkan."),
        ("Exploratory Data Analysis", "Jelajahi distribusi fitur dan hubungan antarvariabel."),
        ("Feature Engineering", "Siapkan fitur dan ubah kolom kategorikal untuk pemodelan."),
        ("Metode Algoritma", "Pilih klasifikasi, regresi, clustering, atau deteksi anomali."),
        ("Evaluasi Model", "Ukur performa model dengan metrik yang sesuai."),
    )

    TEMPLATES = {
        "Data Understanding": '''# Kenali struktur dataset
print(f"Ukuran dataset: {df.shape[0]:,} baris × {df.shape[1]:,} kolom")
display(df.head())
df.info()
display(df.describe(include="all").T)
''',
        "Missing Values": '''# Periksa nilai kosong per kolom
missing = df.isna().sum().sort_values(ascending=False)
display(missing[missing > 0])
print(f"Baris duplikat: {df.duplicated().sum():,}")

# Contoh imputasi kolom numerik dengan median
# df["nama_kolom"] = df["nama_kolom"].fillna(df["nama_kolom"].median())
# Contoh menghapus duplikasi setelah ditinjau
# df = df.drop_duplicates()
''',
        "Outliers (IQR)": '''# Hitung batas outlier dengan metode IQR
numeric = df.select_dtypes(include="number")
q1 = numeric.quantile(0.25)
q3 = numeric.quantile(0.75)
iqr = q3 - q1
outlier_count = ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum()
display(outlier_count.sort_values(ascending=False))
''',
        "EDA & Korelasi": '''# Ringkasan statistik fitur numerik
display(df.select_dtypes(include="number").describe().T)

# Matriks korelasi numerik
correlation = df.select_dtypes(include="number").corr()
display(correlation.round(2))
''',
        "Encoding Kategorikal": '''# Contoh one-hot encoding
# Tentukan kolom kategorikal yang ingin diubah
categorical_columns = df.select_dtypes(include=["str", "category"]).columns
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
display(df_encoded.head())
''',
        "Random Forest": '''# Template ensemble: Random Forest untuk klasifikasi
# Ganti "target" dengan nama kolom target di dataset.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

target_column = "target"
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
report = pd.DataFrame(classification_report(
    y_test, model.predict(X_test), output_dict=True, zero_division=0
)).T
for metric in ("precision", "recall", "f1-score"):
    if metric in report:
        report[metric] = report[metric].map(lambda score: f"{score:.1%}" if pd.notna(score) else "")
display(report)
''',
        "Gradient Boosting": '''# Template ensemble: Gradient Boosting untuk klasifikasi
# Pastikan fitur sudah numerik dan tentukan kolom target yang benar.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

target_column = "target"
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = HistGradientBoostingClassifier(max_iter=150, random_state=42)
model.fit(X_train, y_train)
report = pd.DataFrame(classification_report(
    y_test, model.predict(X_test), output_dict=True, zero_division=0
)).T
for metric in ("precision", "recall", "f1-score"):
    if metric in report:
        report[metric] = report[metric].map(lambda score: f"{score:.1%}" if pd.notna(score) else "")
display(report)
''',
        "XGBoost": '''# Template ensemble: XGBoost untuk klasifikasi
# XGBoost harus dipasang terpisah dan fitur kategorikal perlu di-encode.
# Ganti "target" dengan kolom target dataset.
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

target_column = "target"
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, eval_metric="logloss")
model.fit(X_train, y_train)
report = pd.DataFrame(classification_report(
    y_test, model.predict(X_test), output_dict=True, zero_division=0
)).T
for metric in ("precision", "recall", "f1-score"):
    if metric in report:
        report[metric] = report[metric].map(lambda score: f"{score:.1%}" if pd.notna(score) else "")
display(report)
''',
        "Evaluasi Model": '''# Evaluasi klasifikasi pada data uji
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

y_pred = model.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.1%}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.1%}")
print(f"Recall   : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.1%}")
print(f"F1-score : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.1%}")
report = pd.DataFrame(classification_report(
    y_test, y_pred, output_dict=True, zero_division=0
)).T
for metric in ("precision", "recall", "f1-score"):
    if metric in report:
        report[metric] = report[metric].map(
            lambda score: f"{score:.1%}" if pd.notna(score) else ""
        )
if "support" in report:
    report["support"] = report["support"].map(
        lambda count: f"{count:,.0f}" if pd.notna(count) else ""
    )
display(report)
''',
    }

    STEP_TEMPLATES = {
        0: "Ringkasan dataset",
        1: "Missing Values",
        2: "Heatmap korelasi",
        3: "One-hot encoding",
        4: "Random Forest Classifier",
        5: "Evaluasi klasifikasi + confusion matrix",
    }

    STEP_CATEGORIES = {
        0: "Data Profiling",
        1: "Data Cleaning",
        2: "Exploratory Data Analysis",
        3: "Feature Engineering",
        4: "Klasifikasi",
        5: "Evaluasi Model",
    }

    CATEGORY_DESCRIPTIONS = {
        "Data Profiling": "Pahami ukuran, tipe, nilai unik, dan kelengkapan dataset.",
        "Data Cleaning": "Tinjau missing values, duplikasi, dan outlier sebelum membersihkan data.",
        "Exploratory Data Analysis": "Jelajahi distribusi fitur dan hubungan antarvariabel dengan grafik.",
        "Feature Engineering": "Siapkan fitur, encoding, dan pembagian data untuk pemodelan.",
        "Klasifikasi": "Template klasifikasi dengan preprocessing, metrik, dan confusion matrix.",
        "Regresi": "Template regresi dengan preprocessing, metrik, dan grafik residual.",
        "Clustering": "Template pengelompokan tanpa target dengan grafik PCA dan silhouette score.",
        "Anomaly Detection": "Template untuk menandai baris yang berpotensi anomali.",
        "Evaluasi Model": "Nilai hasil model dengan metrik dan grafik yang sesuai.",
        "Custom": "Tulis kode analisis bebas untuk langkah tambahan.",
    }

    BASE_TEMPLATE_GROUPS = {
        "Data Profiling": {
            "Ringkasan dataset": TEMPLATES["Data Understanding"],
            "Audit tipe data dan nilai kosong": '''# Audit tipe data, nilai unik, nilai kosong, dan duplikasi
summary = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "non_null": df.notna().sum(),
    "missing": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2),
    "unique": df.nunique(dropna=True),
})
display(summary)
print(f"Baris duplikat: {df.duplicated().sum():,}")
''',
            "Distribusi target": '''# Ganti nama kolom target sebelum menjalankan
target_column = "target"
counts = df[target_column].value_counts(dropna=False)
display(counts.to_frame("jumlah"))
ax = counts.plot(kind="bar", figsize=(8, 4), title=f"Distribusi {target_column}")
ax.set_ylabel("Jumlah baris")
plt.tight_layout()
''',
            "Profil memori dan nilai unik": '''# Ringkasan memori dan kardinalitas setiap kolom
memory = df.memory_usage(deep=True).sort_values(ascending=False)
profile = pd.DataFrame({
    "memory_kb": (memory / 1024).round(2),
    "unique_values": df.nunique(dropna=True),
})
display(profile.sort_values("memory_kb", ascending=False))
''',
        },
        "Data Cleaning": {
            "Missing Values": TEMPLATES["Missing Values"],
            "Imputasi numerik dengan median": '''# Buat salinan agar dataset sumber tidak berubah
df_clean = df.copy()
numeric_columns = df_clean.select_dtypes(include="number").columns
df_clean[numeric_columns] = df_clean[numeric_columns].fillna(df_clean[numeric_columns].median())
print(f"Nilai kosong sebelum: {df.isna().sum().sum():,}")
print(f"Nilai kosong numerik sesudah: {df_clean[numeric_columns].isna().sum().sum():,}")
display(df_clean.head())
''',
            "Imputasi kategorikal dengan modus": '''# Isi nilai kosong kategorikal dengan modus kolom
df_clean = df.copy()
categorical_columns = df_clean.select_dtypes(include=["str", "category", "bool"]).columns
for column in categorical_columns:
    modes = df_clean[column].mode(dropna=True)
    if not modes.empty:
        df_clean[column] = df_clean[column].fillna(modes.iloc[0])
print(f"Nilai kosong tersisa: {df_clean.isna().sum().sum():,}")
''',
            "Hapus baris yang kosong": '''# Tinjau jumlah baris yang terdampak sebelum menghapus
rows_before = len(df)
df_clean = df.dropna().copy()
print(f"Baris sebelum: {rows_before:,}")
print(f"Baris sesudah: {len(df_clean):,}")
print(f"Baris dihapus: {rows_before - len(df_clean):,}")
''',
            "Hapus duplikasi": '''# Cek duplikasi, lalu buat salinan tanpa baris identik
duplicate_count = int(df.duplicated().sum())
df_clean = df.drop_duplicates().copy()
print(f"Duplikasi ditemukan: {duplicate_count:,}")
print(f"Ukuran data sesudah: {df_clean.shape}")
''',
            "Deteksi Outliers (IQR)": TEMPLATES["Outliers (IQR)"],
            "Batasi Outliers (IQR capping)": '''# Capping fitur numerik pada batas 1.5 × IQR; tinjau hasil sebelum dipakai
df_clean = df.copy()
numeric_columns = df_clean.select_dtypes(include="number").columns
for column in numeric_columns:
    q1, q3 = df_clean[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    df_clean[column] = df_clean[column].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
display(df_clean[numeric_columns].describe().T)
''',
            "Normalisasi nilai teks": '''# Rapikan spasi pada kolom teks kategorikal
df_clean = df.copy()
text_columns = df_clean.select_dtypes(include=["str", "string"]).columns
for column in text_columns:
    df_clean[column] = df_clean[column].map(
        lambda value: value.strip() if isinstance(value, str) else value
    )
display(df_clean.head())
''',
        },
        "Exploratory Data Analysis": {
            "Histogram fitur numerik": '''# Lihat distribusi beberapa fitur numerik
numeric = df.select_dtypes(include="number")
if numeric.empty:
    print("Tidak ada kolom numerik untuk dibuat histogram.")
else:
    columns = list(numeric.columns[:12])
    axes = numeric[columns].hist(figsize=(12, max(4, 3 * ((len(columns) + 3) // 4))), bins=25)
    plt.suptitle("Distribusi fitur numerik")
    plt.tight_layout()
''',
            "Heatmap korelasi": '''# Heatmap korelasi fitur numerik
correlation = df.select_dtypes(include="number").corr()
if correlation.empty:
    print("Tidak ada fitur numerik untuk menghitung korelasi.")
else:
    fig, ax = plt.subplots(figsize=(max(6, len(correlation.columns) * 0.65), 5))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation.index)), correlation.index)
    fig.colorbar(image, ax=ax, label="Korelasi")
    ax.set_title("Korelasi Pearson")
    plt.tight_layout()
''',
            "Boxplot fitur numerik": '''# Bandingkan rentang dan nilai ekstrem fitur numerik
numeric = df.select_dtypes(include="number")
if numeric.empty:
    print("Tidak ada fitur numerik untuk dibuat boxplot.")
else:
    ax = numeric.iloc[:, :12].plot(kind="box", figsize=(12, 5), rot=35)
    ax.set_title("Boxplot fitur numerik")
    ax.set_ylabel("Nilai")
    plt.tight_layout()
''',
            "Frekuensi kategori": '''# Ganti nama kolom kategori untuk menampilkan frekuensi
column = "nama_kolom_kategori"
counts = df[column].astype("string").fillna("(kosong)").value_counts().head(20)
display(counts.to_frame("jumlah"))
ax = counts.sort_values().plot(kind="barh", figsize=(9, 5), title=f"Frekuensi {column}")
ax.set_xlabel("Jumlah baris")
plt.tight_layout()
''',
            "Scatter plot dua fitur": '''# Ganti nama kedua fitur dengan kolom numerik yang ingin dibandingkan
x_column = "fitur_x"
y_column = "fitur_y"
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df[x_column], df[y_column], alpha=0.55)
ax.set(xlabel=x_column, ylabel=y_column, title=f"{x_column} vs {y_column}")
ax.grid(alpha=0.2)
plt.tight_layout()
''',
            "Missing values per kolom": '''# Grafik jumlah nilai kosong
missing = df.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]
if missing.empty:
    print("Tidak ada nilai kosong pada dataset.")
else:
    display(missing.to_frame("jumlah kosong"))
    ax = missing.sort_values().plot(kind="barh", figsize=(9, max(4, len(missing) * 0.35)))
    ax.set_title("Nilai kosong per kolom")
    plt.tight_layout()
''',
        },
        "Feature Engineering": {
            "One-hot encoding": TEMPLATES["Encoding Kategorikal"],
            "Standardisasi fitur numerik": '''# Standardisasi fitur numerik tanpa mengubah dataset sumber
from sklearn.preprocessing import StandardScaler
numeric_columns = df.select_dtypes(include="number").columns
df_scaled = df.copy()
df_scaled[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])
display(df_scaled.head())
''',
            "Pisahkan fitur dan target": '''# Tentukan kolom target yang ingin diprediksi
target_column = "target"
X = df.drop(columns=[target_column]).copy()
y = df[target_column].copy()
print(f"Jumlah fitur: {X.shape[1]}")
print(f"Jumlah target: {y.shape[0]:,}")
''',
            "Feature importance": '''# Template feature importance setelah variabel model dan X tersedia
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values().tail(20)
display(importance.to_frame("importance"))
ax = importance.plot(kind="barh", figsize=(8, 6), title="Feature importance")
plt.tight_layout()
''',
        },
        "Evaluasi Model": {
            "Evaluasi klasifikasi + confusion matrix": '''# Memerlukan y_test dan y_pred dari model klasifikasi
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
report = pd.DataFrame(classification_report(
    y_test, y_pred, output_dict=True, zero_division=0
)).T
for metric in ("precision", "recall", "f1-score"):
    if metric in report:
        report[metric] = report[metric].map(lambda score: f"{score:.1%}" if pd.notna(score) else "")
display(report)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", xticks_rotation=45)
plt.title("Confusion matrix")
plt.tight_layout()
''',
            "Evaluasi regresi + residual plot": '''# Memerlukan y_test dan y_pred dari model regresi
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2_score(y_test, y_pred):.4f}")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(y_test, y_pred, alpha=0.6)
axes[0].set(xlabel="Aktual", ylabel="Prediksi", title="Aktual vs prediksi")
axes[1].scatter(y_pred, y_test - y_pred, alpha=0.6)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set(xlabel="Prediksi", ylabel="Residual", title="Residual plot")
plt.tight_layout()
''',
            "Validasi silang": '''# Validasi silang untuk model dan fitur yang sudah siap
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"Skor tiap fold: {scores.round(3)}")
print(f"Rata-rata: {scores.mean():.3f} ± {scores.std():.3f}")
''',
        },
        "Custom": {"Kode kosong": "# Tulis kode analisis di sini\n"},
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = parent
        self._steps: list[dict] = []
        self._selected_step = -1
        self._dataframe = None
        self._dataset_name = ""
        self._dataset_scope = "workspace"
        self._workspace_dataframe = None
        self._workspace_dataset_name = ""
        self._custom_dataframe = None
        self._custom_dataset_path = ""
        self._project_key = ""
        self._project_workspace: Path | None = None
        self._project_environment: Path | None = None
        self._environment_process: QProcess | None = None
        self._environment_callback = None
        self._kernel_process: QProcess | None = None
        self._kernel_buffer = bytearray()
        self._kernel_request_id = 0
        self._kernel_ready = False
        self._dataset_pickle_path: Path | None = None
        self._kernel_busy = False
        self._execution_queue: list[int] = []
        self._execution_total = 0
        self._execution_completed = 0
        self._running_step_index = -1
        self._queue_had_error = False
        self._advance_after_run = False
        self._execution_controls_locked = False
        self._package_process: QProcess | None = None
        self._package_log = ""
        self._installing_packages: list[str] = []
        self._package_install_mode = ""
        self._auto_install_context: dict | None = None
        self._auto_install_attempted: set[str] = set()
        self._notebook_path: Path | None = None
        self._notebook_metadata: dict = {}
        self._notebook_minor = 5
        self._notebook_dirty = False
        self._autosave_restored = False
        self._autosave_project_key = ""
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(700)
        self._autosave_timer.timeout.connect(self._save_autosave_state)
        self._initializing = True
        self._run_timeout = QTimer(self)
        self._run_timeout.setSingleShot(True)
        self._run_timeout.timeout.connect(self._on_run_timeout)
        self._graphics: list[QPixmap] = []
        self.template_groups = self._build_template_groups()
        application = QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self._save_autosave_on_exit)
            application.aboutToQuit.connect(self._terminate_process_on_exit)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.page_scroll = QScrollArea()
        self.page_scroll.setObjectName("editorAnalysisScroll")
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setFrameShape(QFrame.NoFrame)
        self.page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        container.setObjectName("editorAnalysisContent")
        self.root_layout = QVBoxLayout(container)
        self.root_layout.setContentsMargins(12, 28, 10, 12)
        self.root_layout.setSpacing(28)

        self.editor_header_container = QWidget()
        header = QHBoxLayout(self.editor_header_container)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(16)
        self.title_block_container = QWidget()
        title_block = QVBoxLayout(self.title_block_container)
        self.title_block = title_block
        title_block.setSpacing(4)
        title = QLabel("Editor Analysis")
        title.setObjectName("editorAnalysisPageTitle")
        title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        subtitle = QLabel(
            "Ikuti tahapan analisis, pilih template, lalu sesuaikan kode dengan kebutuhan risetmu."
        )
        subtitle.setObjectName("editorAnalysisPageSubtitle")
        subtitle.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        subtitle.setWordWrap(True)
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addWidget(self.title_block_container, 1)
        self.dataset_scope_combo = QComboBox(self)
        self.dataset_scope_combo.setObjectName("editorAnalysisDatasetScope")
        self.dataset_scope_combo.addItem("Ikuti Workspace", "workspace")
        self.dataset_scope_combo.addItem("Dataset notebook", "custom")
        self.dataset_scope_combo.setToolTip(
            "Pilih dataset Workspace atau file dataset khusus notebook ini."
        )
        self.dataset_scope_combo.currentIndexChanged.connect(self._dataset_scope_changed)
        self.dataset_scope_combo.hide()
        self.choose_dataset_button = QPushButton("Pilih dataset", self)
        self.choose_dataset_button.setObjectName("secondaryButton")
        self.choose_dataset_button.clicked.connect(self._choose_notebook_dataset)
        self.choose_dataset_button.hide()
        self.dataset_badge = QLabel("Dataset belum dimuat", self)
        self.dataset_badge.setObjectName("editorAnalysisDatasetBadge")
        self.dataset_badge.hide()
        self.dataset_menu_button = QPushButton("Dataset")
        self.dataset_menu_button.setObjectName("editorAnalysisDatasetMenu")
        dataset_menu = QMenu(self.dataset_menu_button)
        use_workspace_action = dataset_menu.addAction("Ikuti dataset Workspace")
        use_workspace_action.triggered.connect(
            lambda: self.dataset_scope_combo.setCurrentIndex(
                self.dataset_scope_combo.findData("workspace")
            )
        )
        use_notebook_action = dataset_menu.addAction("Gunakan dataset notebook")
        use_notebook_action.triggered.connect(
            lambda: self.dataset_scope_combo.setCurrentIndex(
                self.dataset_scope_combo.findData("custom")
            )
        )
        dataset_menu.addSeparator()
        choose_dataset_action = dataset_menu.addAction("Pilih file dataset…")
        choose_dataset_action.triggered.connect(self._choose_notebook_dataset)
        self.dataset_menu_button.setMenu(dataset_menu)
        self.dataset_menu_button.setToolTip(
            "Pilih dataset Workspace atau file dataset khusus notebook ini."
        )
        header.addWidget(self.dataset_menu_button, 0, Qt.AlignTop)
        self.root_layout.addWidget(self.editor_header_container)

        self.info_banner = QLabel(
            "Template dibuat tim DORA, bukan oleh AI. Kode dijalankan lokal dengan salinan dataset; "
            "hasilnya tidak mengubah dataset utama. Kode tidak diisolasi dari file dan jaringan komputer; "
            "jalankan hanya kode yang kamu percaya."
        )
        self.info_banner.setObjectName("editorAnalysisInfoBanner")
        self.info_banner.setWordWrap(True)
        self.info_banner_container = QWidget()
        self.info_banner_container_layout = QVBoxLayout(self.info_banner_container)
        self.info_banner_container_layout.setContentsMargins(0, 0, 0, 0)
        self.info_banner_container_layout.setSpacing(0)
        self.info_banner_container_layout.addWidget(self.info_banner)
        self.root_layout.addWidget(self.info_banner_container)

        self.workspace = QHBoxLayout()
        self.workspace.setSpacing(15)
        self.steps_card = self._build_steps_card()
        self.editor_card = self._build_editor_card()
        self.editor_card.setFixedHeight(ui_px(766))
        self.workspace.addWidget(self.steps_card, 12, Qt.AlignTop)
        self.workspace.addWidget(self.editor_card, 34, Qt.AlignTop)
        self.root_layout.addLayout(self.workspace, 1)
        self.root_layout.setAlignment(self.workspace, Qt.AlignTop)

        self.page_scroll.setWidget(container)
        outer.addWidget(self.page_scroll)
        self._steps = self._build_default_notebook_steps()
        self._populate_step_list()
        self.step_list.setCurrentRow(0)
        self.set_dataset(None, "")
        self._initializing = False
        self._apply_responsive_layout(self.width())

    def _build_default_notebook_steps(self):
        return [
            {
                "title": title,
                "description": description,
                "category": self.STEP_CATEGORIES.get(index, "Data Profiling"),
                "template": self.STEP_TEMPLATES.get(index, "Data Understanding"),
                "cell_type": "code",
                "cell_id": uuid.uuid4().hex[:8],
                "code": self.template_groups[
                    self.STEP_CATEGORIES.get(index, "Data Profiling")
                ][self.STEP_TEMPLATES.get(index, "Ringkasan dataset")],
                "output": "",
                "charts": [],
                "rich_outputs": [],
                "execution_count": None,
                "notebook_outputs": None,
                "notebook_metadata": {},
            }
            for index, (title, description) in enumerate(self.STEPS)
        ]

    def _build_template_groups(self):
        groups = {name: dict(templates) for name, templates in self.BASE_TEMPLATE_GROUPS.items()}
        for category, methods in self.MODEL_METHODS.items():
            groups[category] = {
                name: self._build_model_template(category, import_line, estimator)
                for name, import_line, estimator in methods
            }
        return {
            category: {
                name: self._remove_template_comments(source)
                for name, source in templates.items()
            }
            for category, templates in groups.items()
        }

    @staticmethod
    def _remove_template_comments(source: str) -> str:
        clean_lines = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
        return "\n".join(clean_lines).strip() + "\n"

    @staticmethod
    def _build_model_template(category: str, import_line: str, estimator: str) -> str:
        shared = f'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
{import_line}
'''
        dense_encoder = any(
            model_name in estimator
            for model_name in (
                "GaussianNB(",
                "LinearDiscriminantAnalysis(",
                "HistGradientBoosting",
                "CatBoost",
            )
        )
        if category == "Klasifikasi":
            return shared + f'''from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, f1_score

target_column = "target"
if target_column not in df.columns:
    raise ValueError(f"Kolom target '{{target_column}}' tidak ditemukan. Pilih kolom target dari daftar.")
data = df.dropna(subset=[target_column]).copy()
if len(data) < 5:
    raise ValueError("Diperlukan sedikitnya 5 baris dengan target yang terisi untuk klasifikasi.")
X = data.drop(columns=[target_column])
if X.shape[1] == 0:
    raise ValueError("Dataset harus memiliki sedikitnya 1 kolom fitur selain target.")
X = X.replace([np.inf, -np.inf], np.nan)
y = LabelEncoder().fit_transform(data[target_column].astype(str))
if len(np.unique(y)) < 2:
    raise ValueError("Klasifikasi memerlukan sedikitnya 2 kelas pada kolom target.")
numeric = X.select_dtypes(include="number").columns
categorical = X.select_dtypes(exclude="number").columns
transformers = []
if len(numeric):
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    transformers.append(("numeric", numeric_pipeline, numeric))
if len(categorical):
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output={not dense_encoder},
        )),
    ])
    transformers.append(("category", categorical_pipeline, categorical))
prepare = ColumnTransformer(transformers)
model = Pipeline([("prepare", prepare), ("model", {estimator})])
test_rows = max(1, int(np.ceil(len(y) * 0.2)))
class_counts = pd.Series(y).value_counts()
stratify = None
if (
    class_counts.min() >= 2
    and test_rows >= len(class_counts)
    and len(y) - test_rows >= len(class_counts)
):
    stratify = y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)
if hasattr(model.named_steps["model"], "n_neighbors"):
    model.set_params(model__n_neighbors=min(model.named_steps["model"].n_neighbors, len(X_train)))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {{accuracy_score(y_test, y_pred):.1%}}")
print(f"F1 weighted: {{f1_score(y_test, y_pred, average='weighted', zero_division=0):.1%}}")
report = pd.DataFrame(classification_report(
    y_test, y_pred, output_dict=True, zero_division=0
)).T
for metric in ("precision", "recall", "f1-score"):
    if metric in report:
        report[metric] = report[metric].map(
            lambda score: f"{{score:.1%}}" if pd.notna(score) else ""
        )
if "support" in report:
    report["support"] = report["support"].map(
        lambda count: f"{{count:,.0f}}" if pd.notna(count) else ""
    )
display(report)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title("Confusion matrix · {estimator.split('(')[0]}")
plt.tight_layout()
'''
        if category == "Regresi":
            return shared + f'''from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

target_column = "target"
if target_column not in df.columns:
    raise ValueError(f"Kolom target '{{target_column}}' tidak ditemukan. Pilih kolom target dari daftar.")
data = df.copy()
data[target_column] = pd.to_numeric(data[target_column], errors="coerce")
data = data.dropna(subset=[target_column]).copy()
if len(data) < 3:
    raise ValueError("Diperlukan sedikitnya 3 baris numerik dengan target yang terisi untuk regresi.")
X = data.drop(columns=[target_column])
if X.shape[1] == 0:
    raise ValueError("Dataset harus memiliki sedikitnya 1 kolom fitur selain target.")
X = X.replace([np.inf, -np.inf], np.nan)
y = data[target_column]
numeric = X.select_dtypes(include="number").columns
categorical = X.select_dtypes(exclude="number").columns
transformers = []
if len(numeric):
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    transformers.append(("numeric", numeric_pipeline, numeric))
if len(categorical):
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output={not dense_encoder},
        )),
    ])
    transformers.append(("category", categorical_pipeline, categorical))
prepare = ColumnTransformer(transformers)
model = Pipeline([("prepare", prepare), ("model", {estimator})])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
if hasattr(model.named_steps["model"], "n_neighbors"):
    model.set_params(model__n_neighbors=min(model.named_steps["model"].n_neighbors, len(X_train)))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"MAE: {{mean_absolute_error(y_test, y_pred):.4f}}")
print(f"RMSE: {{rmse:.4f}}")
print(f"R²: {{r2_score(y_test, y_pred):.4f}}")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(y_test, y_pred, alpha=0.6)
axes[0].set(xlabel="Aktual", ylabel="Prediksi", title="Aktual vs prediksi")
axes[1].scatter(y_pred, y_test - y_pred, alpha=0.6)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set(xlabel="Prediksi", ylabel="Residual", title="Residual plot")
plt.tight_layout()
'''
        if category == "Clustering":
            return shared + f'''from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

X = df.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")
labels = np.array([])
coordinates = np.empty((0, 2))
if len(X) < 2 or X.shape[1] == 0:
    print("Clustering memerlukan sedikitnya 2 baris dan 1 kolom numerik.")
else:
    X_scaled = StandardScaler().fit_transform(SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X))
    model = {estimator}
    if hasattr(model, "n_clusters") and model.n_clusters is not None:
        model.set_params(n_clusters=min(model.n_clusters, len(X_scaled)))
    if model.__class__.__name__ == "GaussianMixture":
        model.set_params(n_components=min(model.n_components, len(X_scaled)))
    labels = model.fit_predict(X_scaled)
    print("Jumlah baris per cluster:")
    print(pd.Series(labels).value_counts().sort_index())
    if len(set(labels)) > 1 and len(set(labels)) < len(labels):
        print(f"Silhouette score: {{silhouette_score(X_scaled, labels):.3f}}")
    components = min(2, X_scaled.shape[0], X_scaled.shape[1])
    coordinates = PCA(n_components=components).fit_transform(X_scaled)
    if components == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(len(coordinates))])
if len(labels):
    plt.figure(figsize=(8, 5))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap="tab10", alpha=0.7)
plt.title("Clustering · {estimator.split('(')[0]}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
'''
        return shared + f'''from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = df.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")
labels = np.array([])
coordinates = np.empty((0, 2))
if len(X) < 2 or X.shape[1] == 0:
    print("Deteksi anomali memerlukan sedikitnya 2 baris dan 1 kolom numerik.")
else:
    X_scaled = StandardScaler().fit_transform(SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X))
    model = {estimator}
    if hasattr(model, "n_neighbors"):
        model.set_params(n_neighbors=min(model.n_neighbors, len(X_scaled) - 1))
    labels = model.fit_predict(X_scaled)
    print("Jumlah baris normal/anomali:")
    print(pd.Series(labels).value_counts().sort_index())
    anomaly_count = int((labels == -1).sum())
    print(f"Anomali: {{anomaly_count:,}} dari {{len(labels):,}} baris ({{anomaly_count / max(len(labels), 1):.1%}})")
    components = min(2, X_scaled.shape[0], X_scaled.shape[1])
    coordinates = PCA(n_components=components).fit_transform(X_scaled)
    if components == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(len(coordinates))])
if len(labels):
    plt.figure(figsize=(8, 5))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap="coolwarm", alpha=0.7)
plt.title("Anomaly Detection · {estimator.split('(')[0]}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
'''

    def _build_steps_card(self):
        card = QFrame()
        card.setObjectName("editorAnalysisStepsCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        heading_row = QHBoxLayout()
        heading = QLabel("Tahapan Analisis")
        heading.setObjectName("editorAnalysisSectionTitle")
        heading_row.addWidget(heading, 1)
        self.step_actions_button = QPushButton("...")
        self.step_actions_button.setObjectName("editorAnalysisStepActions")
        self.step_actions_button.setFixedSize(28, 28)
        self.step_actions_button.setToolTip("Kelola langkah analisis dan sel notebook.")
        self.step_actions_menu = QMenu(self.step_actions_button)
        self._step_menu_actions = {}
        for key, label, callback in (
            ("add_code", "Tambah kode", self._add_step),
            ("add_text", "Tambah teks", self._add_markdown_cell),
            ("move_up", "Naik", lambda: self._move_step(-1)),
            ("move_down", "Turun", lambda: self._move_step(1)),
            ("duplicate", "Duplikat", self._duplicate_step),
            ("delete", "Hapus", self._delete_step),
        ):
            action = self.step_actions_menu.addAction(label)
            action.triggered.connect(callback)
            self._step_menu_actions[key] = action
        self.step_actions_button.setMenu(self.step_actions_menu)
        heading_row.addWidget(self.step_actions_button)
        layout.addLayout(heading_row)

        hint = QLabel("Pilih bagian untuk membuka kode template.")
        hint.setObjectName("editorAnalysisMuted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.step_list = QListWidget()
        self.step_list.setObjectName("editorAnalysisStepList")
        self.step_list.currentRowChanged.connect(self._select_step)
        layout.addWidget(self.step_list, 1)

        self.add_step_button = QPushButton("Tambah kode")
        self.add_step_button.setObjectName("secondaryButton")
        self.add_step_button.clicked.connect(self._add_step)
        self.add_markdown_button = QPushButton("Tambah teks")
        self.add_markdown_button.setObjectName("secondaryButton")
        self.add_markdown_button.clicked.connect(self._add_markdown_cell)
        self.move_cell_up_button = QPushButton("Naik")
        self.move_cell_up_button.setObjectName("secondaryButton")
        self.move_cell_up_button.clicked.connect(lambda: self._move_step(-1))
        self.move_cell_down_button = QPushButton("Turun")
        self.move_cell_down_button.setObjectName("secondaryButton")
        self.move_cell_down_button.clicked.connect(lambda: self._move_step(1))
        self.duplicate_cell_button = QPushButton("Duplikat")
        self.duplicate_cell_button.setObjectName("secondaryButton")
        self.duplicate_cell_button.clicked.connect(self._duplicate_step)
        self.delete_cell_button = QPushButton("Hapus")
        self.delete_cell_button.setObjectName("secondaryButton")
        self.delete_cell_button.clicked.connect(self._delete_step)
        for button in (
            self.add_step_button,
            self.add_markdown_button,
            self.move_cell_up_button,
            self.move_cell_down_button,
            self.duplicate_cell_button,
            self.delete_cell_button,
        ):
            button.setParent(card)
            button.hide()
        apply_card_shadow(card)
        return card

    def _build_editor_card(self):
        card = QFrame()
        card.setObjectName("editorAnalysisCodeCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)

        heading_row = QHBoxLayout()
        self.step_title = QLabel("Pilih tahapan analisis")
        self.step_title.setObjectName("editorAnalysisSectionTitle")
        heading_row.addWidget(self.step_title, 1)
        layout.addLayout(heading_row)

        self.notebook_toolbar = QHBoxLayout()
        self.open_notebook_button = QPushButton("Buka notebook")
        self.open_notebook_button.setObjectName("editorAnalysisNotebookButton")
        self.open_notebook_button.clicked.connect(lambda: self._open_notebook())
        self.open_notebook_button.setFixedSize(112, 34)
        self.notebook_toolbar.addWidget(self.open_notebook_button)
        self.reload_notebook_button = QPushButton("Muat ulang notebook")
        self.reload_notebook_button.setObjectName("editorAnalysisNotebookButton")
        self.reload_notebook_button.clicked.connect(self._reload_notebook)
        self.reload_notebook_button.setFixedSize(160, 34)
        self.reload_notebook_button.setEnabled(True)
        self.reload_notebook_button.setToolTip(
            "Pulihkan sel template bawaan DORA dan bersihkan sel dari notebook yang diimpor."
        )
        self.notebook_toolbar.addWidget(self.reload_notebook_button)
        self.save_notebook_button = QPushButton("Simpan notebook")
        self.save_notebook_button.setObjectName("editorAnalysisNotebookButton")
        self.save_notebook_button.clicked.connect(self._save_notebook)
        self.save_notebook_button.setFixedSize(138, 34)
        self.notebook_toolbar.addWidget(self.save_notebook_button)
        self.restart_kernel_button = QPushButton("Mulai ulang Python")
        self.restart_kernel_button.setObjectName("editorAnalysisNotebookButton")
        self.restart_kernel_button.clicked.connect(self._restart_kernel)
        self.restart_kernel_button.setToolTip("Mulai ulang kernel Python notebook.")
        self.restart_kernel_button.setFixedSize(144, 34)
        self.notebook_toolbar.addWidget(self.restart_kernel_button)
        self.notebook_status_label = QLabel("Belum disimpan")
        self.notebook_status_label.setObjectName("editorAnalysisMuted")
        self.notebook_toolbar.addWidget(self.notebook_status_label)
        self.notebook_toolbar.addStretch(1)
        layout.addLayout(self.notebook_toolbar)

        self.cell_type_combo = QComboBox()
        self.cell_type_combo.setObjectName("editorAnalysisCategoryCombo")
        self.cell_type_combo.addItem("Kode", "code")
        self.cell_type_combo.addItem("Markdown", "markdown")
        self.cell_type_combo.addItem("Raw", "raw")
        self.cell_type_combo.currentIndexChanged.connect(self._cell_type_changed)
        self.cell_type_combo.setFixedSize(112, 34)
        self.cell_type_combo.setToolTip("Jenis sel notebook yang sedang dipilih.")

        self.execution_note = QLabel(
            "Ctrl+Enter menjalankan sel. Shift+Enter menjalankan lalu berpindah ke sel berikutnya."
        )
        self.execution_note.setObjectName("editorAnalysisExecutionNote")
        self.execution_note.setWordWrap(True)
        self.execution_note.setParent(card)
        self.execution_note.hide()
        self._run_opacity_effect = QGraphicsOpacityEffect(self.execution_note)
        self._run_opacity_effect.setOpacity(1.0)
        self.execution_note.setGraphicsEffect(self._run_opacity_effect)
        self._run_animation = QPropertyAnimation(self._run_opacity_effect, b"opacity", self)
        self._run_animation.setDuration(850)
        self._run_animation.setStartValue(0.45)
        self._run_animation.setEndValue(1.0)
        self._run_animation.setEasingCurve(QEasingCurve.InOutSine)
        self._run_animation.setLoopCount(-1)
        self.run_button = QPushButton("Jalankan sel")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setEnabled(False)
        self.run_button.setText("Jalankan")
        self.run_button.setFixedSize(130, 34)
        self.run_button.setToolTip("Jalankan kode pada sel yang dipilih.")
        self.run_button.clicked.connect(self._execute_code)
        self.run_all_button = QPushButton("Jalankan semua")
        self.run_all_button.setObjectName("secondaryButton")
        self.run_all_button.setEnabled(False)
        self.run_all_button.setToolTip("Jalankan semua sel kode secara berurutan dalam satu sesi Python.")
        self.run_all_button.clicked.connect(self._run_all)
        self.cancel_button = QPushButton("Hentikan")
        self.cancel_button.setObjectName("secondaryButton")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._terminate_run)

        self.template_toolbar = QHBoxLayout()
        self.template_toolbar.setSpacing(12)
        category_block = QVBoxLayout()
        self.category_label = QLabel("Kelompok template")
        self.category_label.setObjectName("editorAnalysisFieldLabel")
        category_block.addWidget(self.category_label)
        self.template_category_combo = QComboBox()
        self.template_category_combo.setObjectName("editorAnalysisCategoryCombo")
        self.template_category_combo.addItems(self.template_groups.keys())
        self.template_category_combo.currentTextChanged.connect(self._category_changed)
        category_block.addWidget(self.template_category_combo)
        self.template_toolbar.addLayout(category_block, 1)
        template_block = QVBoxLayout()
        self.template_label = QLabel("Pilih template")
        self.template_label.setObjectName("editorAnalysisFieldLabel")
        template_block.addWidget(self.template_label)
        self.template_combo = QComboBox()
        self.template_combo.setObjectName("editorAnalysisTemplateCombo")
        self.template_combo.currentTextChanged.connect(self._apply_template)
        template_block.addWidget(self.template_combo)
        self.template_toolbar.addLayout(template_block, 1)
        layout.addSpacing(9)
        layout.addLayout(self.template_toolbar)

        self.step_description = QLabel("")
        self.step_description.setObjectName("editorAnalysisMuted")
        self.step_description.setWordWrap(True)
        layout.addWidget(self.step_description)

        self.library_row = QHBoxLayout()
        library_label = QLabel("Library")
        library_label.setObjectName("editorAnalysisFieldLabel")
        self.library_row.addWidget(library_label)
        self.library_input = QLineEdit()
        self.library_input.setObjectName("editorAnalysisLibraryInput")
        self.library_input.setPlaceholderText("pip install seaborn pandas atau xgboost==2.1.0")
        self.library_input.setToolTip(
            "Ketik pip install diikuti satu atau beberapa nama paket PyPI."
        )
        self.library_input.returnPressed.connect(self._install_library)
        self.library_row.addWidget(self.library_input, 1)
        self.install_library_button = QPushButton("Pasang")
        self.install_library_button.setObjectName("secondaryButton")
        self.install_library_button.clicked.connect(self._install_library)
        self.library_row.addWidget(self.install_library_button)
        layout.addLayout(self.library_row)
        self.library_status = QLabel("Library tambahan dipisahkan untuk proyek ini.")
        self.library_status.setObjectName("editorAnalysisMuted")
        self.library_status.setWordWrap(True)
        layout.addWidget(self.library_status)

        editor_heading = QHBoxLayout()
        self.editor_code_label = QLabel("Kode Python")
        self.editor_code_label.setObjectName("editorAnalysisFieldLabel")
        editor_heading.addWidget(self.editor_code_label)
        editor_heading.addStretch(1)
        self.copy_button = QPushButton("Salin kode")
        self.copy_button.setObjectName("secondaryButton")
        self.copy_button.clicked.connect(self._copy_code)
        editor_heading.addWidget(self.copy_button)
        self.reset_button = QPushButton("Pulihkan template")
        self.reset_button.setObjectName("secondaryButton")
        self.reset_button.clicked.connect(self._reset_code)
        editor_heading.addWidget(self.reset_button)
        self.more_actions_menu = QMenu(self)
        for action_button in (self.run_all_button, self.cancel_button):
            action = QWidgetAction(self.more_actions_menu)
            action.setDefaultWidget(action_button)
            self.more_actions_menu.addAction(action)
        self.more_actions_button = QPushButton("Lainnya")
        self.more_actions_button.setObjectName("editorAnalysisMoreActions")
        self.more_actions_button.setFixedSize(86, 34)
        self.more_actions_button.setMenu(self.more_actions_menu)
        self.more_actions_button.setToolTip(
            "Ubah jenis sel, jalankan semua sel, atau hentikan eksekusi."
        )
        cell_type_menu = self.more_actions_menu.addMenu("Jenis sel")
        for label, value in (("Kode", "code"), ("Markdown", "markdown"), ("Raw", "raw")):
            action = cell_type_menu.addAction(label)
            action.triggered.connect(
                lambda _checked=False, item=value: self.cell_type_combo.setCurrentIndex(
                    self.cell_type_combo.findData(item)
                )
            )
        self.more_actions_menu.addSeparator()
        editor_heading.addWidget(self.more_actions_button)
        editor_heading.addWidget(self.run_button)
        layout.addLayout(editor_heading)

        self.target_row = QHBoxLayout()
        self.target_label = QLabel("Kolom target")
        self.target_label.setObjectName("editorAnalysisFieldLabel")
        self.target_row.addWidget(self.target_label)
        self.target_combo = QComboBox()
        self.target_combo.setObjectName("editorAnalysisTargetCombo")
        self.target_combo.setMinimumWidth(190)
        self.target_combo.currentTextChanged.connect(self._target_changed)
        self.target_row.addWidget(self.target_combo)
        self.target_row.addStretch(1)
        self.template_toolbar.addLayout(self.target_row)

        self.code_chrome = QFrame()
        self.code_chrome.setObjectName("editorAnalysisCodeChrome")
        chrome_layout = QHBoxLayout(self.code_chrome)
        chrome_layout.setContentsMargins(12, 0, 12, 0)
        chrome_layout.setSpacing(10)
        self.editor_file_label = QLabel("CELL 01 · DATA UNDERSTANDING")
        self.editor_file_label.setObjectName("editorAnalysisFileTab")
        chrome_layout.addWidget(self.editor_file_label)
        chrome_layout.addStretch(1)
        chrome_hint = QLabel("Python · Ctrl+Enter untuk menjalankan")
        chrome_hint.setObjectName("editorAnalysisChromeHint")
        chrome_layout.addWidget(chrome_hint)
        layout.addWidget(self.code_chrome)
        self.code_chrome.setFixedHeight(28)

        self.code_editor = PythonCodeEditor()
        self.code_editor.setObjectName("editorAnalysisCodeEditor")
        self.code_editor.setPlaceholderText("Pilih tahapan untuk memuat template kode.")
        self.code_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        code_font = self.code_editor.font()
        code_font.setStyleHint(QFont.Monospace)
        code_font.setFamily("Consolas")
        code_font.setPixelSize(14)
        self.code_editor.setFont(code_font)
        self.code_editor._update_line_number_width()
        self.code_editor.setTabStopDistance(self.code_editor.fontMetrics().horizontalAdvance(" ") * 4)
        self.code_highlighter = PythonCodeHighlighter(self.code_editor.document())
        self.code_editor.runRequested.connect(self._execute_code)
        self.code_editor.runAndAdvanceRequested.connect(self._execute_code_and_advance)
        self.code_editor.setToolTip("Ctrl+Enter untuk menjalankan kode yang sedang dipilih.")
        self.code_editor.setFixedHeight(125)
        self.code_editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.code_editor.textChanged.connect(self._save_current_code)
        self.code_editor.cursorPositionChanged.connect(self._update_cursor_status)
        layout.addWidget(self.code_editor, 1)

        self.markdown_preview = QTextBrowser()
        self.markdown_preview.setObjectName("editorAnalysisMarkdownPreview")
        self.markdown_preview.setOpenExternalLinks(False)
        self.markdown_preview.setMinimumHeight(120)
        self.markdown_preview.hide()
        layout.addWidget(self.markdown_preview)

        self.code_statusbar = QFrame()
        self.code_statusbar.setObjectName("editorAnalysisCodeStatusbar")
        status_layout = QHBoxLayout(self.code_statusbar)
        status_layout.setContentsMargins(10, 2, 10, 2)
        status_layout.setSpacing(14)
        self.cursor_position_label = QLabel("Ln 1, Col 1")
        self.cursor_position_label.setObjectName("editorAnalysisStatusText")
        status_layout.addWidget(self.cursor_position_label)
        status_layout.addStretch(1)
        self.code_encoding_label = QLabel("UTF-8")
        self.code_encoding_label.setObjectName("editorAnalysisStatusText")
        status_layout.addWidget(self.code_encoding_label)
        self.code_language_label = QLabel("Python 3")
        self.code_language_label.setObjectName("editorAnalysisStatusLanguage")
        status_layout.addWidget(self.code_language_label)
        layout.addWidget(self.code_statusbar)
        self.code_statusbar.setFixedHeight(20)

        output_title = QLabel("Hasil eksekusi")
        output_title.setObjectName("editorAnalysisFieldLabel")
        layout.addWidget(output_title)
        self.output_tabs = QTabWidget()
        self.output_tabs.setObjectName("editorAnalysisOutputTabs")
        self.output_text = QTextBrowser()
        self.output_text.setObjectName("editorAnalysisOutputText")
        self.output_text.setOpenExternalLinks(False)
        self.output_text.setHtml("<p>Teks hasil dan pesan proses akan tampil di sini.</p>")
        self.output_text.setMinimumHeight(80)
        self.output_tabs.addTab(self.output_text, "Output teks")

        self.graphics_scroll = QScrollArea()
        self.graphics_scroll.setObjectName("editorAnalysisGraphicsScroll")
        self.graphics_scroll.setWidgetResizable(True)
        self.graphics_scroll.setFrameShape(QFrame.NoFrame)
        self.graphics_content = QWidget()
        self.graphics_content.setObjectName("editorAnalysisGraphicsContent")
        self.graphics_layout = QVBoxLayout(self.graphics_content)
        self.graphics_layout.setContentsMargins(10, 10, 10, 10)
        self.graphics_layout.setSpacing(12)
        self.graphics_empty = QLabel("Grafik dari kode akan muncul di sini.")
        self.graphics_empty.setObjectName("editorAnalysisGraphicsEmpty")
        self.graphics_empty.setAlignment(Qt.AlignCenter)
        self.graphics_layout.addWidget(self.graphics_empty)
        self.graphics_layout.addStretch(1)
        self.graphics_scroll.setWidget(self.graphics_content)
        self.graphics_scroll.setMinimumHeight(140)
        self.output_tabs.addTab(self.graphics_scroll, "Grafik")

        self.library_output = QPlainTextEdit()
        self.library_output.setObjectName("editorAnalysisOutputText")
        self.library_output.setReadOnly(True)
        self.library_output.setPlaceholderText("Log pemasangan library akan tampil di sini.")
        self.output_tabs.addTab(self.library_output, "Library")
        self.output_tabs.setMinimumHeight(210)
        layout.addWidget(self.output_tabs)

        apply_card_shadow(card)
        return card

    def _populate_step_list(self):
        self.step_list.blockSignals(True)
        self.step_list.clear()
        for index, step in enumerate(self._steps, start=1):
            cell_type = step.get("cell_type", "code")
            type_label = {"code": "Kode", "markdown": "Teks", "raw": "Raw"}.get(cell_type, "Kode")
            run_state = "  !" if step.get("has_error") else (
                f"  [{step['execution_count']}]" if step.get("execution_count") else ""
            )
            item = QListWidgetItem(f"{index}  {step['title']}{run_state}")
            item.setToolTip(f"Jenis sel: {type_label}. {step['description']}")
            self.step_list.addItem(item)
        self.step_list.blockSignals(False)
        self._update_cell_controls()

    def _update_cell_controls(self):
        has_cell = 0 <= self._selected_step < len(self._steps)
        controls_enabled = not self._execution_controls_locked
        if hasattr(self, "move_cell_up_button"):
            self.move_cell_up_button.setEnabled(controls_enabled and has_cell and self._selected_step > 0)
            self.move_cell_down_button.setEnabled(
                controls_enabled and has_cell and self._selected_step < len(self._steps) - 1
            )
            self.duplicate_cell_button.setEnabled(controls_enabled and has_cell)
            self.delete_cell_button.setEnabled(controls_enabled and has_cell and len(self._steps) > 1)
            self.add_step_button.setEnabled(controls_enabled)
            self.add_markdown_button.setEnabled(controls_enabled)
            self.step_list.setEnabled(controls_enabled)
            for key, button in {
                "add_code": self.add_step_button,
                "add_text": self.add_markdown_button,
                "move_up": self.move_cell_up_button,
                "move_down": self.move_cell_down_button,
                "duplicate": self.duplicate_cell_button,
                "delete": self.delete_cell_button,
            }.items():
                action = self._step_menu_actions.get(key)
                if action is not None:
                    action.setEnabled(button.isEnabled())

    def _set_execution_controls_locked(self, locked: bool):
        self._execution_controls_locked = bool(locked)
        for widget_name in (
            "dataset_scope_combo",
            "choose_dataset_button",
            "open_notebook_button",
            "reload_notebook_button",
            "save_notebook_button",
            "restart_kernel_button",
            "cell_type_combo",
            "template_category_combo",
            "template_combo",
            "reset_button",
            "target_combo",
            "code_editor",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(not locked)
        if not locked:
            has_cell = 0 <= self._selected_step < len(self._steps)
            is_code = has_cell and self._steps[self._selected_step].get("cell_type", "code") == "code"
            self.cell_type_combo.setEnabled(has_cell)
            self.code_editor.setEnabled(has_cell)
            self.template_category_combo.setEnabled(is_code)
            self.template_combo.setEnabled(is_code)
            self.reset_button.setEnabled(is_code)
            self.target_combo.setEnabled(
                is_code and self._dataframe is not None and self.target_combo.count() > 0
            )
        if hasattr(self, "reload_notebook_button"):
            self.reload_notebook_button.setEnabled(not locked)
        self._update_cell_controls()

    @Slot()
    def _update_cursor_status(self):
        cursor = self.code_editor.textCursor()
        self.cursor_position_label.setText(
            f"Ln {cursor.blockNumber() + 1}, Col {cursor.positionInBlock() + 1}"
        )

    def _update_editor_file_label(self):
        if 0 <= self._selected_step < len(self._steps):
            step = self._steps[self._selected_step]
            title = re.sub(r"\s+", " ", step.get("title") or "Analisis").strip()
            self.editor_file_label.setText(
                f"CELL {self._selected_step + 1:02d} · {title.upper()}"
            )
        else:
            self.editor_file_label.setText("PILIH CELL ANALISIS")

    def _fill_template_choices(self, category: str, selected: str = ""):
        choices = self.template_groups.get(category, {})
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        self.template_combo.addItems(choices.keys())
        target = selected if selected in choices else next(iter(choices), "")
        if target:
            self.template_combo.setCurrentText(target)
        optional_dependencies = {
            "XGBoost Classifier": "Memerlukan package xgboost.",
            "XGBoost Regressor": "Memerlukan package xgboost.",
            "LightGBM Classifier": "Memerlukan package lightgbm.",
            "LightGBM Regressor": "Memerlukan package lightgbm.",
            "CatBoost Classifier": "Memerlukan package catboost.",
            "CatBoost Regressor": "Memerlukan package catboost.",
        }
        for row in range(self.template_combo.count()):
            name = self.template_combo.itemText(row)
            tooltip = optional_dependencies.get(name, "")
            if tooltip:
                self.template_combo.setItemData(row, tooltip, Qt.ToolTipRole)
        self.template_combo.blockSignals(False)
        return target

    @Slot()
    def _save_current_code(self):
        if 0 <= self._selected_step < len(self._steps):
            step = self._steps[self._selected_step]
            source = self.code_editor.toPlainText()
            if source != step.get("code", ""):
                step["code"] = source
                self._mark_notebook_dirty()
            if step.get("cell_type") == "markdown" and hasattr(self, "markdown_preview"):
                self.markdown_preview.setMarkdown(source)

    def _mark_notebook_dirty(self):
        self._notebook_dirty = True
        if hasattr(self, "notebook_status_label"):
            self.notebook_status_label.setText("Perubahan belum disimpan")
        self._schedule_autosave()

    def _autosave_file_path(self) -> Path:
        app_data = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
        app_data_root = Path(app_data) if app_data else Path.home() / ".dora"
        return app_data_root / "editor_analysis" / "autosave.json"

    def _schedule_autosave(self):
        if not self._initializing and hasattr(self, "_autosave_timer"):
            self._autosave_timer.start()

    @Slot()
    def _save_autosave_state(self):
        if not hasattr(self, "code_editor"):
            return True
        self._save_current_code()
        state = {
            "version": 1,
            "steps": self._steps,
            "selected_step": self._selected_step,
            "notebook_metadata": self._notebook_metadata,
            "notebook_minor": self._notebook_minor,
            "notebook_dirty": self._notebook_dirty,
            "notebook_path": str(self._notebook_path) if self._notebook_path else "",
            "dataset_scope": self._dataset_scope,
            "custom_dataset_path": self._custom_dataset_path,
            "project_key": self._project_key,
        }
        path = self._autosave_file_path()
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        try:
            serialized = json.dumps(state, ensure_ascii=False, indent=2)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(serialized, encoding="utf-8")
            temporary_path.replace(path)
        except (OSError, TypeError, ValueError) as error:
            if hasattr(self, "library_status"):
                self.library_status.setText(f"Autosave lokal gagal: {error}")
            return False
        return True

    @Slot()
    def _save_autosave_on_exit(self):
        self._autosave_timer.stop()
        return self._save_autosave_state()

    def _restore_autosave_state(self):
        if self._autosave_restored:
            return
        self._autosave_restored = True
        path = self._autosave_file_path()
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or state.get("version") != 1:
                return
            steps = state.get("steps")
            if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
                return
            for index, step in enumerate(steps, start=1):
                step.setdefault("title", f"Sel {index:02d}")
                step.setdefault("description", "Dipulihkan dari autosave lokal.")
                cell_type = step.get("cell_type", "code")
                step["cell_type"] = (
                    cell_type if isinstance(cell_type, str) and cell_type in {"code", "markdown", "raw"}
                    else "code"
                )
                category = step.get("category", "Custom")
                step["category"] = (
                    category if isinstance(category, str) and category in self.template_groups else "Custom"
                )
                template = step.get("template", "Kode kosong")
                available_templates = self.template_groups[step["category"]]
                step["template"] = (
                    template if isinstance(template, str) and template in available_templates
                    else next(iter(available_templates))
                )
                step.setdefault("cell_id", uuid.uuid4().hex[:8])
                step["title"] = str(step["title"])
                step["description"] = str(step["description"])
                step["code"] = str(step.get("code", ""))
                step["output"] = str(step.get("output", ""))
                if not isinstance(step.get("charts"), list):
                    step["charts"] = []
                if not isinstance(step.get("rich_outputs"), list):
                    step["rich_outputs"] = []
                step.setdefault("notebook_metadata", {})
            self._steps = steps
            self._selected_step = -1
            try:
                selected_step = int(state.get("selected_step", 0))
            except (TypeError, ValueError):
                selected_step = 0
            self._notebook_metadata = (
                copy.deepcopy(state.get("notebook_metadata"))
                if isinstance(state.get("notebook_metadata"), dict)
                else {}
            )
            try:
                self._notebook_minor = int(state.get("notebook_minor", 5))
            except (TypeError, ValueError):
                self._notebook_minor = 5
            notebook_path = state.get("notebook_path")
            self._notebook_path = Path(notebook_path) if isinstance(notebook_path, str) and notebook_path else None
            self.reload_notebook_button.setEnabled(not self._execution_controls_locked)
            self._notebook_dirty = bool(state.get("notebook_dirty", True))
            self._dataset_scope = (
                "custom" if state.get("dataset_scope") == "custom" else "workspace"
            )
            self.dataset_scope_combo.blockSignals(True)
            self.dataset_scope_combo.setCurrentIndex(
                max(0, self.dataset_scope_combo.findData(self._dataset_scope))
            )
            self.dataset_scope_combo.blockSignals(False)
            custom_path = state.get("custom_dataset_path")
            self._custom_dataset_path = custom_path if isinstance(custom_path, str) else ""
            project_key = state.get("project_key")
            self._autosave_project_key = project_key if isinstance(project_key, str) else ""
            self._populate_step_list()
            if self._steps:
                selected_step = max(0, min(selected_step, len(self._steps) - 1))
                self.step_list.setCurrentRow(selected_step)
            else:
                self._select_step(-1)
            self.notebook_status_label.setText("Progres dipulihkan dari autosave lokal")
            self.library_status.setText(
                f"{len(self._steps)} sel dipulihkan dari sesi sebelumnya."
            )
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            self.library_status.setText(f"Autosave lokal tidak dapat dipulihkan: {error}")

    def _insert_cell(self, cell_type: str, index: int | None = None):
        if cell_type not in {"code", "markdown", "raw"}:
            cell_type = "code"
        if index is None:
            index = self._selected_step + 1 if self._selected_step >= 0 else len(self._steps)
        index = max(0, min(index, len(self._steps)))
        is_code = cell_type == "code"
        step = {
            "title": "Langkah kustom" if is_code else "Catatan",
            "description": "Tulis kode analisis." if is_code else "Tambahkan konteks dan catatan analisis.",
            "category": "Custom",
            "template": "Kode kosong" if is_code else "Teks Markdown",
            "cell_type": cell_type,
            "cell_id": uuid.uuid4().hex[:8],
            "code": self.template_groups["Custom"]["Kode kosong"] if is_code else "",
            "output": "",
            "charts": [],
            "rich_outputs": [],
            "has_error": False,
            "execution_count": None,
            "notebook_outputs": None,
            "notebook_metadata": {},
        }
        self._steps.insert(index, step)
        self._populate_step_list()
        self.step_list.setCurrentRow(index)
        self._mark_notebook_dirty()

    @Slot()
    def _add_markdown_cell(self):
        self._insert_cell("markdown")

    @Slot(int)
    def _cell_type_changed(self, index: int):
        if not 0 <= self._selected_step < len(self._steps):
            return
        cell_type = self.cell_type_combo.itemData(index)
        if cell_type not in {"code", "markdown", "raw"}:
            return
        step = self._steps[self._selected_step]
        if step.get("cell_type", "code") == cell_type:
            return
        self._save_current_code()
        step["cell_type"] = cell_type
        step["output"] = ""
        step["charts"] = []
        step["rich_outputs"] = []
        step["has_error"] = False
        step["execution_count"] = None
        step["notebook_outputs"] = None
        if cell_type == "code" and step.get("category") not in self.template_groups:
            step["category"] = "Custom"
            step["template"] = "Kode kosong"
        if cell_type == "markdown":
            step["template"] = "Teks Markdown"
        self._populate_step_list()
        self.step_list.setCurrentRow(self._selected_step)
        self._mark_notebook_dirty()

    @Slot()
    def _duplicate_step(self):
        if not 0 <= self._selected_step < len(self._steps):
            return
        self._save_current_code()
        duplicate = copy.deepcopy(self._steps[self._selected_step])
        duplicate["cell_id"] = uuid.uuid4().hex[:8]
        duplicate["title"] = f"{duplicate.get('title', 'Sel')} (salinan)"
        duplicate["output"] = ""
        duplicate["charts"] = []
        duplicate["rich_outputs"] = []
        duplicate["has_error"] = False
        duplicate["execution_count"] = None
        duplicate["notebook_outputs"] = None
        index = self._selected_step + 1
        self._steps.insert(index, duplicate)
        self._populate_step_list()
        self.step_list.setCurrentRow(index)
        self._mark_notebook_dirty()

    @Slot()
    def _delete_step(self):
        if len(self._steps) <= 1 or not 0 <= self._selected_step < len(self._steps):
            return
        index = self._selected_step
        self._steps.pop(index)
        self._populate_step_list()
        self.step_list.setCurrentRow(min(index, len(self._steps) - 1))
        self._mark_notebook_dirty()

    def _move_step(self, offset: int):
        index = self._selected_step
        destination = index + offset
        if not 0 <= index < len(self._steps) or not 0 <= destination < len(self._steps):
            return
        self._save_current_code()
        self._steps[index], self._steps[destination] = self._steps[destination], self._steps[index]
        self._populate_step_list()
        self.step_list.setCurrentRow(destination)
        self._mark_notebook_dirty()

    @Slot(int)
    def _select_step(self, index: int):
        self._save_current_code()
        if not 0 <= index < len(self._steps):
            self._selected_step = -1
            self.step_title.setText("Pilih tahapan analisis")
            self._update_editor_file_label()
            self.step_description.clear()
            self.code_editor.clear()
            self.markdown_preview.clear()
            self.markdown_preview.hide()
            self._set_cell_editing_mode("raw")
            self.code_editor.setEnabled(False)
            self.cell_type_combo.setEnabled(False)
            self.template_category_combo.setEnabled(False)
            self.template_combo.setEnabled(False)
            self._update_target_controls("")
            self.target_combo.setEnabled(False)
            self.output_text.clear()
            self._clear_graphs()
            self._update_cell_controls()
            self._schedule_autosave()
            return
        self._selected_step = index
        self.code_editor.setEnabled(not self._execution_controls_locked)
        self.cell_type_combo.setEnabled(not self._execution_controls_locked)
        step = self._steps[index]
        cell_type = step.get("cell_type", "code")
        is_code = cell_type == "code"
        self._update_editor_file_label()
        self.step_title.setText(step["template"])
        self.step_description.setText(step["description"])
        self.cell_type_combo.blockSignals(True)
        self.cell_type_combo.setCurrentIndex(max(0, self.cell_type_combo.findData(cell_type)))
        self.cell_type_combo.blockSignals(False)
        self.template_category_combo.blockSignals(True)
        self.template_category_combo.setCurrentText(step["category"])
        self.template_category_combo.blockSignals(False)
        if is_code:
            self.template_combo.blockSignals(True)
            selected_template = self._fill_template_choices(step["category"], step["template"])
            step["template"] = selected_template
            self.step_title.setText(selected_template)
            self.step_description.setText(self.CATEGORY_DESCRIPTIONS.get(step["category"], step["description"]))
            self.template_combo.blockSignals(False)
        self._set_cell_editing_mode(cell_type)
        self.code_editor.blockSignals(True)
        source = self._bind_target_column(step["code"]) if is_code else step["code"]
        if source != step["code"]:
            step["code"] = source
        self.code_editor.setPlainText(step["code"])
        self.code_editor.blockSignals(False)
        if cell_type == "markdown":
            self.markdown_preview.setMarkdown(step["code"])
        self._update_target_controls(step["code"] if is_code else "")
        self._render_step_output(step)
        self._update_cell_controls()
        self._schedule_autosave()

    def _set_cell_editing_mode(self, cell_type: str):
        is_code = cell_type == "code"
        desired_document = self.code_editor.document() if is_code else None
        if self.code_highlighter.document() is not desired_document:
            signals_were_blocked = self.code_editor.blockSignals(True)
            try:
                self.code_highlighter.setDocument(desired_document)
            finally:
                self.code_editor.blockSignals(signals_were_blocked)
        self.category_label.setVisible(is_code)
        self.template_category_combo.setVisible(is_code)
        self.template_label.setVisible(is_code)
        self.template_combo.setVisible(is_code)
        self.editor_code_label.setText("Kode Python" if is_code else ("Teks Markdown" if cell_type == "markdown" else "Teks Raw"))
        self.reset_button.setVisible(is_code)
        controls_enabled = not self._execution_controls_locked
        self.template_category_combo.setEnabled(controls_enabled and is_code)
        self.template_combo.setEnabled(controls_enabled and is_code)
        self.reset_button.setEnabled(controls_enabled and is_code)
        self.target_combo.setEnabled(controls_enabled and is_code and self.target_combo.count() > 0)
        self.markdown_preview.setVisible(cell_type == "markdown")
        self.output_tabs.setVisible(is_code)
        self.code_editor.setEnabled(not self._execution_controls_locked)
        self.cell_type_combo.setEnabled(not self._execution_controls_locked)
        self.run_button.setEnabled(is_code and self._dataframe is not None and not self._kernel_busy)
        self.run_all_button.setEnabled(self._dataframe is not None and any(
            step.get("cell_type", "code") == "code" for step in self._steps
        ) and not self._kernel_busy)

    @Slot(str)
    def _category_changed(self, category: str):
        if not 0 <= self._selected_step < len(self._steps) or self._steps[self._selected_step].get("cell_type", "code") != "code":
            return
        self._save_current_code()
        step = self._steps[self._selected_step]
        step["category"] = category
        selected_template = self._fill_template_choices(category)
        if not selected_template:
            return
        step["template"] = selected_template
        step["code"] = self._bind_target_column(self.template_groups[category][selected_template])
        self.step_title.setText(selected_template)
        self.step_description.setText(self.CATEGORY_DESCRIPTIONS.get(category, ""))
        self.code_editor.setPlainText(step["code"])
        self._update_target_controls(step["code"])
        step["notebook_outputs"] = None
        self._mark_notebook_dirty()

    @Slot(str)
    def _apply_template(self, template_name: str):
        if not 0 <= self._selected_step < len(self._steps) or self._steps[self._selected_step].get("cell_type", "code") != "code":
            return
        step = self._steps[self._selected_step]
        category = self.template_category_combo.currentText()
        step["template"] = template_name
        step["category"] = category
        step["code"] = self._bind_target_column(self.template_groups[category][template_name])
        self.step_title.setText(template_name)
        self.step_description.setText(self.CATEGORY_DESCRIPTIONS.get(category, ""))
        self.code_editor.blockSignals(True)
        self.code_editor.setPlainText(step["code"])
        self.code_editor.blockSignals(False)
        self._update_target_controls(step["code"])
        step["notebook_outputs"] = None
        self._mark_notebook_dirty()

    @Slot()
    def _reset_code(self):
        if not 0 <= self._selected_step < len(self._steps) or self._steps[self._selected_step].get("cell_type", "code") != "code":
            return
        step = self._steps[self._selected_step]
        step["code"] = self._bind_target_column(
            self.template_groups[step["category"]][step["template"]]
        )
        self.code_editor.setPlainText(step["code"])
        self._update_target_controls(step["code"])
        step["notebook_outputs"] = None
        self._mark_notebook_dirty()

    def _bind_target_column(self, source: str) -> str:
        target = self.target_combo.currentText() if hasattr(self, "target_combo") else ""
        if target and self.TARGET_ASSIGNMENT.search(source):
            return self.TARGET_ASSIGNMENT.sub(f"target_column = {target!r}", source)
        return source

    def _update_target_controls(self, source: str):
        visible = bool(self.TARGET_ASSIGNMENT.search(source))
        self.target_label.setVisible(visible)
        self.target_combo.setVisible(visible)
        self.target_combo.setEnabled(self._dataframe is not None and self.target_combo.count() > 0)

    @Slot(str)
    def _target_changed(self, target: str):
        if not target or not 0 <= self._selected_step < len(self._steps) or self._steps[self._selected_step].get("cell_type", "code") != "code":
            return
        source = self.code_editor.toPlainText()
        if not self.TARGET_ASSIGNMENT.search(source):
            return
        source = self.TARGET_ASSIGNMENT.sub(f"target_column = {target!r}", source)
        self.code_editor.blockSignals(True)
        self.code_editor.setPlainText(source)
        self.code_editor.blockSignals(False)
        self._steps[self._selected_step]["code"] = source
        self._steps[self._selected_step]["notebook_outputs"] = None
        self._mark_notebook_dirty()

    def _preferred_target(self, columns: list[str]) -> str:
        by_normalized = {column.casefold(): column for column in columns}
        for candidate in (
            "target", "label", "class", "survived", "outcome", "diagnosis",
            "churn", "y", "price", "saleprice", "medv", "score", "value", "income",
        ):
            if candidate in by_normalized:
                return by_normalized[candidate]
        return columns[-1] if columns else ""

    @Slot()
    def _copy_code(self):
        QApplication.clipboard().setText(self.code_editor.toPlainText())
        self.execution_note.setText("Kode berhasil disalin ke clipboard.")

    def _execution_interpreter(self) -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable)
        # Reuse the application's already-loaded Python environment. Creating a
        # fresh venv on the first Run adds a large, unnecessary startup delay.
        return Path(sys.executable)

    def _environment_site_packages(self) -> Path:
        if self._project_environment is None:
            raise RuntimeError("Workspace proyek belum disiapkan.")
        if os.name == "nt":
            return self._project_environment / "Lib" / "site-packages"
        return self._project_environment / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

    def _ensure_environment(self, callback):
        if self._project_environment is None:
            self.execution_note.setText("Lingkungan proyek belum tersedia.")
            return
        callback()

    def _read_environment_output(self):
        process = self._environment_process
        if process is None:
            return
        output = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if output.strip():
            self._append_library_log(output)

    def _environment_error(self, error):
        if self._environment_process is None or error != QProcess.FailedToStart:
            return
        self.execution_note.setText("Python gagal menyiapkan lingkungan proyek.")
        self._append_library_log("Gagal membuat lingkungan Python proyek.\n")
        self._environment_process.deleteLater()
        self._environment_process = None
        self._environment_callback = None
        self._finish_execution(False)

    def _environment_finished(self, exit_code: int, _exit_status):
        process = self._environment_process
        if process is None:
            return
        self._read_environment_output()
        self._environment_process = None
        process.deleteLater()
        callback = self._environment_callback
        self._environment_callback = None
        if exit_code != 0 or not self._execution_interpreter().exists():
            self.execution_note.setText("Lingkungan Python proyek gagal dibuat.")
            self._append_library_log("Pembuatan lingkungan Python gagal.\n")
            self._finish_execution(False)
            return
        if callback is not None:
            callback()

    def _queue_execution(self, step_indexes: list[int]):
        if self._dataframe is None:
            self.execution_note.setText("Muat dataset sebelum menjalankan kode.")
            return
        if self._kernel_busy or self._execution_queue or self._package_process is not None:
            return
        self._save_current_code()
        self._execution_queue = [
            index for index in step_indexes
            if 0 <= index < len(self._steps)
            and self._steps[index].get("cell_type", "code") == "code"
        ]
        if not self._execution_queue:
            return
        self._execution_total = len(self._execution_queue)
        self._execution_completed = 0
        self._queue_had_error = False
        self._auto_install_attempted.clear()
        self._auto_install_context = None
        self._advance_after_run = self._advance_after_run and len(self._execution_queue) == 1
        self._set_execution_controls_locked(True)
        self.run_button.setEnabled(False)
        self.run_all_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.install_library_button.setEnabled(False)
        self.execution_note.setText("Menyiapkan sesi Python...")
        self._run_animation.start()
        self._ensure_environment(self._start_kernel)

    @Slot()
    def _execute_code(self):
        if self._selected_step < 0 or self._steps[self._selected_step].get("cell_type", "code") != "code":
            self.execution_note.setText("Pilih bagian analisis yang akan dijalankan.")
            return
        self._queue_execution([self._selected_step])

    @Slot()
    def _execute_code_and_advance(self):
        self._advance_after_run = True
        self._execute_code()
        if not self._execution_queue and not self._kernel_busy:
            self._advance_after_run = False

    def _focus_next_cell(self):
        if self._selected_step + 1 < len(self._steps):
            self.step_list.setCurrentRow(self._selected_step + 1)
        else:
            self._insert_cell("code", len(self._steps))

    @Slot()
    def _run_all(self):
        self._save_current_code()
        self._queue_execution(list(range(len(self._steps))))

    def _start_kernel(self):
        if self._project_workspace is None or self._dataframe is None:
            self._finish_execution(False)
            return
        if self._kernel_process is not None and self._kernel_process.state() != QProcess.NotRunning:
            if self._kernel_ready:
                self._send_next_cell()
            return
        try:
            self._project_workspace.mkdir(parents=True, exist_ok=True)
            dataset_path = self._project_workspace / "active_dataset.pkl"
            self._dataframe.to_pickle(dataset_path)
            self._dataset_pickle_path = dataset_path
            worker_path = Path(__file__).with_name("editor_notebook_kernel.py")
            site_packages = self._environment_site_packages()
            site_packages.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            self._append_library_log(f"Gagal menyiapkan kernel: {error}\n")
            self._finish_execution(False)
            return

        process = QProcess(self)
        self._kernel_process = process
        self._kernel_buffer.clear()
        self._kernel_ready = False
        process.setProgram(str(self._execution_interpreter()))
        if getattr(sys, "frozen", False):
            process.setArguments([
                "--dora-editor-kernel",
                str(site_packages),
                str(dataset_path),
                str(self._project_workspace),
            ])
        else:
            process.setArguments([
                "-I",
                str(worker_path),
                str(site_packages),
                str(dataset_path),
                str(self._project_workspace),
                sysconfig.get_paths().get("purelib", ""),
            ])
        process.setWorkingDirectory(str(self._project_workspace))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUTF8", "1")
        environment.insert("MPLBACKEND", "Agg")
        environment.insert("MPLCONFIGDIR", str(self._project_workspace / "matplotlib"))
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_kernel_output)
        process.finished.connect(
            lambda code, status, source=process: self._kernel_finished(source, code, status)
        )
        process.errorOccurred.connect(
            lambda error, source=process: self._kernel_error(source, error)
        )
        process.start()
        self.execution_note.setText("Menyalakan sesi Python...")

    def _send_next_cell(self):
        process = self._kernel_process
        if process is None or not self._kernel_ready or self._kernel_busy:
            return
        if not self._execution_queue:
            self._finish_execution(not self._queue_had_error)
            return
        index = self._execution_queue.pop(0)
        step = self._steps[index]
        if step.get("cell_type", "code") != "code":
            self._send_next_cell()
            return
        source = self._bind_target_column(step["code"])
        if self.TARGET_ASSIGNMENT.search(source):
            selected_target = self.target_combo.currentText()
            available_columns = {str(column) for column in self._dataframe.columns}
            if not selected_target or selected_target not in available_columns:
                step["output"] = "Pilih kolom target yang tersedia pada dataset sebelum menjalankan bagian ini."
                step["charts"] = []
                step["has_error"] = True
                self._queue_had_error = True
                self._execution_completed += 1
                if index == self._selected_step:
                    self._render_step_output(step)
                self._send_next_cell()
                return
            step["code"] = source
            if index == self._selected_step:
                self.code_editor.blockSignals(True)
                self.code_editor.setPlainText(source)
                self.code_editor.blockSignals(False)

        self._kernel_request_id += 1
        self._running_step_index = index
        self._kernel_busy = True
        self._run_timeout.start(120_000)
        self.execution_note.setText(
            f"Menjalankan {step['title']} ({self._execution_completed + 1}/{self._execution_total})"
        )
        request = {
            "kind": "execute",
            "id": self._kernel_request_id,
            "code": source,
        }
        process.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))

    @Slot()
    def _read_kernel_output(self):
        process = self._kernel_process
        if process is None:
            return
        self._kernel_buffer.extend(bytes(process.readAllStandardOutput()))
        while b"\n" in self._kernel_buffer:
            line, _, remaining = self._kernel_buffer.partition(b"\n")
            self._kernel_buffer = bytearray(remaining)
            if not line.strip():
                continue
            try:
                message = json.loads(line.decode("utf-8", errors="replace"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._append_library_log(line.decode("utf-8", errors="replace") + "\n")
                continue
            self._handle_kernel_message(message)

    def _remove_dataset_snapshot(self):
        path = self._dataset_pickle_path
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return
        self._dataset_pickle_path = None

    def _missing_module_package(self, output: str) -> tuple[str, str] | None:
        match = re.search(
            r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]",
            output,
        )
        if match is None:
            return None
        module = match.group(1).split(".", 1)[0]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", module):
            return module, ""
        if module.casefold() in getattr(sys, "stdlib_module_names", ()):
            return module, ""
        package = self.MISSING_MODULE_PACKAGES.get(module.casefold())
        return module, package or ""

    @staticmethod
    def _dataset_compatibility_hint(output: str) -> str:
        lowered = output.casefold()
        if "could not convert string to float" in lowered or "could not convert string" in lowered:
            detail = "Ada fitur teks/kategori yang perlu diubah menjadi angka sebelum metode ini dijalankan."
        elif "unknown label type: continuous" in lowered or "continuous target" in lowered:
            detail = "Target tampak numerik kontinu; gunakan regresi, atau kelompokkan target menjadi kelas untuk klasifikasi."
        elif (
            "invalid classes inferred" in lowered
            or "at least 2 classes" in lowered
            or "least populated class" in lowered
            or "target contains only one unique value" in lowered
        ):
            detail = "Target klasifikasi perlu memiliki sedikitnya dua kelas dan cukup banyak baris pada tiap kelas."
        elif "contains nan" in lowered or "input y contains nan" in lowered or "missing values" in lowered:
            detail = "Fitur atau target masih berisi NaN; bersihkan atau imputasi nilai kosong sebelum menjalankan metode."
        elif "dtypes for data" in lowered or "categorical" in lowered and "unsupported" in lowered:
            detail = "Metode ini belum menerima tipe kolom kategorikal tersebut; lakukan encoding sebelum pemodelan."
        elif (
            "n_samples" in lowered
            or "minimum of" in lowered
            or "at least 2" in lowered
            or "inconsistent numbers of samples" in lowered
        ):
            detail = "Jumlah baris atau pembagian train/test tidak memenuhi batas minimum metode ini."
        elif "0 feature" in lowered or "no features" in lowered:
            detail = "Tidak ada kolom fitur yang dapat dipakai setelah kolom target dipisahkan."
        elif "keyerror" in lowered or "not in index" in lowered or "column" in lowered and "not found" in lowered:
            detail = "Kolom yang diminta tidak ada di dataset; pilih kolom target dan fitur yang tersedia."
        elif "valueerror" in lowered or "typeerror" in lowered:
            detail = "Periksa kecocokan jenis metode, kolom target, tipe fitur, nilai kosong, dan jumlah baris dataset."
        else:
            return ""
        return f"Pemeriksaan dataset: {detail}\nTraceback lengkap ada di bawah.\n\n"

    def _start_automatic_package_install(
        self, index: int, module: str, package: str, original_output: str
    ):
        continuing_install = self._auto_install_context is not None
        self._auto_install_attempted.add(module)
        self._auto_install_context = {
            "index": index,
            "module": module,
            "package": package,
            "original_output": original_output,
        }
        self._package_install_mode = "auto"
        self._installing_packages = [package]
        if not continuing_install:
            self._package_log = ""
            self.library_output.clear()
        self.output_tabs.setCurrentIndex(2)
        self._append_library_log(
            f"$ python -m pip install --no-deps --target \"{self._environment_site_packages()}\" {package}\n\n"
        )
        self._append_library_log(
            f"Dependency '{module}' belum tersedia. DORA memasang '{package}' otomatis.\n"
        )
        self.library_status.setText(f"Memasang {package} otomatis untuk sel notebook...")
        self.execution_note.setText(f"Memasang library {package}; sel akan dijalankan ulang setelah selesai...")
        self._run_timeout.stop()
        self._kernel_busy = False
        self._running_step_index = -1
        self._execution_queue.insert(0, index)
        self._ensure_environment(lambda: self._start_package_install([package]))

    def _finish_automatic_install_failure(self, exit_code: int):
        context = self._auto_install_context
        self._auto_install_context = None
        if context is None:
            return
        index = context["index"]
        if self._execution_queue and self._execution_queue[0] == index:
            self._execution_queue.pop(0)
        if 0 <= index < len(self._steps):
            step = self._steps[index]
            step["output"] = (
                context["original_output"].rstrip()
                + f"\n\nDORA mencoba memasang '{context['package']}' otomatis, tetapi pemasangan gagal "
                + f"(kode {exit_code}). Buka tab Library untuk melihat log pemasangan."
            )
            step["charts"] = []
            step["has_error"] = True
            step["notebook_outputs"] = None
            self._queue_had_error = True
            self._execution_completed += 1
            self._mark_notebook_dirty()
            self._populate_step_list()
            if index == self._selected_step:
                self._render_step_output(step)
        self._kernel_busy = False
        self._running_step_index = -1
        self._run_timeout.stop()
        self.library_status.setText(
            f"Pemasangan {context['package']} otomatis gagal. Lihat log di tab Library."
        )
        if self._execution_queue:
            self._run_animation.start()
            self._send_next_cell()
        else:
            self._finish_execution(False)

    def _handle_kernel_message(self, message: dict):
        if message.get("kind") == "ready":
            self._kernel_ready = True
            self._remove_dataset_snapshot()
            self._send_next_cell()
            return
        if message.get("id") != self._kernel_request_id or not self._kernel_busy:
            return
        index = self._running_step_index
        if not 0 <= index < len(self._steps):
            return
        step = self._steps[index]
        output = str(message.get("output", ""))
        has_error = bool(message.get("error"))
        retry_context = self._auto_install_context
        is_auto_retry = bool(retry_context and retry_context.get("index") == index)
        missing = self._missing_module_package(output) if has_error else None
        if missing is not None:
            module, package = missing
            if package and module not in self._auto_install_attempted:
                self._start_automatic_package_install(index, module, package, output)
                return
            if package:
                output = (
                    output.rstrip()
                    + f"\n\nLibrary '{module}' masih belum tersedia setelah dicoba dipasang. "
                    + "Periksa versi Python, koneksi internet, dan log pada tab Library."
                )
            else:
                output = (
                    output.rstrip()
                    + f"\n\nModul '{module}' belum dikenali sebagai dependensi yang bisa dipasang otomatis. "
                    + "Pastikan nama paket PyPI-nya, lalu pasang melalui tab Library."
                )
                self.output_tabs.setCurrentIndex(2)
                self.library_status.setText(
                    f"Modul {module} belum ada. Masukkan nama paket PyPI yang sesuai di Library."
                )
        elif has_error:
            output = self._dataset_compatibility_hint(output) + output

        if is_auto_retry:
            if has_error:
                self.library_status.setText(
                    f"{retry_context['package']} sudah dipasang; sel masih gagal. Periksa output sel."
                )
            else:
                self.library_status.setText(
                    f"{retry_context['package']} terpasang otomatis; sel berhasil dijalankan ulang."
                )
            self._auto_install_context = None

        step["output"] = output
        step["charts"] = message.get("charts", [])
        step["rich_outputs"] = message.get("rich_outputs", [])
        step["notebook_outputs"] = None
        step["has_error"] = has_error
        step["execution_count"] = (step.get("execution_count") or 0) + 1
        self._mark_notebook_dirty()
        self._populate_step_list()
        self._queue_had_error = self._queue_had_error or step["has_error"]
        self._kernel_busy = False
        self._running_step_index = -1
        self._run_timeout.stop()
        self._execution_completed += 1
        if index == self._selected_step:
            self._render_step_output(step)
        advance = self._advance_after_run
        if advance:
            self._advance_after_run = False
            QTimer.singleShot(0, self._focus_next_cell)
        if self._execution_queue:
            self._send_next_cell()
        else:
            self._finish_execution(not self._queue_had_error)

    def _kernel_error(self, process: QProcess, error):
        if process is not self._kernel_process or error != QProcess.FailedToStart:
            return
        self.execution_note.setText("Kernel Python gagal dijalankan.")
        self._kernel_process = None
        self._kernel_ready = False
        self._kernel_busy = False
        self._remove_dataset_snapshot()
        process.deleteLater()
        self._finish_execution(False)

    def _kernel_finished(self, process: QProcess, exit_code: int, _exit_status):
        if process is not self._kernel_process:
            return
        self._read_kernel_output()
        self._kernel_process = None
        self._kernel_ready = False
        self._kernel_busy = False
        self._remove_dataset_snapshot()
        process.deleteLater()
        if self._execution_queue or self._running_step_index >= 0:
            index = self._running_step_index
            if index < 0 and self._execution_queue:
                index = self._execution_queue[0]
            if 0 <= index < len(self._steps):
                step = self._steps[index]
                step["output"] = (
                    step.get("output", "")
                    + f"\nSesi Python berhenti sebelum semua kode selesai (kode {exit_code}).\n"
                ).strip()
                step["charts"] = []
                step["has_error"] = True
                if index == self._selected_step:
                    self._render_step_output(step)
            self._execution_queue.clear()
            self._running_step_index = -1
            self._run_timeout.stop()
            self._finish_execution(False)
        else:
            self._finish_execution(exit_code == 0 and not self._queue_had_error)

    @Slot()
    def _on_run_timeout(self):
        index = self._running_step_index
        if 0 <= index < len(self._steps):
            step = self._steps[index]
            step["output"] = (step.get("output", "") + "\nEksekusi melewati batas 120 detik; sesi Python dihentikan.\n").strip()
            step["charts"] = []
            step["has_error"] = True
            if index == self._selected_step:
                self._render_step_output(step)
        self._execution_queue.clear()
        self._queue_had_error = True
        self._stop_kernel()
        self._finish_execution(False)

    @Slot()
    def _terminate_run(self):
        if not self._kernel_busy and not self._execution_queue:
            return
        index = self._running_step_index
        if 0 <= index < len(self._steps):
            step = self._steps[index]
            step["output"] = (step.get("output", "") + "\nProses dihentikan pengguna; sesi Python direset.\n").strip()
            step["charts"] = []
            step["has_error"] = True
            if index == self._selected_step:
                self._render_step_output(step)
        self._execution_queue.clear()
        self._run_timeout.stop()
        self._stop_kernel()
        self._finish_execution(False)

    def _stop_kernel(self):
        process = self._kernel_process
        self._kernel_process = None
        self._kernel_ready = False
        self._kernel_busy = False
        self._execution_queue.clear()
        self._running_step_index = -1
        self._run_timeout.stop()
        self._kernel_buffer.clear()
        if process is not None:
            process.kill()
            process.waitForFinished(1500)
            process.deleteLater()
        self._remove_dataset_snapshot()

    @Slot()
    def _restart_kernel(self):
        if self._kernel_busy or self._execution_queue:
            self.execution_note.setText("Hentikan proses yang berjalan sebelum mereset sesi Python.")
            return
        self._stop_kernel()
        self.execution_note.setText("Sesi Python direset. Variabel dibersihkan saat Run berikutnya.")

    def _finish_execution(self, success: bool):
        self._run_timeout.stop()
        self._run_animation.stop()
        self._run_opacity_effect.setOpacity(1.0)
        active_is_code = (
            0 <= self._selected_step < len(self._steps)
            and self._steps[self._selected_step].get("cell_type", "code") == "code"
        )
        self.run_button.setEnabled(self._dataframe is not None and active_is_code)
        self.run_all_button.setEnabled(
            self._dataframe is not None
            and any(step.get("cell_type", "code") == "code" for step in self._steps)
        )
        self.cancel_button.setEnabled(False)
        self.install_library_button.setEnabled(self._dataframe is not None)
        self._set_execution_controls_locked(False)
        if success:
            self.execution_note.setText("Analisis selesai. Variabel tetap tersedia di sesi Python ini.")
        elif self._execution_completed:
            self.execution_note.setText("Sebagian analisis selesai; periksa output pada bagian yang error.")
        else:
            self.execution_note.setText(
                "Eksekusi belum selesai. Periksa output bagian analisis atau log Library."
            )
        if not self._execution_queue:
            self._running_step_index = -1
            self._advance_after_run = False

    def _append_library_log(self, text: str):
        self._package_log = (self._package_log + text)[-120_000:]
        self.library_output.setPlainText(self._package_log)
        cursor = self.library_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.library_output.setTextCursor(cursor)

    @staticmethod
    def _parse_package_specs(command: str) -> list[str]:
        """Accept package names or a safe, pip-install-shaped command."""
        try:
            parts = shlex.split(command.strip(), posix=True)
        except ValueError:
            return []
        if not parts:
            return []

        lowered = [part.casefold() for part in parts]
        if lowered[:2] in (["pip", "install"], ["pip3", "install"]):
            parts = parts[2:]
        elif len(lowered) >= 4 and lowered[1:4] == ["-m", "pip", "install"] and lowered[0] in {
            "python", "python3", "python.exe", "py",
        }:
            parts = parts[4:]
        elif any(part in {"pip", "pip3", "install", "-m"} for part in lowered[:4]):
            return []

        package_pattern = re.compile(
            r"[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9_,.-]+\])?"
            r"(?:(?:==|~=|>=|<=|!=|>|<)[A-Za-z0-9.*+!-]+)?"
        )
        if not parts or any(not package_pattern.fullmatch(part) for part in parts):
            return []
        return list(dict.fromkeys(parts))

    @Slot()
    def _install_library(self):
        packages = self._parse_package_specs(self.library_input.text())
        if not packages:
            self.library_status.setText(
                "Tulis pip install diikuti nama paket, misalnya pip install seaborn pandas."
            )
            return
        if self._dataframe is None:
            self.library_status.setText("Muat dataset dan proyek sebelum memasang library.")
            return
        if self._kernel_busy or self._execution_queue or self._package_process is not None:
            return
        self._stop_kernel()
        self._installing_packages = packages
        self._package_install_mode = "manual"
        self._package_log = ""
        self.library_output.clear()
        self.output_tabs.setCurrentIndex(2)
        self.library_status.setText(
            f"Memasang {', '.join(packages)} ke lingkungan proyek..."
        )
        self.install_library_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.run_all_button.setEnabled(False)
        self.execution_note.setText("Pemasangan library sedang berjalan...")
        self._run_animation.start()
        self._ensure_environment(lambda: self._start_package_install(packages))

    def _start_package_install(self, packages: list[str]):
        process = QProcess(self)
        self._package_process = process
        process.setProgram(str(self._execution_interpreter()))
        pip_arguments = [
            "install",
            "--disable-pip-version-check",
            "--upgrade",
        ]
        if self._package_install_mode == "auto":
            pip_arguments.extend(
                ["--no-deps", "--target", str(self._environment_site_packages())]
            )
        if getattr(sys, "frozen", False):
            if self._package_install_mode != "auto":
                pip_arguments.extend(["--target", str(self._environment_site_packages())])
            process.setArguments([
                "--dora-editor-pip",
                *pip_arguments,
                *packages,
            ])
        else:
            process.setArguments(["-m", "pip", *pip_arguments, *packages])
        process.setWorkingDirectory(str(self._project_workspace))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PIP_NO_INPUT", "1")
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_package_output)
        process.finished.connect(self._package_finished)
        process.errorOccurred.connect(self._package_error)
        process.start()

    @Slot()
    def _read_package_output(self):
        process = self._package_process
        if process is None:
            return
        output = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_library_log(output)

    @Slot(QProcess.ProcessError)
    def _package_error(self, error):
        if self._package_process is None or error != QProcess.FailedToStart:
            return
        self._append_library_log("pip gagal dijalankan di lingkungan proyek.\n")
        self._package_finished(-1, QProcess.CrashExit)

    @Slot(int, QProcess.ExitStatus)
    def _package_finished(self, exit_code: int, _exit_status):
        process = self._package_process
        if process is None:
            return
        self._read_package_output()
        self._package_process = None
        process.deleteLater()
        self._run_animation.stop()
        self._run_opacity_effect.setOpacity(1.0)
        mode = self._package_install_mode
        self._package_install_mode = ""
        if mode == "auto":
            context = self._auto_install_context
            if (
                exit_code == 0
                and context is not None
                and self._kernel_process is not None
                and self._kernel_ready
            ):
                self.library_status.setText(
                    f"{context['package']} terpasang. Menjalankan ulang sel notebook..."
                )
                self.execution_note.setText("Library siap; mengulangi sel yang tadi gagal...")
                self._run_animation.start()
                self._send_next_cell()
            else:
                self._append_library_log(
                    f"\nPemasangan otomatis berakhir dengan kode {exit_code}.\n"
                )
                self._finish_automatic_install_failure(exit_code if exit_code else -1)
            return
        self.install_library_button.setEnabled(self._dataframe is not None)
        self.run_button.setEnabled(self._dataframe is not None)
        self.run_all_button.setEnabled(self._dataframe is not None)
        if exit_code == 0:
            package_names = ", ".join(self._installing_packages)
            self.library_status.setText(
                f"{package_names} terpasang. Jalankan ulang bagian analisis untuk memakai library ini."
            )
            self.execution_note.setText("Library siap digunakan pada sesi Python berikutnya.")
        else:
            self.library_status.setText("Pemasangan gagal. Periksa log pada tab Library.")
            self.execution_note.setText("Library tidak berhasil dipasang.")

    def _render_step_output(self, step: dict):
        output_parts = []
        text_output = step.get("output", "")
        if text_output:
            output_parts.append(f"<pre>{html.escape(text_output)}</pre>")
        for rich_output in step.get("rich_outputs", []):
            data = rich_output.get("data", {}) if isinstance(rich_output, dict) else {}
            rich_html = data.get("text/html")
            rich_text = data.get("text/plain")
            if isinstance(rich_html, list):
                rich_html = "".join(rich_html)
            if isinstance(rich_text, list):
                rich_text = "".join(rich_text)
            if rich_html:
                output_parts.append(str(rich_html))
            elif rich_text and not text_output:
                output_parts.append(f"<pre>{html.escape(str(rich_text))}</pre>")
        self.output_text.setHtml(
            "<html><body style='font-family:Segoe UI, sans-serif; color:#20324A'>"
            + "<hr style='border:0;border-top:1px solid #e3e8ef'>".join(output_parts)
            + "</body></html>"
            if output_parts
            else "<p>Sel ini belum memiliki output.</p>"
        )
        self._clear_graphs()
        charts = step.get("charts", [])
        for chart in charts[:8]:
            try:
                image = QImage.fromData(base64.b64decode(chart), "PNG")
            except Exception:
                continue
            if image.isNull():
                continue
            pixmap = QPixmap.fromImage(image)
            self._graphics.append(pixmap)
            label = QLabel()
            label.setObjectName("editorAnalysisOutputGraph")
            label.setAlignment(Qt.AlignCenter)
            max_width = max(320, min(900, self.graphics_scroll.viewport().width() - 32))
            label.setPixmap(pixmap.scaledToWidth(max_width, Qt.SmoothTransformation))
            self.graphics_layout.addWidget(label)
        if not self._graphics:
            self.graphics_empty = QLabel("Bagian ini belum menghasilkan grafik.")
            self.graphics_empty.setObjectName("editorAnalysisGraphicsEmpty")
            self.graphics_empty.setAlignment(Qt.AlignCenter)
            self.graphics_layout.addWidget(self.graphics_empty)
            self.graphics_layout.addStretch(1)
        output_title = "Output teks"
        if step.get("execution_count") is not None:
            output_title += f" [{step['execution_count']}]"
        if step.get("has_error"):
            output_title += " - Error"
        self.output_tabs.setTabText(0, output_title)

    def _clear_graphs(self):
        while self.graphics_layout.count():
            item = self.graphics_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._graphics = []

    def _notebook_document(self) -> dict:
        cells = []
        used_cell_ids = set()
        for step in self._steps:
            cell_type = step.get("cell_type", "code")
            if step.get("notebook_outputs") is not None and cell_type == "code":
                outputs = copy.deepcopy(step["notebook_outputs"])
            else:
                outputs = []
                output_text = step.get("output", "")
                if output_text:
                    outputs.append({
                        "output_type": "stream",
                        "name": "stderr" if step.get("has_error") else "stdout",
                        "text": output_text.splitlines(keepends=True),
                    })
                for rich_output in step.get("rich_outputs", []):
                    if rich_output.get("data"):
                        outputs.append({
                            "output_type": "display_data",
                            "data": copy.deepcopy(rich_output["data"]),
                            "metadata": copy.deepcopy(rich_output.get("metadata", {})),
                        })
                for chart in step.get("charts", [])[:8]:
                    outputs.append({
                        "output_type": "display_data",
                        "data": {"image/png": chart, "text/plain": "Grafik analisis"},
                        "metadata": {},
                    })

            metadata = copy.deepcopy(step.get("notebook_metadata") or {})
            existing_dora_metadata = metadata.get("dora")
            dora_metadata = (
                copy.deepcopy(existing_dora_metadata)
                if isinstance(existing_dora_metadata, dict)
                else {}
            )
            dora_metadata.update({
                "title": step.get("title", "Sel notebook"),
                "description": step.get("description", ""),
                "category": step.get("category", "Custom"),
                "template": step.get("template", "Kode kosong"),
                "has_error": bool(step.get("has_error")),
            })
            metadata["dora"] = dora_metadata
            cell = copy.deepcopy(step.get("notebook_extra") or {})
            cell_id = str(step.get("cell_id") or uuid.uuid4().hex[:8])
            if cell_id in used_cell_ids:
                cell_id = uuid.uuid4().hex[:8]
            used_cell_ids.add(cell_id)
            cell.update({
                "cell_type": cell_type,
                "id": cell_id,
                "metadata": metadata,
                "source": step.get("code", "").splitlines(keepends=True),
            })
            if cell_type == "code":
                cell["execution_count"] = step.get("execution_count")
                cell["outputs"] = outputs
            else:
                cell.pop("execution_count", None)
                cell.pop("outputs", None)
            cells.append(cell)
        notebook_metadata = copy.deepcopy(self._notebook_metadata)
        notebook_metadata.setdefault(
            "kernelspec",
            {"display_name": "Python 3", "language": "python", "name": "python3"},
        )
        notebook_metadata.setdefault(
            "language_info", {"name": "python", "version": sys.version.split()[0]}
        )
        return {
            "cells": cells,
            "metadata": notebook_metadata,
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    @Slot()
    def _save_notebook(self):
        default_notebook = (
            self._project_workspace / "analisis.ipynb"
            if self._project_workspace is not None
            else Path.home() / "analisis.ipynb"
        )
        default_path = str(self._notebook_path or default_notebook)
        filename, _ = QFileDialog.getSaveFileName(
            self, "Simpan notebook analisis", default_path, "Jupyter Notebook (*.ipynb)"
        )
        if not filename:
            return
        self._save_current_code()
        path = Path(filename)
        if path.suffix.lower() != ".ipynb":
            path = path.with_suffix(".ipynb")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._notebook_document(), ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as error:
            self.library_status.setText(f"Notebook gagal disimpan: {error}")
            return
        self._notebook_path = path
        self.reload_notebook_button.setEnabled(not self._execution_controls_locked)
        self._notebook_dirty = False
        self.notebook_status_label.setText(f"Tersimpan: {path.name}")
        self.library_status.setText(f"Notebook disimpan: {path.name}")
        self._autosave_timer.stop()
        self._save_autosave_state()
        self.notebook_status_label.setText(f"Tersimpan: {path.name}")

    @Slot()
    def _reload_notebook(self):
        if self._kernel_busy or self._execution_queue:
            self.execution_note.setText("Hentikan proses sebelum memuat ulang notebook.")
            return

        if self._notebook_path is not None or self._notebook_dirty:
            response = QMessageBox.question(
                self,
                "Pulihkan notebook DORA",
                (
                    "Pulihkan sel bawaan DORA? Sel dari notebook .ipynb yang terbuka "
                    "dan perubahan saat ini akan dihapus dari editor. File sumber tetap aman."
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if response != QMessageBox.Yes:
                return

        self._stop_kernel()
        self._steps = self._build_default_notebook_steps()
        self._selected_step = -1
        self._notebook_metadata = {}
        self._notebook_minor = 5
        self._notebook_path = None
        self._notebook_dirty = False
        self._populate_step_list()
        if self._steps:
            self.step_list.setCurrentRow(0)
        else:
            self._select_step(-1)
        self.notebook_status_label.setText("Notebook bawaan DORA")
        self.library_status.setText(
            f"Notebook bawaan DORA dipulihkan: {len(self._steps)} sel template."
        )
        self._autosave_timer.stop()
        self._save_autosave_state()

    @Slot()
    def _open_notebook(self, filename=None):
        if self._kernel_busy or self._execution_queue:
            self.execution_note.setText("Hentikan proses sebelum membuka notebook lain.")
            return
        if self._notebook_dirty:
            response = QMessageBox.question(
                self,
                "Perubahan belum disimpan",
                "Buka notebook lain dan buang perubahan yang belum disimpan?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if response != QMessageBox.Yes:
                return
        if filename is None:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Buka notebook", str(self._notebook_path or ""), "Jupyter Notebook (*.ipynb)"
            )
        if not filename:
            return
        filename = str(filename)
        try:
            document = json.loads(Path(filename).read_text(encoding="utf-8"))
            if (
                not isinstance(document, dict)
                or document.get("nbformat") != 4
                or not isinstance(document.get("cells"), list)
            ):
                raise ValueError("Format notebook tidak didukung.")
            steps = []
            for index, cell in enumerate(document["cells"], start=1):
                if not isinstance(cell, dict):
                    raise ValueError(f"Sel pada urutan {index} bukan objek notebook yang valid.")
                cell_type = cell.get("cell_type")
                if cell_type not in {"code", "markdown", "raw"}:
                    raise ValueError(f"Jenis sel pada urutan {index} tidak didukung: {cell_type!r}.")
                source = cell.get("source", "")
                code = "".join(source) if isinstance(source, list) else str(source)
                notebook_metadata = cell.get("metadata") or {}
                if not isinstance(notebook_metadata, dict):
                    notebook_metadata = {}
                metadata = notebook_metadata.get("dora", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                fallback_title = "Catatan" if cell_type == "markdown" else f"Sel {index:02d}"
                title = str(metadata.get("title", fallback_title))
                description = str(metadata.get("description", "Diimpor dari notebook Jupyter."))
                category = metadata.get("category", "Custom")
                if not isinstance(category, str) or category not in self.template_groups:
                    category = "Custom"
                default_template = "Teks Markdown" if cell_type == "markdown" else "Kode kosong"
                template = metadata.get("template", default_template)
                if not isinstance(template, str):
                    template = default_template
                if template not in self.template_groups[category]:
                    template = next(iter(self.template_groups[category]))
                output_parts = []
                charts = []
                rich_outputs = []
                has_error = bool(metadata.get("has_error"))
                outputs = cell.get("outputs", []) if cell_type == "code" else []
                if not isinstance(outputs, list):
                    outputs = []
                for output in outputs:
                    if not isinstance(output, dict):
                        continue
                    if output.get("output_type") == "stream":
                        text = output.get("text", "")
                        output_parts.append("".join(text) if isinstance(text, list) else str(text))
                    if output.get("output_type") == "error":
                        has_error = True
                        error_name = output.get("ename", "Error")
                        error_value = output.get("evalue", "")
                        traceback_lines = output.get("traceback", [])
                        output_parts.append(f"{error_name}: {error_value}\n" + "\n".join(traceback_lines))
                    data = output.get("data") or {}
                    if not isinstance(data, dict):
                        continue
                    image = data.get("image/png")
                    if image:
                        charts.append(image)
                    display_data = {}
                    for mime_type in ("text/html", "text/plain"):
                        value = data.get(mime_type)
                        if isinstance(value, list):
                            value = "".join(value)
                        if isinstance(value, str):
                            display_data[mime_type] = value
                    if display_data:
                        rich_outputs.append({
                            "data": display_data,
                            "metadata": copy.deepcopy(output.get("metadata") or {}),
                        })
                base_keys = {"cell_type", "id", "metadata", "source", "execution_count", "outputs"}
                steps.append({
                    "title": str(title),
                    "description": description,
                    "category": category,
                    "template": template,
                    "cell_type": cell_type,
                    "cell_id": str(cell.get("id") or uuid.uuid4().hex[:8]),
                    "code": code,
                    "output": "".join(output_parts),
                    "charts": charts[:8],
                    "rich_outputs": rich_outputs,
                    "has_error": has_error,
                    "execution_count": cell.get("execution_count") if cell_type == "code" else None,
                    "notebook_outputs": copy.deepcopy(outputs) if cell_type == "code" else None,
                    "notebook_metadata": copy.deepcopy(notebook_metadata),
                    "notebook_extra": {
                        key: copy.deepcopy(value)
                        for key, value in cell.items()
                        if key not in base_keys
                    },
                })
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            self.library_status.setText(f"Notebook gagal dibuka: {error}")
            return
        self._stop_kernel()
        self._steps = steps
        self._selected_step = -1
        notebook_metadata = document.get("metadata") or {}
        self._notebook_metadata = copy.deepcopy(notebook_metadata) if isinstance(notebook_metadata, dict) else {}
        try:
            self._notebook_minor = int(document.get("nbformat_minor", 5))
        except (TypeError, ValueError):
            self._notebook_minor = 5
        self._populate_step_list()
        if steps:
            self.step_list.setCurrentRow(0)
        else:
            self._select_step(-1)
        self._notebook_path = Path(filename)
        self.reload_notebook_button.setEnabled(not self._execution_controls_locked)
        self._notebook_dirty = False
        self.notebook_status_label.setText(f"Tersimpan: {Path(filename).name}")
        self.library_status.setText(
            f"{len(steps)} sel dimuat ke notebook DORA dari {Path(filename).name}."
        )
        self._schedule_autosave()

    @Slot()
    def _terminate_process_on_exit(self):
        self._run_timeout.stop()
        self._stop_kernel()
        for process in (self._environment_process, self._package_process):
            if process is not None and process.state() != QProcess.NotRunning:
                process.kill()
                process.waitForFinished(1500)

    @Slot()
    def _add_step(self):
        self._insert_cell("code")

    def set_dataset(self, dataframe, filename: str = ""):
        self._workspace_dataframe = dataframe
        self._workspace_dataset_name = filename or "Dataset"
        if not self._initializing:
            self._restore_autosave_state()
        if self._dataset_scope == "custom" and self._custom_dataframe is not None:
            self._apply_dataset(
                self._custom_dataframe,
                Path(self._custom_dataset_path).name or "Dataset notebook",
            )
            self._schedule_autosave()
            return
        if self._dataset_scope == "custom" and self._custom_dataset_path:
            loader = getattr(self._main_window, "loader", None)
            if loader is not None and Path(self._custom_dataset_path).is_file():
                try:
                    self._custom_dataframe = loader.load(self._custom_dataset_path)
                except Exception as error:
                    self.library_status.setText(
                        f"Dataset notebook tidak dapat dipulihkan: {error}"
                    )
                else:
                    self._apply_dataset(
                        self._custom_dataframe,
                        Path(self._custom_dataset_path).name or "Dataset notebook",
                    )
                    self._schedule_autosave()
                    return
            self._dataset_scope = "workspace"
            self.library_status.setText(
                "Progres notebook dipulihkan; pilih ulang dataset notebook yang tersimpan."
            )
        self._dataset_scope = "workspace"
        self.dataset_scope_combo.blockSignals(True)
        self.dataset_scope_combo.setCurrentIndex(self.dataset_scope_combo.findData("workspace"))
        self.dataset_scope_combo.blockSignals(False)
        self._apply_dataset(dataframe, filename)
        self._schedule_autosave()

    def _sync_dataset_menu_button(self):
        if hasattr(self, "dataset_menu_button"):
            self.dataset_menu_button.setToolTip(
                f"Dataset aktif: {self.dataset_badge.text()}\n"
                "Pilih dataset Workspace atau file notebook melalui menu."
            )

    @Slot(int)
    def _dataset_scope_changed(self, index: int):
        scope = self.dataset_scope_combo.itemData(index)
        if scope == self._dataset_scope:
            return
        if scope == "custom":
            if self._custom_dataframe is None:
                self._choose_notebook_dataset()
            else:
                self._dataset_scope = "custom"
                self._apply_dataset(
                    self._custom_dataframe,
                    Path(self._custom_dataset_path).name or "Dataset notebook",
                )
                self._schedule_autosave()
            return
        if scope == "workspace":
            self._dataset_scope = "workspace"
            self._apply_dataset(self._workspace_dataframe, self._workspace_dataset_name)
            self._schedule_autosave()

    @Slot()
    def _choose_notebook_dataset(self):
        loader = getattr(self._main_window, "loader", None)
        if loader is None:
            QMessageBox.information(self, "Dataset notebook", "Loader dataset aplikasi tidak tersedia.")
            self.dataset_scope_combo.blockSignals(True)
            self.dataset_scope_combo.setCurrentIndex(self.dataset_scope_combo.findData(self._dataset_scope))
            self.dataset_scope_combo.blockSignals(False)
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih dataset untuk notebook",
            self._custom_dataset_path,
            "Dataset (*.csv *.tsv *.xlsx *.xls *.json *.parquet);;Semua file (*)",
        )
        if not filename:
            self.dataset_scope_combo.blockSignals(True)
            self.dataset_scope_combo.setCurrentIndex(self.dataset_scope_combo.findData(self._dataset_scope))
            self.dataset_scope_combo.blockSignals(False)
            return
        try:
            dataframe = loader.load(filename)
        except Exception as error:
            QMessageBox.critical(self, "Dataset tidak dapat dimuat", str(error))
            self.dataset_scope_combo.blockSignals(True)
            self.dataset_scope_combo.setCurrentIndex(self.dataset_scope_combo.findData(self._dataset_scope))
            self.dataset_scope_combo.blockSignals(False)
            return
        self._custom_dataframe = dataframe
        self._custom_dataset_path = str(Path(filename).resolve())
        self._dataset_scope = "custom"
        self.dataset_scope_combo.blockSignals(True)
        self.dataset_scope_combo.setCurrentIndex(self.dataset_scope_combo.findData("custom"))
        self.dataset_scope_combo.blockSignals(False)
        self._apply_dataset(dataframe, Path(filename).name)
        self._schedule_autosave()

    def _apply_dataset(self, dataframe, filename: str = ""):
        previous_dataframe = self._dataframe
        previous_project_key = self._project_key
        self._dataframe = dataframe
        scope_label = "Dataset notebook" if self._dataset_scope == "custom" else "Workspace"
        self._dataset_name = f"{scope_label}: {filename or 'Dataset'}"
        if dataframe is None:
            self._cancel_environment_jobs()
            self._stop_kernel()
            self._set_execution_controls_locked(False)
            self._project_key = ""
            self._project_workspace = None
            self._project_environment = None
            self.dataset_badge.setText("Dataset belum dimuat")
            self._sync_dataset_menu_button()
            self.target_combo.blockSignals(True)
            self.target_combo.clear()
            self.target_combo.blockSignals(False)
            self._update_target_controls(self.code_editor.toPlainText())
            self.run_button.setEnabled(False)
            self.run_all_button.setEnabled(False)
            self.install_library_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self._run_animation.stop()
            self._run_opacity_effect.setOpacity(1.0)
            return

        project_id = getattr(self._main_window, "current_project_id", None)
        if project_id is not None:
            project_key = f"project_{project_id}_{self._dataset_scope}"
        else:
            identity = f"{self._dataset_name}|{dataframe.shape}|{list(map(str, dataframe.columns))}"
            project_key = "dataset_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        dataset_changed = previous_dataframe is not dataframe or previous_project_key != project_key
        if dataset_changed:
            self._cancel_environment_jobs()
            self._stop_kernel()
            self._set_execution_controls_locked(False)
            self.cancel_button.setEnabled(False)
            had_output = any(
                step.get("output")
                or step.get("charts")
                or step.get("rich_outputs")
                or step.get("notebook_outputs")
                for step in self._steps
            )
            was_restored = self._autosave_restored
            restored_dataset_matches = bool(
                self._autosave_project_key and self._autosave_project_key == project_key
            )
            self._autosave_project_key = ""
            self._autosave_restored = False
            if not restored_dataset_matches:
                for step in self._steps:
                    step["output"] = ""
                    step["charts"] = []
                    step["rich_outputs"] = []
                    step["has_error"] = False
                    step["execution_count"] = None
                    step["notebook_outputs"] = None
            if had_output and not restored_dataset_matches:
                self._mark_notebook_dirty()
                if was_restored:
                    self.notebook_status_label.setText(
                        "Kode dipulihkan; output perlu dijalankan ulang karena dataset berubah."
                    )
        self._project_key = project_key
        app_data = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
        app_data_root = Path(app_data) if app_data else Path.home() / ".dora"
        self._project_workspace = app_data_root / "editor_analysis" / project_key
        self._project_environment = self._project_workspace / "python-env"
        columns = [str(column) for column in dataframe.columns]
        current_target = self.target_combo.currentText()
        selected_target = current_target if current_target in columns else self._preferred_target(columns)
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItems(columns)
        if selected_target:
            self.target_combo.setCurrentText(selected_target)
        self.target_combo.blockSignals(False)
        for step in self._steps:
            if step.get("cell_type", "code") != "code":
                continue
            existing = re.search(r"(?m)^target_column\s*=\s*(['\"])(.*?)\1\s*$", step["code"])
            if self.TARGET_ASSIGNMENT.search(step["code"]):
                existing_target = existing.group(2) if existing else ""
                if existing_target not in columns:
                    step["code"] = self.TARGET_ASSIGNMENT.sub(
                        f"target_column = {selected_target!r}", step["code"]
                    )
        if 0 <= self._selected_step < len(self._steps):
            step = self._steps[self._selected_step]
            if step.get("cell_type", "code") == "code":
                step["code"] = self._bind_target_column(step["code"])
            self.code_editor.blockSignals(True)
            self.code_editor.setPlainText(step["code"])
            self.code_editor.blockSignals(False)
            self._update_target_controls(step["code"] if step.get("cell_type", "code") == "code" else "")
        self.dataset_badge.setText(f"{self._dataset_name} · {len(dataframe):,} baris")
        active_is_code = (
            0 <= self._selected_step < len(self._steps)
            and self._steps[self._selected_step].get("cell_type", "code") == "code"
        )
        self.run_button.setEnabled(active_is_code and self._kernel_busy is False and self._package_process is None)
        self.run_all_button.setEnabled(
            self._kernel_busy is False
            and self._package_process is None
            and any(step.get("cell_type", "code") == "code" for step in self._steps)
        )
        self.install_library_button.setEnabled(self._package_process is None)
        self._refresh_library_status()
        self._sync_dataset_menu_button()
        if 0 <= self._selected_step < len(self._steps):
            self._render_step_output(self._steps[self._selected_step])

    def _cancel_environment_jobs(self):
        self._environment_callback = None
        self._run_animation.stop()
        self._run_opacity_effect.setOpacity(1.0)
        for name in ("_environment_process", "_package_process"):
            process = getattr(self, name)
            setattr(self, name, None)
            if process is not None and process.state() != QProcess.NotRunning:
                process.kill()
                process.deleteLater()

    def _refresh_library_status(self):
        if self._project_environment is None:
            self.library_status.setText("Paket tambahan dipisahkan untuk setiap proyek.")
            return
        site_packages = self._environment_site_packages()
        installed = []
        if site_packages.exists():
            for metadata in site_packages.glob("*.dist-info"):
                package = re.sub(r"-[0-9].*$", "", metadata.name[:-10]).replace("_", "-")
                if package:
                    installed.append(package)
        installed = sorted(set(installed), key=str.casefold)
        if installed:
            self.library_status.setText("Library proyek: " + ", ".join(installed[:24]))
        else:
            self.library_status.setText("Belum ada library tambahan di lingkungan proyek ini.")

    def _apply_responsive_layout(self, width: int):
        compact = logical_width(width) < 1080
        self.root_layout.setContentsMargins(
            16 if logical_width(width) < 760 else 12,
            28,
            16 if logical_width(width) < 760 else 10,
            12,
        )
        self.root_layout.setSpacing(28)
        self.title_block.setContentsMargins(
            0 if logical_width(width) < 1080 else 36, 0, 0, 0
        )
        self.workspace.setDirection(QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight)
        self.workspace.setSpacing(15)
        if compact:
            self.editor_header_container.setMinimumHeight(0)
            self.editor_header_container.setMaximumHeight(16777215)
            self.info_banner.setMinimumHeight(0)
            self.info_banner.setMaximumHeight(16777215)
            self.info_banner_container.setMinimumHeight(0)
            self.info_banner_container.setMaximumHeight(16777215)
        else:
            self.editor_header_container.setFixedHeight(ui_px(66))
            self.info_banner.setFixedHeight(ui_px(56))
            self.info_banner_container.setFixedHeight(ui_px(56))
        if compact:
            self.editor_card.setMinimumHeight(0)
            self.editor_card.setMaximumHeight(16777215)
        else:
            self.editor_card.setFixedHeight(ui_px(766))
        self.template_toolbar.setDirection(
            QBoxLayout.TopToBottom if logical_width(width) < 650 else QBoxLayout.LeftToRight
        )
        self.notebook_toolbar.setDirection(
            QBoxLayout.TopToBottom if logical_width(width) < 650 else QBoxLayout.LeftToRight
        )
        self.library_row.setDirection(
            QBoxLayout.TopToBottom if logical_width(width) < 650 else QBoxLayout.LeftToRight
        )
        if compact:
            self.steps_card.setMinimumWidth(0)
            self.steps_card.setMaximumWidth(16777215)
            self.steps_card.setMinimumHeight(0)
            self.steps_card.setMaximumHeight(16777215)
        else:
            self.steps_card.setFixedWidth(264)
            self.steps_card.setFixedHeight(393)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())
