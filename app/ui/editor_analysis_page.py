"""Guided, editable analysis templates for the active dataset."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import sys
import sysconfig
from pathlib import Path

from app.ui.scale import logical_width

from PySide6.QtCore import (
    QEasingCurve,
    QProcess,
    QProcessEnvironment,
    QPropertyAnimation,
    QRegularExpression,
    Signal,
    QStandardPaths,
    QTimer,
    Qt,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPixmap,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
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
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QPlainTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
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

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and event.modifiers() & Qt.ControlModifier:
            self.runRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class EditorAnalysisPage(QWidget):
    """Guided notebook-style editor with editable templates and local run output."""

    TARGET_ASSIGNMENT = re.compile(r"(?m)^target_column\s*=.*$")

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
print(classification_report(y_test, model.predict(X_test)))
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
print(classification_report(y_test, model.predict(X_test)))
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
print(classification_report(y_test, model.predict(X_test)))
''',
        "Evaluasi Model": '''# Evaluasi klasifikasi pada data uji
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = model.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
print(f"Recall   : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
print(f"F1-score : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
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
print(classification_report(y_test, y_pred, zero_division=0))
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
        self._steps: list[dict[str, str]] = []
        self._selected_step = -1
        self._dataframe = None
        self._dataset_name = ""
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
        self._package_process: QProcess | None = None
        self._package_log = ""
        self._installing_packages: list[str] = []
        self._notebook_path: Path | None = None
        self._run_timeout = QTimer(self)
        self._run_timeout.setSingleShot(True)
        self._run_timeout.timeout.connect(self._on_run_timeout)
        self._graphics: list[QPixmap] = []
        self.template_groups = self._build_template_groups()
        application = QApplication.instance()
        if application is not None:
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
        self.root_layout.setContentsMargins(30, 22, 28, 20)
        self.root_layout.setSpacing(16)

        header = QHBoxLayout()
        header.setSpacing(16)
        title_block = QVBoxLayout()
        title_block.setSpacing(4)
        title = QLabel("Editor Analysis")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Ikuti tahapan analisis, pilih template, lalu sesuaikan kode dengan kebutuhan risetmu."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block, 1)
        self.dataset_badge = QLabel("Dataset belum dimuat")
        self.dataset_badge.setObjectName("editorAnalysisDatasetBadge")
        header.addWidget(self.dataset_badge, 0, Qt.AlignTop)
        self.root_layout.addLayout(header)

        self.info_banner = QLabel(
            "Template dibuat tim DORA, bukan oleh AI. Kode dijalankan lokal dengan salinan dataset; "
            "hasilnya tidak mengubah dataset utama. Kode tidak diisolasi dari file dan jaringan komputer; "
            "jalankan hanya kode yang kamu percaya."
        )
        self.info_banner.setObjectName("editorAnalysisInfoBanner")
        self.info_banner.setWordWrap(True)
        self.root_layout.addWidget(self.info_banner)

        self.workspace = QHBoxLayout()
        self.workspace.setSpacing(16)
        self.steps_card = self._build_steps_card()
        self.editor_card = self._build_editor_card()
        self.workspace.addWidget(self.steps_card, 1)
        self.workspace.addWidget(self.editor_card, 3)
        self.root_layout.addLayout(self.workspace, 1)

        self.page_scroll.setWidget(container)
        outer.addWidget(self.page_scroll)
        self._steps = [
            {
                "title": title,
                "description": description,
                "category": self.STEP_CATEGORIES.get(index, "Data Profiling"),
                "template": self.STEP_TEMPLATES.get(index, "Data Understanding"),
                "code": self.template_groups[
                    self.STEP_CATEGORIES.get(index, "Data Profiling")
                ][self.STEP_TEMPLATES.get(index, "Ringkasan dataset")],
                "output": "",
                "charts": [],
                "execution_count": None,
            }
            for index, (title, description) in enumerate(self.STEPS)
        ]
        self._populate_step_list()
        self.step_list.setCurrentRow(0)
        self.set_dataset(None, "")
        self._apply_responsive_layout(self.width())

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
print(f"Accuracy: {{accuracy_score(y_test, y_pred):.3f}}")
print(f"F1 weighted: {{f1_score(y_test, y_pred, average='weighted', zero_division=0):.3f}}")
print(classification_report(y_test, y_pred, zero_division=0))
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
        heading = QLabel("Tahapan Analisis")
        heading.setObjectName("editorAnalysisSectionTitle")
        layout.addWidget(heading)
        hint = QLabel("Pilih bagian untuk membuka kode template.")
        hint.setObjectName("editorAnalysisMuted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.step_list = QListWidget()
        self.step_list.setObjectName("editorAnalysisStepList")
        self.step_list.currentRowChanged.connect(self._select_step)
        layout.addWidget(self.step_list, 1)
        self.add_step_button = QPushButton("Tambah langkah")
        self.add_step_button.setObjectName("secondaryButton")
        self.add_step_button.clicked.connect(self._add_step)
        layout.addWidget(self.add_step_button)
        apply_card_shadow(card)
        return card

    def _build_editor_card(self):
        card = QFrame()
        card.setObjectName("editorAnalysisCodeCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        heading_row = QHBoxLayout()
        self.step_title = QLabel("Pilih tahapan analisis")
        self.step_title.setObjectName("editorAnalysisSectionTitle")
        heading_row.addWidget(self.step_title, 1)
        layout.addLayout(heading_row)

        self.notebook_toolbar = QHBoxLayout()
        self.open_notebook_button = QPushButton("Buka notebook")
        self.open_notebook_button.setObjectName("secondaryButton")
        self.open_notebook_button.clicked.connect(self._open_notebook)
        self.notebook_toolbar.addWidget(self.open_notebook_button)
        self.save_notebook_button = QPushButton("Simpan notebook")
        self.save_notebook_button.setObjectName("secondaryButton")
        self.save_notebook_button.clicked.connect(self._save_notebook)
        self.notebook_toolbar.addWidget(self.save_notebook_button)
        self.restart_kernel_button = QPushButton("Mulai ulang Python")
        self.restart_kernel_button.setObjectName("secondaryButton")
        self.restart_kernel_button.clicked.connect(self._restart_kernel)
        self.notebook_toolbar.addWidget(self.restart_kernel_button)
        self.notebook_toolbar.addStretch(1)
        layout.addLayout(self.notebook_toolbar)

        self.template_toolbar = QHBoxLayout()
        self.template_toolbar.setSpacing(12)
        category_block = QVBoxLayout()
        category_label = QLabel("Kelompok template")
        category_label.setObjectName("editorAnalysisFieldLabel")
        category_block.addWidget(category_label)
        self.template_category_combo = QComboBox()
        self.template_category_combo.setObjectName("editorAnalysisCategoryCombo")
        self.template_category_combo.addItems(self.template_groups.keys())
        self.template_category_combo.currentTextChanged.connect(self._category_changed)
        category_block.addWidget(self.template_category_combo)
        self.template_toolbar.addLayout(category_block, 1)
        template_block = QVBoxLayout()
        template_label = QLabel("Pilih template")
        template_label.setObjectName("editorAnalysisFieldLabel")
        template_block.addWidget(template_label)
        self.template_combo = QComboBox()
        self.template_combo.setObjectName("editorAnalysisTemplateCombo")
        self.template_combo.currentTextChanged.connect(self._apply_template)
        template_block.addWidget(self.template_combo)
        self.template_toolbar.addLayout(template_block, 1)
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
        code_label = QLabel("Kode Python")
        code_label.setObjectName("editorAnalysisFieldLabel")
        editor_heading.addWidget(code_label)
        editor_heading.addStretch(1)
        self.copy_button = QPushButton("Salin kode")
        self.copy_button.setObjectName("secondaryButton")
        self.copy_button.clicked.connect(self._copy_code)
        editor_heading.addWidget(self.copy_button)
        self.reset_button = QPushButton("Pulihkan template")
        self.reset_button.setObjectName("secondaryButton")
        self.reset_button.clicked.connect(self._reset_code)
        editor_heading.addWidget(self.reset_button)
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
        layout.addLayout(self.target_row)

        self.code_editor = PythonCodeEditor()
        self.code_editor.setObjectName("editorAnalysisCodeEditor")
        self.code_editor.setPlaceholderText("Pilih tahapan untuk memuat template kode.")
        self.code_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        code_font = self.code_editor.font()
        code_font.setStyleHint(QFont.Monospace)
        code_font.setFamily("Consolas")
        code_font.setPixelSize(14)
        self.code_editor.setFont(code_font)
        self.code_editor.setTabStopDistance(self.code_editor.fontMetrics().horizontalAdvance(" ") * 4)
        self.code_highlighter = PythonCodeHighlighter(self.code_editor.document())
        self.code_editor.runRequested.connect(self._execute_code)
        self.code_editor.setToolTip("Ctrl+Enter untuk menjalankan kode yang sedang dipilih.")
        self.code_editor.setMinimumHeight(300)
        self.code_editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.code_editor.textChanged.connect(self._save_current_code)
        layout.addWidget(self.code_editor, 1)

        output_title = QLabel("Hasil eksekusi")
        output_title.setObjectName("editorAnalysisFieldLabel")
        layout.addWidget(output_title)
        self.output_tabs = QTabWidget()
        self.output_tabs.setObjectName("editorAnalysisOutputTabs")
        self.output_text = QPlainTextEdit()
        self.output_text.setObjectName("editorAnalysisOutputText")
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("Teks hasil dan pesan proses akan tampil di sini.")
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
        self.output_tabs.addTab(self.graphics_scroll, "Grafik")

        self.library_output = QPlainTextEdit()
        self.library_output.setObjectName("editorAnalysisOutputText")
        self.library_output.setReadOnly(True)
        self.library_output.setPlaceholderText("Log pemasangan library akan tampil di sini.")
        self.output_tabs.addTab(self.library_output, "Library")
        self.output_tabs.setMinimumHeight(210)
        layout.addWidget(self.output_tabs)

        execution_row = QHBoxLayout()
        self.execution_note = QLabel(
            "Eksekusi berjalan maksimal 120 detik. Perubahan hanya berlaku pada salinan dataset."
        )
        self.execution_note.setObjectName("editorAnalysisExecutionNote")
        self.execution_note.setWordWrap(True)
        self._run_opacity_effect = QGraphicsOpacityEffect(self.execution_note)
        self._run_opacity_effect.setOpacity(1.0)
        self.execution_note.setGraphicsEffect(self._run_opacity_effect)
        self._run_animation = QPropertyAnimation(self._run_opacity_effect, b"opacity", self)
        self._run_animation.setDuration(850)
        self._run_animation.setStartValue(0.45)
        self._run_animation.setEndValue(1.0)
        self._run_animation.setEasingCurve(QEasingCurve.InOutSine)
        self._run_animation.setLoopCount(-1)
        execution_row.addWidget(self.execution_note, 1)
        self.run_button = QPushButton("Jalankan kode")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setEnabled(False)
        self.run_button.setToolTip("Jalankan kode yang sedang dipilih pada salinan dataset aktif.")
        self.run_button.clicked.connect(self._execute_code)
        execution_row.addWidget(self.run_button)
        self.run_all_button = QPushButton("Jalankan semua")
        self.run_all_button.setObjectName("secondaryButton")
        self.run_all_button.setEnabled(False)
        self.run_all_button.setToolTip("Jalankan semua tahapan secara berurutan dalam satu sesi Python.")
        self.run_all_button.clicked.connect(self._run_all)
        execution_row.addWidget(self.run_all_button)
        self.cancel_button = QPushButton("Hentikan")
        self.cancel_button.setObjectName("secondaryButton")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._terminate_run)
        execution_row.addWidget(self.cancel_button)
        layout.addLayout(execution_row)
        apply_card_shadow(card)
        return card

    def _populate_step_list(self):
        self.step_list.blockSignals(True)
        self.step_list.clear()
        for step in self._steps:
            item = QListWidgetItem(step["title"])
            item.setToolTip(step["description"])
            self.step_list.addItem(item)
        self.step_list.blockSignals(False)

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

    def _save_current_code(self):
        if 0 <= self._selected_step < len(self._steps):
            self._steps[self._selected_step]["code"] = self.code_editor.toPlainText()

    def _select_step(self, index: int):
        self._save_current_code()
        if not 0 <= index < len(self._steps):
            self._selected_step = -1
            self.step_title.setText("Pilih tahapan analisis")
            self.step_description.clear()
            self.code_editor.clear()
            self._update_target_controls("")
            self.output_text.clear()
            self._clear_graphs()
            return
        self._selected_step = index
        step = self._steps[index]
        self.step_title.setText(step["template"])
        self.step_description.setText(step["description"])
        self.template_category_combo.blockSignals(True)
        self.template_category_combo.setCurrentText(step["category"])
        self.template_category_combo.blockSignals(False)
        self.template_combo.blockSignals(True)
        selected_template = self._fill_template_choices(step["category"], step["template"])
        step["template"] = selected_template
        self.step_title.setText(selected_template)
        self.step_description.setText(self.CATEGORY_DESCRIPTIONS.get(step["category"], step["description"]))
        self.template_combo.blockSignals(False)
        self.code_editor.blockSignals(True)
        step["code"] = self._bind_target_column(step["code"])
        self.code_editor.setPlainText(step["code"])
        self.code_editor.blockSignals(False)
        self._update_target_controls(step["code"])
        self._render_step_output(step)

    def _category_changed(self, category: str):
        if not 0 <= self._selected_step < len(self._steps):
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

    def _apply_template(self, template_name: str):
        if not 0 <= self._selected_step < len(self._steps):
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

    def _reset_code(self):
        if not 0 <= self._selected_step < len(self._steps):
            return
        step = self._steps[self._selected_step]
        step["code"] = self._bind_target_column(
            self.template_groups[step["category"]][step["template"]]
        )
        self.code_editor.setPlainText(step["code"])
        self._update_target_controls(step["code"])

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

    def _target_changed(self, target: str):
        if not target or not 0 <= self._selected_step < len(self._steps):
            return
        source = self.code_editor.toPlainText()
        if not self.TARGET_ASSIGNMENT.search(source):
            return
        source = self.TARGET_ASSIGNMENT.sub(f"target_column = {target!r}", source)
        self.code_editor.blockSignals(True)
        self.code_editor.setPlainText(source)
        self.code_editor.blockSignals(False)
        self._steps[self._selected_step]["code"] = source

    def _preferred_target(self, columns: list[str]) -> str:
        by_normalized = {column.casefold(): column for column in columns}
        for candidate in (
            "target", "label", "class", "survived", "outcome", "diagnosis",
            "churn", "y", "price", "saleprice", "medv", "score", "value", "income",
        ):
            if candidate in by_normalized:
                return by_normalized[candidate]
        return columns[-1] if columns else ""

    def _copy_code(self):
        QApplication.clipboard().setText(self.code_editor.toPlainText())
        self.execution_note.setText("Kode berhasil disalin ke clipboard.")

    def _execution_interpreter(self) -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable)
        if self._project_environment is None:
            raise RuntimeError("Lingkungan Python proyek belum disiapkan.")
        if os.name == "nt":
            return self._project_environment / "Scripts" / "python.exe"
        return self._project_environment / "bin" / "python"

    def _environment_site_packages(self) -> Path:
        if self._project_environment is None:
            raise RuntimeError("Lingkungan Python proyek belum disiapkan.")
        if os.name == "nt":
            return self._project_environment / "Lib" / "site-packages"
        return self._project_environment / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

    def _ensure_environment(self, callback):
        if self._project_environment is None:
            self.execution_note.setText("Lingkungan proyek belum tersedia.")
            return
        if self._execution_interpreter().exists():
            callback()
            return
        if self._environment_process is not None:
            return
        self._environment_callback = callback
        self._project_environment.parent.mkdir(parents=True, exist_ok=True)
        process = QProcess(self)
        self._environment_process = process
        process.setProgram(sys.executable)
        process.setArguments(["-m", "venv", str(self._project_environment)])
        process.setWorkingDirectory(str(self._project_workspace))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_environment_output)
        process.finished.connect(self._environment_finished)
        process.errorOccurred.connect(self._environment_error)
        process.start()
        self.execution_note.setText("Menyiapkan lingkungan Python khusus proyek...")

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
        self._execution_queue = [index for index in step_indexes if 0 <= index < len(self._steps)]
        if not self._execution_queue:
            return
        self._execution_total = len(self._execution_queue)
        self._execution_completed = 0
        self._queue_had_error = False
        self.run_button.setEnabled(False)
        self.run_all_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.install_library_button.setEnabled(False)
        self.execution_note.setText("Menyiapkan sesi Python...")
        self._run_animation.start()
        self._ensure_environment(self._start_kernel)

    def _execute_code(self):
        if self._selected_step < 0:
            self.execution_note.setText("Pilih bagian analisis yang akan dijalankan.")
            return
        self._queue_execution([self._selected_step])

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
                str(worker_path),
                str(site_packages),
                str(dataset_path),
                str(self._project_workspace),
            ])
        process.setWorkingDirectory(str(self._project_workspace))
        environment = QProcessEnvironment.systemEnvironment()
        app_site_packages = "" if getattr(sys, "frozen", False) else sysconfig.get_paths().get("purelib", "")
        existing_pythonpath = environment.value("PYTHONPATH")
        pythonpath = [path for path in (app_site_packages, existing_pythonpath) if path]
        environment.insert("PYTHONPATH", os.pathsep.join(pythonpath))
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
        step["output"] = message.get("output", "")
        step["charts"] = message.get("charts", [])
        step["has_error"] = bool(message.get("error"))
        step["execution_count"] = (step.get("execution_count") or 0) + 1
        self._queue_had_error = self._queue_had_error or step["has_error"]
        self._kernel_busy = False
        self._running_step_index = -1
        self._run_timeout.stop()
        self._execution_completed += 1
        if index == self._selected_step:
            self._render_step_output(step)
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
        self.run_button.setEnabled(self._dataframe is not None)
        self.run_all_button.setEnabled(self._dataframe is not None)
        self.cancel_button.setEnabled(False)
        self.install_library_button.setEnabled(self._dataframe is not None)
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
        if getattr(sys, "frozen", False):
            process.setArguments([
                "--dora-editor-pip",
                "install",
                "--disable-pip-version-check",
                "--upgrade",
                "--target",
                str(self._environment_site_packages()),
                *packages,
            ])
        else:
            process.setArguments([
                "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", *packages,
            ])
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

    def _read_package_output(self):
        process = self._package_process
        if process is None:
            return
        output = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_library_log(output)

    def _package_error(self, error):
        if self._package_process is None or error != QProcess.FailedToStart:
            return
        self._append_library_log("pip gagal dijalankan di lingkungan proyek.\n")
        self._package_finished(-1, QProcess.CrashExit)

    def _package_finished(self, exit_code: int, _exit_status):
        process = self._package_process
        if process is None:
            return
        self._read_package_output()
        self._package_process = None
        process.deleteLater()
        self._run_animation.stop()
        self._run_opacity_effect.setOpacity(1.0)
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
        self.output_text.setPlainText(step.get("output", ""))
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
        self.output_tabs.setTabText(1, f"Grafik ({len(self._graphics)})" if self._graphics else "Grafik")

    def _clear_graphs(self):
        while self.graphics_layout.count():
            item = self.graphics_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._graphics = []

    def _notebook_document(self) -> dict:
        cells = []
        for step in self._steps:
            outputs = []
            output_text = step.get("output", "")
            if output_text:
                output_type = "stderr" if step.get("has_error") else "stdout"
                outputs.append({
                    "output_type": "stream",
                    "name": output_type,
                    "text": output_text.splitlines(keepends=True),
                })
            for chart in step.get("charts", [])[:8]:
                outputs.append({
                    "output_type": "display_data",
                    "data": {"image/png": chart, "text/plain": "Grafik analisis"},
                    "metadata": {},
                })
            cells.append({
                "cell_type": "code",
                "execution_count": step.get("execution_count"),
                "metadata": {
                    "dora": {
                        "title": step["title"],
                        "description": step["description"],
                        "category": step["category"],
                        "template": step["template"],
                        "has_error": bool(step.get("has_error")),
                    }
                },
                "outputs": outputs,
                "source": step["code"].splitlines(keepends=True),
            })
        return {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": sys.version.split()[0]},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    def _save_notebook(self):
        if self._project_workspace is None:
            self.library_status.setText("Pilih atau muat dataset sebelum menyimpan notebook.")
            return
        default_path = str(self._notebook_path or (self._project_workspace / "analisis.ipynb"))
        filename, _ = QFileDialog.getSaveFileName(
            self, "Simpan notebook analisis", default_path, "Jupyter Notebook (*.ipynb)"
        )
        if not filename:
            return
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
        self.library_status.setText(f"Notebook disimpan: {path.name}")

    def _open_notebook(self):
        if self._kernel_busy or self._execution_queue:
            self.execution_note.setText("Hentikan proses sebelum membuka notebook lain.")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Buka notebook", str(self._notebook_path or ""), "Jupyter Notebook (*.ipynb)"
        )
        if not filename:
            return
        try:
            document = json.loads(Path(filename).read_text(encoding="utf-8"))
            if document.get("nbformat") != 4 or not isinstance(document.get("cells"), list):
                raise ValueError("Format notebook tidak didukung.")
            steps = []
            for cell in document["cells"]:
                if cell.get("cell_type") != "code":
                    continue
                source = cell.get("source", "")
                code = "".join(source) if isinstance(source, list) else str(source)
                metadata = (cell.get("metadata") or {}).get("dora", {})
                title = metadata.get("title", "Bagian notebook")
                category = metadata.get("category", "Custom")
                if category not in self.template_groups:
                    category = "Custom"
                template = metadata.get("template", "Kode kosong")
                if template not in self.template_groups[category]:
                    template = next(iter(self.template_groups[category]))
                output_parts = []
                charts = []
                has_error = bool(metadata.get("has_error"))
                for output in cell.get("outputs", []):
                    if output.get("output_type") == "stream":
                        text = output.get("text", "")
                        output_parts.append("".join(text) if isinstance(text, list) else str(text))
                    if output.get("output_type") == "error":
                        has_error = True
                        output_parts.extend(output.get("traceback", []))
                    image = (output.get("data") or {}).get("image/png")
                    if image:
                        charts.append(image)
                steps.append({
                    "title": title,
                    "description": metadata.get("description", "Diimpor dari notebook Jupyter."),
                    "category": category,
                    "template": template,
                    "code": code,
                    "output": "".join(output_parts),
                    "charts": charts[:8],
                    "has_error": has_error,
                    "execution_count": cell.get("execution_count"),
                })
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            self.library_status.setText(f"Notebook gagal dibuka: {error}")
            return
        if not steps:
            self.library_status.setText("Notebook tidak memiliki bagian kode Python.")
            return
        self._stop_kernel()
        self._steps = steps
        self._populate_step_list()
        self.step_list.setCurrentRow(0)
        self._notebook_path = Path(filename)
        self.library_status.setText(f"{len(steps)} bagian dimuat dari {Path(filename).name}.")

    def _terminate_process_on_exit(self):
        self._run_timeout.stop()
        self._stop_kernel()
        for process in (self._environment_process, self._package_process):
            if process is not None and process.state() != QProcess.NotRunning:
                process.kill()
                process.waitForFinished(1500)

    def _add_step(self):
        index = len(self._steps) + 1
        title = f"Langkah Kustom {index - len(self.STEPS)}"
        step = {
            "title": title,
            "description": "Tulis catatan atau kode analisis sesuai kebutuhanmu.",
            "category": "Custom",
            "template": "Kode kosong",
            "code": self.template_groups["Custom"]["Kode kosong"],
            "output": "",
            "charts": [],
            "has_error": False,
            "execution_count": None,
        }
        self._steps.append(step)
        self._populate_step_list()
        self.step_list.setCurrentRow(len(self._steps) - 1)

    def set_dataset(self, dataframe, filename: str = ""):
        previous_dataframe = self._dataframe
        previous_project_key = self._project_key
        self._dataframe = dataframe
        self._dataset_name = filename or "Dataset"
        if dataframe is None:
            self._cancel_environment_jobs()
            self._stop_kernel()
            self._project_key = ""
            self._project_workspace = None
            self._project_environment = None
            self.dataset_badge.setText("Dataset belum dimuat")
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
            project_key = f"project_{project_id}"
        else:
            identity = f"{self._dataset_name}|{dataframe.shape}|{list(map(str, dataframe.columns))}"
            project_key = "dataset_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        dataset_changed = previous_dataframe is not dataframe or previous_project_key != project_key
        if dataset_changed:
            self._cancel_environment_jobs()
            self._stop_kernel()
            self.cancel_button.setEnabled(False)
            for step in self._steps:
                step["output"] = ""
                step["charts"] = []
                step["has_error"] = False
                step["execution_count"] = None
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
            existing = re.search(r"(?m)^target_column\s*=\s*(['\"])(.*?)\1\s*$", step["code"])
            if self.TARGET_ASSIGNMENT.search(step["code"]):
                existing_target = existing.group(2) if existing else ""
                if existing_target not in columns:
                    step["code"] = self.TARGET_ASSIGNMENT.sub(
                        f"target_column = {selected_target!r}", step["code"]
                    )
        if 0 <= self._selected_step < len(self._steps):
            step = self._steps[self._selected_step]
            step["code"] = self._bind_target_column(step["code"])
            self.code_editor.blockSignals(True)
            self.code_editor.setPlainText(step["code"])
            self.code_editor.blockSignals(False)
            self._update_target_controls(step["code"])
        self.dataset_badge.setText(f"{self._dataset_name} · {len(dataframe):,} baris")
        self.run_button.setEnabled(self._kernel_busy is False and self._package_process is None)
        self.run_all_button.setEnabled(self._kernel_busy is False and self._package_process is None)
        self.install_library_button.setEnabled(self._package_process is None)
        self._refresh_library_status()
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
        self.root_layout.setContentsMargins(16 if logical_width(width) < 760 else 30, 16, 16 if logical_width(width) < 760 else 28, 18)
        self.workspace.setDirection(QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight)
        self.template_toolbar.setDirection(
            QBoxLayout.TopToBottom if logical_width(width) < 650 else QBoxLayout.LeftToRight
        )
        self.notebook_toolbar.setDirection(
            QBoxLayout.TopToBottom if logical_width(width) < 650 else QBoxLayout.LeftToRight
        )
        self.library_row.setDirection(
            QBoxLayout.TopToBottom if logical_width(width) < 650 else QBoxLayout.LeftToRight
        )
        self.steps_card.setMinimumWidth(0 if compact else 240)
        self.steps_card.setMaximumWidth(16777215 if compact else 360)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())
