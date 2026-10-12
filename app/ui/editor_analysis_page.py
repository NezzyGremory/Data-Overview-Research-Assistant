"""Guided, editable analysis templates for the active dataset."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.ui.style_utils import apply_card_shadow


class EditorAnalysisPage(QWidget):
    """Notebook-style template editor; code execution is intentionally not wired here."""

    STEPS = (
        ("Data Understanding", "Kenali ukuran, tipe kolom, dan contoh isi dataset."),
        ("Data Cleaning", "Periksa nilai kosong, duplikasi, dan data yang perlu dibersihkan."),
        ("Exploratory Data Analysis", "Jelajahi distribusi fitur dan hubungan antarvariabel."),
        ("Feature Engineering", "Siapkan fitur dan ubah kolom kategorikal untuk pemodelan."),
        ("Ensemble Modeling", "Pilih template Random Forest, Gradient Boosting, atau XGBoost."),
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
categorical_columns = df.select_dtypes(include=["object", "category"]).columns
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
        0: "Data Understanding",
        1: "Missing Values",
        2: "EDA & Korelasi",
        3: "Encoding Kategorikal",
        4: "Random Forest",
        5: "Evaluasi Model",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = parent
        self._steps: list[dict[str, str]] = []
        self._selected_step = -1

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
            "Template disusun sebagai contoh analisis. Periksa nama kolom dan sesuaikan kode sebelum digunakan."
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
                "template": self.STEP_TEMPLATES.get(index, "Data Understanding"),
                "code": self.TEMPLATES[self.STEP_TEMPLATES.get(index, "Data Understanding")],
            }
            for index, (title, description) in enumerate(self.STEPS)
        ]
        self._populate_step_list()
        self.step_list.setCurrentRow(0)
        self.set_dataset(None, "")
        self._apply_responsive_layout(self.width())

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
        self.template_combo = QComboBox()
        self.template_combo.setObjectName("editorAnalysisTemplateCombo")
        self.template_combo.addItems(self.TEMPLATES.keys())
        self.template_combo.currentTextChanged.connect(self._apply_template)
        heading_row.addWidget(self.template_combo)
        layout.addLayout(heading_row)

        self.step_description = QLabel("")
        self.step_description.setObjectName("editorAnalysisMuted")
        self.step_description.setWordWrap(True)
        layout.addWidget(self.step_description)

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

        self.code_editor = QPlainTextEdit()
        self.code_editor.setObjectName("editorAnalysisCodeEditor")
        self.code_editor.setPlaceholderText("Pilih tahapan untuk memuat template kode.")
        self.code_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.code_editor.setTabStopDistance(32)
        self.code_editor.setMinimumHeight(300)
        self.code_editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.code_editor.textChanged.connect(self._save_current_code)
        layout.addWidget(self.code_editor, 1)

        execution_row = QHBoxLayout()
        self.execution_note = QLabel(
            "Kode bisa diedit dan disalin. Eksekusi kode manual belum tersedia."
        )
        self.execution_note.setObjectName("editorAnalysisExecutionNote")
        self.execution_note.setWordWrap(True)
        execution_row.addWidget(self.execution_note, 1)
        self.run_button = QPushButton("Jalankan kode")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setEnabled(False)
        self.run_button.setToolTip("Eksekusi kode belum disambungkan ke lingkungan yang aman.")
        execution_row.addWidget(self.run_button)
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
            return
        self._selected_step = index
        step = self._steps[index]
        self.step_title.setText(step["title"])
        self.step_description.setText(step["description"])
        self.template_combo.blockSignals(True)
        self.template_combo.setCurrentText(step["template"])
        self.template_combo.blockSignals(False)
        self.code_editor.blockSignals(True)
        self.code_editor.setPlainText(step["code"])
        self.code_editor.blockSignals(False)

    def _apply_template(self, template_name: str):
        if not 0 <= self._selected_step < len(self._steps):
            return
        step = self._steps[self._selected_step]
        step["template"] = template_name
        step["code"] = self.TEMPLATES[template_name]
        self.code_editor.blockSignals(True)
        self.code_editor.setPlainText(step["code"])
        self.code_editor.blockSignals(False)

    def _reset_code(self):
        if not 0 <= self._selected_step < len(self._steps):
            return
        template = self._steps[self._selected_step]["template"]
        self.code_editor.setPlainText(self.TEMPLATES[template])

    def _copy_code(self):
        QApplication.clipboard().setText(self.code_editor.toPlainText())
        self.execution_note.setText("Kode berhasil disalin ke clipboard.")

    def _add_step(self):
        index = len(self._steps) + 1
        title = f"Langkah Kustom {index - len(self.STEPS)}"
        step = {
            "title": title,
            "description": "Tulis catatan atau kode analisis sesuai kebutuhanmu.",
            "template": "Data Understanding",
            "code": "# Tulis kode analisis di sini\n",
        }
        self._steps.append(step)
        self._populate_step_list()
        self.step_list.setCurrentRow(len(self._steps) - 1)

    def set_dataset(self, dataframe, filename: str = ""):
        if dataframe is None:
            self.dataset_badge.setText("Dataset belum dimuat")
            self.run_button.setEnabled(False)
            return
        self.dataset_badge.setText(f"{filename or 'Dataset'} · {len(dataframe):,} baris")
        self.run_button.setEnabled(False)

    def _apply_responsive_layout(self, width: int):
        compact = width < 1080
        self.root_layout.setContentsMargins(16 if width < 760 else 30, 16, 16 if width < 760 else 28, 18)
        self.workspace.setDirection(QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight)
        self.steps_card.setMinimumWidth(0 if compact else 240)
        self.steps_card.setMaximumWidth(16777215 if compact else 360)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())
