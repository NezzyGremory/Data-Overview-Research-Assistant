"""Tanya DORA local research assistant page."""

from __future__ import annotations

import json
import re
import socket
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from PySide6.QtCore import (
    QEvent, QPoint, QSettings, Qt, QThread, QTimer, QPropertyAnimation,
    QEasingCurve, Signal, Slot,
)
from PySide6.QtGui import QColor, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.ai.conversation_store import (
    ConversationMessage,
    ConversationStore,
    ConversationStoreError,
)
from app.ai.knowledge_base import DoraKnowledgeBase
from app.ai.llm_manager import OllamaLLMManager, OllamaStatus
from app.ai.model_registry import LocalModel
from app.ai.model_versioning import ModelVersionStore
from app.ai.rag_pipeline import RAGPipeline, RAGResponse
from app.ai.tool_calling import ToolExecution, build_dora_tool_registry
from app.ui.scale import logical_width, ui_px


def _local_test_answer(question: str) -> str:
    """Return a canned, clearly identified response for offline chat testing."""
    normalized = re.sub(r"[^a-z0-9]+", " ", str(question or "").casefold()).strip()
    terms = set(normalized.split())

    if any(term in normalized for term in ("data science", "data sains", "sains data", "ilmu data")):
        return (
            "Data sains adalah disiplin ilmu yang mengintegrasikan statistika, pemrograman, "
            "dan pemahaman domain untuk mengekstrak wawasan bermakna dari data. "
            "Sebagai asisten data Anda, saya dapat membantu eksplorasi dataset, "
            "analisis statistik, pemodelan, hingga interpretasi hasil riset."
        )
    if any(term in normalized for term in ("tanya dora", "apa itu dora", "dora")):
        return (
            "Saya DORA — Data Overview & Research Assistant, asisten data Anda. "
            "Saya dirancang untuk membantu analisis dataset, mendukung riset akademik, "
            "menjawab pertanyaan seputar data sains dan machine learning, "
            "serta memberikan rekomendasi analisis yang relevan dengan kebutuhan Anda."
        )
    if any(term in normalized for term in ("dataset", "data set")):
        return (
            "Dataset adalah kumpulan data terstruktur yang siap untuk dianalisis. "
            "Setiap baris biasanya merepresentasikan satu observasi, sementara setiap kolom "
            "merepresentasikan variabel seperti usia, tanggal, atau nilai numerik. "
            "Saya dapat membantu Anda memuat, memeriksa kualitas, dan menganalisis dataset."
        )
    if any(term in normalized for term in ("analisis data", "analisa data", "data analysis")):
        return (
            "Analisis data adalah proses sistematis untuk memeriksa, merangkum, dan "
            "menafsirkan data guna menjawab pertanyaan riset. Tahapannya meliputi "
            "pemeriksaan kualitas data, statistik deskriptif, visualisasi, hingga "
            "penarikan kesimpulan yang didukung bukti."
        )
    if any(term in normalized for term in ("machine learning", "pembelajaran mesin")):
        return (
            "Machine learning adalah metode komputasi yang memungkinkan sistem mempelajari "
            "pola dari data secara otomatis. Teknik ini mencakup klasifikasi, regresi, "
            "klasterisasi, dan deteksi anomali — semua dapat diterapkan pada dataset "
            "yang Anda miliki."
        )
    if "kecerdasan buatan" in normalized or "artificial intelligence" in normalized or "ai" in terms:
        return (
            "Kecerdasan buatan (AI) adalah bidang ilmu komputer yang mengembangkan sistem "
            "mampu melakukan tugas yang biasanya memerlukan kecerdasan manusia — seperti "
            "memahami bahasa, mengenali pola, dan membuat prediksi berbasis data."
        )
    if any(term in normalized for term in ("training", "pelatihan model", "melatih model")):
        return (
            "Pelatihan model adalah proses mengoptimalkan parameter model machine learning "
            "menggunakan data latih. Proses ini mencakup pemilihan algoritma, tuning "
            "hyperparameter, dan validasi performa pada data yang belum pernah dilihat model."
        )
    if terms.intersection({"halo", "hai", "hello", "hi"}):
        return (
            "Halo! Saya DORA, asisten data Anda. Saya siap membantu analisis dataset, "
            "menjawab pertanyaan seputar data sains, machine learning, dan riset akademik. "
            "Silakan ajukan pertanyaan Anda."
        )
    if any(term in normalized for term in ("bisa apa", "apa yang bisa kamu", "bantuan", "help")):
        return (
            "Sebagai asisten data Anda, saya dapat membantu: analisis dan profiling dataset, "
            "penjelasan konsep data sains dan machine learning, rekomendasi metode analisis, "
            "interpretasi hasil statistik, serta dukungan riset akademik. "
            "Silakan ajukan pertanyaan spesifik Anda."
        )
    return (
        "Pertanyaan Anda telah diterima. Untuk pertanyaan mendalam, saya memerlukan "
        "koneksi ke model AI. Saat ini saya dapat membantu topik: data sains, dataset, "
        "analisis data, machine learning, kecerdasan buatan, dan riset akademik. "
        "Silakan coba salah satu topik tersebut."
    )


class _ModelRefreshThread(QThread):
    completed = Signal(object, object)

    def __init__(self, manager: OllamaLLMManager, parent=None) -> None:
        super().__init__(parent)
        self.manager = manager

    def run(self) -> None:
        base_url = getattr(self.manager, "base_url", None)
        if base_url:
            try:
                endpoint = urlsplit(str(base_url))
                host = endpoint.hostname or "127.0.0.1"
                port = endpoint.port or 11434
                with socket.create_connection((host, port), timeout=0.5):
                    pass
            except (OSError, ValueError):
                status = OllamaStatus(
                    available=False,
                    selected_model=getattr(self.manager, "selected_model", None),
                    installed_model=False,
                    running_model=False,
                    installed_model_count=0,
                    detail="Ollama lokal tidak dapat dijangkau.",
                )
                self.completed.emit((), status)
                return
        status = self.manager.get_status()
        self.completed.emit(self.manager.list_models(), status)


class _ChatThread(QThread):
    delta = Signal(str)
    tool_activity = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        knowledge_base: DoraKnowledgeBase,
        manager: OllamaLLMManager,
        *,
        question: str,
        model: str,
        embedding_model: str | None,
        temperature: float,
        history: Sequence[Mapping[str, str]],
        tool_registry,
        include_knowledge_base: bool,
        custom_instructions: str,
        cancel_event: threading.Event,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.knowledge_base = knowledge_base
        self.manager = manager
        self.question = question
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.history = list(history)
        self.tool_registry = tool_registry
        self.include_knowledge_base = include_knowledge_base
        self.custom_instructions = custom_instructions
        self.cancel_event = cancel_event

    def run(self) -> None:
        try:
            pipeline = RAGPipeline(
                self.knowledge_base,
                self.manager,
                chat_model=self.model,
                embedding_model=self.embedding_model,
                temperature=self.temperature,
                tool_registry=self.tool_registry,
            )
            response = pipeline.answer(
                self.question,
                conversation_history=self.history,
                include_knowledge_base=self.include_knowledge_base,
                allow_general_answers=True,
                custom_instructions=self.custom_instructions,
                on_delta=self.delta.emit,
                on_tool_call=lambda name, _args: self.tool_activity.emit(name),
                cancel_event=self.cancel_event,
            )
        except Exception as error:
            self.failed.emit(str(error))
        else:
            self.completed.emit(response)


class TanyaDoraPage(QWidget):
    """A local-only chat workspace grounded in the Dora Knowledge Base."""

    TEMPORARY_CONVERSATIONS_SETTING = "tanya_dora/temporary_conversation_ids"

    def __init__(
        self,
        *,
        conversation_store: ConversationStore | None = None,
        knowledge_base: DoraKnowledgeBase | None = None,
        llm_manager: OllamaLLMManager | None = None,
        model_version_store: ModelVersionStore | None = None,
        tool_context_provider=None,
        compact_mode: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("tanyaDoraPage")
        self.compact_mode = bool(compact_mode)
        self.setProperty("compactMode", self.compact_mode)
        self.conversation_store = conversation_store or ConversationStore()
        self.knowledge_base = knowledge_base or DoraKnowledgeBase()
        self.llm_manager = llm_manager or OllamaLLMManager()
        self.model_version_store = model_version_store or ModelVersionStore()
        self._settings = QSettings("DORA", "Dataset Research")
        self._owns_manager = llm_manager is None
        self.tool_context_provider = tool_context_provider
        self._model_thread: _ModelRefreshThread | None = None
        self._chat_thread: _ChatThread | None = None
        self._cancel_event: threading.Event | None = None
        self._active_conversation_id: str | None = None
        self._temporary_conversation_ids: set[str] = set()
        self._is_shutting_down = False
        self._shutdown_complete = False
        self._remove_stale_temporary_conversations()
        self._current_models: tuple[LocalModel, ...] = ()
        self._streaming_text = ""
        self._streaming_started = False
        self._last_response: RAGResponse | None = None
        self.message_list_layout = None
        self.message_list_host = None
        # Typing indicator state (local-mode delay animation)
        self._drag_pos: "QPoint | None" = None
        self._typing_timer: QTimer | None = None
        self._typing_dot_timer: QTimer | None = None
        self._typing_dot_frame: int = 0
        self._typing_bubble: QWidget | None = None
        self._pending_local_answer: str | None = None
        self._pending_conversation_id: str | None = None
        self._build_ui()
        if self.compact_mode and self.parentWidget() is not None:
            self.parentWidget().installEventFilter(self)
            self._position_compact_overlay()
            QTimer.singleShot(0, self._position_compact_overlay)
        self._load_conversations()
        self._initialize_local_test_mode()
        application = QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(lambda: self.shutdown())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(
            ui_px(12) if self.compact_mode else ui_px(26),
            ui_px(10) if self.compact_mode else ui_px(18),
            ui_px(12) if self.compact_mode else ui_px(26),
            ui_px(12) if self.compact_mode else ui_px(24),
        )
        root.setSpacing(ui_px(8) if self.compact_mode else ui_px(14))

        header = QHBoxLayout()
        header.setSpacing(ui_px(10))
        if self.compact_mode:
            self.avatar_label = QLabel()
            self.avatar_label.setObjectName("tanyaDoraAvatar")
            self.avatar_label.setFixedSize(ui_px(42), ui_px(42))
            avatar_path = Path(__file__).resolve().parent / "assets" / "icons" / "dora-robot-avatar.svg"
            self.avatar_label.setPixmap(QIcon(str(avatar_path)).pixmap(ui_px(34), ui_px(34)))
            self.avatar_label.setAlignment(Qt.AlignCenter)
            self.avatar_label.setAccessibleName("Logo robot Tanya DORA")
            header.addWidget(self.avatar_label)
        title_block = QVBoxLayout()
        title_block.setSpacing(ui_px(3))
        title = QLabel("Tanya DORA")
        title.setObjectName("tanyaDoraTitle")
        subtitle = QLabel("Uji chat lokal; tanpa Ollama tersedia balasan contoh")
        subtitle.setObjectName("tanyaDoraSubtitle")
        title_block.addWidget(title)
        if not self.compact_mode:
            title_block.addWidget(subtitle)
        header.addLayout(title_block, 1)

        self.model_status = QLabel("Mode uji lokal siap")
        self.model_status.setObjectName("tanyaDoraStatus")
        self.model_status.setProperty("state", "pending")
        self.model_status.setWordWrap(True)
        self.model_status.setMinimumWidth(ui_px(160))
        if not self.compact_mode:
            header.addWidget(self.model_status, 0, Qt.AlignRight | Qt.AlignVCenter)
        else:
            self.model_status.hide()
        self.settings_toggle_button = QPushButton("Tutup pengaturan")
        self.settings_toggle_button.setObjectName("tanyaDoraSettingsToggle")
        if self.compact_mode:
            self.settings_toggle_button.setText("⋯")
            self.settings_toggle_button.setObjectName("tanyaDoraMenuButton")
            self.settings_toggle_button.setAccessibleName("Riwayat dan pengaturan chat")
            self.settings_toggle_button.setToolTip("Riwayat dan pengaturan chat")
            self.settings_toggle_button.setFixedSize(ui_px(38), ui_px(38))
            self.settings_toggle_button.clicked.connect(
                lambda _checked=False: self._show_compact_menu()
            )
        else:
            self.settings_toggle_button.setCheckable(True)
            self.settings_toggle_button.setChecked(True)
            self.settings_toggle_button.toggled.connect(
                lambda expanded: self._toggle_settings_panel(expanded)
            )
        header.addWidget(self.settings_toggle_button)
        if self.compact_mode:
            self.header_new_conversation_button = QPushButton("+")
            self.header_new_conversation_button.setObjectName("tanyaDoraHeaderButton")
            self.header_new_conversation_button.setAccessibleName("Percakapan baru")
            self.header_new_conversation_button.setToolTip("Percakapan baru")
            self.header_new_conversation_button.clicked.connect(
                lambda _checked=False: self._new_conversation()
            )
            header.addWidget(self.header_new_conversation_button)
            self.close_overlay_button = QPushButton("×")
            self.close_overlay_button.setObjectName("tanyaDoraCloseButton")
            self.close_overlay_button.setAccessibleName("Tutup chat Tanya DORA")
            self.close_overlay_button.setToolTip("Tutup chat")
            self.close_overlay_button.clicked.connect(
                lambda _checked=False: self._close_compact_overlay()
            )
            header.addWidget(self.close_overlay_button)
        root.addLayout(header)

        # Thin separator line below header in compact mode
        if self.compact_mode:
            sep = QFrame()
            sep.setObjectName("tanyaDoraHeaderSep")
            sep.setFrameShape(QFrame.HLine)
            sep.setFixedHeight(1)
            root.addWidget(sep)

        controls = QFrame()
        controls.setObjectName("tanyaDoraControls")
        self.settings_panel = controls
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(ui_px(14), ui_px(12), ui_px(14), ui_px(12))
        controls_layout.setSpacing(ui_px(9))

        model_row = QHBoxLayout()
        model_row.setSpacing(ui_px(9))

        model_label = QLabel("Model lokal")
        model_label.setObjectName("tanyaDoraFieldLabel")
        model_row.addWidget(model_label)
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("tanyaDoraModelCombo")
        self.model_combo.setMinimumWidth(ui_px(260))
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.model_combo.currentIndexChanged.connect(
            lambda index: self._on_model_changed(index)
        )
        model_row.addWidget(self.model_combo, 1)

        temperature_label = QLabel("Temperature")
        temperature_label.setObjectName("tanyaDoraFieldLabel")
        model_row.addWidget(temperature_label)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setObjectName("tanyaDoraTemperature")
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setDecimals(1)
        self.temperature_spin.setValue(float(self._settings.value("tanya_dora/temperature", 0.2)))
        self.temperature_spin.setFixedWidth(ui_px(82))
        model_row.addWidget(self.temperature_spin)

        self.refresh_button = QPushButton("Perbarui model")
        self.refresh_button.setObjectName("secondaryButton")
        self.refresh_button.setToolTip("Periksa model yang sudah terpasang di Ollama lokal")
        self.refresh_button.clicked.connect(
            lambda _checked=False: self._refresh_models()
        )
        model_row.addWidget(self.refresh_button)
        controls_layout.addLayout(model_row)

        source_row = QHBoxLayout()
        source_row.setSpacing(ui_px(9))
        self.include_dkb_checkbox = QCheckBox("Sertakan dokumen lokal DKB sebagai konteks")
        self.include_dkb_checkbox.setObjectName("tanyaDoraIncludeDkb")
        self.include_dkb_checkbox.setChecked(
            self._settings.value("tanya_dora/include_dkb", False, type=bool)
        )
        self.include_dkb_checkbox.toggled.connect(
            lambda enabled: self._on_dkb_setting_changed(enabled)
        )
        source_row.addWidget(self.include_dkb_checkbox)

        embedding_label = QLabel("Embedding")
        embedding_label.setObjectName("tanyaDoraFieldLabel")
        source_row.addWidget(embedding_label)
        self.embedding_combo = QComboBox()
        self.embedding_combo.setObjectName("tanyaDoraEmbeddingCombo")
        self.embedding_combo.setMinimumWidth(ui_px(168))
        self.embedding_combo.setToolTip(
            "Pilih model embedding Ollama yang memang mendukung /api/embed. "
            "Tanpa pilihan, Tanya DORA memakai pencarian keyword FTS5."
        )
        source_row.addWidget(self.embedding_combo, 1)
        source_row.addStretch(1)
        controls_layout.addLayout(source_row)

        instructions_heading = QLabel("Arahan jawaban")
        instructions_heading.setObjectName("tanyaDoraFieldLabel")
        controls_layout.addWidget(instructions_heading)
        self.instructions_edit = QPlainTextEdit()
        self.instructions_edit.setObjectName("tanyaDoraInstructions")
        self.instructions_edit.setPlaceholderText(
            "Tulis arahan gaya, peran, atau tingkat detail. Pertanyaan yang kamu ajukan tetap bebas."
        )
        self.instructions_edit.setPlainText(
            str(self._settings.value("tanya_dora/instructions", "") or "")
        )
        self.instructions_edit.setFixedHeight(ui_px(76))
        controls_layout.addWidget(self.instructions_edit)

        settings_footer = QHBoxLayout()
        settings_note = QLabel(
            "Jika Ollama belum tersedia, mode uji lokal memberi balasan contoh. Balasan uji bukan hasil AI; tidak ada training atau pencarian online."
        )
        settings_note.setObjectName("tanyaDoraSettingsNote")
        settings_note.setWordWrap(True)
        settings_footer.addWidget(settings_note, 1)
        self.save_settings_button = QPushButton("Simpan pengaturan")
        self.save_settings_button.setObjectName("tanyaDoraSmallButton")
        self.save_settings_button.clicked.connect(
            lambda _checked=False: self._save_chat_settings()
        )
        settings_footer.addWidget(self.save_settings_button)
        controls_layout.addLayout(settings_footer)
        root.addWidget(controls)
        if self.compact_mode:
            controls.hide()
        self.embedding_combo.setEnabled(self.include_dkb_checkbox.isChecked())

        workspace = QSplitter(Qt.Horizontal)
        workspace.setObjectName("tanyaDoraWorkspace")
        workspace.setChildrenCollapsible(False)
        root.addWidget(workspace, 1)

        history_panel = QFrame()
        history_panel.setObjectName("tanyaDoraHistoryPanel")
        self.history_panel = history_panel
        history_panel.setMinimumWidth(ui_px(188))
        history_panel.setMaximumWidth(ui_px(300))
        history_layout = QVBoxLayout(history_panel)
        history_layout.setContentsMargins(ui_px(14), ui_px(14), ui_px(14), ui_px(14))
        history_layout.setSpacing(ui_px(10))
        history_heading = QHBoxLayout()
        history_title = QLabel("Percakapan")
        history_title.setObjectName("tanyaDoraSectionTitle")
        history_heading.addWidget(history_title, 1)
        self.new_conversation_button = QPushButton("+")
        self.new_conversation_button.setObjectName("tanyaDoraIconButton")
        self.new_conversation_button.setAccessibleName("Percakapan baru")
        self.new_conversation_button.setToolTip("Percakapan baru")
        self.new_conversation_button.clicked.connect(
            lambda _checked=False: self._new_conversation()
        )
        history_heading.addWidget(self.new_conversation_button)
        history_layout.addLayout(history_heading)

        self.history_list = QListWidget()
        self.history_list.setObjectName("tanyaDoraHistoryList")
        self.history_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(
            lambda position: self._history_context_menu(position)
        )
        self.history_list.currentItemChanged.connect(
            lambda current, previous: self._on_history_selected(current, previous)
        )
        history_layout.addWidget(self.history_list, 1)

        history_actions = QHBoxLayout()
        self.rename_button = QPushButton("Ganti judul")
        self.rename_button.setObjectName("tanyaDoraSmallButton")
        self.rename_button.clicked.connect(
            lambda _checked=False: self._rename_conversation()
        )
        history_actions.addWidget(self.rename_button)
        self.delete_button = QPushButton("Hapus")
        self.delete_button.setObjectName("tanyaDoraSmallButton")
        self.delete_button.clicked.connect(
            lambda _checked=False: self._delete_conversation()
        )
        history_actions.addWidget(self.delete_button)
        history_layout.addLayout(history_actions)
        privacy_note = QLabel("Riwayat tersimpan lokal di SQLite.")
        privacy_note.setObjectName("tanyaDoraPrivacyNote")
        privacy_note.setWordWrap(True)
        history_layout.addWidget(privacy_note)
        workspace.addWidget(history_panel)

        chat_panel = QFrame()
        chat_panel.setObjectName("tanyaDoraChatPanel")
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(ui_px(18), ui_px(16), ui_px(18), ui_px(16))
        chat_layout.setSpacing(ui_px(10))

        chat_actions = QHBoxLayout()
        chat_actions.addStretch(1)
        self.copy_button = QPushButton("Salin jawaban")
        self.copy_button.setObjectName("tanyaDoraSmallButton")
        self.copy_button.clicked.connect(
            lambda _checked=False: self._copy_last_answer()
        )
        chat_actions.addWidget(self.copy_button)
        self.regenerate_button = QPushButton("Buat ulang")
        self.regenerate_button.setObjectName("tanyaDoraSmallButton")
        self.regenerate_button.clicked.connect(
            lambda _checked=False: self._regenerate()
        )
        chat_actions.addWidget(self.regenerate_button)
        chat_layout.addLayout(chat_actions)

        self.transcript = QTextEdit()
        self.transcript.setObjectName("tanyaDoraTranscript")
        self.transcript.setReadOnly(True)
        self.transcript.setAcceptRichText(False)
        self.transcript.setPlaceholderText(
            "Mulai percakapan. Tanpa Ollama, coba: halo, apa itu DORA, atau apa itu data sains."
        )
        chat_layout.addWidget(self.transcript, 1)

        references_frame = QFrame()
        references_frame.setObjectName("tanyaDoraReferences")
        self.references_frame = references_frame
        references_frame.setMinimumHeight(ui_px(116))
        references_layout = QHBoxLayout(references_frame)
        references_layout.setContentsMargins(ui_px(12), ui_px(9), ui_px(12), ui_px(9))
        references_layout.setSpacing(ui_px(12))
        references_title = QLabel("Sumber & hasil tool")
        references_title.setObjectName("tanyaDoraReferenceHeading")
        references_title.setAlignment(Qt.AlignTop)
        references_layout.addWidget(references_title)
        self.source_list = QListWidget()
        self.source_list.setObjectName("tanyaDoraSourceList")
        self.source_list.setMaximumHeight(ui_px(90))
        self.source_list.currentItemChanged.connect(
            lambda current, previous: self._on_source_selected(current, previous)
        )
        references_layout.addWidget(self.source_list, 1)
        self.source_detail = QLabel("Sumber yang dipakai akan muncul di sini.")
        self.source_detail.setObjectName("tanyaDoraSourceDetail")
        self.source_detail.setWordWrap(True)
        self.source_detail.setMinimumWidth(ui_px(170))
        self.source_detail.setMaximumWidth(ui_px(260))
        references_layout.addWidget(self.source_detail)
        chat_layout.addWidget(references_frame)

        self.activity_label = QLabel(
            "Tanpa Ollama, Tanya DORA memakai balasan contoh lokal untuk pengujian."
        )
        self.activity_label.setObjectName("tanyaDoraActivity")
        self.activity_label.setWordWrap(True)
        chat_layout.addWidget(self.activity_label)

        input_row = QHBoxLayout()
        input_row.setSpacing(ui_px(10))
        self.message_input = QPlainTextEdit()
        self.message_input.setObjectName("tanyaDoraInput")
        self.message_input.setPlaceholderText(
            "Tanpa Ollama, coba: halo, apa itu DORA, atau apa itu data sains…"
        )
        self.message_input.setFixedHeight(ui_px(88))
        self.message_input.installEventFilter(self)
        self.message_input.setTabChangesFocus(True)
        input_row.addWidget(self.message_input, 1)
        send_buttons = QVBoxLayout()
        send_buttons.setSpacing(ui_px(7))
        self.send_button = QPushButton("Kirim")
        self.send_button.setObjectName("primaryButton")
        self.send_button.setMinimumWidth(ui_px(94))
        self.send_button.clicked.connect(
            lambda _checked=False: self._send_message()
        )
        send_buttons.addWidget(self.send_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("tanyaDoraStopButton")
        self.stop_button.setMinimumWidth(ui_px(94))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(
            lambda _checked=False: self._stop_generation()
        )
        send_buttons.addWidget(self.stop_button)
        send_buttons.addStretch(1)
        input_row.addLayout(send_buttons)
        chat_layout.addLayout(input_row)
        workspace.addWidget(chat_panel)
        workspace.setStretchFactor(0, 0)
        workspace.setStretchFactor(1, 1)
        workspace.setSizes([ui_px(230), ui_px(900)])
        self.workspace = workspace
        self.chat_panel = chat_panel
        if self.compact_mode:
            history_panel.hide()
            self.copy_button.hide()
            self.regenerate_button.hide()
            references_frame.hide()
            self.activity_label.hide()
            self.transcript.hide()

            self.compact_messages_scroll = QScrollArea()
            self.compact_messages_scroll.setObjectName("tanyaDoraMessagesScroll")
            self.compact_messages_scroll.setWidgetResizable(True)
            self.compact_messages_scroll.setFrameShape(QFrame.NoFrame)
            self.message_list_host = QWidget()
            self.message_list_host.setObjectName("tanyaDoraMessageList")
            self.message_list_layout = QVBoxLayout(self.message_list_host)
            self.message_list_layout.setContentsMargins(ui_px(4), ui_px(8), ui_px(4), ui_px(8))
            self.message_list_layout.setSpacing(ui_px(8))
            self.compact_messages_scroll.setWidget(self.message_list_host)
            chat_layout.removeWidget(self.transcript)
            chat_layout.insertWidget(1, self.compact_messages_scroll, 1)

            self.message_input.setFixedHeight(ui_px(48))
            self.message_input.setPlaceholderText("Tulis pesan...")
            self.send_button.setText("→")
            self.send_button.setAccessibleName("Kirim pesan")
            self.send_button.setToolTip("Kirim pesan")
            self.send_button.setMinimumWidth(ui_px(44))
            self.send_button.setFixedSize(ui_px(44), ui_px(44))
            self.stop_button.hide()
            chat_layout.setContentsMargins(ui_px(10), ui_px(8), ui_px(10), ui_px(10))
            chat_layout.setSpacing(ui_px(7))
            root.setStretch(1, 1)
            self._render_messages(())

        shortcut = QShortcut(QKeySequence("Ctrl+Return"), self.message_input)
        shortcut.activated.connect(lambda: self._send_message())
        shortcut_alt = QShortcut(QKeySequence("Ctrl+Enter"), self.message_input)
        shortcut_alt.activated.connect(lambda: self._send_message())

    def _initialize_local_test_mode(self) -> None:
        """Prepare offline chat without starting a background Ollama request."""
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("Balasan uji lokal (tanpa Ollama)", None)
        self.model_combo.setEnabled(False)
        self.model_combo.blockSignals(False)
        self.refresh_button.setEnabled(True)
        self.send_button.setEnabled(True)
        self._set_model_status(
            "Mode uji lokal siap · klik Perbarui model untuk memeriksa Ollama",
            "warning",
        )

    @Slot(bool)
    def _toggle_settings_panel(self, expanded: bool) -> None:
        self.settings_panel.setVisible(bool(expanded))
        self.settings_toggle_button.setText(
            "Tutup pengaturan" if expanded else "Pengaturan chat"
        )

    @Slot(bool)
    def _on_dkb_setting_changed(self, enabled: bool) -> None:
        self.embedding_combo.setEnabled(bool(enabled))

    @Slot()
    def _save_chat_settings(self) -> None:
        self._settings.setValue("tanya_dora/model", str(self.model_combo.currentData() or ""))
        self._settings.setValue("tanya_dora/temperature", self.temperature_spin.value())
        self._settings.setValue("tanya_dora/include_dkb", self.include_dkb_checkbox.isChecked())
        self._settings.setValue(
            "tanya_dora/embedding_model", str(self.embedding_combo.currentData() or "")
        )
        self._settings.setValue(
            "tanya_dora/instructions", self.instructions_edit.toPlainText().strip()[:4000]
        )
        self._settings.sync()
        if hasattr(self, "activity_label"):
            self.activity_label.setText("Pengaturan chat lokal tersimpan.")

    # ── Drag to reposition compact overlay ───────────────────────────────
    def _header_contains(self, pos: "QPoint") -> bool:
        """Return True if pos (widget-local) is within the header area."""
        sep = self.findChild(QFrame, "tanyaDoraHeaderSep")
        if sep is not None:
            return pos.y() <= sep.geometry().bottom()
        return pos.y() <= 60

    def mousePressEvent(self, event) -> None:
        if (
            self.compact_mode
            and event.button() == Qt.LeftButton
            and self._header_contains(event.position().toPoint())
        ):
            self._drag_pos = event.globalPosition().toPoint() - self.parentWidget().frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.compact_mode and self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            parent = self.parentWidget()
            if parent is not None:
                new_top_left = event.globalPosition().toPoint() - self._drag_pos
                # Clamp so widget stays within parent bounds
                pw, ph = parent.width(), parent.height()
                sw, sh = self.width(), self.height()
                x = max(0, min(new_top_left.x(), pw - sw))
                y = max(0, min(new_top_left.y(), ph - sh))
                self.move(x, y)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_pos = None
        super().mouseReleaseEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if self.compact_mode and watched is self.parentWidget() and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._position_compact_overlay)
        if watched is self.message_input and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if event.modifiers() & Qt.ShiftModifier:
                    # Shift+Enter: insert newline
                    return False
                self._send_message()
                return True
        return super().eventFilter(watched, event)

    @Slot()
    def _refresh_models(self) -> None:
        if self._is_shutting_down:
            return
        if self._model_thread is not None and self._model_thread.isRunning():
            return
        self.refresh_button.setEnabled(False)
        self.model_status.setText("Memeriksa Ollama dan model lokal…")
        self.model_status.setProperty("state", "pending")
        self._polish_status()
        thread = _ModelRefreshThread(self.llm_manager, self)
        thread.completed.connect(
            lambda models, status: self._queue_ui_callback(
                self._on_models_refreshed, models, status
            )
        )
        thread.finished.connect(
            lambda thread=thread: self._queue_ui_callback(
                self._model_refresh_finished, thread
            )
        )
        self._model_thread = thread
        thread.start()

    @Slot(object, object)
    def _on_models_refreshed(self, models: object, status: object) -> None:
        model_items = tuple(item for item in models if isinstance(item, LocalModel))
        self._current_models = model_items
        previous_chat = self.model_combo.currentData()
        active_runtime = self.model_version_store.get_active_runtime_model()
        active_model_name = str(active_runtime.get("model_name") or "") if active_runtime else ""
        previous_embedding = self.embedding_combo.currentData() or self._settings.value(
            "tanya_dora/embedding_model", ""
        )
        self.model_combo.blockSignals(True)
        self.embedding_combo.blockSignals(True)
        self.model_combo.clear()
        self.embedding_combo.clear()
        self.embedding_combo.addItem("Keyword saja", None)
        if not model_items:
            self.model_combo.addItem("Tidak ada model lokal", None)
            self.model_combo.setEnabled(False)
        else:
            self.model_combo.setEnabled(True)
            self.model_combo.addItem("Pilih model lokal", None)
        for model in model_items:
            display = f"{model.name} · {model.size_label}"
            self.model_combo.addItem(display, model.name)
            self.model_combo.setItemData(self.model_combo.count() - 1, model.name, Qt.ToolTipRole)
            self.embedding_combo.addItem(model.name, model.name)
        chat_index = self.model_combo.findData(previous_chat) if previous_chat else -1
        if chat_index < 0 and active_model_name:
            chat_index = self.model_combo.findData(active_model_name)
        if chat_index < 0:
            saved_model_name = str(self._settings.value("tanya_dora/model", "") or "")
            chat_index = self.model_combo.findData(saved_model_name)
        if chat_index < 0 and model_items:
            chat_index = 0
        if chat_index >= 0:
            self.model_combo.setCurrentIndex(chat_index)
        embedding_index = self.embedding_combo.findData(previous_embedding)
        if embedding_index < 0:
            embedding_index = 0
        self.embedding_combo.setCurrentIndex(embedding_index)
        self.model_combo.blockSignals(False)
        self.embedding_combo.blockSignals(False)
        self.refresh_button.setEnabled(True)
        self._on_model_changed(self.model_combo.currentIndex())

        ollama_status = status if isinstance(status, OllamaStatus) else None
        if ollama_status is None or not ollama_status.available:
            detail = ollama_status.detail if ollama_status else "Status Ollama tidak tersedia."
            self._set_model_status("Mode uji lokal aktif · Ollama tidak tersedia", "warning")
            self.model_status.setToolTip(
                f"{detail} Balasan contoh tersedia tanpa Ollama; balasan bukan hasil model AI."
            )
            self.send_button.setEnabled(not self._is_generating())
        elif not model_items:
            self._set_model_status(
                "Mode uji lokal aktif · belum ada model Ollama terpasang", "warning"
            )
            self.send_button.setEnabled(not self._is_generating())
        else:
            if self.model_combo.currentData():
                self._set_model_status(
                    f"Ollama lokal aktif · {len(model_items)} model terpasang", "ready"
                )
                self.send_button.setEnabled(not self._is_generating())
            else:
                active_detail = (
                    f"Model aktif tersimpan {active_model_name} tidak tersedia; pilih model baru secara eksplisit"
                    if active_model_name else "Pilih model lokal sebelum chat"
                )
                self._set_model_status(
                    f"Ollama aktif · {len(model_items)} model · {active_detail} · mode uji juga tersedia",
                    "warning",
                )
                self.send_button.setEnabled(not self._is_generating())
        self._update_action_buttons()

    def set_active_model(self, model_name: str) -> bool:
        """Apply an explicitly selected local model to the open chat page."""
        normalized = str(model_name or "").strip()
        index = self.model_combo.findData(normalized)
        if index < 0:
            self._refresh_models()
            return False
        self.model_combo.setCurrentIndex(index)
        return str(self.model_combo.currentData() or "") == normalized

    def _model_refresh_finished(self, thread: _ModelRefreshThread) -> None:
        if self._model_thread is thread:
            self._model_thread = None
        thread.deleteLater()

    def _queue_ui_callback(self, callback, *args) -> None:
        """Run worker notifications on this page's Qt thread, while it is alive."""
        if self._is_shutting_down:
            return
        QTimer.singleShot(
            0,
            self,
            lambda callback=callback, args=args: self._run_ui_callback(callback, args),
        )

    def _persist_temporary_conversation_ids(self) -> None:
        if self._temporary_conversation_ids:
            self._settings.setValue(
                self.TEMPORARY_CONVERSATIONS_SETTING,
                sorted(self._temporary_conversation_ids),
            )
        else:
            self._settings.remove(self.TEMPORARY_CONVERSATIONS_SETTING)
        self._settings.sync()

    def _mark_temporary_conversation(self, conversation_id: str) -> None:
        normalized_id = str(conversation_id or "").strip()
        if not normalized_id:
            return
        self._temporary_conversation_ids.add(normalized_id)
        self._persist_temporary_conversation_ids()

    def _remove_stale_temporary_conversations(self) -> None:
        stale_ids = self._settings.value(
            self.TEMPORARY_CONVERSATIONS_SETTING,
            [],
            type=list,
        )
        if isinstance(stale_ids, str):
            stale_ids = [stale_ids]
        failed_ids = []
        for conversation_id in stale_ids:
            normalized_id = str(conversation_id or "").strip()
            if not normalized_id:
                continue
            try:
                self.conversation_store.delete_conversation(normalized_id)
            except Exception:
                failed_ids.append(normalized_id)
        if failed_ids:
            self._settings.setValue(self.TEMPORARY_CONVERSATIONS_SETTING, failed_ids)
        else:
            self._settings.remove(self.TEMPORARY_CONVERSATIONS_SETTING)
        self._settings.sync()

    def _run_ui_callback(self, callback, args) -> None:
        if self._is_shutting_down:
            return
        try:
            callback(*args)
        except Exception as error:
            detail = str(error).strip() or error.__class__.__name__
            self.activity_label.setText(f"Pembaruan Tanya DORA gagal: {detail[:240]}")
            self._finish_generation_state()

    @Slot(int)
    def _on_model_changed(self, _index: int) -> None:
        model_name = self.model_combo.currentData()
        if model_name:
            self._settings.setValue("tanya_dora/model", str(model_name))
            try:
                model = self.llm_manager.select_cached_model(str(model_name))
            except Exception:
                self._set_model_status("Model belum masuk inventaris terbaru.", "warning")
            else:
                self._set_model_status(
                    f"Model dipilih · {model.name} · {model.size_label}", "ready"
                )
                self.send_button.setEnabled(not self._is_generating())
        else:
            active_runtime = self.model_version_store.get_active_runtime_model()
            if active_runtime:
                self._set_model_status(
                    f"Model aktif {active_runtime.get('model_name')} tidak tersedia; balasan contoh mode uji siap dipakai.",
                    "warning",
                )
            else:
                self._set_model_status("Mode uji lokal siap · pilih model Ollama untuk jawaban AI", "warning")
            self.send_button.setEnabled(not self._is_generating())

    def _set_model_status(self, text: str, state: str) -> None:
        self.model_status.setText(text)
        self.model_status.setToolTip(text)
        self.model_status.setProperty("state", state)
        self._polish_status()

    def _polish_status(self) -> None:
        self.model_status.style().unpolish(self.model_status)
        self.model_status.style().polish(self.model_status)

    def _load_conversations(self, select_id: str | None = None) -> None:
        conversations = self.conversation_store.list_conversations()
        self.history_list.blockSignals(True)
        self.history_list.clear()
        target = select_id or self._active_conversation_id
        for conversation in conversations:
            item = QListWidgetItem(conversation.title)
            item.setData(Qt.UserRole, conversation.conversation_id)
            item.setToolTip(conversation.title)
            self.history_list.addItem(item)
            if conversation.conversation_id == target:
                self.history_list.setCurrentItem(item)
        if self.history_list.currentItem() is None and conversations:
            self.history_list.setCurrentRow(0)
        self.history_list.blockSignals(False)
        selected = self.history_list.currentItem()
        if selected is not None:
            self._activate_conversation(str(selected.data(Qt.UserRole)))
        else:
            self._active_conversation_id = None
            self._render_messages(())
        self._update_action_buttons()

    def _load_conversations_sidebar_only(self, select_id: str | None = None) -> None:
        """Refresh the history sidebar without re-rendering the chat area.

        Use this after a local-mode send where the chat area has already been
        rendered, to avoid an extra _render_compact_messages call that would
        trigger another widget create/destroy cycle while the MainWindow scale
        timer (_refresh_new_widget_scale) may still hold references to the
        previous round of chat bubble widgets.
        """
        conversations = self.conversation_store.list_conversations()
        target = select_id or self._active_conversation_id
        self.history_list.blockSignals(True)
        self.history_list.clear()
        for conversation in conversations:
            item = QListWidgetItem(conversation.title)
            item.setData(Qt.UserRole, conversation.conversation_id)
            item.setToolTip(conversation.title)
            self.history_list.addItem(item)
            if conversation.conversation_id == target:
                self.history_list.setCurrentItem(item)
        if self.history_list.currentItem() is None and conversations:
            self.history_list.setCurrentRow(0)
        self.history_list.blockSignals(False)
        # Keep _active_conversation_id in sync without calling _activate_conversation
        # (which would re-render the chat area).
        selected = self.history_list.currentItem()
        if selected is not None:
            self._active_conversation_id = str(selected.data(Qt.UserRole))
        self._update_action_buttons()

    def _activate_conversation(self, conversation_id: str) -> None:
        self._active_conversation_id = conversation_id
        self._last_response = None
        messages = self.conversation_store.get_messages(conversation_id)
        self._render_messages(messages)
        last_assistant = next((message for message in reversed(messages) if message.role == "assistant"), None)
        self._show_sources(last_assistant.sources if last_assistant else ())
        self._update_action_buttons()

    def _render_messages(
        self,
        messages: Sequence[ConversationMessage],
        *,
        streaming_text: str = "",
    ) -> None:
        if self.compact_mode:
            self._render_compact_messages(messages, streaming_text=streaming_text)
            return
        self.transcript.clear()
        if not messages:
            self.transcript.setPlainText(
                "Selamat datang di Tanya DORA.\n\n"
                "Tanpa Ollama, kamu tetap bisa mencoba balasan contoh lokal: halo, apa itu DORA, "
                "data sains, dataset, analisis data, machine learning, dan AI. Balasan uji "
                "bukan hasil model AI. Pilih model Ollama untuk jawaban AI yang lebih bebas.\n\n"
                "Dokumen percakapan disimpan di database lokal yang terpisah dari Knowledge Base."
            )
            return
        for index, message in enumerate(messages):
            if index:
                self.transcript.append("")
            if message.role != "user" and message.model == "mode-uji-lokal":
                label = "Tanya DORA · balasan uji"
            elif message.role == "user":
                label = "Anda"
            elif message.status == "interrupted":
                label = "Tanya DORA · dihentikan"
            elif message.status == "error":
                label = "Tanya DORA · perlu diperiksa"
            else:
                label = "Tanya DORA"
            self.transcript.append(f"{label}\n{message.content}")
        cursor = self.transcript.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    def _render_compact_messages(
        self,
        messages: Sequence[ConversationMessage],
        *,
        streaming_text: str = "",
    ) -> None:
        if self.message_list_layout is None or self.message_list_host is None:
            return
        # Use setParent(None) instead of deleteLater() so each widget's C++ object
        # is destroyed immediately and synchronously.  deleteLater() defers destruction
        # to the next event-loop iteration; during that window the MainWindow
        # _refresh_new_widget_scale timer can try to access the C++ object via
        # _apply_widget_scale / findChildren, causing an access violation.
        while self.message_list_layout.count():
            item = self.message_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)  # immediate C++ destruction, no deferred deletion

        entries = [(message.role, message.content, message) for message in messages]
        if not entries:
            entries = [
                (
                    "assistant",
                    "Halo! Aku Tanya DORA. Tanyakan apa saja, misalnya tentang DORA atau data sains.",
                    None,
                )
            ]
        if streaming_text:
            entries.append(("assistant", streaming_text, None))

        available_width = self.compact_messages_scroll.viewport().width()
        if available_width > 0:
            self.message_list_host.setMinimumWidth(available_width)
        bubble_width = min(
            ui_px(330),
            max(ui_px(170), (available_width or ui_px(380)) - ui_px(38)),
        )
        for role, content, message in entries:
            row = QWidget(self.message_list_host)
            row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(0)
            user_message = role == "user"
            if user_message:
                row_layout.addStretch(1)

            bubble = QFrame(row)
            bubble.setObjectName(
                "tanyaDoraUserBubble" if user_message else "tanyaDoraAssistantBubble"
            )
            bubble_layout = QVBoxLayout(bubble)
            bubble_layout.setContentsMargins(ui_px(12), ui_px(9), ui_px(12), ui_px(9))
            bubble_layout.setSpacing(ui_px(3))

            if message is not None and not user_message:
                if message.model == "mode-uji-lokal":
                    tag_text = "BALASAN UJI"
                elif message.status == "interrupted":
                    tag_text = "DIHENTIKAN"
                elif message.status == "error":
                    tag_text = "PERLU DIPERIKSA"
                else:
                    tag_text = ""
                if tag_text:
                    tag = QLabel(tag_text)
                    tag.setObjectName("tanyaDoraDemoTag")
                    bubble_layout.addWidget(tag)

            text_label = QLabel(str(content))
            text_label.setObjectName("tanyaDoraBubbleText")
            text_label.setWordWrap(True)
            text_label.setTextInteractionFlags(Qt.NoTextInteraction)
            text_width = min(
                bubble_width - ui_px(24),
                max(ui_px(48), text_label.fontMetrics().horizontalAdvance(str(content)[:34])),
            )
            bubble.setFixedWidth(text_width + ui_px(24))
            text_label.setMinimumWidth(text_width)
            text_label.setMaximumWidth(text_width)
            bubble_layout.addWidget(text_label)
            row_layout.addWidget(bubble)
            if not user_message:
                row_layout.addStretch(1)
            self.message_list_layout.addWidget(row)

        self.message_list_layout.addStretch(1)
        QTimer.singleShot(0, self, self._scroll_compact_messages_to_bottom)

    def _scroll_compact_messages_to_bottom(self) -> None:
        if self.compact_mode and self.compact_messages_scroll is not None:
            scrollbar = self.compact_messages_scroll.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    # ------------------------------------------------------------------
    # Typing indicator (local-mode delay animation)
    # ------------------------------------------------------------------

    def _start_typing_indicator(self) -> None:
        """Add an animated 'mengetik…' bubble and start the dot-pulse timer."""
        if self.message_list_layout is None or self.message_list_host is None:
            return
        self._stop_typing_indicator()          # clear any stale indicator
        self._typing_dot_frame = 0

        # Build the typing bubble row
        row = QWidget(self.message_list_host)
        row.setObjectName("tanyaDoraTypingRow")
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)

        bubble = QFrame(row)
        bubble.setObjectName("tanyaDoraTypingBubble")
        bubble_layout = QHBoxLayout(bubble)
        bubble_layout.setContentsMargins(ui_px(14), ui_px(10), ui_px(14), ui_px(10))
        bubble_layout.setSpacing(ui_px(5))

        # Three dot labels
        self._typing_dots: list[QLabel] = []
        for _ in range(3):
            dot = QLabel("●")
            dot.setObjectName("tanyaDoraTypingDot")
            bubble_layout.addWidget(dot)
            self._typing_dots.append(dot)

        row_layout.addWidget(bubble)
        row_layout.addStretch(1)

        # Remove trailing stretch, add row, re-add stretch
        last = self.message_list_layout.count() - 1
        if last >= 0:
            stretch_item = self.message_list_layout.takeAt(last)
            del stretch_item
        self.message_list_layout.addWidget(row)
        self.message_list_layout.addStretch(1)
        self._typing_bubble = row
        QTimer.singleShot(0, self, self._scroll_compact_messages_to_bottom)

        # Animate dots: each dot lights up in turn at 350 ms intervals
        self._typing_dot_timer = QTimer(self)
        self._typing_dot_timer.setInterval(350)
        self._typing_dot_timer.timeout.connect(self._tick_typing_dots)
        self._typing_dot_timer.start()

    def _tick_typing_dots(self) -> None:
        """Cycle the lit dot through the three dots."""
        if not hasattr(self, "_typing_dots") or not self._typing_dots:
            return
        n = len(self._typing_dots)
        self._typing_dot_frame = (self._typing_dot_frame + 1) % n
        for i, dot in enumerate(self._typing_dots):
            dot.setProperty("active", i == self._typing_dot_frame)
            dot.style().unpolish(dot)
            dot.style().polish(dot)

    def _stop_typing_indicator(self) -> None:
        """Remove the typing bubble and stop the dot timer."""
        if self._typing_dot_timer is not None:
            self._typing_dot_timer.stop()
            self._typing_dot_timer.deleteLater()
            self._typing_dot_timer = None
        if self._typing_bubble is not None:
            # Remove from layout then destroy immediately (avoid deleteLater race)
            if self.message_list_layout is not None:
                idx = self.message_list_layout.indexOf(self._typing_bubble)
                if idx >= 0:
                    item = self.message_list_layout.takeAt(idx)
                    del item
            self._typing_bubble.setParent(None)
            self._typing_bubble = None
        self._typing_dots = []

    @Slot()
    def _deliver_local_answer(self) -> None:
        """Called after the typing delay: save the answer and render it."""
        if self._typing_timer is not None:
            self._typing_timer.deleteLater()
            self._typing_timer = None
        self._stop_typing_indicator()

        answer = self._pending_local_answer or ""
        conversation_id = self._pending_conversation_id
        self._pending_local_answer = None
        self._pending_conversation_id = None

        if not conversation_id or self._is_shutting_down:
            self.send_button.setEnabled(True)
            self._update_action_buttons()
            return

        self.conversation_store.append_message(
            conversation_id,
            "assistant",
            answer,
            model="mode-uji-lokal",
        )
        final_messages = self.conversation_store.get_messages(conversation_id)
        self._render_messages(final_messages)
        self._load_conversations_sidebar_only(conversation_id)
        self.activity_label.setText(
            "Balasan contoh untuk pengujian · tanpa Ollama, tanpa training, dan tanpa pencarian online."
        )
        self.send_button.setEnabled(True)
        self._update_action_buttons()

    @Slot()
    def _show_compact_menu(self) -> None:
        menu = QMenu(self)
        settings_text = "Sembunyikan pengaturan" if self.settings_panel.isVisible() else "Pengaturan chat"
        settings_action = menu.addAction(settings_text)
        settings_action.triggered.connect(
            lambda: self.settings_panel.setVisible(not self.settings_panel.isVisible())
        )
        history_menu = menu.addMenu("Riwayat percakapan")
        if self.history_list.count() == 0:
            empty_action = history_menu.addAction("Belum ada percakapan")
            empty_action.setEnabled(False)
        else:
            for index in range(self.history_list.count()):
                item = self.history_list.item(index)
                conversation_id = str(item.data(Qt.UserRole))
                action = history_menu.addAction(item.text())
                action.setCheckable(True)
                action.setChecked(conversation_id == self._active_conversation_id)
                action.triggered.connect(
                    lambda _checked=False, target_id=conversation_id: self._select_compact_conversation(target_id)
                )
        menu.addSeparator()
        copy_action = menu.addAction("Salin jawaban terakhir")
        copy_action.setEnabled(self.copy_button.isEnabled())
        copy_action.triggered.connect(
            lambda _checked=False: self._copy_last_answer()
        )
        regenerate_action = menu.addAction("Buat ulang jawaban")
        regenerate_action.setEnabled(self.regenerate_button.isEnabled())
        regenerate_action.triggered.connect(
            lambda _checked=False: self._regenerate()
        )
        menu.exec(self.settings_toggle_button.mapToGlobal(self.settings_toggle_button.rect().bottomLeft()))

    def _select_compact_conversation(self, conversation_id: str) -> None:
        for index in range(self.history_list.count()):
            item = self.history_list.item(index)
            if str(item.data(Qt.UserRole)) == conversation_id:
                self.history_list.setCurrentItem(item)
                return
        self._activate_conversation(conversation_id)

    @Slot()
    def _close_compact_overlay(self) -> None:
        self.hide()
        main_window = self.window()
        current_key = getattr(main_window, "_current_page_key", "dashboard")
        if hasattr(main_window, "set_active_button"):
            main_window.set_active_button(current_key or "dashboard")

    def _position_compact_overlay(self) -> None:
        if not self.compact_mode:
            return
        parent = self.parentWidget()
        if parent is None or parent.width() <= 0 or parent.height() <= 0:
            return
        margin = ui_px(18)
        width = min(ui_px(420), max(ui_px(300), parent.width() - margin * 2))
        height = min(ui_px(640), max(ui_px(360), parent.height() - margin * 2))
        width = min(width, parent.width() - margin * 2)
        height = min(height, parent.height() - margin * 2)
        self.setGeometry(
            parent.width() - width - margin,
            parent.height() - height - margin,
            width,
            height,
        )
        if getattr(self, "_compact_shadow", None) is None:
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(ui_px(24))
            shadow.setOffset(0, ui_px(6))
            shadow.setColor(QColor(23, 56, 86, 48))
            self.setGraphicsEffect(shadow)
            self._compact_shadow = shadow
        if self.isVisible():
            self.raise_()

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_history_selected(self, current: QListWidgetItem | None, _previous) -> None:
        if current is not None:
            self._activate_conversation(str(current.data(Qt.UserRole)))

    @Slot()
    def _new_conversation(self) -> None:
        if self._is_generating():
            return
        conversation = self.conversation_store.create_conversation()
        self._mark_temporary_conversation(conversation.conversation_id)
        self._active_conversation_id = conversation.conversation_id
        self._load_conversations(conversation.conversation_id)
        self.message_input.setFocus()

    @Slot()
    def _rename_conversation(self) -> None:
        conversation = self._selected_conversation()
        if conversation is None:
            return
        title, accepted = QInputDialog.getText(
            self,
            "Ganti judul percakapan",
            "Judul:",
            text=conversation.title,
        )
        if not accepted:
            return
        try:
            self.conversation_store.rename_conversation(conversation.conversation_id, title)
        except (ValueError, ConversationStoreError) as error:
            QMessageBox.warning(self, "Judul tidak dapat disimpan", str(error))
            return
        self._load_conversations(conversation.conversation_id)

    @Slot()
    def _delete_conversation(self) -> None:
        conversation = self._selected_conversation()
        if conversation is None:
            return
        answer = QMessageBox.question(
            self,
            "Hapus percakapan",
            f"Hapus '{conversation.title}' dan seluruh pesannya dari penyimpanan lokal?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.conversation_store.delete_conversation(conversation.conversation_id)
        self._temporary_conversation_ids.discard(conversation.conversation_id)
        self._persist_temporary_conversation_ids()
        self._active_conversation_id = None
        self._load_conversations()

    @Slot(QPoint)
    def _history_context_menu(self, position) -> None:
        item = self.history_list.itemAt(position)
        if item is None:
            return
        self.history_list.setCurrentItem(item)
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        rename_action = menu.addAction("Ganti judul")
        delete_action = menu.addAction("Hapus percakapan")
        chosen = menu.exec(self.history_list.mapToGlobal(position))
        if chosen is rename_action:
            self._rename_conversation()
        elif chosen is delete_action:
            self._delete_conversation()

    def _selected_conversation(self):
        item = self.history_list.currentItem()
        if item is None:
            return None
        return self.conversation_store.get_conversation(str(item.data(Qt.UserRole)))

    @Slot()
    def _send_message(self, *, regenerate: bool = False) -> None:
        try:
            self._send_message_impl(regenerate=regenerate)
        except Exception as error:
            detail = str(error).strip() or error.__class__.__name__
            self.activity_label.setText(f"Pesan gagal diproses: {detail[:240]}")
            self.send_button.setEnabled(not self._is_generating())
            self.stop_button.setEnabled(self._is_generating())
            self._update_action_buttons()

    def _send_message_impl(self, *, regenerate: bool = False) -> None:
        if self._is_generating():
            return
        model_name = str(self.model_combo.currentData() or "").strip()

        conversation_id = self._active_conversation_id
        if conversation_id is None:
            conversation = self.conversation_store.create_conversation()
            conversation_id = conversation.conversation_id
            self._active_conversation_id = conversation_id
        self._mark_temporary_conversation(conversation_id)
        messages = self.conversation_store.get_messages(conversation_id)
        question = ""
        if regenerate:
            last_user = next((message for message in reversed(messages) if message.role == "user"), None)
            if last_user is None:
                return
            question = last_user.content
            trailing = [message for message in messages if message.message_id > last_user.message_id]
            for message in trailing:
                self.conversation_store.delete_message(message.message_id)
            messages = self.conversation_store.get_messages(conversation_id)
        else:
            question = self.message_input.toPlainText().strip()
            if not question:
                return
            self.conversation_store.append_message(conversation_id, "user", question)
            self.message_input.clear()
            messages = self.conversation_store.get_messages(conversation_id)
            self._load_conversations(conversation_id)

        history = [
            {"role": message.role, "content": message.content}
            for message in messages[:-1]
            if message.role in {"user", "assistant"}
        ][-8:]
        if not model_name:
            self._save_chat_settings()
            self._last_response = None
            self._show_sources(())
            answer = _local_test_answer(question)
            # Show user message immediately first, then animate typing delay.
            current_messages = self.conversation_store.get_messages(conversation_id)
            self._render_messages(current_messages)
            self._load_conversations_sidebar_only(conversation_id)
            # Disable send while "typing" animation plays
            self.send_button.setEnabled(False)
            self._pending_local_answer = answer
            self._pending_conversation_id = conversation_id
            # Start the animated typing bubble, deliver reply after 3-4 s
            self._start_typing_indicator()
            import random as _random
            delay_ms = _random.randint(1000, 2000)
            self._typing_timer = QTimer(self)
            self._typing_timer.setSingleShot(True)
            self._typing_timer.timeout.connect(self._deliver_local_answer)
            self._typing_timer.start(delay_ms)
            return
        try:
            self._save_chat_settings()
            include_knowledge_base = self.include_dkb_checkbox.isChecked()
            tool_context = (
                self.tool_context_provider()
                if callable(self.tool_context_provider)
                else None
            )
            tool_registry = build_dora_tool_registry(
                tool_context,
                self.knowledge_base,
                allow_online_research=False,
                include_knowledge_base=include_knowledge_base,
            )
        except Exception as error:
            self.activity_label.setText(f"Konteks DORA tidak dapat dibaca: {error}")
            self.send_button.setEnabled(bool(self.model_combo.currentData()))
            return
        self._streaming_text = ""
        self._streaming_started = False
        self._last_response = None
        self._show_sources(())
        self.activity_label.setText(
            "Mencari konteks DKB dan menyiapkan jawaban lokal..."
            if include_knowledge_base
            else "Menyiapkan jawaban dengan model lokal..."
        )
        self.send_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        if self.compact_mode:
            self.stop_button.show()
        self.refresh_button.setEnabled(False)
        self._update_action_buttons()
        cancel_event = threading.Event()
        self._cancel_event = cancel_event
        embedding_model = self.embedding_combo.currentData() if include_knowledge_base else None
        thread = _ChatThread(
            self.knowledge_base,
            self.llm_manager,
            question=question,
            model=model_name,
            embedding_model=str(embedding_model) if embedding_model else None,
            temperature=self.temperature_spin.value(),
            history=history,
            tool_registry=tool_registry,
            include_knowledge_base=include_knowledge_base,
            custom_instructions=self.instructions_edit.toPlainText().strip()[:4000],
            cancel_event=cancel_event,
            parent=self,
        )
        thread.delta.connect(
            lambda delta: self._queue_ui_callback(self._on_stream_delta, delta)
        )
        thread.tool_activity.connect(
            lambda name: self._queue_ui_callback(self._on_tool_activity, name)
        )
        thread.completed.connect(
            lambda response: self._queue_ui_callback(self._on_answer_completed, response)
        )
        thread.failed.connect(
            lambda detail: self._queue_ui_callback(self._on_answer_failed, detail)
        )
        thread.finished.connect(
            lambda thread=thread: self._queue_ui_callback(
                self._chat_thread_finished, thread
            )
        )
        self._chat_thread = thread
        thread.start()
        self._update_action_buttons()

    @Slot(str)
    def _on_stream_delta(self, delta: str) -> None:
        self._streaming_text += delta
        if self.compact_mode:
            messages = (
                self.conversation_store.get_messages(self._active_conversation_id)
                if self._active_conversation_id
                else ()
            )
            self._render_messages(messages, streaming_text=self._streaming_text)
            self._streaming_started = True
            return
        if not self._streaming_started:
            self._streaming_started = True
            self.transcript.append("Tanya DORA · sedang menjawab")
        cursor = self.transcript.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(delta)
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    @Slot(object)
    def _on_answer_completed(self, response: object) -> None:
        if not isinstance(response, RAGResponse):
            self._on_answer_failed("Respons RAG memiliki format yang tidak dikenal.")
            return
        conversation_id = self._active_conversation_id
        if conversation_id is None:
            return
        self._last_response = response
        self._show_sources(response.sources, response.tool_results)
        if response.answer:
            status = "interrupted" if response.cancelled else "complete"
            saved_sources = tuple(self._serialize_source(source) for source in response.sources)
            saved_sources += tuple(self._serialize_tool_result(result) for result in response.tool_results)
            self.conversation_store.append_message(
                conversation_id,
                "assistant",
                response.answer,
                model=response.model,
                sources=saved_sources,
                status=status,
            )
        self._render_messages(self.conversation_store.get_messages(conversation_id))
        if response.cancelled:
            self.activity_label.setText("Generasi dihentikan. Teks yang sudah dibuat tersimpan.")
        elif not response.grounded and response.answer:
            self.activity_label.setText(
                "Jawaban dibuat model lokal tanpa sumber DKB yang terverifikasi. Periksa fakta penting."
            )
        elif not response.grounded:
            self.activity_label.setText("DKB tidak menemukan sumber relevan untuk jawaban ini.")
        elif response.sources:
            tool_count = len(response.tool_results)
            tool_detail = f" · {tool_count} hasil tool" if tool_count else ""
            self.activity_label.setText(
                f"Jawaban berbasis {len(response.sources)} sumber{tool_detail} · {response.retrieval_mode}"
            )
        elif any(result.provides_evidence for result in response.tool_results):
            tool_names = ", ".join(dict.fromkeys(result.name for result in response.tool_results))
            self.activity_label.setText(f"Jawaban memakai hasil tool DORA: {tool_names}")
        else:
            self.activity_label.setText("DKB dan tool DORA belum memberi bukti yang cukup.")
        # Use sidebar-only refresh to avoid an extra _render_compact_messages call
        # immediately after the one above.  The Ollama path renders chat via streaming
        # deltas already; this call only needs to update the history list title.
        self._load_conversations_sidebar_only(conversation_id)
        self._finish_generation_state()

    @Slot(str)
    def _on_answer_failed(self, detail: str) -> None:
        conversation_id = self._active_conversation_id
        safe_detail = str(detail or "Terjadi kesalahan pada model lokal.").strip()
        if conversation_id and self._streaming_text.strip():
            self.conversation_store.append_message(
                conversation_id,
                "assistant",
                self._streaming_text.strip(),
                model=str(self.model_combo.currentData() or "") or None,
                status="error",
            )
            self._render_messages(self.conversation_store.get_messages(conversation_id))
        self.activity_label.setText(f"Generasi gagal: {safe_detail}")
        self._finish_generation_state()

    def _chat_thread_finished(self, thread: _ChatThread) -> None:
        if self._chat_thread is thread:
            self._chat_thread = None
        self._cancel_event = None
        self._update_action_buttons()
        thread.deleteLater()

    def _finish_generation_state(self) -> None:
        self.send_button.setEnabled(not self._is_generating())
        self.stop_button.setEnabled(False)
        if self.compact_mode:
            self.stop_button.hide()
        self.refresh_button.setEnabled(True)
        self._update_action_buttons()

    @Slot()
    def _stop_generation(self) -> None:
        if self._cancel_event is None:
            return
        self._cancel_event.set()
        self.llm_manager.cancel_generation()
        self.stop_button.setEnabled(False)
        self.activity_label.setText("Menghentikan generasi lokal…")

    @Slot()
    def _regenerate(self) -> None:
        self._send_message(regenerate=True)

    @Slot()
    def _copy_last_answer(self) -> None:
        messages = (
            self.conversation_store.get_messages(self._active_conversation_id)
            if self._active_conversation_id
            else ()
        )
        answer = next((item.content for item in reversed(messages) if item.role == "assistant"), "")
        if answer:
            QApplication.clipboard().setText(answer)
            self.activity_label.setText("Jawaban terakhir disalin ke clipboard.")

    @Slot(str)
    def _on_tool_activity(self, name: str) -> None:
        self.activity_label.setText(f"Menjalankan tool read-only: {name}…")

    def _show_sources(
        self,
        sources: Sequence[Any],
        tool_results: Sequence[ToolExecution] = (),
    ) -> None:
        self.source_list.clear()
        source_items = []
        for source in sources:
            if isinstance(source, Mapping):
                source_id = str(source.get("source_id") or "S")
                citation = str(source.get("citation") or source.get("title") or "Sumber DKB")
                detail = source.get("excerpt")
                if not detail and "result" in source:
                    detail = json.dumps(source["result"], ensure_ascii=False, indent=2)
                item = QListWidgetItem(f"[{source_id}] {citation}")
                item.setData(Qt.UserRole, str(detail or "Metadata sumber tersedia."))
            else:
                result = getattr(source, "result", None)
                if result is None:
                    continue
                source_id = str(getattr(source, "source_id", ""))
                citation = str(getattr(source, "citation", ""))
                item = QListWidgetItem(f"[{source_id}] {citation}")
                item.setData(Qt.UserRole, str(result.text))
            item.setToolTip(citation)
            self.source_list.addItem(item)
            source_items.append(item)
        for execution in tool_results:
            label = (
                "Online · OpenAlex/Crossref"
                if execution.name == "search_academic_papers"
                else f"DORA · {execution.name}"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, execution.to_json())
            item.setToolTip(
                "Hasil pencarian paper online" if execution.name == "search_academic_papers"
                else "Hasil tool read-only DORA"
            )
            self.source_list.addItem(item)
            source_items.append(item)
        if not source_items:
            self.source_list.addItem("Belum ada sumber yang digunakan.")
            self.source_detail.setText("Sumber yang dipakai akan muncul di sini.")
        else:
            self.source_list.setCurrentRow(0)

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_source_selected(self, current: QListWidgetItem | None, _previous) -> None:
        if current is not None and current.data(Qt.UserRole):
            text = str(current.data(Qt.UserRole))
            if text.startswith("{"):
                try:
                    text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
                except (TypeError, ValueError):
                    pass
            self.source_detail.setText(text[:900] + ("…" if len(text) > 900 else ""))

    @staticmethod
    def _serialize_source(source: Any) -> dict[str, Any]:
        result = source.result
        return {
            "source_id": source.source_id,
            "citation": source.citation,
            "title": result.title,
            "author_or_org": result.author_or_org,
            "year": result.publication_year,
            "page_or_section": result.page_or_section,
            "doi": result.doi,
            "source_url": result.source_url,
            "excerpt": result.text[:2000],
            "relevance_score": result.relevance_score,
        }

    @staticmethod
    def _serialize_tool_result(execution: ToolExecution) -> dict[str, Any]:
        return {
            "source_id": "ONLINE" if execution.name == "search_academic_papers" else "DORA",
            "citation": (
                "OpenAlex and Crossref search results"
                if execution.name == "search_academic_papers"
                else f"DORA read-only tool: {execution.name}"
            ),
            "tool": execution.name,
            "arguments": execution.arguments,
            "result": execution.result,
            "success": execution.success,
        }

    def _update_action_buttons(self) -> None:
        generating = self._is_generating()
        has_conversation = self._selected_conversation() is not None
        messages = (
            self.conversation_store.get_messages(self._active_conversation_id)
            if self._active_conversation_id
            else ()
        )
        self.rename_button.setEnabled(has_conversation and not self._is_generating())
        self.delete_button.setEnabled(has_conversation and not self._is_generating())
        self.new_conversation_button.setEnabled(not generating)
        if self.compact_mode:
            self.header_new_conversation_button.setEnabled(not generating)
        self.history_list.setEnabled(not generating)
        self.copy_button.setEnabled(
            any(message.role == "assistant" for message in messages)
        )
        self.regenerate_button.setEnabled(
            bool(messages)
            and any(message.role == "user" for message in messages)
            and not generating
        )

    def _is_generating(self) -> bool:
        # Also treat the local-mode typing delay as "generating" so that
        # the send button stays disabled and re-send is blocked.
        if self._typing_timer is not None and self._typing_timer.isActive():
            return True
        return self._chat_thread is not None and self._chat_thread.isRunning()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.compact_mode and self.message_list_layout is not None:
            # Use a flag to avoid stacking multiple pending timer callbacks when
            # many resize events fire in quick succession (e.g. during layout
            # geometry passes triggered by _render_compact_messages itself).
            if not getattr(self, "_resize_render_pending", False):
                self._resize_render_pending = True
                QTimer.singleShot(0, self, self._render_compact_messages_from_state)
        narrow = logical_width(event.size().width()) < 900
        self.source_detail.setVisible(not narrow)
        if narrow:
            self.source_detail.setText("Pilih sumber untuk melihat kutipan.")

    def _render_compact_messages_from_state(self) -> None:
        self._resize_render_pending = False
        if not self.compact_mode or self._is_shutting_down:
            return
        messages = (
            self.conversation_store.get_messages(self._active_conversation_id)
            if self._active_conversation_id
            else ()
        )
        self._render_compact_messages(
            messages,
            streaming_text=self._streaming_text if self._is_generating() else "",
        )

    def apply_scale(self, _scale=None) -> None:
        if self.compact_mode:
            self.layout().setContentsMargins(ui_px(12), ui_px(10), ui_px(12), ui_px(12))
            self._position_compact_overlay()
        else:
            self.layout().setContentsMargins(ui_px(26), ui_px(18), ui_px(26), ui_px(24))

    @Slot()
    def shutdown(self) -> None:
        if self._shutdown_complete:
            return
        self._is_shutting_down = True
        if self._cancel_event is not None:
            self._cancel_event.set()
        try:
            self.llm_manager.cancel_generation()
        except Exception:
            pass
        thread = self._chat_thread
        if thread is not None and thread.isRunning():
            thread.wait()
        model_thread = self._model_thread
        if model_thread is not None and model_thread.isRunning():
            model_thread.wait()
        # Cancel any pending typing-delay timer
        if self._typing_timer is not None:
            self._typing_timer.stop()
            self._typing_timer = None
        self._stop_typing_indicator()
        failed_ids = []
        for conversation_id in tuple(self._temporary_conversation_ids):
            try:
                self.conversation_store.delete_conversation(conversation_id)
            except Exception:
                failed_ids.append(conversation_id)
        self._temporary_conversation_ids = set(failed_ids)
        self._persist_temporary_conversation_ids()
        if self._owns_manager:
            try:
                self.llm_manager.close()
            except Exception:
                pass
        self._shutdown_complete = True

