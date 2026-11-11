"""Tanya DORA local research assistant page."""

from __future__ import annotations

import json
import threading
from typing import Any, Mapping, Sequence

from PySide6.QtCore import QEvent, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
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


class _ModelRefreshThread(QThread):
    completed = Signal(object, object)

    def __init__(self, manager: OllamaLLMManager, parent=None) -> None:
        super().__init__(parent)
        self.manager = manager

    def run(self) -> None:
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

    def __init__(
        self,
        *,
        conversation_store: ConversationStore | None = None,
        knowledge_base: DoraKnowledgeBase | None = None,
        llm_manager: OllamaLLMManager | None = None,
        model_version_store: ModelVersionStore | None = None,
        tool_context_provider=None,
    ) -> None:
        super().__init__()
        self.setObjectName("tanyaDoraPage")
        self.conversation_store = conversation_store or ConversationStore()
        self.knowledge_base = knowledge_base or DoraKnowledgeBase()
        self.llm_manager = llm_manager or OllamaLLMManager()
        self.model_version_store = model_version_store or ModelVersionStore()
        self._owns_manager = llm_manager is None
        self.tool_context_provider = tool_context_provider
        self._model_thread: _ModelRefreshThread | None = None
        self._chat_thread: _ChatThread | None = None
        self._cancel_event: threading.Event | None = None
        self._active_conversation_id: str | None = None
        self._current_models: tuple[LocalModel, ...] = ()
        self._streaming_text = ""
        self._streaming_started = False
        self._last_response: RAGResponse | None = None
        self._build_ui()
        self._load_conversations()
        self._refresh_models()
        application = QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.shutdown)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(ui_px(26), ui_px(18), ui_px(26), ui_px(24))
        root.setSpacing(ui_px(14))

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title_block.setSpacing(ui_px(3))
        title = QLabel("Tanya DORA")
        title.setObjectName("tanyaDoraTitle")
        subtitle = QLabel("Asisten riset lokal dengan jawaban berbasis sumber DKB")
        subtitle.setObjectName("tanyaDoraSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block, 1)

        self.model_status = QLabel("Memeriksa Ollama lokal…")
        self.model_status.setObjectName("tanyaDoraStatus")
        self.model_status.setProperty("state", "pending")
        self.model_status.setWordWrap(True)
        self.model_status.setMinimumWidth(ui_px(160))
        header.addWidget(self.model_status, 0, Qt.AlignRight | Qt.AlignVCenter)
        root.addLayout(header)

        controls = QFrame()
        controls.setObjectName("tanyaDoraControls")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(ui_px(14), ui_px(10), ui_px(14), ui_px(10))
        controls_layout.setSpacing(ui_px(10))

        model_label = QLabel("Model lokal")
        model_label.setObjectName("tanyaDoraFieldLabel")
        controls_layout.addWidget(model_label)
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("tanyaDoraModelCombo")
        self.model_combo.setMinimumWidth(ui_px(200))
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        controls_layout.addWidget(self.model_combo, 2)

        embedding_label = QLabel("Embedding")
        embedding_label.setObjectName("tanyaDoraFieldLabel")
        controls_layout.addWidget(embedding_label)
        self.embedding_combo = QComboBox()
        self.embedding_combo.setObjectName("tanyaDoraEmbeddingCombo")
        self.embedding_combo.setMinimumWidth(ui_px(168))
        self.embedding_combo.setToolTip(
            "Pilih model embedding Ollama yang memang mendukung /api/embed. "
            "Tanpa pilihan, Tanya DORA memakai pencarian keyword FTS5."
        )
        controls_layout.addWidget(self.embedding_combo, 1)

        self.online_research_checkbox = QCheckBox("Cari paper online")
        self.online_research_checkbox.setObjectName("tanyaDoraOnlineResearch")
        self.online_research_checkbox.setToolTip(
            "Jika diaktifkan, query paper dikirim ke OpenAlex dan Crossref. "
            "Inferensi tetap memakai Ollama lokal."
        )
        self.online_research_checkbox.setChecked(False)
        controls_layout.addWidget(self.online_research_checkbox)

        temperature_label = QLabel("Temperature")
        temperature_label.setObjectName("tanyaDoraFieldLabel")
        controls_layout.addWidget(temperature_label)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setObjectName("tanyaDoraTemperature")
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setDecimals(1)
        self.temperature_spin.setValue(0.2)
        self.temperature_spin.setFixedWidth(ui_px(82))
        controls_layout.addWidget(self.temperature_spin)

        self.refresh_button = QPushButton("Perbarui")
        self.refresh_button.setObjectName("secondaryButton")
        self.refresh_button.setToolTip("Periksa ulang model yang sudah terpasang di Ollama")
        self.refresh_button.clicked.connect(self._refresh_models)
        controls_layout.addWidget(self.refresh_button)
        root.addWidget(controls)

        workspace = QSplitter(Qt.Horizontal)
        workspace.setObjectName("tanyaDoraWorkspace")
        workspace.setChildrenCollapsible(False)
        root.addWidget(workspace, 1)

        history_panel = QFrame()
        history_panel.setObjectName("tanyaDoraHistoryPanel")
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
        self.new_conversation_button.clicked.connect(self._new_conversation)
        history_heading.addWidget(self.new_conversation_button)
        history_layout.addLayout(history_heading)

        self.history_list = QListWidget()
        self.history_list.setObjectName("tanyaDoraHistoryList")
        self.history_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self._history_context_menu)
        self.history_list.currentItemChanged.connect(self._on_history_selected)
        history_layout.addWidget(self.history_list, 1)

        history_actions = QHBoxLayout()
        self.rename_button = QPushButton("Ganti judul")
        self.rename_button.setObjectName("tanyaDoraSmallButton")
        self.rename_button.clicked.connect(self._rename_conversation)
        history_actions.addWidget(self.rename_button)
        self.delete_button = QPushButton("Hapus")
        self.delete_button.setObjectName("tanyaDoraSmallButton")
        self.delete_button.clicked.connect(self._delete_conversation)
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
        self.copy_button.clicked.connect(self._copy_last_answer)
        chat_actions.addWidget(self.copy_button)
        self.regenerate_button = QPushButton("Buat ulang")
        self.regenerate_button.setObjectName("tanyaDoraSmallButton")
        self.regenerate_button.clicked.connect(self._regenerate)
        chat_actions.addWidget(self.regenerate_button)
        chat_layout.addLayout(chat_actions)

        self.transcript = QTextEdit()
        self.transcript.setObjectName("tanyaDoraTranscript")
        self.transcript.setReadOnly(True)
        self.transcript.setAcceptRichText(False)
        self.transcript.setPlaceholderText(
            "Mulai percakapan. Jawaban Tanya DORA memakai sumber DKB yang ditemukan."
        )
        chat_layout.addWidget(self.transcript, 1)

        references_frame = QFrame()
        references_frame.setObjectName("tanyaDoraReferences")
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
        self.source_list.currentItemChanged.connect(self._on_source_selected)
        references_layout.addWidget(self.source_list, 1)
        self.source_detail = QLabel("Sumber yang dipakai akan muncul di sini.")
        self.source_detail.setObjectName("tanyaDoraSourceDetail")
        self.source_detail.setWordWrap(True)
        self.source_detail.setMinimumWidth(ui_px(170))
        self.source_detail.setMaximumWidth(ui_px(260))
        references_layout.addWidget(self.source_detail)
        chat_layout.addWidget(references_frame)

        self.activity_label = QLabel(
            "Jawaban tanpa sumber relevan akan ditandai dan tidak dibuat-buat."
        )
        self.activity_label.setObjectName("tanyaDoraActivity")
        self.activity_label.setWordWrap(True)
        chat_layout.addWidget(self.activity_label)

        input_row = QHBoxLayout()
        input_row.setSpacing(ui_px(10))
        self.message_input = QPlainTextEdit()
        self.message_input.setObjectName("tanyaDoraInput")
        self.message_input.setPlaceholderText("Tanyakan tentang data, statistik, atau riset…")
        self.message_input.setFixedHeight(ui_px(88))
        self.message_input.installEventFilter(self)
        self.message_input.setTabChangesFocus(True)
        input_row.addWidget(self.message_input, 1)
        send_buttons = QVBoxLayout()
        send_buttons.setSpacing(ui_px(7))
        self.send_button = QPushButton("Kirim")
        self.send_button.setObjectName("primaryButton")
        self.send_button.setMinimumWidth(ui_px(94))
        self.send_button.clicked.connect(self._send_message)
        send_buttons.addWidget(self.send_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("tanyaDoraStopButton")
        self.stop_button.setMinimumWidth(ui_px(94))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_generation)
        send_buttons.addWidget(self.stop_button)
        send_buttons.addStretch(1)
        input_row.addLayout(send_buttons)
        chat_layout.addLayout(input_row)
        workspace.addWidget(chat_panel)
        workspace.setStretchFactor(0, 0)
        workspace.setStretchFactor(1, 1)
        workspace.setSizes([ui_px(230), ui_px(900)])

        shortcut = QShortcut(QKeySequence("Ctrl+Return"), self.message_input)
        shortcut.activated.connect(self._send_message)
        shortcut_alt = QShortcut(QKeySequence("Ctrl+Enter"), self.message_input)
        shortcut_alt.activated.connect(self._send_message)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.message_input and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter) and event.modifiers() & Qt.ControlModifier:
                self._send_message()
                return True
        return super().eventFilter(watched, event)

    def _refresh_models(self) -> None:
        if self._model_thread is not None and self._model_thread.isRunning():
            return
        self.refresh_button.setEnabled(False)
        self.model_status.setText("Memeriksa Ollama dan model lokal…")
        self.model_status.setProperty("state", "pending")
        self._polish_status()
        thread = _ModelRefreshThread(self.llm_manager, self)
        thread.completed.connect(self._on_models_refreshed)
        thread.finished.connect(lambda thread=thread: self._model_refresh_finished(thread))
        self._model_thread = thread
        thread.start()

    def _on_models_refreshed(self, models: object, status: object) -> None:
        model_items = tuple(item for item in models if isinstance(item, LocalModel))
        self._current_models = model_items
        previous_chat = self.model_combo.currentData()
        active_runtime = self.model_version_store.get_active_runtime_model()
        active_model_name = str(active_runtime.get("model_name") or "") if active_runtime else ""
        previous_embedding = self.embedding_combo.currentData()
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
        chat_index = self.model_combo.findData(previous_chat)
        if chat_index < 0 and active_model_name:
            chat_index = self.model_combo.findData(active_model_name)
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
            self._set_model_status("Ollama tidak tersedia", "error")
            self.model_status.setToolTip(detail)
            self.send_button.setEnabled(False)
        elif not model_items:
            self._set_model_status("Ollama aktif, belum ada model lokal terpasang.", "warning")
            self.send_button.setEnabled(False)
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
                    f"Ollama aktif · {len(model_items)} model terpasang · {active_detail}", "warning"
                )
                self.send_button.setEnabled(False)
        self._update_action_buttons()

    def set_active_model(self, model_name: str) -> bool:
        """Apply an explicitly confirmed Studio model switch to the open chat page."""
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

    def _on_model_changed(self, _index: int) -> None:
        model_name = self.model_combo.currentData()
        if model_name:
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
                    f"Model aktif tersimpan {active_runtime.get('model_name')} tidak ada di Ollama lokal; pilih dan aktifkan model baru dari Studio.",
                    "warning",
                )
            else:
                self._set_model_status("Pilih model Ollama lokal sebelum chat.", "warning")
            self.send_button.setEnabled(False)

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

    def _activate_conversation(self, conversation_id: str) -> None:
        self._active_conversation_id = conversation_id
        self._last_response = None
        messages = self.conversation_store.get_messages(conversation_id)
        self._render_messages(messages)
        last_assistant = next((message for message in reversed(messages) if message.role == "assistant"), None)
        self._show_sources(last_assistant.sources if last_assistant else ())
        self._update_action_buttons()

    def _render_messages(self, messages: Sequence[ConversationMessage]) -> None:
        self.transcript.clear()
        if not messages:
            self.transcript.setPlainText(
                "Selamat datang di Tanya DORA.\n\n"
                "Pilih model Ollama lokal, lalu ajukan pertanyaan. Jawaban akan memakai sumber "
                "DKB yang ditemukan dan menampilkan referensinya di bawah percakapan.\n\n"
                "Dokumen percakapan disimpan di database lokal yang terpisah dari Knowledge Base."
            )
            return
        for index, message in enumerate(messages):
            if index:
                self.transcript.append("")
            if message.role == "user":
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

    def _on_history_selected(self, current: QListWidgetItem | None, _previous) -> None:
        if current is not None:
            self._activate_conversation(str(current.data(Qt.UserRole)))

    def _new_conversation(self) -> None:
        if self._is_generating():
            return
        conversation = self.conversation_store.create_conversation()
        self._active_conversation_id = conversation.conversation_id
        self._load_conversations(conversation.conversation_id)
        self.message_input.setFocus()

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
        self._active_conversation_id = None
        self._load_conversations()

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

    def _send_message(self, *, regenerate: bool = False) -> None:
        if self._is_generating():
            return
        model_name = str(self.model_combo.currentData() or "").strip()
        if not model_name:
            self.activity_label.setText("Pilih model Ollama lokal yang sudah terpasang.")
            return

        conversation_id = self._active_conversation_id
        if conversation_id is None:
            conversation = self.conversation_store.create_conversation()
            conversation_id = conversation.conversation_id
            self._active_conversation_id = conversation_id
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
        try:
            tool_context = (
                self.tool_context_provider()
                if callable(self.tool_context_provider)
                else None
            )
            tool_registry = build_dora_tool_registry(
                tool_context,
                self.knowledge_base,
                allow_online_research=self.online_research_checkbox.isChecked(),
            )
        except Exception as error:
            self.activity_label.setText(f"Konteks DORA tidak dapat dibaca: {error}")
            self.send_button.setEnabled(bool(self.model_combo.currentData()))
            return
        self._streaming_text = ""
        self._streaming_started = False
        self._last_response = None
        self._show_sources(())
        self.activity_label.setText("Mencari sumber lokal di DKB…")
        self.send_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.refresh_button.setEnabled(False)
        self._update_action_buttons()
        cancel_event = threading.Event()
        self._cancel_event = cancel_event
        embedding_model = self.embedding_combo.currentData()
        thread = _ChatThread(
            self.knowledge_base,
            self.llm_manager,
            question=question,
            model=model_name,
            embedding_model=str(embedding_model) if embedding_model else None,
            temperature=self.temperature_spin.value(),
            history=history,
            tool_registry=tool_registry,
            cancel_event=cancel_event,
            parent=self,
        )
        thread.delta.connect(self._on_stream_delta)
        thread.tool_activity.connect(self._on_tool_activity)
        thread.completed.connect(self._on_answer_completed)
        thread.failed.connect(self._on_answer_failed)
        thread.finished.connect(lambda thread=thread: self._chat_thread_finished(thread))
        self._chat_thread = thread
        thread.start()
        self._update_action_buttons()

    def _on_stream_delta(self, delta: str) -> None:
        if not self._streaming_started:
            self._streaming_started = True
            self.transcript.append("Tanya DORA · sedang menjawab")
        cursor = self.transcript.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(delta)
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()
        self._streaming_text += delta

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
        elif not response.grounded:
            self.activity_label.setText(
                "DKB tidak menemukan sumber relevan. Jawaban tidak diteruskan ke model."
            )
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
        self._load_conversations(conversation_id)
        self._finish_generation_state()

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
        self.send_button.setEnabled(bool(self.model_combo.currentData()))
        self.stop_button.setEnabled(False)
        self.refresh_button.setEnabled(True)
        self._update_action_buttons()

    def _stop_generation(self) -> None:
        if self._cancel_event is None:
            return
        self._cancel_event.set()
        self.llm_manager.cancel_generation()
        self.stop_button.setEnabled(False)
        self.activity_label.setText("Menghentikan generasi lokal…")

    def _regenerate(self) -> None:
        self._send_message(regenerate=True)

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
        self.history_list.setEnabled(not generating)
        self.copy_button.setEnabled(
            any(message.role == "assistant" for message in messages)
        )
        self.regenerate_button.setEnabled(
            bool(messages)
            and any(message.role == "user" for message in messages)
            and not self._is_generating()
            and bool(self.model_combo.currentData())
        )

    def _is_generating(self) -> bool:
        return self._chat_thread is not None and self._chat_thread.isRunning()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        narrow = logical_width(event.size().width()) < 900
        self.source_detail.setVisible(not narrow)
        if narrow:
            self.source_detail.setText("Pilih sumber untuk melihat kutipan.")

    def apply_scale(self, _scale=None) -> None:
        self.layout().setContentsMargins(ui_px(26), ui_px(18), ui_px(26), ui_px(24))

    def shutdown(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        self.llm_manager.cancel_generation()
        thread = self._chat_thread
        if thread is not None and thread.isRunning():
            thread.wait(2500)
        model_thread = self._model_thread
        if model_thread is not None and model_thread.isRunning():
            model_thread.wait(1500)
        if self._owns_manager:
            self.llm_manager.close()

