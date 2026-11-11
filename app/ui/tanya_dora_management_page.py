"""Working local management studio for Tanya DORA services."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ai.adapter_runtime import AdapterRuntimeError, AdapterRuntimeService
from app.ai.conversation_store import ConversationStore
from app.ai.evaluation_benchmark import (
    BenchmarkTarget,
    TDEBBenchmarkBuilder,
    TDEBBenchmarkRunner,
    TDEBBenchmarkStore,
)
from app.ai.fine_tuning import (
    FineTuningConfig,
    FineTuningEngine,
    FineTuningError,
    FineTuningPreflight,
    FineTuningResult,
    TrainingProgress,
    detect_training_hardware,
)
from app.ai.knowledge_base import (
    DoraKnowledgeBase,
    KnowledgeBaseError,
    KnowledgeDocumentMetadata,
)
from app.ai.llm_manager import LocalLLMError, OllamaLLMManager, OllamaStatus
from app.ai.model_registry import LocalModel
from app.ai.model_versioning import ModelVersionError, ModelVersionStore
from app.ai.rag_pipeline import RAGPipeline
from app.ai.tdid_dataset import (
    TDIDDatasetBuilder,
    TDIDError,
    TDIDVersionStore,
)
from app.ai.tool_calling import DoraToolRegistry, build_dora_tool_registry
from app.core.config import get_data_dir
from app.ui.scale import ui_px


class _StatusThread(QThread):
    completed = Signal(object, object, object)

    def __init__(self, manager: OllamaLLMManager, parent=None) -> None:
        super().__init__(parent)
        self.manager = manager

    def run(self) -> None:
        try:
            status = self.manager.get_status()
            running = self.manager.running_models() if status.available else ()
            self.completed.emit(self.manager.list_models(), status, running)
        except Exception as error:
            self.completed.emit((), error, ())


class _ModelTestThread(QThread):
    delta = Signal(str)
    completed = Signal(str, float)
    failed = Signal(str, float)

    def __init__(self, manager, model: str, question: str, cancel: threading.Event, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.model = model
        self.question = question
        self.cancel = cancel

    def run(self) -> None:
        started = time.perf_counter()
        try:
            answer = self.manager.chat_stream(
                [{"role": "user", "content": self.question}],
                model=self.model,
                temperature=0.2,
                on_delta=self.delta.emit,
                cancel_event=self.cancel,
            )
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}", time.perf_counter() - started)
        else:
            self.completed.emit(answer, time.perf_counter() - started)


class _PreflightThread(QThread):
    completed = Signal(object)

    def __init__(self, engine: FineTuningEngine, config: FineTuningConfig, parent=None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.config = config

    def run(self) -> None:
        try:
            self.completed.emit(self.engine.preflight(self.config))
        except Exception as error:
            self.completed.emit(error)


class _TrainingThread(QThread):
    progress = Signal(object)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, engine: FineTuningEngine, config: FineTuningConfig, cancel: threading.Event, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.config = config
        self.cancel = cancel

    def run(self) -> None:
        try:
            result = self.engine.run(
                self.config,
                cancel_event=self.cancel,
                progress_callback=self.progress.emit,
            )
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.completed.emit(result)


class _AdapterImportThread(QThread):
    completed = Signal(str, str)
    failed = Signal(str)

    def __init__(self, service, version_id: str, parent=None) -> None:
        super().__init__(parent)
        self.service = service
        self.version_id = version_id

    def run(self) -> None:
        try:
            tag, output_dir = self.service.merge_and_import_to_ollama(
                self.version_id,
                user_confirmed=True,
            )
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.completed.emit(tag, str(output_dir))


class _EvaluationThread(QThread):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, runner, version_id: str, targets, cancel, parent=None) -> None:
        super().__init__(parent)
        self.runner = runner
        self.version_id = version_id
        self.targets = targets
        self.cancel = cancel

    def run(self) -> None:
        runs = []
        try:
            for target in self.targets:
                if self.cancel.is_set():
                    break
                self.progress.emit(f"Menjalankan {target.mode}: {target.model_name}")
                runs.append(self.runner.run(self.version_id, target, cancel_event=self.cancel))
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.completed.emit(tuple(runs))


class _LossChart(QWidget):
    """Tiny loss plot that only draws values emitted by the real trainer."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(ui_px(130))
        self._values: list[tuple[int, float]] = []

    def set_values(self, values: Sequence[tuple[int, float]]) -> None:
        self._values = [(int(step), float(loss)) for step, loss in values if loss is not None]
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        area = self.rect().adjusted(ui_px(28), ui_px(12), -ui_px(12), -ui_px(22))
        painter.setPen(QPen(self.palette().mid().color(), ui_px(1)))
        painter.drawLine(area.bottomLeft(), area.bottomRight())
        painter.drawLine(area.bottomLeft(), area.topLeft())
        if len(self._values) < 2:
            painter.setPen(self.palette().text().color())
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "Loss akan tampil dari log training nyata")
            return
        losses = [value for _step, value in self._values]
        low, high = min(losses), max(losses)
        if abs(high - low) < 1e-9:
            high = low + 1.0
        first_step, last_step = self._values[0][0], self._values[-1][0]
        path = QPainterPath()
        for index, (step, loss) in enumerate(self._values):
            x_ratio = (step - first_step) / max(last_step - first_step, 1)
            y_ratio = (loss - low) / (high - low)
            point_x = area.left() + x_ratio * area.width()
            point_y = area.bottom() - y_ratio * area.height()
            path.moveTo(point_x, point_y) if index == 0 else path.lineTo(point_x, point_y)
        painter.setPen(QPen(self.palette().highlight().color(), ui_px(2)))
        painter.drawPath(path)
        painter.setPen(self.palette().text().color())
        painter.drawText(0, self.height() - ui_px(4), f"train loss: {losses[-1]:.4f}")


def _button(text: str, handler, *, primary: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("primaryButton" if primary else "secondaryButton")
    button.clicked.connect(handler)
    return button


def _table(columns: Sequence[str], *, row_height: int = 34) -> QTableWidget:
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(list(columns))
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(ui_px(row_height))
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def _set_cell(table: QTableWidget, row: int, column: int, text: Any, data: Any = None) -> QTableWidgetItem:
    item = QTableWidgetItem(str(text if text is not None else ""))
    if data is not None:
        item.setData(Qt.ItemDataRole.UserRole, data)
    table.setItem(row, column, item)
    return item


class KnowledgeBaseManagerTab(QWidget):
    CATEGORIES = tuple(sorted(DoraKnowledgeBase.CATEGORIES))

    def __init__(self, knowledge_base: DoraKnowledgeBase, parent=None) -> None:
        super().__init__(parent)
        self.knowledge_base = knowledge_base
        self._documents = ()
        root = QVBoxLayout(self)
        root.setContentsMargins(ui_px(16), ui_px(14), ui_px(16), ui_px(16))
        toolbar = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Cari judul atau isi dokumen")
        self.search_input.returnPressed.connect(self.refresh)
        toolbar.addWidget(self.search_input, 1)
        self.category_combo = QComboBox()
        self.category_combo.addItem("Semua kategori", "")
        for category in self.CATEGORIES:
            self.category_combo.addItem(category.replace("_", " ").title(), category)
        self.category_combo.currentIndexChanged.connect(self.refresh)
        toolbar.addWidget(self.category_combo)
        toolbar.addWidget(_button("Cari / Perbarui", self.refresh))
        toolbar.addWidget(_button("Impor dokumen", self.import_documents, primary=True))
        root.addLayout(toolbar)
        split = QSplitter(Qt.Orientation.Vertical)
        self.table = _table(("Judul", "Kategori", "Status", "Chunks", "Diverifikasi", "Sumber"))
        self.table.itemSelectionChanged.connect(self.show_preview)
        split.addWidget(self.table)
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        actions = QHBoxLayout()
        self.edit_category = QComboBox()
        for category in self.CATEGORIES:
            self.edit_category.addItem(category.replace("_", " ").title(), category)
        self.verified = QCheckBox("Dokumen sudah diverifikasi")
        actions.addWidget(QLabel("Kategori dokumen"))
        actions.addWidget(self.edit_category)
        actions.addWidget(self.verified)
        actions.addWidget(_button("Simpan metadata", self.save_metadata))
        actions.addWidget(_button("Re-index", self.reindex))
        actions.addWidget(_button("Hapus", self.delete_document))
        preview_layout.addLayout(actions)
        metadata_form = QGridLayout()
        self.metadata_title = QLineEdit(); self.metadata_author = QLineEdit()
        self.metadata_year = QLineEdit(); self.metadata_doi = QLineEdit()
        self.metadata_url = QLineEdit(); self.metadata_license = QLineEdit()
        metadata_form.addWidget(QLabel("Judul"), 0, 0); metadata_form.addWidget(self.metadata_title, 0, 1)
        metadata_form.addWidget(QLabel("Penulis/organisasi"), 0, 2); metadata_form.addWidget(self.metadata_author, 0, 3)
        metadata_form.addWidget(QLabel("Tahun"), 1, 0); metadata_form.addWidget(self.metadata_year, 1, 1)
        metadata_form.addWidget(QLabel("DOI"), 1, 2); metadata_form.addWidget(self.metadata_doi, 1, 3)
        metadata_form.addWidget(QLabel("URL sumber"), 2, 0); metadata_form.addWidget(self.metadata_url, 2, 1)
        metadata_form.addWidget(QLabel("Lisensi"), 2, 2); metadata_form.addWidget(self.metadata_license, 2, 3)
        preview_layout.addLayout(metadata_form)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Pilih dokumen untuk melihat metadata dan cuplikan sumber.")
        preview_layout.addWidget(self.preview, 1)
        split.addWidget(preview_panel)
        split.setSizes([ui_px(360), ui_px(260)])
        root.addWidget(split, 1)
        self.refresh()

    def refresh(self, *_args) -> None:
        try:
            category = str(self.category_combo.currentData() or "")
            documents = self.knowledge_base.list_documents(category=category or None)
            query = self.search_input.text().strip()
            if query:
                matching = {item.document_id for item in self.knowledge_base.search(query, category=category or None, limit=100)}
                documents = tuple(item for item in documents if item.document_id in matching or query.casefold() in item.title.casefold())
        except Exception as error:
            QMessageBox.warning(self, "Knowledge Base", str(error))
            return
        self._documents = documents
        self.table.setRowCount(len(documents))
        for row, document in enumerate(documents):
            _set_cell(self.table, row, 0, document.title, document.document_id)
            _set_cell(self.table, row, 1, document.category)
            _set_cell(self.table, row, 2, f"{document.status}: {document.status_detail}".strip())
            _set_cell(self.table, row, 3, document.chunk_count)
            _set_cell(self.table, row, 4, document.verification_status)
            _set_cell(self.table, row, 5, document.source_name)
        if documents and not self.table.currentRow() >= 0:
            self.table.selectRow(0)

    def import_documents(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Impor dokumen ke Knowledge Base", "",
            "Dokumen (*.pdf *.docx *.txt *.md *.rst *.html *.htm)",
        )
        if not paths:
            return
        category = str(self.category_combo.currentData() or "dora_documentation")
        successes, failures = [], []
        for path_text in paths:
            path = Path(path_text)
            metadata = KnowledgeDocumentMetadata(title=path.stem, category=category, verified=False)
            try:
                document = self.knowledge_base.import_document(path, metadata)
            except Exception as error:
                failures.append(f"{path.name}: {error}")
            else:
                successes.append(f"{document.title} ({document.chunk_count} chunks)")
        self.refresh()
        summary = "Dokumen terindeks:\n" + "\n".join(successes) if successes else "Tidak ada dokumen yang berhasil diindeks."
        if failures:
            summary += "\n\nGagal:\n" + "\n".join(failures)
        QMessageBox.information(self, "Hasil impor", summary)

    def _selected_id(self) -> str | None:
        row = self.table.currentRow()
        if row < 0 or self.table.item(row, 0) is None:
            return None
        return str(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) or "") or None

    def show_preview(self) -> None:
        document_id = self._selected_id()
        if not document_id:
            self.preview.clear()
            return
        document = self.knowledge_base.get_document(document_id)
        if document is None:
            self.preview.setPlainText("Dokumen sudah tidak tersedia.")
            return
        index = self.edit_category.findData(document.category)
        if index >= 0:
            self.edit_category.setCurrentIndex(index)
        self.verified.setChecked(document.verification_status == "VERIFIED")
        self.metadata_title.setText(document.title)
        self.metadata_author.setText(document.author_or_org)
        self.metadata_year.setText(str(document.publication_year or ""))
        self.metadata_doi.setText(document.doi)
        self.metadata_url.setText(document.source_url)
        self.metadata_license.setText(document.license)
        chunks = self.knowledge_base.get_chunks(document_id)
        excerpt = "\n\n---\n\n".join(
            f"[{chunk.page_or_section or f'Chunk {chunk.chunk_index + 1}'}]\n{chunk.text[:1600]}"
            for chunk in chunks[:3]
        )
        self.preview.setPlainText(
            f"Judul: {document.title}\nSumber: {document.source_name}\nKategori: {document.category}\n"
            f"Ukuran: {document.chunk_count} chunks\nStatus: {document.status} {document.status_detail}\n"
            f"Lisensi: {document.license or 'Belum dicatat'}\nDOI: {document.doi or '—'}\n\n{excerpt}"
        )

    def save_metadata(self) -> None:
        document_id = self._selected_id()
        document = self.knowledge_base.get_document(document_id) if document_id else None
        if not document:
            return
        try:
            publication_year = int(self.metadata_year.text()) if self.metadata_year.text().strip() else None
        except ValueError:
            QMessageBox.warning(self, "Metadata", "Tahun publikasi harus berupa angka.")
            return
        metadata = KnowledgeDocumentMetadata(
            title=self.metadata_title.text().strip(),
            author_or_org=self.metadata_author.text().strip(),
            category=str(self.edit_category.currentData()),
            publication_year=publication_year,
            source_url=self.metadata_url.text().strip(),
            doi=self.metadata_doi.text().strip(),
            license=self.metadata_license.text().strip(),
            version=document.version,
            verified=self.verified.isChecked(),
        )
        try:
            self.knowledge_base.update_metadata(document_id, metadata)
        except Exception as error:
            QMessageBox.warning(self, "Metadata", str(error))
        self.refresh()

    def reindex(self) -> None:
        document_id = self._selected_id()
        if not document_id:
            return
        try:
            document = self.knowledge_base.reindex_document(document_id)
        except Exception as error:
            QMessageBox.warning(self, "Re-index", str(error))
        else:
            QMessageBox.information(self, "Re-index", f"{document.title}: {document.chunk_count} chunks")
        self.refresh()

    def delete_document(self) -> None:
        document_id = self._selected_id()
        if not document_id:
            return
        answer = QMessageBox.question(self, "Hapus dokumen", "Hapus dokumen, sumber lokal, chunks dan embeddings?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.knowledge_base.delete_document(document_id)
        except Exception as error:
            QMessageBox.warning(self, "Knowledge Base", str(error))
        self.preview.clear()
        self.refresh()


class _RecordEditor(QDialog):
    def __init__(self, record: Mapping[str, Any] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit record TDID" if record else "Tambah record TDID")
        self.resize(ui_px(760), ui_px(560))
        layout = QVBoxLayout(self)
        help_text = QLabel("Edit satu record JSON lengkap. Record baru berstatus belum diverifikasi sampai sumber diperiksa manusia.")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.editor = QPlainTextEdit()
        self.editor.setObjectName("tanyaDoraStudioCode")
        default_record = {
            "messages": [
                {"role": "user", "content": "Pertanyaan yang sudah ditinjau manusia"},
                {"role": "assistant", "content": "Jawaban yang sudah ditinjau dengan sumber yang dapat diperiksa"},
            ],
            "metadata": {
                "category": "statistics", "difficulty": "beginner", "instruction_type": "knowledge",
                "verified": False, "source": "", "source_url": "", "source_group": "",
            },
        }
        self.editor.setPlainText(json.dumps(dict(record) if record else default_record, ensure_ascii=False, indent=2))
        layout.addWidget(self.editor, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def record(self) -> dict[str, Any]:
        value = json.loads(self.editor.toPlainText())
        if not isinstance(value, Mapping):
            raise ValueError("TDID record harus berupa objek JSON.")
        return dict(value)


class DatasetManagerTab(QWidget):
    def __init__(self, store: TDIDVersionStore, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self.builder = TDIDDatasetBuilder()
        root = QVBoxLayout(self)
        root.setContentsMargins(ui_px(16), ui_px(14), ui_px(16), ui_px(16))
        actions = QHBoxLayout()
        for label, handler, primary in (
            ("Impor JSONL/CSV", self.import_dataset, False),
            ("Tambah", self.add_record, False), ("Edit", self.edit_record, False),
            ("Hapus", self.delete_record, False), ("Simpan draft", self.save_draft, False),
            ("Simpan final", self.save_final, True), ("Ekspor JSONL", self.export_dataset, False),
        ):
            actions.addWidget(_button(label, handler, primary=primary))
        root.addLayout(actions)
        self.table = _table(("Record ID", "Prompt", "Jawaban", "Kategori", "Kesulitan", "Verifikasi"))
        root.addWidget(self.table, 1)
        self.summary = QLabel("Belum ada dataset. JSONL/CSV dapat diimpor dari file lokal.")
        self.summary.setObjectName("tanyaDoraStudioHint")
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)
        self.versions = QListWidget()
        self.versions.setMaximumHeight(ui_px(84))
        root.addWidget(self.versions)
        self.refresh_versions()

    def import_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Impor dataset instruksi", "", "Dataset (*.jsonl *.csv)")
        if not path:
            return
        try:
            imported = TDIDDatasetBuilder.from_jsonl(path) if path.casefold().endswith(".jsonl") else TDIDDatasetBuilder.from_csv(path)
        except Exception as error:
            QMessageBox.warning(self, "Dataset TDID", str(error))
            return
        self.builder = imported
        self.refresh()

    def refresh(self) -> None:
        records = self.builder.records
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            messages = record.get("messages", ())
            prompt = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            answer = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"), "")
            metadata = record.get("metadata", {})
            _set_cell(self.table, row, 0, record.get("record_id", ""), row)
            _set_cell(self.table, row, 1, str(prompt)[:180])
            _set_cell(self.table, row, 2, str(answer)[:180])
            _set_cell(self.table, row, 3, metadata.get("category", ""))
            _set_cell(self.table, row, 4, metadata.get("difficulty", ""))
            _set_cell(self.table, row, 5, "Terverifikasi" if metadata.get("verified") is True else "Belum diverifikasi")
        issues = self.builder.validate()
        duplicates = self.builder.duplicate_groups()
        repeated = self.builder.near_duplicate_prompt_groups()
        self.summary.setText(
            f"{len(records)} records · {len(issues)} masalah validasi · {len(duplicates)} duplikat exact · "
            f"{len(repeated)} kelompok prompt berulang · kategori {self.builder.category_counts()} · "
            f"kesulitan {self.builder.difficulty_counts()} · verifikasi {self.builder.verification_counts()}. "
            "Record yang belum diverifikasi akan dilewati saat split dan tidak dapat masuk versi final."
        )

    def _selected_row(self) -> int | None:
        row = self.table.currentRow()
        return int(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)) if row >= 0 else None

    def add_record(self) -> None:
        dialog = _RecordEditor(parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.builder.add_record(dialog.record())
        except Exception as error:
            QMessageBox.warning(self, "Record TDID", str(error))
        self.refresh()

    def edit_record(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        dialog = _RecordEditor(self.builder.records[row], self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.builder.edit_record(row, dialog.record())
        except Exception as error:
            QMessageBox.warning(self, "Record TDID", str(error))
        self.refresh()

    def delete_record(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        self.builder.delete_record(row)
        self.refresh()

    def _save_version(self, status: str) -> None:
        if not self.builder.records:
            QMessageBox.information(self, "Dataset TDID", "Impor atau tambah record terlebih dahulu.")
            return
        name, accepted = QInputDialog.getText(self, "Versi dataset", "Nama versi:")
        if not accepted or not name.strip():
            return
        if status == "final" and self.builder.verification_counts().get("unverified", 0):
            QMessageBox.warning(self, "Dataset belum terverifikasi", "Semua record harus diverifikasi sebelum versi final disimpan.")
            return
        try:
            split = self.builder.split()
            version = self.store.save_version(name, self.builder, status=status, split=split)
        except Exception as error:
            QMessageBox.warning(self, "Versi dataset", str(error))
            return
        QMessageBox.information(
            self,
            "Versi tersimpan",
            f"{version.name}\nTrain: {len(split.train)} · validation: {len(split.validation)} · test: {len(split.test)}\n"
            f"Dilewati (belum diverifikasi): {len(split.excluded_unverified)} · duplikat: {len(split.excluded_duplicates)}",
        )
        self.refresh_versions()

    def save_draft(self) -> None:
        self._save_version("draft")

    def save_final(self) -> None:
        self._save_version("final")

    def export_dataset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Ekspor dataset JSONL", "tdid.jsonl", "JSONL (*.jsonl)")
        if not path:
            return
        try:
            self.builder.export_jsonl(path)
        except Exception as error:
            QMessageBox.warning(self, "Ekspor dataset", str(error))

    def refresh_versions(self) -> None:
        self.versions.clear()
        try:
            versions = self.store.list_versions()
        except Exception as error:
            self.versions.addItem(f"Gagal memuat versi: {error}")
            return
        for version in versions:
            self.versions.addItem(
                f"{version.name} · {version.status} · {version.record_count} records · "
                f"{version.verified_count} verified · split {bool(version.split_manifest)}"
            )


class FineTuningStudioTab(QWidget):
    def __init__(self, store: TDIDVersionStore, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self.engine = FineTuningEngine(store)
        self._preflight_thread: _PreflightThread | None = None
        self._training_thread: _TrainingThread | None = None
        self._cancel = threading.Event()
        self._loss_values: list[tuple[int, float]] = []
        self._last_config: FineTuningConfig | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(ui_px(16), ui_px(14), ui_px(16), ui_px(16))
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        form = QFormLayout(left)
        self.base_path = QLineEdit()
        base_row = QHBoxLayout()
        base_row.addWidget(self.base_path, 1)
        base_row.addWidget(_button("Pilih", self.choose_base))
        form.addRow("Base model lokal", base_row)
        self.dataset_combo = QComboBox()
        form.addRow("Versi TDID", self.dataset_combo)
        self.output_path = QLineEdit(str(get_data_dir() / "tanya_dora" / "training_runs"))
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_path, 1)
        out_row.addWidget(_button("Pilih", self.choose_output))
        form.addRow("Folder output", out_row)
        self.epochs = QDoubleSpinBox(); self.epochs.setRange(0.1, 100); self.epochs.setValue(1.0); self.epochs.setSingleStep(0.5)
        form.addRow("Epoch", self.epochs)
        self.learning_rate = QDoubleSpinBox(); self.learning_rate.setRange(0.000001, 1); self.learning_rate.setDecimals(6); self.learning_rate.setValue(0.0002)
        form.addRow("Learning rate", self.learning_rate)
        self.batch_size = QSpinBox(); self.batch_size.setRange(1, 128); self.batch_size.setValue(1)
        form.addRow("Batch / device", self.batch_size)
        self.grad_accum = QSpinBox(); self.grad_accum.setRange(1, 256); self.grad_accum.setValue(8)
        form.addRow("Gradient accumulation", self.grad_accum)
        self.max_length = QSpinBox(); self.max_length.setRange(64, 32768); self.max_length.setSingleStep(64); self.max_length.setValue(1024)
        form.addRow("Maximum sequence length", self.max_length)
        self.quantization = QComboBox(); self.quantization.addItems(["none", "4bit"])
        form.addRow("Quantization", self.quantization)
        self.rank = QSpinBox(); self.rank.setRange(1, 256); self.rank.setValue(8)
        form.addRow("LoRA rank", self.rank)
        self.preflight_button = _button("Periksa hardware & dependencies", self.check_preflight)
        self.start_button = _button("Mulai training", self.start_training, primary=True)
        self.start_button.setEnabled(False)
        self.stop_button = _button("Stop training", self.stop_training)
        self.stop_button.setEnabled(False)
        self.export_button = _button("Ekspor workspace untuk perangkat lain", self.export_workspace)
        action_row = QHBoxLayout(); action_row.addWidget(self.preflight_button); action_row.addWidget(self.start_button); action_row.addWidget(self.stop_button)
        form.addRow(action_row)
        form.addRow(self.export_button)
        splitter.addWidget(left)
        right = QWidget(); right_layout = QVBoxLayout(right)
        self.readiness = QLabel("Preflight belum dijalankan. Pilih model dan dataset tersimpan.")
        self.readiness.setObjectName("tanyaDoraStudioStatus"); self.readiness.setWordWrap(True)
        right_layout.addWidget(self.readiness)
        self.progress = QProgressBar(); self.progress.setRange(0, 0); self.progress.setValue(0); self.progress.setVisible(False)
        right_layout.addWidget(self.progress)
        self.progress_label = QLabel("Belum ada training")
        right_layout.addWidget(self.progress_label)
        self.loss_chart = _LossChart()
        right_layout.addWidget(self.loss_chart)
        self.logs = QPlainTextEdit(); self.logs.setReadOnly(True); self.logs.setPlaceholderText("Log preflight, langkah, loss dan checkpoint nyata tampil di sini.")
        right_layout.addWidget(self.logs, 1)
        splitter.addWidget(right); splitter.setSizes([ui_px(590), ui_px(660)])
        root.addWidget(splitter, 1)
        self.refresh_versions()

    def refresh_versions(self) -> None:
        self.dataset_combo.clear()
        try:
            versions = self.store.list_versions()
        except Exception as error:
            self.readiness.setText(f"Gagal membaca dataset version: {error}")
            return
        for version in versions:
            self.dataset_combo.addItem(
                f"{version.name} · {version.status} · {version.verified_count}/{version.record_count} verified",
                version.version_id,
            )

    def choose_base(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pilih direktori base model lokal")
        if path: self.base_path.setText(path)

    def choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pilih direktori training output")
        if path: self.output_path.setText(path)

    def _config(self) -> FineTuningConfig:
        return FineTuningConfig(
            base_model_path=self.base_path.text().strip(),
            dataset_version_id=str(self.dataset_combo.currentData() or ""),
            output_dir=self.output_path.text().strip(),
            epochs=self.epochs.value(),
            learning_rate=self.learning_rate.value(),
            per_device_batch_size=self.batch_size.value(),
            gradient_accumulation_steps=self.grad_accum.value(),
            max_sequence_length=self.max_length.value(),
            lora_rank=self.rank.value(),
            quantization=self.quantization.currentText(),
        )

    def check_preflight(self) -> None:
        if self._preflight_thread and self._preflight_thread.isRunning(): return
        self.preflight_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.readiness.setText("Memeriksa model lokal, split dataset, dependency, hardware dan storage…")
        thread = _PreflightThread(self.engine, self._config(), self)
        thread.completed.connect(self._on_preflight)
        thread.finished.connect(lambda: self.preflight_button.setEnabled(True))
        self._preflight_thread = thread
        thread.start()

    def _on_preflight(self, result: object) -> None:
        self._last_config = self._config()
        if isinstance(result, Exception):
            self.readiness.setText(f"Preflight gagal: {result}")
            self.start_button.setEnabled(False)
            return
        preflight = result
        if not isinstance(preflight, FineTuningPreflight):
            self.readiness.setText("Preflight mengembalikan hasil yang tidak dikenal.")
            self.start_button.setEnabled(False)
            return
        hardware = preflight.hardware
        lines = [
            f"Status: {'SIAP' if preflight.ready else 'TERHALANG'}",
            f"CPU {hardware.cpu_count} · RAM tersedia { _bytes(preflight.hardware.available_ram_bytes) } · "
            f"VRAM tersedia { _bytes(preflight.hardware.vram_available_bytes) }",
            f"Estimasi memori training: {_bytes(preflight.estimated_memory_bytes)} · disk bebas: {_bytes(preflight.available_disk_bytes)}",
        ]
        lines += [f"Blocker: {value}" for value in preflight.blockers]
        lines += [f"Peringatan: {value}" for value in preflight.warnings]
        self.readiness.setText("\n".join(lines))
        self.logs.appendPlainText("\n".join(lines))
        self.start_button.setEnabled(preflight.ready)

    def start_training(self) -> None:
        config = self._config()
        if self._last_config != config:
            self.check_preflight()
            return
        self._cancel = threading.Event()
        self._loss_values = []
        self.loss_chart.set_values(())
        self.logs.appendPlainText("Memulai FineTuningEngine.run(); progress berasal dari callback trainer.")
        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.preflight_button.setEnabled(False); self.progress.setVisible(True); self.progress.setRange(0, 0)
        thread = _TrainingThread(self.engine, config, self._cancel, self)
        thread.progress.connect(self._on_training_progress)
        thread.completed.connect(self._on_training_completed)
        thread.failed.connect(self._on_training_failed)
        thread.finished.connect(self._training_finished)
        self._training_thread = thread
        thread.start()

    def _on_training_progress(self, progress: object) -> None:
        if not isinstance(progress, TrainingProgress): return
        if progress.total_steps:
            self.progress.setRange(0, progress.total_steps); self.progress.setValue(min(progress.step, progress.total_steps))
        self.progress_label.setText(
            f"{progress.status} · langkah {progress.step}/{progress.total_steps or '?'} · "
            f"epoch {progress.epoch or 0:.2f} · elapsed {progress.elapsed_seconds:.0f}s"
        )
        self.logs.appendPlainText(
            f"step={progress.step} loss={progress.train_loss} eval_loss={progress.validation_loss} "
            f"lr={progress.learning_rate} checkpoint={progress.checkpoint_dir}"
        )
        if progress.train_loss is not None:
            self._loss_values.append((progress.step, progress.train_loss))
            self.loss_chart.set_values(self._loss_values)

    def stop_training(self) -> None:
        self._cancel.set()
        self.stop_button.setEnabled(False)
        self.logs.appendPlainText("Stop diminta; trainer akan berhenti pada batas langkah berikutnya.")

    def _on_training_completed(self, result: object) -> None:
        if isinstance(result, FineTuningResult):
            self.logs.appendPlainText(f"Training {result.status}: {result.job_dir}\nManifest: {result.manifest_path}")
            self.progress_label.setText(f"Training {result.status} · {result.metrics}")

    def _on_training_failed(self, detail: str) -> None:
        self.logs.appendPlainText(f"Training gagal: {detail}")
        self.progress_label.setText("Training gagal; periksa blocker dan manifest job.")

    def _training_finished(self) -> None:
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.preflight_button.setEnabled(True)
        self.progress.setVisible(False)
        self._training_thread = None
        self._last_config = None

    def export_workspace(self) -> None:
        version_id = str(self.dataset_combo.currentData() or "")
        if not version_id:
            QMessageBox.information(self, "Training workspace", "Simpan versi dataset dengan split terlebih dahulu.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Ekspor workspace training", "tanya-dora-training.zip", "ZIP (*.zip)")
        if not path: return
        try:
            output = self.engine.export_workspace(version_id, path, config=self._config())
        except Exception as error:
            QMessageBox.warning(self, "Training workspace", str(error))
        else:
            QMessageBox.information(self, "Training workspace", f"Workspace lokal siap: {output}")


def _bytes(value: int | None) -> str:
    if value is None: return "tidak tersedia"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB": return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


class ModelManagerTab(QWidget):
    model_activated = Signal(str)

    def __init__(self, manager: OllamaLLMManager, version_store: ModelVersionStore, parent=None) -> None:
        super().__init__(parent)
        self.manager = manager
        self.version_store = version_store
        self.adapter_service = AdapterRuntimeService(version_store)
        self._models: tuple[LocalModel, ...] = ()
        self._status_thread: _StatusThread | None = None
        self._test_thread: _ModelTestThread | None = None
        self._adapter_thread: _AdapterImportThread | None = None
        self._test_cancel = threading.Event()
        root = QVBoxLayout(self); root.setContentsMargins(ui_px(16), ui_px(14), ui_px(16), ui_px(16))
        self.status = QLabel("Memeriksa Ollama lokal…"); self.status.setObjectName("tanyaDoraStudioStatus"); self.status.setWordWrap(True)
        root.addWidget(self.status)
        refresh_row = QHBoxLayout(); refresh_row.addWidget(_button("Periksa koneksi & model", self.refresh_models)); refresh_row.addStretch(1)
        self.active_label = QLabel("Model produksi: belum dipilih"); self.active_label.setObjectName("tanyaDoraStudioHint"); refresh_row.addWidget(self.active_label)
        root.addLayout(refresh_row)
        split = QSplitter(Qt.Orientation.Vertical)
        self.model_table = _table(("Model lokal", "Family", "Parameter", "Quantization", "Ukuran", "Sedang berjalan"))
        split.addWidget(self.model_table)
        test_box = QFrame(); test_box.setObjectName("tanyaDoraStudioPanel")
        test_layout = QVBoxLayout(test_box)
        title = QLabel("Panel uji inference lokal"); title.setObjectName("tanyaDoraStudioHeading"); test_layout.addWidget(title)
        control = QHBoxLayout(); self.test_model = QComboBox(); self.test_model.setMinimumWidth(ui_px(230)); control.addWidget(self.test_model)
        self.test_input = QLineEdit("Jelaskan secara singkat kapan median lebih tepat digunakan daripada mean.")
        self.test_input.returnPressed.connect(self.run_model_test); control.addWidget(self.test_input, 1)
        self.test_button = _button("Uji model", self.run_model_test, primary=True); control.addWidget(self.test_button)
        self.stop_test_button = _button("Stop", self.stop_model_test); self.stop_test_button.setEnabled(False); control.addWidget(self.stop_test_button)
        test_layout.addLayout(control)
        self.test_output = QPlainTextEdit(); self.test_output.setReadOnly(True); self.test_output.setMinimumHeight(ui_px(110)); test_layout.addWidget(self.test_output)
        self.test_timing = QLabel("Durasi akan dicatat ketika inference dijalankan."); self.test_timing.setObjectName("tanyaDoraStudioHint"); test_layout.addWidget(self.test_timing)
        model_actions = QHBoxLayout(); model_actions.addWidget(_button("Jadikan model chat aktif…", self.activate_base_model)); model_actions.addStretch(1)
        test_layout.addLayout(model_actions)
        split.addWidget(test_box); split.setSizes([ui_px(230), ui_px(300)])
        root.addWidget(split, 2)
        adapter_header = QHBoxLayout(); adapter_header.addWidget(QLabel("Adapter registry"), 1)
        adapter_header.addWidget(_button("Impor adapter ZIP", self.import_adapter))
        adapter_header.addWidget(_button("Periksa kompatibilitas", self.inspect_adapter))
        adapter_header.addWidget(_button("Ekspor", self.export_adapter))
        adapter_header.addWidget(_button("Konversi & aktifkan…", self.activate_adapter, primary=True))
        adapter_header.addWidget(_button("Nonaktifkan adapter", self.deactivate_adapter))
        adapter_header.addWidget(_button("Rollback model", self.rollback_model))
        root.addLayout(adapter_header)
        self.adapter_table = _table(("Nama versi", "Base model", "Format", "Training", "Dibuat", "Aktif"))
        self.adapter_table.setMaximumHeight(ui_px(170)); root.addWidget(self.adapter_table)
        self.adapter_detail = QLabel("Adapter Safetensors tidak langsung dimuat ke Ollama. Konversi akan merge LoRA ke model lokal, lalu meminta Ollama mengimpor full model.")
        self.adapter_detail.setObjectName("tanyaDoraStudioHint"); self.adapter_detail.setWordWrap(True); root.addWidget(self.adapter_detail)
        self.refresh_adapters()
        self.refresh_models()

    def refresh_models(self) -> None:
        if self._status_thread and self._status_thread.isRunning(): return
        self.status.setText("Memeriksa koneksi, inventaris model dan model yang sedang berjalan…")
        thread = _StatusThread(self.manager, self)
        thread.completed.connect(self._models_refreshed)
        thread.finished.connect(lambda: setattr(self, "_status_thread", None))
        self._status_thread = thread; thread.start()

    def _models_refreshed(self, models: object, status: object, running_models: object) -> None:
        self._models = tuple(models) if isinstance(models, Sequence) else ()
        self.model_table.setRowCount(len(self._models))
        self.test_model.clear(); active = self.version_store.get_active_runtime_model()
        active_name = str(active.get("model_name")) if active else ""
        running = set(running_models) if isinstance(running_models, Sequence) else set()
        for row, model in enumerate(self._models):
            if not isinstance(model, LocalModel): continue
            _set_cell(self.model_table, row, 0, model.name)
            _set_cell(self.model_table, row, 1, model.family or "—")
            _set_cell(self.model_table, row, 2, model.parameter_size or "—")
            _set_cell(self.model_table, row, 3, model.quantization_level or "—")
            _set_cell(self.model_table, row, 4, model.size_label)
            _set_cell(self.model_table, row, 5, "Ya" if model.name in running else "Tidak")
            self.test_model.addItem(model.name, model.name)
        status_text = status.detail if isinstance(status, OllamaStatus) else str(status)
        if isinstance(status, OllamaStatus) and status.available:
            self.status.setText(f"Tersambung · {len(self._models)} model lokal · {status_text}")
        else:
            self.status.setText(
                "Ollama tidak tersedia. Instal Ollama dan mulai server lokal, lalu pasang model pilihan dari Ollama Library. "
                "Tidak ada unduhan dimulai dari Tanya DORA. Ukuran unduhan bergantung pada tag dan kuantisasi; periksa ukuran tag serta ruang disk sebelum memasang."
            )
        self.active_label.setText(f"Model produksi: {active_name or 'belum dipilih'}")
        if active_name:
            index = self.test_model.findData(active_name)
            if index >= 0: self.test_model.setCurrentIndex(index)

    def run_model_test(self) -> None:
        model = str(self.test_model.currentData() or "")
        question = self.test_input.text().strip()
        if not model or not question:
            self.test_timing.setText("Pilih model lokal yang tersedia dan masukkan pertanyaan.")
            return
        if self._test_thread and self._test_thread.isRunning(): return
        self._test_cancel = threading.Event(); self.test_output.clear()
        self.test_timing.setText(f"Mengirim inference ke {model}…")
        self.test_button.setEnabled(False); self.stop_test_button.setEnabled(True)
        thread = _ModelTestThread(self.manager, model, question, self._test_cancel, self)
        thread.delta.connect(self.test_output.insertPlainText)
        thread.completed.connect(self._test_complete)
        thread.failed.connect(self._test_failed)
        thread.finished.connect(self._test_finished)
        self._test_thread = thread; thread.start()

    def _test_complete(self, _answer: str, elapsed: float) -> None:
        self.test_timing.setText(f"Inference selesai · {elapsed:.2f} detik")

    def _test_failed(self, detail: str, elapsed: float) -> None:
        self.test_timing.setText(f"Inference gagal setelah {elapsed:.2f} detik · {detail}")

    def _test_finished(self) -> None:
        self.test_button.setEnabled(True); self.stop_test_button.setEnabled(False); self._test_thread = None

    def stop_model_test(self) -> None:
        self._test_cancel.set(); self.manager.cancel_generation(); self.stop_test_button.setEnabled(False)

    def activate_base_model(self) -> None:
        model = str(self.test_model.currentData() or "")
        if not model: return
        answer = QMessageBox.question(
            self, "Ganti model chat aktif?",
            f"Jadikan {model} sebagai model chat aktif Tanya DORA?\n\nPergantian dicatat dan dapat di-rollback.",
        )
        if answer != QMessageBox.StandardButton.Yes: return
        try:
            self.version_store.set_active_runtime_model(model, user_confirmed=True)
            self.manager.select_model(model)
        except Exception as error:
            QMessageBox.warning(self, "Model manager", str(error)); return
        self.active_label.setText(f"Model produksi: {model}")
        self.model_activated.emit(model)

    def refresh_adapters(self) -> None:
        try: versions = self.version_store.list_versions()
        except Exception as error:
            self.adapter_detail.setText(f"Registry adapter gagal dibaca: {error}"); return
        self.adapter_table.setRowCount(len(versions))
        for row, version in enumerate(versions):
            config = version.training_config
            fmt = "Safetensors" if (self.version_store._resolve_adapter_path(version.adapter_path) / "adapter_model.safetensors").is_file() else "PyTorch bin"
            _set_cell(self.adapter_table, row, 0, version.name, version.version_id)
            _set_cell(self.adapter_table, row, 1, Path(version.base_model_path).name)
            _set_cell(self.adapter_table, row, 2, fmt)
            _set_cell(self.adapter_table, row, 3, str(config.get("quantization", "none")))
            _set_cell(self.adapter_table, row, 4, version.created_at)
            _set_cell(self.adapter_table, row, 5, "Ya" if version.selected else "Tidak")

    def _selected_adapter(self) -> str | None:
        row = self.adapter_table.currentRow()
        return str(self.adapter_table.item(row, 0).data(Qt.ItemDataRole.UserRole)) if row >= 0 else None

    def import_adapter(self) -> None:
        archive, _ = QFileDialog.getOpenFileName(self, "Impor adapter Tanya DORA", "", "ZIP (*.zip)")
        if not archive: return
        base = QFileDialog.getExistingDirectory(self, "Pilih base model lokal adapter")
        if not base: return
        name, accepted = QInputDialog.getText(self, "Impor adapter", "Nama versi:")
        if not accepted: return
        try:
            version = self.version_store.import_adapter_archive(archive, base, name=name or None)
        except Exception as error:
            QMessageBox.warning(self, "Impor adapter", str(error)); return
        self.refresh_adapters()
        QMessageBox.information(self, "Adapter diimpor", f"{version.name} · perlu pemeriksaan kompatibilitas sebelum aktivasi.")

    def inspect_adapter(self) -> None:
        version_id = self._selected_adapter()
        if not version_id: return
        try:
            report = self.adapter_service.inspect(version_id)
        except Exception as error:
            self.adapter_detail.setText(f"Pemeriksaan gagal: {error}"); return
        lines = [
            f"Base: {Path(report.base_model_path).name} · model_type={report.model_type} · PEFT={report.peft_type} · "
            f"adapter={report.adapter_format} · QLoRA={report.training_quantization}",
            f"Kompatibel: {report.compatible} · Ollama Safetensors import family: {report.ollama_import_supported} · "
            f"Merge dependencies tersedia: {report.runtime_ready}",
        ]
        lines.extend(f"Blocker: {item}" for item in report.issues)
        lines.extend(f"Catatan: {item}" for item in report.warnings)
        self.adapter_detail.setText("\n".join(lines))

    def activate_adapter(self) -> None:
        version_id = self._selected_adapter()
        if not version_id: return
        try: report = self.adapter_service.inspect(version_id)
        except Exception as error:
            QMessageBox.warning(self, "Adapter", str(error)); return
        if not report.compatible or not report.ollama_import_supported or not report.runtime_ready:
            self.inspect_adapter(); return
        free = shutil_disk_free(self.version_store.storage_dir)
        answer = QMessageBox.question(
            self, "Merge dan ganti model aktif?",
            f"Adapter {version_id[:10]} akan di-merge menjadi full model Safetensors.\n"
            f"Perkiraan ruang bebas sekarang: {_bytes(free)}. Hasil import masuk ke Ollama.\n\n"
            "Setelah berhasil, model chat aktif Tanya DORA akan berpindah ke model hasil fine-tuning. Lanjutkan?",
        )
        if answer != QMessageBox.StandardButton.Yes: return
        self.adapter_detail.setText("Merge lokal berjalan. Model aktif saat ini tetap dipakai sampai import berhasil.")
        thread = _AdapterImportThread(self.adapter_service, version_id, self)
        thread.completed.connect(lambda tag, path: self._adapter_imported(version_id, tag, path))
        thread.failed.connect(lambda detail: self.adapter_detail.setText(f"Konversi gagal; model aktif tetap sama. {detail}"))
        thread.finished.connect(lambda: setattr(self, "_adapter_thread", None))
        self._adapter_thread = thread; thread.start()

    def _adapter_imported(self, version_id: str, tag: str, output_dir: str) -> None:
        try:
            self.version_store.select_version(version_id, user_confirmed=True)
            self.version_store.set_active_runtime_model(
                tag, model_kind="ollama_adapter", version_id=version_id, user_confirmed=True
            )
            self.manager.refresh_models(); self.manager.select_model(tag)
        except Exception as error:
            self.adapter_detail.setText(f"Model dibuat, tetapi aktivasi belum tersimpan: {error}. Folder: {output_dir}")
            self.refresh_models(); return
        self.active_label.setText(f"Model produksi: {tag} · adapter {version_id[:10]}")
        self.adapter_detail.setText(f"Model merged berhasil diimpor ke Ollama dan dipilih setelah persetujuan. File: {output_dir}")
        self.model_activated.emit(tag); self.refresh_models(); self.refresh_adapters()

    def export_adapter(self) -> None:
        version_id = self._selected_adapter()
        if not version_id: return
        path, _ = QFileDialog.getSaveFileName(self, "Ekspor adapter", "tanya-dora-adapter.zip", "ZIP (*.zip)")
        if not path: return
        try: output = self.version_store.export_adapter(version_id, path)
        except Exception as error: QMessageBox.warning(self, "Ekspor adapter", str(error))
        else: QMessageBox.information(self, "Ekspor adapter", f"Adapter tersimpan: {output}")

    def rollback_model(self) -> None:
        history = self.version_store.list_activation_history(limit=1)
        if not history or not history[0].get("previous"):
            QMessageBox.information(self, "Rollback", "Belum ada model sebelumnya."); return
        previous = history[0]["previous"]
        target = str(previous.get("model_name") or "")
        if target not in {item.name for item in self._models}:
            QMessageBox.warning(self, "Rollback", f"Model sebelumnya {target or '—'} tidak tersedia di Ollama lokal."); return
        answer = QMessageBox.question(self, "Rollback model", f"Kembalikan model aktif ke {target}?")
        if answer != QMessageBox.StandardButton.Yes: return
        try:
            self.version_store.rollback_active_runtime_model(user_confirmed=True)
            self.manager.select_model(target)
        except Exception as error: QMessageBox.warning(self, "Rollback", str(error)); return
        self.active_label.setText(f"Model produksi: {target}"); self.model_activated.emit(target); self.refresh_adapters()

    def deactivate_adapter(self) -> None:
        active = self.version_store.get_active_runtime_model()
        if not active or active.get("model_kind") != "ollama_adapter":
            QMessageBox.information(self, "Adapter", "Tidak ada adapter hasil merge yang aktif.")
            return
        base_model = str(self.test_model.currentData() or "")
        if not base_model:
            QMessageBox.warning(self, "Adapter", "Pilih base model lokal untuk kembali ke model dasar.")
            return
        answer = QMessageBox.question(
            self,
            "Nonaktifkan adapter?",
            f"Kembali ke model dasar {base_model}? Model hasil fine-tuning tetap tersimpan di Ollama.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.version_store.set_active_runtime_model(base_model, user_confirmed=True)
            self.version_store.clear_selection(user_confirmed=True)
            self.manager.select_model(base_model)
        except Exception as error:
            QMessageBox.warning(self, "Adapter", str(error))
            return
        self.active_label.setText(f"Model produksi: {base_model}")
        self.model_activated.emit(base_model)
        self.refresh_adapters()

    def shutdown(self) -> None:
        self._test_cancel.set()
        self.manager.cancel_generation()
        for thread in (self._status_thread, self._test_thread, self._adapter_thread):
            if thread and thread.isRunning(): thread.wait(1200)


def shutil_disk_free(path: Path) -> int | None:
    try:
        import shutil
        return shutil.disk_usage(path).free
    except OSError:
        return None


class TrainingHistoryTab(QWidget):
    def __init__(self, version_store: ModelVersionStore, parent=None) -> None:
        super().__init__(parent)
        self.version_store = version_store
        root = QVBoxLayout(self); root.setContentsMargins(ui_px(16), ui_px(14), ui_px(16), ui_px(16))
        row = QHBoxLayout(); self.root_path = QLineEdit(str(get_data_dir() / "tanya_dora" / "training_runs")); row.addWidget(self.root_path, 1)
        row.addWidget(_button("Pilih folder", self.choose_folder)); row.addWidget(_button("Muat riwayat", self.refresh, primary=True)); root.addLayout(row)
        self.table = _table(("Status", "Job", "Base model", "TDID", "Loss", "Dibuat", "Folder")); self.table.itemSelectionChanged.connect(self.show_logs); root.addWidget(self.table, 1)
        self.log_view = QPlainTextEdit(); self.log_view.setReadOnly(True); self.log_view.setMaximumHeight(ui_px(190)); root.addWidget(self.log_view)
        actions = QHBoxLayout(); actions.addWidget(_button("Buka folder job", self.open_job)); actions.addWidget(_button("Salin path checkpoint", self.copy_checkpoint)); actions.addWidget(_button("Daftarkan adapter…", self.register_adapter)); actions.addStretch(1); root.addLayout(actions)
        self._jobs: list[dict[str, Any]] = []
        self.refresh()

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pilih folder riwayat training", self.root_path.text())
        if path: self.root_path.setText(path)

    def refresh(self, *_args) -> None:
        root = Path(self.root_path.text()).expanduser()
        self._jobs = []
        if root.is_dir():
            for manifest_path in root.rglob("training_manifest.json"):
                try:
                    if manifest_path.stat().st_size > 2 * 1024 * 1024: continue
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, ValueError): continue
                if not isinstance(manifest, Mapping): continue
                manifest = dict(manifest); manifest["_path"] = str(manifest_path); self._jobs.append(manifest)
        self._jobs.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        self.table.setRowCount(len(self._jobs))
        for row, job in enumerate(self._jobs):
            metrics = job.get("metrics", {}) if isinstance(job.get("metrics"), Mapping) else {}
            loss = metrics.get("train_loss", metrics.get("loss", "—"))
            _set_cell(self.table, row, 0, job.get("status", "unknown"), row)
            _set_cell(self.table, row, 1, job.get("job_id", ""))
            _set_cell(self.table, row, 2, Path(str(job.get("base_model_path", ""))).name)
            _set_cell(self.table, row, 3, job.get("dataset_version_id", ""))
            _set_cell(self.table, row, 4, loss)
            _set_cell(self.table, row, 5, job.get("created_at", ""))
            _set_cell(self.table, row, 6, str(Path(str(job["_path"])).parent))

    def _selected_job(self) -> dict[str, Any] | None:
        row = self.table.currentRow()
        return self._jobs[row] if 0 <= row < len(self._jobs) else None

    def show_logs(self) -> None:
        job = self._selected_job()
        if not job: return
        path = Path(str(job["_path"])).parent / "training_progress.jsonl"
        try: text = path.read_text(encoding="utf-8")[-80_000:] if path.exists() else "Belum ada file progress log pada job ini."
        except OSError as error: text = str(error)
        self.log_view.setPlainText(text)

    def open_job(self) -> None:
        job = self._selected_job()
        if job: QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(str(job["_path"])).parent)))

    def copy_checkpoint(self) -> None:
        job = self._selected_job()
        if not job: return
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(str(Path(str(job["_path"])).parent / "checkpoints"))

    def register_adapter(self) -> None:
        job = self._selected_job()
        if not job or not job.get("adapter_dir") or not Path(str(job["adapter_dir"])).is_dir():
            QMessageBox.information(self, "Adapter registry", "Job ini belum menghasilkan adapter yang dapat didaftarkan."); return
        name, accepted = QInputDialog.getText(self, "Daftarkan adapter", "Nama versi:")
        if not accepted or not name.strip(): return
        try:
            version = self.version_store.register_adapter(
                name,
                str(job["adapter_dir"]),
                str(job["base_model_path"]),
                dataset_version_id=str(job.get("dataset_version_id") or "") or None,
                training_config=job.get("config", {}),
                evaluation_metrics=job.get("metrics", {}),
            )
        except Exception as error:
            QMessageBox.warning(self, "Adapter registry", str(error)); return
        QMessageBox.information(self, "Adapter registry", f"{version.name} tersimpan. Adapter belum diaktifkan.")


class EvaluationStudioTab(QWidget):
    def __init__(
        self,
        manager: OllamaLLMManager,
        knowledge_base: DoraKnowledgeBase,
        tdid_store: TDIDVersionStore,
        benchmark_store: TDEBBenchmarkStore,
        tool_context_provider=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.manager = manager; self.kb = knowledge_base; self.tdid_store = tdid_store
        self.benchmark_store = benchmark_store; self.tool_context_provider = tool_context_provider
        self.runner = TDEBBenchmarkRunner(benchmark_store)
        self.builder = TDEBBenchmarkBuilder()
        self._cancel = threading.Event(); self._thread: _EvaluationThread | None = None
        root = QVBoxLayout(self); root.setContentsMargins(ui_px(16), ui_px(14), ui_px(16), ui_px(16))
        import_row = QHBoxLayout(); import_row.addWidget(_button("Impor benchmark JSONL", self.import_benchmark)); import_row.addWidget(QLabel("TDID pembanding"))
        self.tdid_combo = QComboBox(); import_row.addWidget(self.tdid_combo, 1)
        import_row.addWidget(QLabel("Benchmark")); self.benchmark_combo = QComboBox(); import_row.addWidget(self.benchmark_combo, 1)
        import_row.addWidget(_button("Muat versi", self.refresh_versions)); root.addLayout(import_row)
        model_row = QHBoxLayout(); model_row.addWidget(QLabel("Base model")); self.base_model = QComboBox(); model_row.addWidget(self.base_model, 1)
        model_row.addWidget(QLabel("Fine-tuned Ollama model")); self.ft_model = QComboBox(); model_row.addWidget(self.ft_model, 1)
        self.embedding = QComboBox(); self.embedding.addItem("RAG keyword FTS5", None); model_row.addWidget(self.embedding); root.addLayout(model_row)
        self.run_button = _button("Jalankan perbandingan 5 kondisi", self.run_comparison, primary=True); self.stop_button = _button("Stop", self.stop_comparison); self.stop_button.setEnabled(False)
        action = QHBoxLayout(); action.addWidget(self.run_button); action.addWidget(self.stop_button); action.addWidget(_button("Ekspor hasil JSON", self.export_results)); action.addStretch(1); root.addLayout(action)
        self.progress = QLabel("Impor benchmark terpisah, pilih TDID untuk leakage check, lalu pilih model lokal."); self.progress.setObjectName("tanyaDoraStudioHint"); self.progress.setWordWrap(True); root.addWidget(self.progress)
        self.table = _table(("Kondisi", "Model", "Status", "Kasus", "Exact match¹", "Term recall¹", "Latency", "Run ID")); root.addWidget(self.table, 1)
        self.disclaimer = QLabel("¹ Metrik otomatis ini proxy lexical. Factual accuracy, reasoning, citation faithfulness, hallucination dan kualitas tools membutuhkan review manusia.")
        self.disclaimer.setObjectName("tanyaDoraStudioHint"); self.disclaimer.setWordWrap(True); root.addWidget(self.disclaimer)
        self.refresh_models(); self.refresh_versions()

    def refresh_versions(self) -> None:
        self.tdid_combo.clear()
        for version in self.tdid_store.list_versions():
            self.tdid_combo.addItem(f"{version.name} · {version.status}", version.version_id)
        selected_benchmark = self.benchmark_combo.currentData()
        self.benchmark_combo.clear()
        for version in self.benchmark_store.list_versions():
            self.benchmark_combo.addItem(f"{version.name} · {version.case_count} kasus", version.version_id)
        if selected_benchmark:
            index = self.benchmark_combo.findData(selected_benchmark)
            if index >= 0: self.benchmark_combo.setCurrentIndex(index)
        if hasattr(self, "builder"):
            self.progress.setText(f"{len(self.builder.cases)} kasus benchmark belum disimpan.")

    def refresh_models(self) -> None:
        try: models = self.manager.refresh_models()
        except Exception as error:
            self.progress.setText(f"Ollama belum tersedia: {error}"); return
        self.base_model.clear(); self.ft_model.clear()
        self.embedding.clear(); self.embedding.addItem("RAG keyword FTS5", None)
        for model in models:
            self.base_model.addItem(model.name, model.name); self.ft_model.addItem(model.name, model.name)
            self.embedding.addItem(f"Embedding · {model.name}", model.name)

    def import_benchmark(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Impor TDEB benchmark JSONL", "", "JSONL (*.jsonl)")
        if not path: return
        try: builder = TDEBBenchmarkBuilder.from_jsonl(path)
        except Exception as error: QMessageBox.warning(self, "TDEB", str(error)); return
        issues = builder.validate()
        self.builder = builder
        if issues:
            self.progress.setText("Benchmark belum valid: " + " · ".join(f"{item.case_id}:{item.field} {item.message}" for item in issues[:8]))
            return
        self.progress.setText(f"{len(builder.cases)} kasus valid · kebocoran diperiksa saat menyimpan versi.")
        name, accepted = __import__("PySide6.QtWidgets", fromlist=["QInputDialog"]).QInputDialog.getText(self, "Versi benchmark", "Nama versi:")
        if not accepted or not name.strip(): return
        tdid_id = str(self.tdid_combo.currentData() or "")
        if not tdid_id:
            QMessageBox.warning(self, "TDEB", "Pilih versi TDID untuk memeriksa leakage."); return
        try:
            tdid_builder = self.tdid_store.load_version(tdid_id)
            version = self.benchmark_store.save_version(name, builder, training_dataset=tdid_builder)
        except Exception as error:
            QMessageBox.warning(self, "TDEB", str(error)); return
        self.progress.setText(f"Benchmark {version.name} disimpan setelah leakage check independent.")
        self.refresh_versions()
        index = self.benchmark_combo.findData(version.version_id)
        if index >= 0: self.benchmark_combo.setCurrentIndex(index)
        self.refresh_saved_runs(version.version_id)

    def _target_bundle(self):
        base = str(self.base_model.currentData() or "")
        fine_tuned = str(self.ft_model.currentData() or "")
        if not base or not fine_tuned: raise ValueError("Pilih base model dan model hasil fine-tuning dari Ollama lokal.")
        if base == fine_tuned: raise ValueError("Pilih model base dan model fine-tuned yang berbeda.")
        version_id = str(self.tdid_combo.currentData() or "")
        if not version_id: raise ValueError("Pilih TDID pembanding.")
        benchmark_version = self._selected_benchmark_version()
        if not benchmark_version: raise ValueError("Pilih/import versi TDEB terlebih dahulu.")
        tdeb = self.benchmark_store.get_version(benchmark_version)
        if not tdeb.leakage_checked or not tdeb.leakage_report.get("independent"):
            raise ValueError("Benchmark belum lolos leakage check terhadap TDID.")
        embedding = str(self.embedding.currentData() or "") or None
        tool_registry = None
        if callable(self.tool_context_provider):
            try: tool_registry = build_dora_tool_registry(self.tool_context_provider(), self.kb)
            except Exception as error: raise ValueError(f"Tool context tidak tersedia: {error}") from error

        def rag_target(model_name: str, mode: str, tools_enabled: bool) -> BenchmarkTarget:
            registry = tool_registry if tools_enabled else None
            def generate(case):
                response = RAGPipeline(
                    self.kb, self.manager, chat_model=model_name, embedding_model=embedding,
                    temperature=0, tool_registry=registry,
                ).answer(case.prompt)
                return response
            return BenchmarkTarget(name=f"{model_name} ({mode})", model_name=model_name, mode=mode, generate=generate)

        targets = (
            BenchmarkTarget.ollama(self.manager, base, mode="base"),
            rag_target(base, "base_rag", False),
            BenchmarkTarget.ollama(self.manager, fine_tuned, mode="fine_tuned"),
            rag_target(fine_tuned, "fine_tuned_rag", False),
            rag_target(fine_tuned, "fine_tuned_rag_tools", True),
        )
        return benchmark_version, targets

    def _selected_benchmark_version(self) -> str | None:
        selected = self.benchmark_combo.currentData()
        return str(selected) if selected else None

    def run_comparison(self) -> None:
        if self._thread and self._thread.isRunning(): return
        try: version_id, targets = self._target_bundle()
        except Exception as error: QMessageBox.warning(self, "TDEB comparison", str(error)); return
        self._cancel = threading.Event(); self.run_button.setEnabled(False); self.stop_button.setEnabled(True)
        thread = _EvaluationThread(self.runner, version_id, targets, self._cancel, self)
        thread.progress.connect(self.progress.setText); thread.completed.connect(self._comparison_complete); thread.failed.connect(self._comparison_failed)
        thread.finished.connect(self._comparison_finished); self._thread = thread; thread.start()

    def stop_comparison(self) -> None:
        self._cancel.set(); self.stop_button.setEnabled(False)

    def _comparison_complete(self, runs: object) -> None:
        for run in runs:
            summary = run.summary
            self.progress.setText(f"{run.mode} selesai · status={run.status} · rata-rata latency={summary.get('mean_latency_seconds')}s")
        if runs: self.refresh_saved_runs(runs[0].version_id)

    def _comparison_failed(self, detail: str) -> None:
        self.progress.setText(f"Perbandingan gagal: {detail}")

    def _comparison_finished(self) -> None:
        self.run_button.setEnabled(True); self.stop_button.setEnabled(False); self._thread = None

    def refresh_saved_runs(self, version_id: str) -> None:
        self._benchmark_version_id = version_id
        index = self.benchmark_combo.findData(version_id)
        if index >= 0: self.benchmark_combo.setCurrentIndex(index)
        runs = self.benchmark_store.list_runs(version_id)
        self.table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            summary = run.summary
            _set_cell(self.table, row, 0, run.mode, run.run_id)
            _set_cell(self.table, row, 1, run.model_name)
            _set_cell(self.table, row, 2, run.status)
            _set_cell(self.table, row, 3, summary.get("case_count", len(run.case_results)))
            _set_cell(self.table, row, 4, summary.get("exact_match_rate"))
            _set_cell(self.table, row, 5, summary.get("required_term_recall"))
            _set_cell(self.table, row, 6, summary.get("mean_latency_seconds"))
            _set_cell(self.table, row, 7, run.run_id)

    def export_results(self) -> None:
        version_id = self._selected_benchmark_version()
        if not version_id:
            QMessageBox.information(self, "Ekspor TDEB", "Belum ada benchmark version atau hasil."); return
        path, _ = QFileDialog.getSaveFileName(self, "Ekspor hasil evaluasi", "tdeb-results.json", "JSON (*.json)")
        if not path: return
        try:
            runs = self.benchmark_store.list_runs(version_id)
            payload = [asdict(run) for run in runs]
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as error: QMessageBox.warning(self, "Ekspor TDEB", str(error))
        else: QMessageBox.information(self, "Ekspor TDEB", f"{len(runs)} hasil diekspor.")


class TanyaDoraManagementPage(QWidget):
    """One Studio page with six functional service-backed management tabs."""

    model_activated = Signal(str)

    def __init__(
        self,
        *,
        knowledge_base: DoraKnowledgeBase | None = None,
        llm_manager: OllamaLLMManager | None = None,
        tdid_store: TDIDVersionStore | None = None,
        version_store: ModelVersionStore | None = None,
        benchmark_store: TDEBBenchmarkStore | None = None,
        tool_context_provider=None,
    ) -> None:
        super().__init__()
        self.setObjectName("tanyaDoraManagementPage")
        self.knowledge_base = knowledge_base or DoraKnowledgeBase()
        self.llm_manager = llm_manager or OllamaLLMManager()
        self.tdid_store = tdid_store or TDIDVersionStore()
        self.version_store = version_store or ModelVersionStore()
        self.benchmark_store = benchmark_store or TDEBBenchmarkStore()
        self._owns_manager = llm_manager is None
        self.model_tab: ModelManagerTab | None = None
        root = QVBoxLayout(self); root.setContentsMargins(ui_px(26), ui_px(18), ui_px(26), ui_px(24)); root.setSpacing(ui_px(14))
        title = QLabel("Tanya DORA Management Studio"); title.setObjectName("tanyaDoraStudioTitle")
        subtitle = QLabel("Kelola sumber lokal, dataset terverifikasi, training, model, dan evaluasi dari service Tanya DORA yang sama.")
        subtitle.setObjectName("tanyaDoraStudioSubtitle"); subtitle.setWordWrap(True)
        root.addWidget(title); root.addWidget(subtitle)
        self.tabs = QTabWidget(); self.tabs.setObjectName("tanyaDoraStudioTabs")
        self.kb_tab = KnowledgeBaseManagerTab(self.knowledge_base)
        self.dataset_tab = DatasetManagerTab(self.tdid_store)
        self.training_tab = FineTuningStudioTab(self.tdid_store)
        self.model_tab = ModelManagerTab(self.llm_manager, self.version_store)
        self.history_tab = TrainingHistoryTab(self.version_store)
        self.evaluation_tab = EvaluationStudioTab(
            self.llm_manager, self.knowledge_base, self.tdid_store, self.benchmark_store,
            tool_context_provider=tool_context_provider,
        )
        for title_text, page in (
            ("Knowledge Base", self.kb_tab), ("Dataset TDID", self.dataset_tab),
            ("Fine-Tuning", self.training_tab), ("Model Manager", self.model_tab),
            ("Training History", self.history_tab), ("AI Evaluation", self.evaluation_tab),
        ):
            self.tabs.addTab(page, title_text)
        root.addWidget(self.tabs, 1)
        self.model_tab.model_activated.connect(self.model_activated)
        app = __import__("PySide6.QtWidgets", fromlist=["QApplication"]).QApplication.instance()
        if app is not None: app.aboutToQuit.connect(self.shutdown)

    def apply_scale(self, _scale=None) -> None:
        self.layout().setContentsMargins(ui_px(26), ui_px(18), ui_px(26), ui_px(24))

    def shutdown(self) -> None:
        if self.model_tab: self.model_tab.shutdown()
        for thread in (
            self.training_tab._preflight_thread,
            self.training_tab._training_thread,
            self.evaluation_tab._thread,
        ):
            if thread and thread.isRunning(): thread.wait(1500)
        if self._owns_manager: self.llm_manager.close()

