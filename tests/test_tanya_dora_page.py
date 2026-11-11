from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QEvent

from app.ai.conversation_store import ConversationStore
from app.ai.knowledge_base import DoraKnowledgeBase
from app.ai.llm_manager import OllamaStatus
from app.ai.model_versioning import ModelVersionStore
from app.ui.tanya_dora_page import TanyaDoraPage


class UnavailableLocalLLM:
    def get_status(self):
        return OllamaStatus(
            available=False,
            selected_model=None,
            installed_model=False,
            running_model=False,
            installed_model_count=0,
            detail="Local service is stopped.",
        )

    def list_models(self):
        return ()

    def cancel_generation(self):
        return None


class TestTanyaDoraPage(unittest.TestCase):
    def test_page_reports_unavailable_local_model_without_demo_chat(self):
        application = QApplication.instance() or QApplication([])
        temporary = tempfile.TemporaryDirectory(prefix="tanya-page-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        page = TanyaDoraPage(
            conversation_store=ConversationStore(root / "history"),
            knowledge_base=DoraKnowledgeBase(root / "knowledge"),
            llm_manager=UnavailableLocalLLM(),
            model_version_store=ModelVersionStore(root / "models"),
        )
        def cleanup_page():
            page.shutdown()
            page.close()
            page.deleteLater()
            application.sendPostedEvents(None, QEvent.DeferredDelete)
            application.processEvents()

        self.addCleanup(cleanup_page)
        page.resize(1120, 760)
        page.show()

        refresh_thread = page._model_thread
        self.assertIsNotNone(refresh_thread)
        refresh_thread.wait(3000)
        application.processEvents()

        self.assertIn("Ollama tidak tersedia", page.model_status.text())
        self.assertEqual(page.model_combo.currentText(), "Tidak ada model lokal")
        self.assertFalse(page.send_button.isEnabled())
        self.assertFalse(page.online_research_checkbox.isChecked())
        self.assertIn("Selamat datang di Tanya DORA", page.transcript.toPlainText())
        self.assertEqual(page.conversation_store.list_conversations(), ())


if __name__ == "__main__":
    unittest.main()
