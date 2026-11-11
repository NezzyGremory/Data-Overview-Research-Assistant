from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import httpx
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

from app.ai.evaluation_benchmark import TDEBBenchmarkStore
from app.ai.knowledge_base import DoraKnowledgeBase
from app.ai.llm_manager import OllamaLLMManager
from app.ai.model_versioning import ModelVersionStore
from app.ai.tdid_dataset import TDIDVersionStore
from app.ui.tanya_dora_management_page import TanyaDoraManagementPage


def _ollama(payload):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": payload})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(404, json={"error": "unknown endpoint"})
    return handler


class TestTanyaDoraManagementPage(unittest.TestCase):
    def test_six_studio_tabs_bind_to_local_service_stores(self):
        application = QApplication.instance() or QApplication([])
        temporary = tempfile.TemporaryDirectory(prefix="tanya-studio-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        manager = OllamaLLMManager(transport=httpx.MockTransport(_ollama([])))
        page = TanyaDoraManagementPage(
            knowledge_base=DoraKnowledgeBase(root / "knowledge"),
            llm_manager=manager,
            tdid_store=TDIDVersionStore(root / "tdid"),
            version_store=ModelVersionStore(root / "models"),
            benchmark_store=TDEBBenchmarkStore(root / "benchmarks"),
        )

        def cleanup_page():
            page.shutdown()
            manager.close()
            page.close()
            page.deleteLater()
            application.sendPostedEvents(None, QEvent.DeferredDelete)
            application.processEvents()

        self.addCleanup(cleanup_page)
        page.resize(1440, 900)
        page.show()
        status_thread = page.model_tab._status_thread
        if status_thread is not None:
            status_thread.wait(3000)
        application.processEvents()

        self.assertEqual(page.tabs.count(), 6)
        self.assertEqual(
            [page.tabs.tabText(index) for index in range(page.tabs.count())],
            ["Knowledge Base", "Dataset TDID", "Fine-Tuning", "Model Manager", "Training History", "AI Evaluation"],
        )
        self.assertIn("model lokal", page.model_tab.status.text())
        self.assertEqual(page.kb_tab.table.rowCount(), 0)
        self.assertEqual(page.dataset_tab.table.rowCount(), 0)


if __name__ == "__main__":
    unittest.main()
