from __future__ import annotations

import json
import threading
import unittest

import httpx

from app.ai.llm_manager import (
    LocalLLMError,
    ModelNotInstalledError,
    ModelNotSelectedError,
    OllamaLLMManager,
    OllamaUnavailableError,
)


MODEL = {
    "name": "qwen2.5:7b",
    "model": "qwen2.5:7b",
    "size": 4 * 1024**3,
    "modified_at": "2026-01-01T00:00:00Z",
    "details": {
        "family": "qwen2",
        "parameter_size": "7.6B",
        "quantization_level": "Q4_K_M",
    },
}


class TestOllamaLLMManager(unittest.TestCase):
    def manager(self, handler):
        manager = OllamaLLMManager(transport=httpx.MockTransport(handler))
        self.addCleanup(manager.close)
        return manager

    @staticmethod
    def json_response(payload, status_code=200):
        return httpx.Response(
            status_code,
            json=payload,
            request=httpx.Request("GET", "http://127.0.0.1:11434"),
        )

    def test_refresh_models_parses_installed_model_metadata(self):
        manager = self.manager(
            lambda request: self.json_response({"models": [MODEL]})
        )

        models = manager.refresh_models()

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "qwen2.5:7b")
        self.assertEqual(models[0].family, "qwen2")
        self.assertEqual(models[0].parameter_size, "7.6B")
        self.assertEqual(models[0].quantization_level, "Q4_K_M")
        self.assertEqual(models[0].size_label, "4.0 GiB")

    def test_select_model_requires_model_to_be_installed(self):
        manager = self.manager(
            lambda request: self.json_response({"models": [MODEL]})
        )

        selected = manager.select_model("qwen2.5:7b")

        self.assertEqual(selected.name, "qwen2.5:7b")
        self.assertEqual(manager.selected_model, "qwen2.5:7b")

    def test_missing_model_is_not_pulled_automatically(self):
        requested_paths = []

        def handler(request):
            requested_paths.append(request.url.path)
            return self.json_response({"models": []})

        manager = self.manager(handler)

        with self.assertRaises(ModelNotInstalledError):
            manager.select_model("not-installed:latest")

        self.assertEqual(requested_paths, ["/api/tags"])
        self.assertNotIn("/api/pull", requested_paths)

    def test_generate_sends_non_streaming_chat_request(self):
        requests = []

        def handler(request):
            requests.append(request)
            if request.url.path == "/api/tags":
                return self.json_response({"models": [MODEL]})
            payload = json.loads(request.content.decode("utf-8"))
            self.assertEqual(payload["model"], "qwen2.5:7b")
            self.assertEqual(payload["stream"], False)
            self.assertEqual(payload["options"]["temperature"], 0.2)
            self.assertEqual(
                payload["messages"],
                [
                    {"role": "system", "content": "Jawab ringkas."},
                    {"role": "user", "content": "Apa itu median?"},
                ],
            )
            return self.json_response(
                {"message": {"role": "assistant", "content": "Nilai tengah data."}}
            )

        manager = self.manager(handler)
        manager.select_model("qwen2.5:7b")

        answer = manager.generate(
            "Apa itu median?", system="Jawab ringkas.", temperature=0.2
        )

        self.assertEqual(answer, "Nilai tengah data.")
        self.assertEqual([request.url.path for request in requests], ["/api/tags"] * 2 + ["/api/chat"])

    def test_embed_texts_uses_installed_local_model_and_validates_vectors(self):
        requested_paths = []

        def handler(request):
            requested_paths.append(request.url.path)
            if request.url.path == "/api/tags":
                return self.json_response({"models": [MODEL]})
            payload = json.loads(request.content.decode("utf-8"))
            self.assertEqual(payload["model"], "qwen2.5:7b")
            self.assertEqual(payload["input"], ["median", "variance"])
            self.assertFalse(payload["truncate"])
            return httpx.Response(
                200,
                json={"embeddings": [[0.25, 0.75], [0.5, -0.5]]},
                request=request,
            )

        manager = self.manager(handler)

        vectors = manager.embed_texts(
            ["median", "variance"],
            model="qwen2.5:7b",
        )

        self.assertEqual(vectors, ((0.25, 0.75), (0.5, -0.5)))
        self.assertEqual(requested_paths, ["/api/tags", "/api/embed"])

    def test_chat_stream_emits_local_deltas(self):
        requests = []

        def handler(request):
            requests.append(request)
            if request.url.path == "/api/tags":
                return self.json_response({"models": [MODEL]})
            self.assertEqual(request.url.path, "/api/chat")
            payload = json.loads(request.content.decode("utf-8"))
            self.assertTrue(payload["stream"])
            body = (
                '{"message":{"role":"assistant","content":"Halo"},"done":false}\n'
                '{"message":{"role":"assistant","content":" DORA"},"done":false}\n'
                '{"message":{"role":"assistant","content":""},"done":true}\n'
            )
            return httpx.Response(200, content=body.encode(), request=request)

        manager = self.manager(handler)
        deltas = []
        answer = manager.chat_stream(
            [{"role": "user", "content": "Halo"}],
            model="qwen2.5:7b",
            temperature=0.1,
            on_delta=deltas.append,
        )

        self.assertEqual(answer, "Halo DORA")
        self.assertEqual(deltas, ["Halo", " DORA"])
        self.assertEqual([request.url.path for request in requests], ["/api/tags", "/api/chat"])

    def test_chat_stream_stops_after_cancel_event(self):
        cancel_event = threading.Event()

        def handler(request):
            if request.url.path == "/api/tags":
                return self.json_response({"models": [MODEL]})
            body = (
                '{"message":{"role":"assistant","content":"Parsial"},"done":false}\n'
                '{"message":{"role":"assistant","content":" diabaikan"},"done":false}\n'
            )
            return httpx.Response(200, content=body.encode(), request=request)

        manager = self.manager(handler)
        deltas = []

        def stop_after_first(delta):
            deltas.append(delta)
            cancel_event.set()

        answer = manager.chat_stream(
            [{"role": "user", "content": "Halo"}],
            model="qwen2.5:7b",
            on_delta=stop_after_first,
            cancel_event=cancel_event,
        )

        self.assertEqual(answer, "Parsial")
        self.assertEqual(deltas, ["Parsial"])

    def test_chat_requires_selected_or_explicit_model(self):
        manager = self.manager(
            lambda request: self.json_response({"models": [MODEL]})
        )

        with self.assertRaises(ModelNotSelectedError):
            manager.chat([{"role": "user", "content": "Halo"}])

    def test_chat_refuses_model_removed_after_selection(self):
        inventory = {"models": [MODEL]}
        manager = self.manager(lambda request: self.json_response(inventory))
        manager.select_model("qwen2.5:7b")
        inventory["models"] = []

        with self.assertRaises(ModelNotInstalledError):
            manager.chat([{"role": "user", "content": "Halo"}])

    def test_status_reports_connection_and_running_model(self):
        def handler(request):
            if request.url.path == "/api/tags":
                return self.json_response({"models": [MODEL]})
            if request.url.path == "/api/ps":
                return self.json_response({"models": [{"name": "qwen2.5:7b"}]})
            raise AssertionError(request.url.path)

        manager = self.manager(handler)
        manager.select_model("qwen2.5:7b")

        status = manager.get_status()

        self.assertTrue(status.available)
        self.assertTrue(status.installed_model)
        self.assertTrue(status.running_model)
        self.assertEqual(status.installed_model_count, 1)
        self.assertEqual(status.label, "Model is running")

    def test_connection_failure_is_reported_as_local_ollama_unavailable(self):
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        manager = self.manager(handler)

        with self.assertRaises(OllamaUnavailableError):
            manager.refresh_models()

        self.assertFalse(manager.get_status().available)

    def test_server_http_error_is_reported(self):
        def handler(request):
            return httpx.Response(
                500,
                json={"error": "server failure"},
                request=request,
            )

        manager = self.manager(handler)

        with self.assertRaisesRegex(LocalLLMError, "server failure"):
            manager.refresh_models()

    def test_invalid_temperature_and_messages_are_rejected(self):
        manager = self.manager(
            lambda request: self.json_response({"models": [MODEL]})
        )

        with self.assertRaises(ValueError):
            manager.generate("Hello", temperature=float("nan"))
        with self.assertRaises(ValueError):
            manager.chat([{"role": "developer", "content": "Hello"}])
        with self.assertRaises(ValueError):
            manager.generate("  ")

    def test_non_loopback_server_url_is_rejected(self):
        for url in (
            "https://ollama.example.com",
            "http://0.0.0.0:11434",
            "http://localhost:11434/path",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                OllamaLLMManager(base_url=url)


if __name__ == "__main__":
    unittest.main()
