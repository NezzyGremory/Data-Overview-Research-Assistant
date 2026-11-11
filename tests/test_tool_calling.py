from __future__ import annotations

import json
import unittest

import httpx

from app.ai.llm_manager import OllamaLLMManager
from app.ai.tool_calling import (
    DoraTool,
    DoraToolRegistry,
    ToolExecution,
    build_dora_tool_registry,
)


class FakeKnowledgeBase:
    CATEGORIES = {"statistics", "dora_documentation"}

    def __init__(self):
        self.queries = []

    def search(self, query, **kwargs):
        self.queries.append((query, kwargs))
        return ()


class TestDoraToolRegistry(unittest.TestCase):
    def test_registers_only_bounded_read_only_dora_tools(self):
        registry = build_dora_tool_registry(
            {
                "active_dataset": {
                    "name": "study.csv",
                    "rows": 42,
                    "columns": 2,
                    "column_types": [
                        {"name": "score", "dtype": "float64"},
                        {"name": "group", "dtype": "object"},
                    ],
                },
                "analysis": {
                    "profile": {"rows": 42, "columns": 2},
                    "statistics": [
                        {"column": "score", "mean": 0.75},
                        {"column": "group", "unique_count": 2},
                    ],
                },
            },
            FakeKnowledgeBase(),
        )

        names = {
            item["function"]["name"]
            for item in registry.definitions()
        }
        self.assertIn("get_active_dataset", names)
        self.assertIn("get_dataset_profile", names)
        self.assertIn("get_statistical_summary", names)
        self.assertIn("search_knowledge_base", names)
        self.assertNotIn("search_academic_papers", names)
        self.assertFalse(any("write" in name or "delete" in name for name in names))

        stats = registry.execute("get_statistical_summary", {"column": "score"})
        self.assertTrue(stats.provides_evidence)
        self.assertEqual(stats.result["result"][0]["mean"], 0.75)
        self.assertEqual(json.loads(stats.to_json())["name"], "get_statistical_summary")

    def test_unavailable_analysis_and_empty_search_are_not_evidence(self):
        kb = FakeKnowledgeBase()
        registry = build_dora_tool_registry(None, kb)

        profile = registry.execute("get_dataset_profile", {})
        search = registry.execute("search_knowledge_base", {"query": "uji statistik"})

        self.assertTrue(profile.success)
        self.assertFalse(profile.provides_evidence)
        self.assertFalse(search.provides_evidence)
        self.assertEqual(kb.queries[0][0], "uji statistik")

    def test_rejects_unknown_tools_unexpected_arguments_and_pending_confirmation(self):
        registry = DoraToolRegistry()
        registry.register(
            DoraTool(
                "read_summary",
                "Read a summary.",
                {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                lambda _args: {"rows": 2},
            )
        )
        registry.register(
            DoraTool(
                "write_project",
                "Example future write tool.",
                {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                lambda _args: {"written": True},
                requires_confirmation=True,
            )
        )

        self.assertFalse(registry.execute("python_eval", {}).success)
        self.assertFalse(registry.execute("read_summary", {"path": "x"}).success)
        self.assertFalse(
            build_dora_tool_registry(None, FakeKnowledgeBase(), allow_online_research=True)
            .execute("search_academic_papers", {"query": "AI", "limit": 2.5})
            .success
        )
        pending = registry.execute("write_project", {})
        self.assertFalse(pending.success)
        self.assertTrue(pending.requires_confirmation)

    def test_online_academic_search_is_registered_only_after_opt_in(self):
        kb = FakeKnowledgeBase()
        offline = build_dora_tool_registry(None, kb)
        online = build_dora_tool_registry(None, kb, allow_online_research=True)

        self.assertFalse(any(
            item["function"]["name"] == "search_academic_papers"
            for item in offline.definitions()
        ))
        self.assertTrue(any(
            item["function"]["name"] == "search_academic_papers"
            for item in online.definitions()
        ))

    def test_ollama_tool_loop_executes_registered_function_then_streams_final_answer(self):
        requests = []
        model = {
            "name": "qwen3:8b",
            "model": "qwen3:8b",
            "size": 1234,
        }

        def json_response(request, payload):
            return httpx.Response(200, json=payload, request=request)

        def stream_response(request, messages):
            content = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in messages)
            return httpx.Response(200, content=content.encode("utf-8"), request=request)

        def handler(request):
            requests.append(request)
            if request.url.path == "/api/tags":
                return json_response(request, {"models": [model]})
            payload = json.loads(request.content.decode("utf-8"))
            self.assertTrue(payload["stream"])
            self.assertTrue(payload["tools"])
            if not any(message.get("role") == "tool" for message in payload["messages"]):
                return stream_response(request, [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "type": "function",
                                "function": {
                                    "name": "get_active_dataset",
                                    "arguments": {},
                                },
                            }],
                        },
                        "done": False,
                    },
                    {"message": {"role": "assistant", "content": ""}, "done": True},
                ])
            tool_message = next(message for message in payload["messages"] if message.get("role") == "tool")
            result = json.loads(tool_message["content"])
            self.assertEqual(result["result"]["rows"], 42)
            self.assertEqual(tool_message["tool_name"], "get_active_dataset")
            return stream_response(request, [
                {"message": {"role": "assistant", "content": "Dataset memiliki 42 baris."}, "done": False},
                {"message": {"role": "assistant", "content": " [DORA]"}, "done": True},
            ])

        manager = OllamaLLMManager(transport=httpx.MockTransport(handler))
        self.addCleanup(manager.close)
        registry = build_dora_tool_registry(
            {"active_dataset": {"name": "scores.csv", "rows": 42, "columns": 2}},
            FakeKnowledgeBase(),
        )
        deltas = []
        tool_calls = []

        response = manager.chat_with_tools(
            [{"role": "user", "content": "Berapa baris dataset saya?"}],
            model="qwen3:8b",
            tool_registry=registry,
            on_delta=deltas.append,
            on_tool_call=lambda name, args: tool_calls.append((name, dict(args))),
        )

        self.assertEqual(response.answer, "Dataset memiliki 42 baris. [DORA]")
        self.assertEqual(len(response.executions), 1)
        self.assertTrue(response.executions[0].provides_evidence)
        self.assertEqual(tool_calls, [("get_active_dataset", {})])
        self.assertEqual(deltas, ["Dataset memiliki 42 baris.", " [DORA]"])
        self.assertEqual([request.url.path for request in requests], ["/api/tags", "/api/chat", "/api/chat"])


if __name__ == "__main__":
    unittest.main()
