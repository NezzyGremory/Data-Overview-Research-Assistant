from __future__ import annotations

import unittest
import threading

from app.ai.knowledge_base import KnowledgeSearchResult
from app.ai.rag_pipeline import RAGPipeline
from app.ai.tool_calling import ToolChatResult, build_dora_tool_registry


def source_result(*, text="Median is the middle value.", document_id="doc-1"):
    return KnowledgeSearchResult(
        document_id=document_id,
        chunk_id=1,
        title="Statistics Guide",
        author_or_org="DORA Docs",
        category="statistics",
        publication_year=2026,
        source_url="https://example.test/statistics",
        doi="10.1234/dora.2026.1",
        license="CC BY 4.0",
        version="1.0",
        verification_status="VERIFIED",
        text=text,
        page_or_section="Page 2",
        chunk_index=0,
        relevance_score=0.42,
    )


class FakeKnowledgeBase:
    def __init__(self, results):
        self.results = tuple(results)
        self.search_calls = []
        self.hybrid_calls = []

    def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return self.results

    def search_hybrid(self, query, embedding, model, **kwargs):
        self.hybrid_calls.append((query, embedding, model, kwargs))
        return self.results


class FakeLLM:
    def __init__(self, answer="Median is the middle value. [S1] [S8]"):
        self.answer_text = answer
        self.embedding_calls = []
        self.chat_calls = []

    def embed_texts(self, texts, *, model):
        self.embedding_calls.append((texts, model))
        return ((1.0, 0.0),)

    def chat(self, messages, *, model, temperature):
        self.chat_calls.append((messages, model, temperature))
        return self.answer_text

    def chat_stream(self, messages, *, model, temperature, on_delta, cancel_event=None):
        self.chat_calls.append((messages, model, temperature))
        on_delta("Jawaban")
        if cancel_event is not None:
            cancel_event.set()
        return "Jawaban [S1]"

    def chat_with_tools(
        self,
        messages,
        *,
        model,
        temperature,
        on_delta,
        on_tool_call,
        cancel_event,
        tool_registry,
    ):
        self.chat_calls.append((messages, model, temperature))
        if cancel_event is not None and cancel_event.is_set():
            return ToolChatResult("", (), True)
        if on_tool_call is not None:
            on_tool_call("get_active_dataset", {})
        execution = tool_registry.execute("get_active_dataset", {})
        if execution.provides_evidence:
            answer = "Dataset memiliki 42 baris. [DORA]"
            return ToolChatResult(answer, (execution,))
        return ToolChatResult("Jawaban tanpa bukti yang ditolak.", (execution,))


class TestRAGPipeline(unittest.TestCase):
    def test_answer_uses_retrieved_source_and_strips_unknown_source_markers(self):
        knowledge_base = FakeKnowledgeBase([source_result()])
        llm = FakeLLM()
        pipeline = RAGPipeline(
            knowledge_base,
            llm,
            chat_model="qwen2.5:7b",
        )

        response = pipeline.answer("Apa itu median?")

        self.assertTrue(response.grounded)
        self.assertEqual(response.retrieval_mode, "keyword_fts5")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].source_id, "S1")
        self.assertIn("[S1]", response.answer)
        self.assertNotIn("[S8]", response.answer)
        self.assertIn("DOI: 10.1234/dora.2026.1", response.sources[0].citation)
        self.assertEqual(knowledge_base.search_calls[0][0], "Apa itu median?")
        self.assertIn("[S1]", llm.chat_calls[0][0][1]["content"])

    def test_hybrid_mode_embeds_question_and_uses_local_hybrid_retrieval(self):
        knowledge_base = FakeKnowledgeBase([source_result()])
        llm = FakeLLM("Sumber mendukung jawaban ini. [S1]")
        pipeline = RAGPipeline(
            knowledge_base,
            llm,
            chat_model="qwen2.5:7b",
            embedding_model="embeddinggemma:latest",
        )

        response = pipeline.answer("Median?", category="statistics", max_sources=3)

        self.assertEqual(response.retrieval_mode, "hybrid_fts_cosine")
        self.assertEqual(llm.embedding_calls, [(["Median?"], "embeddinggemma:latest")])
        self.assertEqual(knowledge_base.hybrid_calls[0][2], "embeddinggemma:latest")
        self.assertEqual(knowledge_base.hybrid_calls[0][3]["category"], "statistics")
        self.assertTrue(response.grounded)

    def test_no_source_returns_honest_message_without_calling_chat_model(self):
        knowledge_base = FakeKnowledgeBase([])
        llm = FakeLLM()
        pipeline = RAGPipeline(knowledge_base, llm, chat_model="qwen2.5:7b")

        response = pipeline.answer("Pertanyaan tanpa sumber")

        self.assertFalse(response.grounded)
        self.assertEqual(response.sources, ())
        self.assertIn("belum menemukan sumber", response.answer)
        self.assertEqual(llm.chat_calls, [])

    def test_rejects_empty_question_and_missing_chat_model(self):
        with self.assertRaises(ValueError):
            RAGPipeline(FakeKnowledgeBase([]), FakeLLM(), chat_model="")
        pipeline = RAGPipeline(FakeKnowledgeBase([]), FakeLLM(), chat_model="qwen2.5:7b")
        with self.assertRaises(ValueError):
            pipeline.answer("  ")

    def test_context_assembly_respects_configured_character_bound(self):
        result = source_result(text="Source text. " * 1000)
        knowledge_base = FakeKnowledgeBase([result])
        llm = FakeLLM("Answer [S1]")
        pipeline = RAGPipeline(
            knowledge_base,
            llm,
            chat_model="qwen2.5:7b",
            max_context_chars=500,
        )

        pipeline.answer("Explain this source")

        user_message = llm.chat_calls[0][0][1]["content"]
        context = user_message.rsplit("\n\n", 1)[-1]
        self.assertLessEqual(len(context), 500)

    def test_answer_streams_and_includes_recent_conversation_history(self):
        knowledge_base = FakeKnowledgeBase([source_result()])
        llm = FakeLLM("Unused")
        pipeline = RAGPipeline(knowledge_base, llm, chat_model="qwen2.5:7b")
        deltas = []
        cancel_event = threading.Event()

        response = pipeline.answer(
            "Lalu bagaimana dengan modus?",
            conversation_history=[
                {"role": "user", "content": "Apa itu median?"},
                {"role": "assistant", "content": "Median adalah nilai tengah."},
            ],
            on_delta=deltas.append,
            cancel_event=cancel_event,
        )

        self.assertEqual(deltas, ["Jawaban"])
        self.assertTrue(response.cancelled)
        messages = llm.chat_calls[0][0]
        self.assertEqual(messages[1]["content"], "Apa itu median?")
        self.assertEqual(messages[2]["content"], "Median adalah nilai tengah.")
        self.assertIn("Lalu bagaimana", messages[-1]["content"])

    def test_read_only_dora_tool_can_ground_dataset_answer_without_dkb_source(self):
        knowledge_base = FakeKnowledgeBase([])
        llm = FakeLLM()
        registry = build_dora_tool_registry(
            {"active_dataset": {"name": "scores.csv", "rows": 42, "columns": 2}},
            knowledge_base,
        )
        pipeline = RAGPipeline(
            knowledge_base,
            llm,
            chat_model="qwen2.5:7b",
            tool_registry=registry,
        )
        deltas = []
        tool_names = []

        response = pipeline.answer(
            "Berapa baris dataset aktif?",
            on_delta=deltas.append,
            on_tool_call=lambda name, _arguments: tool_names.append(name),
        )

        self.assertTrue(response.grounded)
        self.assertEqual(response.sources, ())
        self.assertEqual(response.tool_results[0].name, "get_active_dataset")
        self.assertEqual(tool_names, ["get_active_dataset"])
        self.assertEqual(deltas, [response.answer])
        self.assertIn("42 baris", response.answer)

    def test_tool_loop_without_any_evidence_keeps_honest_no_source_response(self):
        knowledge_base = FakeKnowledgeBase([])
        llm = FakeLLM()
        registry = build_dora_tool_registry(None, knowledge_base)
        pipeline = RAGPipeline(
            knowledge_base,
            llm,
            chat_model="qwen2.5:7b",
            tool_registry=registry,
        )

        response = pipeline.answer("Berapa baris dataset aktif?")

        self.assertFalse(response.grounded)
        self.assertIn("belum menemukan sumber", response.answer)


if __name__ == "__main__":
    unittest.main()
