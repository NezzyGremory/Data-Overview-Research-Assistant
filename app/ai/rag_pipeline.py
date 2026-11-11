"""Source-grounded answer assembly for Tanya DORA."""

from __future__ import annotations

from dataclasses import dataclass
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from app.ai.knowledge_base import DoraKnowledgeBase, KnowledgeSearchResult
from app.ai.tool_calling import DoraToolRegistry, ToolExecution


@dataclass(frozen=True, slots=True)
class RAGSource:
    source_id: str
    result: KnowledgeSearchResult

    @property
    def citation(self) -> str:
        return self.result.citation


@dataclass(frozen=True, slots=True)
class RAGResponse:
    answer: str
    sources: tuple[RAGSource, ...]
    grounded: bool
    retrieval_mode: str
    model: str
    cancelled: bool = False
    tool_results: tuple[ToolExecution, ...] = ()


class RAGPipeline:
    """Retrieve DKB chunks, assemble a bounded prompt, and call local Ollama.

    When retrieval returns no source, this pipeline returns that fact directly
    instead of asking a language model to fill the gap from unverified memory.
    """

    DEFAULT_SYSTEM_PROMPT = """Anda adalah Tanya DORA, asisten riset dan Data Science.
Jawab dalam Bahasa Indonesia yang jelas dan akurat. Gunakan hanya fakta dari konteks sumber yang diberikan.
Gunakan juga hasil tool DORA yang benar-benar tersedia; tool DORA adalah sumber angka analisis dataset.
Jangan mengarang angka, paper, DOI, URL, kutipan, atau hasil eksperimen.
Gunakan penanda sumber [S1], [S2], dan seterusnya hanya untuk sumber yang sesuai.
Jika konteks tidak cukup untuk menjawab, nyatakan bagian yang belum didukung sumber.
Jangan mengklaim bahwa jawaban berasal dari sumber yang tidak ada dalam konteks."""

    NO_SOURCE_ANSWER = (
        "DKB belum menemukan sumber terindeks yang relevan untuk pertanyaan ini. "
        "Saya belum dapat memberikan jawaban berbasis sumber."
    )

    def __init__(
        self,
        knowledge_base: DoraKnowledgeBase,
        llm_manager: Any,
        *,
        chat_model: str,
        embedding_model: str | None = None,
        temperature: float = 0.2,
        max_context_chars: int = 12000,
        system_prompt: str | None = None,
        tool_registry: DoraToolRegistry | None = None,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.llm_manager = llm_manager
        self.chat_model = str(chat_model or "").strip()
        self.embedding_model = str(embedding_model or "").strip() or None
        self.temperature = temperature
        self.max_context_chars = int(max_context_chars)
        self.system_prompt = str(system_prompt or self.DEFAULT_SYSTEM_PROMPT).strip()
        self.tool_registry = tool_registry
        if not self.chat_model:
            raise ValueError("An installed local chat model is required.")
        if self.max_context_chars < 500:
            raise ValueError("max_context_chars must be at least 500.")

    def answer(
        self,
        question: str,
        *,
        category: str | None = None,
        max_sources: int = 5,
        conversation_history: Sequence[Mapping[str, str]] = (),
        on_delta: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, Mapping[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> RAGResponse:
        normalized_question = str(question or "").strip()
        if not normalized_question:
            raise ValueError("Question must contain text.")
        safe_source_count = max(1, min(int(max_sources), 10))

        if self.embedding_model:
            query_embedding = self.llm_manager.embed_texts(
                [normalized_question],
                model=self.embedding_model,
            )[0]
            results = self.knowledge_base.search_hybrid(
                normalized_question,
                query_embedding,
                self.embedding_model,
                category=category,
                limit=safe_source_count,
                candidate_limit=max(safe_source_count, min(100, safe_source_count * 5)),
            )
            retrieval_mode = "hybrid_fts_cosine"
        else:
            results = self.knowledge_base.search(
                normalized_question,
                category=category,
                limit=safe_source_count,
            )
            retrieval_mode = "keyword_fts5"

        if cancel_event is not None and cancel_event.is_set():
            return RAGResponse(
                answer="",
                sources=(),
                grounded=False,
                retrieval_mode=retrieval_mode,
                model=self.chat_model,
                cancelled=True,
            )

        has_tools = self.tool_registry is not None and self.tool_registry.has_tools()
        if not results and not has_tools:
            return RAGResponse(
                answer=self.NO_SOURCE_ANSWER,
                sources=(),
                grounded=False,
                retrieval_mode=retrieval_mode,
                model=self.chat_model,
            )

        sources = tuple(
            RAGSource(source_id=f"S{index}", result=result)
            for index, result in enumerate(results, start=1)
        )
        context = self._assemble_context(sources) if sources else "DKB tidak menemukan sumber relevan."
        user_content = (
            f"Pertanyaan pengguna:\n{normalized_question}\n\n"
            "Gunakan hanya bukti dari chunk DKB yang disertakan dan hasil tool lokal yang benar-benar "
            "dipanggil. Angka dataset hanya boleh berasal dari tool DORA. Jangan menebak jika bukti "
            "tidak tersedia. Cantumkan penanda [S#] untuk sumber DKB yang relevan.\n\n"
            f"{context}"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        messages.extend(self._normalize_history(conversation_history))
        messages.append({"role": "user", "content": user_content})
        if cancel_event is not None and cancel_event.is_set():
            return RAGResponse(
                answer="",
                sources=sources,
                grounded=True,
                retrieval_mode=retrieval_mode,
                model=self.chat_model,
                cancelled=True,
            )
        tool_results: tuple[ToolExecution, ...] = ()
        if has_tools:
            tool_response = self.llm_manager.chat_with_tools(
                messages,
                model=self.chat_model,
                temperature=self.temperature,
                on_delta=on_delta if sources else None,
                on_tool_call=on_tool_call,
                cancel_event=cancel_event,
                tool_registry=self.tool_registry,
            )
            generated = tool_response.answer
            tool_results = tool_response.executions
            cancelled = tool_response.cancelled
        elif on_delta is not None:
            generated = self.llm_manager.chat_stream(
                messages,
                model=self.chat_model,
                temperature=self.temperature,
                on_delta=on_delta,
                cancel_event=cancel_event,
            )
            cancelled = bool(cancel_event is not None and cancel_event.is_set())
        else:
            generated = self.llm_manager.chat(
                messages,
                model=self.chat_model,
                temperature=self.temperature,
            )
            cancelled = bool(cancel_event is not None and cancel_event.is_set())
        answer = self._remove_unknown_source_markers(str(generated or ""), sources).strip()
        grounded = bool(sources) or any(item.provides_evidence for item in tool_results)
        has_tool_evidence = any(item.provides_evidence for item in tool_results)
        if cancelled and not sources and not has_tool_evidence:
            answer = ""
        elif not sources and not has_tool_evidence:
            answer = self.NO_SOURCE_ANSWER
        elif not answer and not cancelled:
            answer = "Model lokal tidak menghasilkan jawaban teks untuk sumber yang ditemukan."
        if not sources and grounded and on_delta is not None and answer:
            on_delta(answer)
        return RAGResponse(
            answer=answer,
            sources=sources,
            grounded=grounded,
            retrieval_mode=retrieval_mode,
            model=self.chat_model,
            cancelled=cancelled,
            tool_results=tool_results,
        )

    @staticmethod
    def _normalize_history(
        history: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
            raise ValueError("Conversation history must be a sequence of messages.")
        normalized: list[dict[str, str]] = []
        for entry in history[-8:]:
            if not isinstance(entry, Mapping):
                continue
            role = str(entry.get("role") or "").strip()
            content = entry.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            content = content.strip()
            if content:
                normalized.append({"role": role, "content": content[:4000]})
        return normalized

    def _assemble_context(self, sources: tuple[RAGSource, ...]) -> str:
        sections: list[str] = []
        remaining = self.max_context_chars
        for source in sources:
            result = source.result
            metadata = [
                f"Judul: {result.title}",
                f"Penulis/organisasi: {result.author_or_org or 'Tidak dicatat'}",
                f"Tahun: {result.publication_year if result.publication_year is not None else 'Tidak dicatat'}",
                f"Kategori: {result.category}",
                f"Verifikasi: {result.verification_status}",
            ]
            if result.page_or_section:
                metadata.append(f"Halaman/bagian: {result.page_or_section}")
            if result.doi:
                metadata.append(f"DOI: {result.doi}")
            elif result.source_url:
                metadata.append(f"URL: {result.source_url}")
            header = f"[{source.source_id}] " + " | ".join(metadata) + "\nIsi: "
            available = remaining - len(header) - 2
            if available <= 0:
                break
            excerpt = result.text[:available].rstrip()
            section = header + excerpt
            sections.append(section)
            remaining -= len(section) + 2
            if remaining <= 0:
                break
        return "\n\n".join(sections)

    @staticmethod
    def _remove_unknown_source_markers(
        answer: str,
        sources: tuple[RAGSource, ...],
    ) -> str:
        allowed = {source.source_id for source in sources}
        return re.sub(
            r"\[(S\d+)\]",
            lambda match: match.group(0) if match.group(1) in allowed else "",
            answer,
        )
