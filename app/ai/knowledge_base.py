"""Local document store and keyword retrieval for Dora Knowledge Base."""

from __future__ import annotations

import hashlib
import html
from html.parser import HTMLParser
import math
import re
import sqlite3
import struct
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


class KnowledgeBaseError(RuntimeError):
    """Raised when a document cannot be safely imported or indexed."""


@dataclass(frozen=True, slots=True)
class KnowledgeDocumentMetadata:
    title: str
    author_or_org: str = ""
    category: str = "dora_documentation"
    publication_year: int | None = None
    source_url: str = ""
    doi: str = ""
    license: str = ""
    version: str = ""
    verified: bool = False


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    document_id: str
    title: str
    author_or_org: str
    category: str
    publication_year: int | None
    source_url: str
    doi: str
    license: str
    version: str
    indexed_at: str
    verification_status: str
    status: str
    status_detail: str
    chunk_count: int
    source_name: str


@dataclass(frozen=True, slots=True)
class KnowledgeSearchResult:
    document_id: str
    chunk_id: int
    title: str
    author_or_org: str
    category: str
    publication_year: int | None
    source_url: str
    doi: str
    license: str
    version: str
    verification_status: str
    text: str
    page_or_section: str | None
    chunk_index: int
    relevance_score: float

    @property
    def citation(self) -> str:
        parts = [self.title]
        if self.author_or_org:
            parts.append(self.author_or_org)
        if self.publication_year is not None:
            parts.append(str(self.publication_year))
        if self.page_or_section:
            parts.append(self.page_or_section)
        if self.doi:
            parts.append(f"DOI: {self.doi}")
        elif self.source_url:
            parts.append(self.source_url)
        return ". ".join(parts)


@dataclass(frozen=True, slots=True)
class _TextSegment:
    text: str
    page_or_section: str | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    chunk_id: int
    document_id: str
    chunk_index: int
    page_or_section: str | None
    text: str


class _HTMLTextParser(HTMLParser):
    _IGNORED_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._IGNORED_TAGS:
            self._ignored_depth += 1
        elif tag.lower() in {"p", "div", "li", "br", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag.lower() in {"p", "div", "li", "br", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


class DoraKnowledgeBase:
    """Store source documents locally and index chunks with SQLite FTS5.

    Chunk boundaries use a character budget rather than a model tokenizer.
    The default 3,200-character chunk and 480-character overlap are a starting
    heuristic; exact token counts vary by language and model.
    """

    CATEGORIES = frozenset(
        {
            "data_science",
            "data_analysis",
            "statistics",
            "mathematics",
            "machine_learning",
            "deep_learning",
            "computer_vision",
            "natural_language_processing",
            "programming",
            "software_engineering",
            "database_engineering",
            "academic_research",
            "research_methodology",
            "research_gap",
            "research_novelty",
            "scientific_writing",
            "dora_documentation",
        }
    )
    SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".rst", ".html", ".htm", ".pdf", ".docx"})
    DEFAULT_CHUNK_SIZE_CHARS = 3200
    DEFAULT_OVERLAP_CHARS = 480
    DEFAULT_MAX_FILE_BYTES = 32 * 1024 * 1024
    MAX_PDF_PAGES = 5000
    MAX_DOCX_ENTRIES = 10000
    MAX_DOCX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
    MAX_EXACT_VECTOR_SEARCH = 10000

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        *,
        db_path: str | Path | None = None,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
        overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    ) -> None:
        if storage_dir is None:
            from app.core.config import get_data_dir

            storage_dir = get_data_dir() / "tanya_dora" / "knowledge"
        self.storage_dir = Path(storage_dir).expanduser().resolve()
        self.sources_dir = self.storage_dir / "sources"
        self.staging_dir = self.storage_dir / ".staging"
        self.db_path = Path(db_path).expanduser().resolve() if db_path else self.storage_dir / "knowledge.db"
        self.max_file_bytes = int(max_file_bytes)
        self.chunk_size_chars = int(chunk_size_chars)
        self.overlap_chars = int(overlap_chars)
        self._validate_chunk_settings()

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    author_or_org TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL,
                    publication_year INTEGER,
                    source_url TEXT NOT NULL DEFAULT '',
                    doi TEXT NOT NULL DEFAULT '',
                    license TEXT NOT NULL DEFAULT '',
                    version TEXT NOT NULL DEFAULT '',
                    verified INTEGER NOT NULL DEFAULT 0,
                    indexed_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    status_detail TEXT NOT NULL DEFAULT '',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    source_name TEXT NOT NULL,
                    source_relative_path TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    source_size_bytes INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_knowledge_documents_category
                ON knowledge_documents(category, status);

                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks USING fts5(
                    text,
                    title,
                    category,
                    author_or_org,
                    publication_year UNINDEXED,
                    document_id UNINDEXED,
                    chunk_index UNINDEXED,
                    page_or_section UNINDEXED,
                    tokenize = 'unicode61'
                );

                CREATE TABLE IF NOT EXISTS knowledge_embeddings (
                    chunk_id INTEGER NOT NULL,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    embedded_at TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, embedding_model)
                );

                CREATE INDEX IF NOT EXISTS idx_knowledge_embeddings_model
                ON knowledge_embeddings(embedding_model, document_id, chunk_index);
                """
            )
            connection.commit()
        except sqlite3.OperationalError as error:
            connection.rollback()
            if "fts5" in str(error).lower() or "virtual table" in str(error).lower():
                raise KnowledgeBaseError(
                    "This Python SQLite build does not provide FTS5, which the local knowledge index requires."
                ) from error
            raise
        finally:
            connection.close()

    def import_document(
        self,
        source_path: str | Path,
        metadata: KnowledgeDocumentMetadata | None = None,
    ) -> KnowledgeDocument:
        """Copy, extract, chunk, and index one user-selected source document."""
        source = Path(source_path).expanduser()
        if not source.is_file():
            raise KnowledgeBaseError("The selected document does not exist or is not a file.")
        suffix = source.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise KnowledgeBaseError(f"Unsupported document type. Supported extensions: {supported}.")

        try:
            file_size = source.stat().st_size
            if file_size > self.max_file_bytes:
                raise KnowledgeBaseError(
                    f"Document is larger than the configured limit of {self.max_file_bytes:,} bytes."
                )
            content = source.read_bytes()
        except OSError as error:
            raise KnowledgeBaseError(f"The document could not be read: {error}") from error
        if len(content) > self.max_file_bytes:
            raise KnowledgeBaseError(
                f"Document is larger than the configured limit of {self.max_file_bytes:,} bytes."
            )

        normalized_metadata = self._normalize_metadata(metadata, source.stem)
        document_id = uuid4().hex
        relative_path = f"sources/{document_id}{suffix}"
        staged_path = self.staging_dir / f"{document_id}{suffix}"
        stored_path = self._resolve_source_path(relative_path)
        stored_path_written = False
        try:
            staged_path.write_bytes(content)
            segments = self._extract_segments(staged_path)
            chunks = self._make_chunks(segments)
            if not chunks:
                raise KnowledgeBaseError("No readable text was found in the selected document.")

            staged_path.replace(stored_path)
            stored_path_written = True
            now = _utc_now()
            connection = self._connect()
            try:
                with connection:
                    self._insert_document(
                        connection,
                        document_id=document_id,
                        metadata=normalized_metadata,
                        indexed_at=now,
                        status="INDEXED",
                        status_detail="",
                        chunk_count=len(chunks),
                        source_name=source.name,
                        source_relative_path=relative_path,
                        source_sha256=hashlib.sha256(content).hexdigest(),
                        source_size_bytes=len(content),
                    )
                    self._insert_chunks(connection, document_id, normalized_metadata, chunks)
            except Exception:
                stored_path.unlink(missing_ok=True)
                raise
            finally:
                connection.close()
        except Exception:
            if stored_path_written:
                stored_path.unlink(missing_ok=True)
            raise
        finally:
            staged_path.unlink(missing_ok=True)

        result = self.get_document(document_id)
        if result is None:
            raise KnowledgeBaseError("The document was indexed but could not be loaded from storage.")
        return result

    def get_document(self, document_id: str) -> KnowledgeDocument | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM knowledge_documents WHERE document_id = ?",
                (str(document_id),),
            ).fetchone()
            return _document_from_row(row) if row else None
        finally:
            connection.close()

    def list_documents(
        self,
        *,
        category: str | None = None,
        status: str | None = None,
    ) -> tuple[KnowledgeDocument, ...]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if category:
            category = self._normalize_category(category)
            clauses.append("category = ?")
            parameters.append(category)
        if status:
            clauses.append("status = ?")
            parameters.append(str(status).upper())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        connection = self._connect()
        try:
            rows = connection.execute(
                f"SELECT * FROM knowledge_documents {where} ORDER BY indexed_at DESC, title COLLATE NOCASE",
                tuple(parameters),
            ).fetchall()
            return tuple(_document_from_row(row) for row in rows)
        finally:
            connection.close()

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 10,
    ) -> tuple[KnowledgeSearchResult, ...]:
        terms = re.findall(r"[\w]+", str(query or ""), flags=re.UNICODE)
        if not terms:
            return ()
        normalized_category = self._normalize_category(category) if category else None
        safe_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
        safe_limit = max(1, min(int(limit), 100))

        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT
                    d.document_id, d.title, d.author_or_org, d.category,
                    d.publication_year, d.source_url, d.doi, d.license,
                    d.version, d.verified, k.rowid AS chunk_id, k.text, k.page_or_section,
                    k.chunk_index, bm25(knowledge_chunks, 1.0, 2.0, 0.5, 0.5) AS rank_score
                FROM knowledge_chunks AS k
                JOIN knowledge_documents AS d ON d.document_id = k.document_id
                WHERE knowledge_chunks MATCH ?
                    AND d.status = 'INDEXED'
                    AND (? IS NULL OR d.category = ?)
                ORDER BY rank_score ASC, d.title COLLATE NOCASE, k.chunk_index
                LIMIT ?
                """,
                (safe_query, normalized_category, normalized_category, safe_limit),
            ).fetchall()
            return tuple(_search_result_from_row(row) for row in rows)
        finally:
            connection.close()

    def get_chunks(self, document_id: str) -> tuple[KnowledgeChunk, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT rowid AS chunk_id, document_id, chunk_index, page_or_section, text
                   FROM knowledge_chunks WHERE document_id = ? ORDER BY chunk_index""",
                (str(document_id),),
            ).fetchall()
            return tuple(
                KnowledgeChunk(
                    chunk_id=int(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    chunk_index=int(row["chunk_index"]),
                    page_or_section=row["page_or_section"],
                    text=str(row["text"]),
                )
                for row in rows
            )
        finally:
            connection.close()

    def embed_document(
        self,
        document_id: str,
        embedding_model: str,
        embedding_provider: Any,
        *,
        batch_size: int = 32,
    ) -> int:
        """Create and store local embeddings in bounded batches."""
        normalized_model = str(embedding_model or "").strip()
        if not normalized_model:
            raise ValueError("An installed local embedding model is required.")
        safe_batch_size = max(1, min(int(batch_size), 128))
        chunks = self.get_chunks(document_id)
        if not chunks:
            raise KnowledgeBaseError(f"Document '{document_id}' has no indexed chunks.")

        vectors: dict[int, tuple[float, ...]] = {}
        for start in range(0, len(chunks), safe_batch_size):
            batch = chunks[start : start + safe_batch_size]
            batch_vectors = embedding_provider.embed_texts(
                [chunk.text for chunk in batch],
                model=normalized_model,
            )
            if len(batch_vectors) != len(batch):
                raise KnowledgeBaseError("Embedding provider returned an incomplete batch.")
            for chunk, vector in zip(batch, batch_vectors):
                vectors[chunk.chunk_index] = _normalize_vector(vector)

        return self.save_embeddings(document_id, normalized_model, vectors)

    def save_embeddings(
        self,
        document_id: str,
        embedding_model: str,
        vectors: Mapping[int, Any],
    ) -> int:
        """Store one vector per current chunk for a specific model version."""
        normalized_model = str(embedding_model or "").strip()
        if not normalized_model:
            raise ValueError("An embedding model name is required.")
        chunks = self.get_chunks(document_id)
        if not chunks:
            raise KnowledgeBaseError(f"Document '{document_id}' has no indexed chunks.")

        normalized_vectors: dict[int, tuple[float, ...]] = {}
        for chunk_index, vector in vectors.items():
            try:
                normalized_index = int(chunk_index)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Embedding chunk indexes must be integers.") from error
            normalized_vectors[normalized_index] = _normalize_vector(vector)

        expected_indexes = {chunk.chunk_index for chunk in chunks}
        if set(normalized_vectors) != expected_indexes:
            raise KnowledgeBaseError("Embedding vectors must cover every current document chunk exactly once.")
        dimensions = {len(vector) for vector in normalized_vectors.values()}
        if len(dimensions) != 1:
            raise KnowledgeBaseError("Embedding vectors for one model must have consistent dimensions.")
        dimension = next(iter(dimensions))
        chunk_by_index = {chunk.chunk_index: chunk for chunk in chunks}
        embedded_at = _utc_now()

        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "DELETE FROM knowledge_embeddings WHERE document_id = ? AND embedding_model = ?",
                    (str(document_id), normalized_model),
                )
                for chunk_index, vector in normalized_vectors.items():
                    chunk = chunk_by_index[chunk_index]
                    try:
                        packed_vector = struct.pack(f"<{dimension}f", *vector)
                    except (OverflowError, struct.error) as error:
                        raise ValueError("Embedding values exceed the supported float32 range.") from error
                    connection.execute(
                        """INSERT INTO knowledge_embeddings (
                            chunk_id, document_id, chunk_index, embedding_model,
                            dimensions, vector, embedded_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            chunk.chunk_id,
                            str(document_id),
                            chunk_index,
                            normalized_model,
                            dimension,
                            packed_vector,
                            embedded_at,
                        ),
                    )
        finally:
            connection.close()
        return len(chunks)

    def search_hybrid(
        self,
        query: str,
        query_embedding: Any,
        embedding_model: str,
        *,
        category: str | None = None,
        limit: int = 10,
        candidate_limit: int = 100,
        semantic_weight: float = 0.7,
    ) -> tuple[KnowledgeSearchResult, ...]:
        """Fuse local FTS5 and exact cosine rankings with reciprocal rank fusion."""
        if not str(query or "").strip():
            raise ValueError("A non-empty query is required for hybrid search.")
        model_name = str(embedding_model or "").strip()
        if not model_name:
            raise ValueError("An embedding model name is required for hybrid search.")
        query_vector = _normalize_vector(query_embedding)
        try:
            semantic_weight = float(semantic_weight)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("semantic_weight must be between 0 and 1.") from error
        if not math.isfinite(semantic_weight) or not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1.")

        normalized_category = self._normalize_category(category) if category else None
        safe_limit = max(1, min(int(limit), 100))
        safe_candidates = max(safe_limit, min(int(candidate_limit), 100))

        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT
                       d.document_id, d.title, d.author_or_org, d.category,
                       d.publication_year, d.source_url, d.doi, d.license,
                       d.version, d.verified, k.rowid AS chunk_id, k.text,
                       k.page_or_section, k.chunk_index, e.dimensions, e.vector
                   FROM knowledge_chunks AS k
                   JOIN knowledge_documents AS d ON d.document_id = k.document_id
                   LEFT JOIN knowledge_embeddings AS e
                       ON e.chunk_id = k.rowid AND e.embedding_model = ?
                   WHERE d.status = 'INDEXED'
                       AND (? IS NULL OR d.category = ?)
                   ORDER BY d.title COLLATE NOCASE, k.chunk_index""",
                (model_name, normalized_category, normalized_category),
            ).fetchall()
        finally:
            connection.close()

        if not rows:
            return ()
        if len(rows) > self.MAX_EXACT_VECTOR_SEARCH:
            raise KnowledgeBaseError(
                f"Exact local vector search is limited to {self.MAX_EXACT_VECTOR_SEARCH:,} chunks."
            )
        missing_count = sum(row["vector"] is None for row in rows)
        if missing_count:
            raise KnowledgeBaseError(
                f"Embedding index for '{model_name}' is incomplete ({missing_count:,} chunk(s) missing)."
            )

        semantic_results: list[KnowledgeSearchResult] = []
        for row in rows:
            dimensions = int(row["dimensions"])
            if dimensions != len(query_vector):
                raise KnowledgeBaseError(
                    "Query and stored embedding dimensions do not match. Re-index with the same model."
                )
            packed = bytes(row["vector"])
            if len(packed) != dimensions * 4:
                raise KnowledgeBaseError("Stored embedding data is corrupt or has an invalid size.")
            vector = struct.unpack(f"<{dimensions}f", packed)
            score = _cosine_similarity(query_vector, vector)
            semantic_results.append(_search_result_from_embedding_row(row, score))
        semantic_results.sort(
            key=lambda result: (-result.relevance_score, result.title.casefold(), result.chunk_index)
        )
        semantic_results = semantic_results[:safe_candidates]

        keyword_results = self.search(query, category=normalized_category, limit=safe_candidates)
        semantic_rank = {result.chunk_id: index + 1 for index, result in enumerate(semantic_results)}
        keyword_rank = {result.chunk_id: index + 1 for index, result in enumerate(keyword_results)}
        by_chunk_id = {result.chunk_id: result for result in (*semantic_results, *keyword_results)}
        fused: list[KnowledgeSearchResult] = []
        for chunk_id, result in by_chunk_id.items():
            score = 0.0
            if chunk_id in semantic_rank:
                score += semantic_weight / (60 + semantic_rank[chunk_id])
            if chunk_id in keyword_rank:
                score += (1 - semantic_weight) / (60 + keyword_rank[chunk_id])
            fused.append(replace(result, relevance_score=score))
        fused.sort(key=lambda result: (-result.relevance_score, result.title.casefold(), result.chunk_index))
        return tuple(fused[:safe_limit])

    def reindex_document(self, document_id: str) -> KnowledgeDocument:
        """Re-extract a managed source file while preserving its metadata."""
        document = self.get_document(document_id)
        if document is None:
            raise KnowledgeBaseError(f"Document '{document_id}' was not found.")

        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM knowledge_documents WHERE document_id = ?",
                (str(document_id),),
            ).fetchone()
        finally:
            connection.close()
        source_relative_path = str(row["source_relative_path"])
        source_path = self._resolve_source_path(source_relative_path)

        try:
            source_size_bytes = source_path.stat().st_size
            if source_size_bytes > self.max_file_bytes:
                raise KnowledgeBaseError(
                    f"Stored document is larger than the configured limit of {self.max_file_bytes:,} bytes."
                )
            source_content = source_path.read_bytes()
            if len(source_content) > self.max_file_bytes:
                raise KnowledgeBaseError(
                    f"Stored document is larger than the configured limit of {self.max_file_bytes:,} bytes."
                )
            source_size_bytes = len(source_content)
            source_sha256 = hashlib.sha256(source_content).hexdigest()
            segments = self._extract_segments(source_path)
            chunks = self._make_chunks(segments)
            if not chunks:
                raise KnowledgeBaseError("No readable text was found in the stored source document.")

            metadata = self._metadata_from_row(row)
            connection = self._connect()
            try:
                with connection:
                    connection.execute(
                        "DELETE FROM knowledge_embeddings WHERE document_id = ?",
                        (str(document_id),),
                    )
                    connection.execute(
                        "DELETE FROM knowledge_chunks WHERE document_id = ?",
                        (str(document_id),),
                    )
                    self._insert_chunks(connection, str(document_id), metadata, chunks)
                    connection.execute(
                        """UPDATE knowledge_documents
                           SET indexed_at = ?, status = 'INDEXED', status_detail = '', chunk_count = ?,
                               source_sha256 = ?, source_size_bytes = ?
                           WHERE document_id = ?""",
                        (_utc_now(), len(chunks), source_sha256, source_size_bytes, str(document_id)),
                    )
            finally:
                connection.close()
        except Exception as error:
            connection = self._connect()
            try:
                with connection:
                    connection.execute(
                        "UPDATE knowledge_documents SET status = 'FAILED', status_detail = ? WHERE document_id = ?",
                        (str(error)[:500], str(document_id)),
                    )
            finally:
                connection.close()
            if isinstance(error, KnowledgeBaseError):
                raise
            raise KnowledgeBaseError(f"Re-indexing failed: {error}") from error

        refreshed = self.get_document(str(document_id))
        if refreshed is None:
            raise KnowledgeBaseError("The document was re-indexed but could not be loaded from storage.")
        return refreshed

    def delete_document(self, document_id: str) -> bool:
        """Remove a document's indexed chunks, metadata, and managed source."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT source_relative_path FROM knowledge_documents WHERE document_id = ?",
                (str(document_id),),
            ).fetchone()
            if row is None:
                return False
            source_path = self._resolve_source_path(str(row["source_relative_path"]))
            with connection:
                connection.execute(
                    "DELETE FROM knowledge_embeddings WHERE document_id = ?",
                    (str(document_id),),
                )
                connection.execute("DELETE FROM knowledge_chunks WHERE document_id = ?", (str(document_id),))
                connection.execute("DELETE FROM knowledge_documents WHERE document_id = ?", (str(document_id),))
        finally:
            connection.close()

        try:
            source_path.unlink(missing_ok=True)
        except OSError as error:
            raise KnowledgeBaseError(
                f"Document metadata was removed, but the managed source file could not be deleted: {error}"
            ) from error
        return True

    def update_metadata(
        self,
        document_id: str,
        metadata: KnowledgeDocumentMetadata,
    ) -> KnowledgeDocument:
        """Update source metadata and refresh the text-search fields."""
        normalized = self._normalize_metadata(metadata, metadata.title)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT 1 FROM knowledge_documents WHERE document_id = ?",
                (str(document_id),),
            ).fetchone()
            if row is None:
                raise KnowledgeBaseError(f"Document '{document_id}' was not found.")
            with connection:
                connection.execute(
                    """UPDATE knowledge_documents SET title = ?, author_or_org = ?, category = ?,
                       publication_year = ?, source_url = ?, doi = ?, license = ?, version = ?, verified = ?
                       WHERE document_id = ?""",
                    (
                        normalized.title,
                        normalized.author_or_org,
                        normalized.category,
                        normalized.publication_year,
                        normalized.source_url,
                        normalized.doi,
                        normalized.license,
                        normalized.version,
                        int(normalized.verified),
                        str(document_id),
                    ),
                )
                connection.execute(
                    """UPDATE knowledge_chunks SET title = ?, category = ?, author_or_org = ?, publication_year = ?
                       WHERE document_id = ?""",
                    (
                        normalized.title,
                        normalized.category,
                        normalized.author_or_org,
                        normalized.publication_year,
                        str(document_id),
                    ),
                )
        finally:
            connection.close()

        refreshed = self.get_document(str(document_id))
        if refreshed is None:
            raise KnowledgeBaseError("Updated document metadata could not be loaded from storage.")
        return refreshed

    def _insert_document(
        self,
        connection: sqlite3.Connection,
        *,
        document_id: str,
        metadata: KnowledgeDocumentMetadata,
        indexed_at: str,
        status: str,
        status_detail: str,
        chunk_count: int,
        source_name: str,
        source_relative_path: str,
        source_sha256: str,
        source_size_bytes: int,
    ) -> None:
        connection.execute(
            """INSERT INTO knowledge_documents (
                document_id, title, author_or_org, category, publication_year,
                source_url, doi, license, version, verified, indexed_at, status,
                status_detail, chunk_count, source_name, source_relative_path,
                source_sha256, source_size_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document_id,
                metadata.title,
                metadata.author_or_org,
                metadata.category,
                metadata.publication_year,
                metadata.source_url,
                metadata.doi,
                metadata.license,
                metadata.version,
                int(metadata.verified),
                indexed_at,
                status,
                status_detail,
                chunk_count,
                source_name,
                source_relative_path,
                source_sha256,
                source_size_bytes,
            ),
        )

    @staticmethod
    def _insert_chunks(
        connection: sqlite3.Connection,
        document_id: str,
        metadata: KnowledgeDocumentMetadata,
        chunks: list[tuple[str, str | None]],
    ) -> tuple[KnowledgeChunk, ...]:
        inserted: list[KnowledgeChunk] = []
        for chunk_index, (text, page_or_section) in enumerate(chunks):
            cursor = connection.execute(
                """INSERT INTO knowledge_chunks (
                    text, title, category, author_or_org, publication_year,
                    document_id, chunk_index, page_or_section
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    text,
                    metadata.title,
                    metadata.category,
                    metadata.author_or_org,
                    metadata.publication_year,
                    document_id,
                    chunk_index,
                    page_or_section,
                ),
            )
            inserted.append(
                KnowledgeChunk(
                    chunk_id=int(cursor.lastrowid),
                    document_id=document_id,
                    chunk_index=chunk_index,
                    page_or_section=page_or_section,
                    text=text,
                )
            )
        return tuple(inserted)

    def _extract_segments(self, path: Path) -> list[_TextSegment]:
        suffix = path.suffix.lower()
        try:
            if suffix in {".txt", ".md", ".rst"}:
                text = path.read_text(encoding="utf-8-sig", errors="replace")
                return [_TextSegment(text)]
            if suffix in {".html", ".htm"}:
                parser = _HTMLTextParser()
                parser.feed(path.read_text(encoding="utf-8-sig", errors="replace"))
                return [_TextSegment(html.unescape(" ".join(parser.parts)))]
            if suffix == ".pdf":
                return self._extract_pdf(path)
            if suffix == ".docx":
                return self._extract_docx(path)
        except KnowledgeBaseError:
            raise
        except Exception as error:
            raise KnowledgeBaseError(f"Could not extract text from '{path.name}': {error}") from error
        raise KnowledgeBaseError(f"Unsupported document type: {suffix or '(no extension)'}.")

    def _extract_pdf(self, path: Path) -> list[_TextSegment]:
        try:
            import pymupdf
        except ImportError as error:
            raise KnowledgeBaseError(
                "PDF import requires PyMuPDF. Install the optional dependencies in requirements_ai.txt."
            ) from error

        segments: list[_TextSegment] = []
        with pymupdf.open(path) as document:
            if getattr(document, "is_encrypted", False):
                raise KnowledgeBaseError("Encrypted PDFs cannot be indexed.")
            if document.page_count > self.MAX_PDF_PAGES:
                raise KnowledgeBaseError(
                    f"PDF exceeds the page limit of {self.MAX_PDF_PAGES:,}."
                )
            for page_index, page in enumerate(document, start=1):
                segments.append(_TextSegment(page.get_text("text"), f"Page {page_index}"))
        return segments

    @staticmethod
    def _extract_docx(path: Path) -> list[_TextSegment]:
        try:
            from docx import Document
        except ImportError as error:
            raise KnowledgeBaseError(
                "DOCX import requires python-docx. Install the optional dependencies in requirements_ai.txt."
            ) from error

        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                if len(entries) > DoraKnowledgeBase.MAX_DOCX_ENTRIES:
                    raise KnowledgeBaseError(
                        f"DOCX contains more than {DoraKnowledgeBase.MAX_DOCX_ENTRIES:,} archive entries."
                    )
                expanded_size = sum(entry.file_size for entry in entries)
                if expanded_size > DoraKnowledgeBase.MAX_DOCX_UNCOMPRESSED_BYTES:
                    raise KnowledgeBaseError(
                        "DOCX expands beyond the configured 128 MiB extraction limit."
                    )
        except zipfile.BadZipFile as error:
            raise KnowledgeBaseError("The DOCX file is not a valid Office document archive.") from error

        document = Document(str(path))
        parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cell_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cell_text:
                    parts.append(" | ".join(cell_text))
        return [_TextSegment("\n".join(parts))]

    def _make_chunks(self, segments: list[_TextSegment]) -> list[tuple[str, str | None]]:
        chunks: list[tuple[str, str | None]] = []
        for segment in segments:
            text = re.sub(r"\s+", " ", str(segment.text or "")).strip()
            if not text:
                continue
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size_chars, len(text))
                if end < len(text):
                    break_at = text.rfind(" ", start + self.chunk_size_chars // 2, end)
                    if break_at > start:
                        end = break_at
                piece = text[start:end].strip()
                if piece:
                    chunks.append((piece, segment.page_or_section))
                if end >= len(text):
                    break
                start = max(start + 1, end - self.overlap_chars)
                while start < len(text) and text[start].isspace():
                    start += 1
        return chunks

    def _resolve_source_path(self, relative_path: str) -> Path:
        candidate = (self.storage_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.sources_dir.resolve())
        except ValueError as error:
            raise KnowledgeBaseError("Stored document path is outside the managed source directory.") from error
        return candidate

    def _validate_chunk_settings(self) -> None:
        if self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be greater than zero.")
        if self.chunk_size_chars < 100:
            raise ValueError("chunk_size_chars must be at least 100.")
        if self.overlap_chars < 0 or self.overlap_chars >= self.chunk_size_chars:
            raise ValueError("overlap_chars must be non-negative and smaller than chunk_size_chars.")

    @classmethod
    def _normalize_category(cls, category: str) -> str:
        normalized = str(category or "").strip().lower()
        if normalized not in cls.CATEGORIES:
            raise ValueError(f"Unknown Dora Knowledge Base category: {category!r}.")
        return normalized

    @classmethod
    def _normalize_metadata(
        cls,
        metadata: KnowledgeDocumentMetadata | None,
        default_title: str,
    ) -> KnowledgeDocumentMetadata:
        metadata = metadata or KnowledgeDocumentMetadata(title=default_title)
        category = cls._normalize_category(metadata.category)
        title = str(metadata.title or default_title).strip()
        if not title:
            raise ValueError("Document title cannot be empty.")
        year = metadata.publication_year
        if year is not None:
            try:
                year = int(year)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Publication year must be an integer.") from error
            if year < 1000 or year > 9999:
                raise ValueError("Publication year must be between 1000 and 9999.")
        return KnowledgeDocumentMetadata(
            title=title,
            author_or_org=str(metadata.author_or_org or "").strip(),
            category=category,
            publication_year=year,
            source_url=str(metadata.source_url or "").strip(),
            doi=str(metadata.doi or "").strip(),
            license=str(metadata.license or "").strip(),
            version=str(metadata.version or "").strip(),
            verified=bool(metadata.verified),
        )

    @staticmethod
    def _metadata_from_row(row: sqlite3.Row) -> KnowledgeDocumentMetadata:
        return KnowledgeDocumentMetadata(
            title=str(row["title"]),
            author_or_org=str(row["author_or_org"]),
            category=str(row["category"]),
            publication_year=row["publication_year"],
            source_url=str(row["source_url"]),
            doi=str(row["doi"]),
            license=str(row["license"]),
            version=str(row["version"]),
            verified=bool(row["verified"]),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _document_from_row(row: sqlite3.Row) -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id=str(row["document_id"]),
        title=str(row["title"]),
        author_or_org=str(row["author_or_org"]),
        category=str(row["category"]),
        publication_year=row["publication_year"],
        source_url=str(row["source_url"]),
        doi=str(row["doi"]),
        license=str(row["license"]),
        version=str(row["version"]),
        indexed_at=str(row["indexed_at"]),
        verification_status="VERIFIED" if row["verified"] else "UNVERIFIED",
        status=str(row["status"]),
        status_detail=str(row["status_detail"]),
        chunk_count=int(row["chunk_count"]),
        source_name=str(row["source_name"]),
    )


def _search_result_from_row(row: sqlite3.Row) -> KnowledgeSearchResult:
    return KnowledgeSearchResult(
        document_id=str(row["document_id"]),
        chunk_id=int(row["chunk_id"]),
        title=str(row["title"]),
        author_or_org=str(row["author_or_org"]),
        category=str(row["category"]),
        publication_year=row["publication_year"],
        source_url=str(row["source_url"]),
        doi=str(row["doi"]),
        license=str(row["license"]),
        version=str(row["version"]),
        verification_status="VERIFIED" if row["verified"] else "UNVERIFIED",
        text=str(row["text"]),
        page_or_section=row["page_or_section"],
        chunk_index=int(row["chunk_index"]),
        relevance_score=float(-row["rank_score"]),
    )


def _normalize_vector(values: Any) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("Embedding vectors must be numeric sequences.")
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Embedding vectors must be numeric sequences.") from error
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("Embedding vectors must contain finite numeric values.")
    return vector


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise KnowledgeBaseError("Cannot compare embeddings with different dimensions.")
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = math.fsum(first * second for first, second in zip(left, right))
    return dot / (left_norm * right_norm)


def _search_result_from_embedding_row(
    row: sqlite3.Row,
    score: float,
) -> KnowledgeSearchResult:
    return KnowledgeSearchResult(
        document_id=str(row["document_id"]),
        chunk_id=int(row["chunk_id"]),
        title=str(row["title"]),
        author_or_org=str(row["author_or_org"]),
        category=str(row["category"]),
        publication_year=row["publication_year"],
        source_url=str(row["source_url"]),
        doi=str(row["doi"]),
        license=str(row["license"]),
        version=str(row["version"]),
        verification_status="VERIFIED" if row["verified"] else "UNVERIFIED",
        text=str(row["text"]),
        page_or_section=row["page_or_section"],
        chunk_index=int(row["chunk_index"]),
        relevance_score=float(score),
    )
