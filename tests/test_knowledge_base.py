from __future__ import annotations

import hashlib
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from app.ai.knowledge_base import (
    DoraKnowledgeBase,
    KnowledgeBaseError,
    KnowledgeDocumentMetadata,
)


class TestDoraKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.kb = DoraKnowledgeBase(
            self.root / "kb",
            chunk_size_chars=120,
            overlap_chars=20,
        )

    def write_source(self, filename="stats.md", content="""# Statistics
Median is the middle value after observations are ordered. A median is robust
to extreme values and can describe a skewed distribution. Mean is the arithmetic
average, and it can move when outliers are present. Quartiles summarize spread.
"""):
        path = self.root / filename
        path.write_text(content, encoding="utf-8")
        return path

    def test_imports_markdown_and_persists_document_metadata(self):
        source = self.write_source()
        metadata = KnowledgeDocumentMetadata(
            title="Descriptive Statistics",
            author_or_org="DORA Docs",
            category="statistics",
            publication_year=2026,
            source_url="https://example.test/statistics",
            license="CC BY 4.0",
            version="1.0",
            verified=True,
        )

        document = self.kb.import_document(source, metadata)

        self.assertEqual(document.title, "Descriptive Statistics")
        self.assertEqual(document.category, "statistics")
        self.assertEqual(document.publication_year, 2026)
        self.assertEqual(document.verification_status, "VERIFIED")
        self.assertEqual(document.status, "INDEXED")
        self.assertGreaterEqual(document.chunk_count, 1)
        self.assertTrue((self.root / "kb" / "sources" / f"{document.document_id}.md").is_file())

    def test_search_returns_real_chunk_and_source_citation(self):
        source = self.write_source()
        document = self.kb.import_document(
            source,
            KnowledgeDocumentMetadata(
                title="Descriptive Statistics",
                author_or_org="DORA Docs",
                category="statistics",
                publication_year=2026,
                doi="10.1234/dora.2026.1",
            ),
        )

        results = self.kb.search("median outliers", category="statistics")

        self.assertTrue(results)
        self.assertEqual(results[0].document_id, document.document_id)
        self.assertIn("median", results[0].text.lower())
        self.assertIn("DORA Docs", results[0].citation)
        self.assertIn("10.1234/dora.2026.1", results[0].citation)
        self.assertEqual(results[0].verification_status, "UNVERIFIED")

    def test_category_filter_and_empty_query(self):
        self.kb.import_document(
            self.write_source("stats.md"),
            KnowledgeDocumentMetadata(title="Statistics", category="statistics"),
        )
        self.kb.import_document(
            self.write_source("programming.md", "Python functions can group reusable operations."),
            KnowledgeDocumentMetadata(title="Python", category="programming"),
        )

        results = self.kb.search("python median")

        self.assertEqual({result.category for result in results}, {"statistics", "programming"})
        self.assertEqual(self.kb.search("---"), ())
        self.assertEqual(self.kb.search("median", category="statistics")[0].category, "statistics")

    def test_html_extraction_omits_script_and_style_content(self):
        source = self.write_source(
            "guide.html",
            "<html><style>hidden style</style><body><h1>Local search</h1>"
            "<p>Visible knowledge text.</p><script>secret script</script></body></html>",
        )
        self.kb.import_document(
            source,
            KnowledgeDocumentMetadata(title="Local Search", category="dora_documentation"),
        )

        results = self.kb.search("visible knowledge")

        self.assertTrue(results)
        indexed_text = " ".join(result.text for result in results)
        self.assertNotIn("hidden", indexed_text)
        self.assertNotIn("secret", indexed_text)

    def test_reindex_reads_managed_copy_and_replaces_search_chunks(self):
        source = self.write_source(content="Original phrase for index refresh.")
        document = self.kb.import_document(
            source,
            KnowledgeDocumentMetadata(title="Re-index", category="software_engineering"),
        )
        self.kb.save_embeddings(document.document_id, "embed:test", {0: (1.0, 0.0)})
        managed_source = self.root / "kb" / "sources" / f"{document.document_id}.md"
        with closing(sqlite3.connect(self.kb.db_path)) as connection:
            old_hash = connection.execute(
                "SELECT source_sha256 FROM knowledge_documents WHERE document_id = ?",
                (document.document_id,),
            ).fetchone()[0]
        managed_source.write_text("Updated phrase for index refresh.", encoding="utf-8")

        refreshed = self.kb.reindex_document(document.document_id)

        self.assertEqual(refreshed.status, "INDEXED")
        self.assertEqual(self.kb.search("original"), ())
        self.assertTrue(self.kb.search("updated phrase"))
        with closing(sqlite3.connect(self.kb.db_path)) as connection:
            new_hash = connection.execute(
                "SELECT source_sha256 FROM knowledge_documents WHERE document_id = ?",
                (document.document_id,),
            ).fetchone()[0]
        self.assertNotEqual(new_hash, old_hash)
        self.assertEqual(new_hash, hashlib.sha256(managed_source.read_bytes()).hexdigest())
        with self.assertRaisesRegex(KnowledgeBaseError, "incomplete"):
            self.kb.search_hybrid("updated", (1.0, 0.0), "embed:test")

    def test_hybrid_search_ranks_semantic_match_and_requires_complete_model_coverage(self):
        statistics = self.kb.import_document(
            self.write_source("stats.md", "Medians describe a robust center for skewed data."),
            KnowledgeDocumentMetadata(title="Statistics", category="statistics"),
        )
        programming = self.kb.import_document(
            self.write_source("python.md", "Python functions organize reusable operations."),
            KnowledgeDocumentMetadata(title="Programming", category="programming"),
        )
        self.kb.save_embeddings(statistics.document_id, "embed:test", {0: (1.0, 0.0)})

        with self.assertRaisesRegex(KnowledgeBaseError, "incomplete"):
            self.kb.search_hybrid("unmatchedterm", (1.0, 0.0), "embed:test")

        self.kb.save_embeddings(programming.document_id, "embed:test", {0: (0.0, 1.0)})
        results = self.kb.search_hybrid(
            "unmatchedterm",
            (1.0, 0.0),
            "embed:test",
            limit=2,
        )

        self.assertEqual(results[0].document_id, statistics.document_id)
        self.assertEqual(results[1].document_id, programming.document_id)

    def test_embed_document_uses_bounded_provider_batches(self):
        source = self.write_source(
            "long.md",
            " ".join(f"token{index}" for index in range(80)),
        )
        document = self.kb.import_document(
            source,
            KnowledgeDocumentMetadata(title="Long Guide", category="programming"),
        )

        class Provider:
            def __init__(self):
                self.batch_sizes = []

            def embed_texts(self, texts, *, model):
                self.batch_sizes.append(len(texts))
                return [(float(len(text)), 1.0) for text in texts]

        provider = Provider()
        embedded_count = self.kb.embed_document(
            document.document_id,
            "embed:test",
            provider,
            batch_size=2,
        )

        self.assertEqual(embedded_count, document.chunk_count)
        self.assertGreater(len(provider.batch_sizes), 1)
        self.assertLessEqual(max(provider.batch_sizes), 2)

    def test_update_metadata_refreshes_search_filters_and_verification(self):
        document = self.kb.import_document(
            self.write_source(),
            KnowledgeDocumentMetadata(title="Old title", category="data_science"),
        )

        updated = self.kb.update_metadata(
            document.document_id,
            KnowledgeDocumentMetadata(
                title="Verified Statistics Guide",
                author_or_org="NIST",
                category="statistics",
                publication_year=2024,
                source_url="https://example.test/nist",
                license="Public domain",
                verified=True,
            ),
        )

        results = self.kb.search("median", category="statistics")
        self.assertEqual(updated.title, "Verified Statistics Guide")
        self.assertEqual(updated.verification_status, "VERIFIED")
        self.assertTrue(results)
        self.assertEqual(results[0].author_or_org, "NIST")
        self.assertEqual(self.kb.search("median", category="data_science"), ())

    def test_delete_removes_document_source_and_search_results(self):
        document = self.kb.import_document(
            self.write_source(),
            KnowledgeDocumentMetadata(title="Delete me", category="statistics"),
        )
        managed_source = self.root / "kb" / "sources" / f"{document.document_id}.md"

        self.assertTrue(self.kb.delete_document(document.document_id))

        self.assertFalse(managed_source.exists())
        self.assertIsNone(self.kb.get_document(document.document_id))
        self.assertEqual(self.kb.search("median"), ())
        self.assertFalse(self.kb.delete_document(document.document_id))

    def test_rejects_unsupported_files_oversized_files_and_invalid_categories(self):
        source = self.write_source("archive.zip", "not a document")
        with self.assertRaises(KnowledgeBaseError):
            self.kb.import_document(source)

        large_kb = DoraKnowledgeBase(self.root / "small-kb", max_file_bytes=100)
        large_source = self.write_source("large.txt", "x" * 101)
        with self.assertRaises(KnowledgeBaseError):
            large_kb.import_document(large_source)

        with self.assertRaises(ValueError):
            self.kb.import_document(
                self.write_source("bad-category.txt"),
                KnowledgeDocumentMetadata(title="Bad", category="unknown"),
            )

    def test_pdf_and_docx_dependencies_are_optional_and_reported(self):
        for filename, dependency, category in (
            ("reference.pdf", "PyMuPDF", "academic_research"),
            ("reference.docx", "python-docx", "academic_research"),
        ):
            with self.subTest(filename=filename):
                source = self.root / filename
                source.write_bytes(b"placeholder")
                with self.assertRaisesRegex(KnowledgeBaseError, dependency):
                    self.kb.import_document(
                        source,
                        KnowledgeDocumentMetadata(title=filename, category=category),
                    )


if __name__ == "__main__":
    unittest.main()
