# Dora Knowledge Base (DKB)

DKB menyimpan dokumen dan index lokal di bawah `get_data_dir()/tanya_dora/knowledge/`. Dokumen sumber yang diimpor disalin ke direktori `sources/`; teks hasil ekstraksi dan metadata disimpan pada SQLite `knowledge.db`. File tidak dikirim ke layanan eksternal oleh modul ini.

## Format dan batas

Format yang didukung: TXT, Markdown, reStructuredText, HTML, PDF, dan DOCX. Ukuran file maksimum default 32 MiB. PDF dibatasi 5.000 halaman. DOCX dibatasi 10.000 entri ZIP dan 128 MiB ukuran ekspansi.

TXT, Markdown, RST, dan HTML dapat diproses dengan standard library Python. PDF memerlukan PyMuPDF dan DOCX memerlukan python-docx; pasang dependency opsional melalui:

```powershell
python -m pip install -r requirements_ai.txt
```

Dokumen PDF berbasis gambar memerlukan OCR dan belum dapat diindeks. PDF/DOCX yang terenkripsi atau rusak ditolak dengan error. Parser PDF dicoba pada fixture dua halaman yang dibuat untuk audit; PDF/DOCX terhadap dokumen independen dan DOCX nyata belum diuji. `.venv` tidak memiliki PyMuPDF atau python-docx meskipun PyMuPDF tersedia di global Python.

## Metadata

Gunakan `KnowledgeDocumentMetadata` untuk mencatat judul, penulis/organisasi, kategori, tahun, URL sumber, DOI, lisensi, versi, dan status verifikasi. Kategori mengikuti slug yang ditetapkan DKB, seperti `statistics`, `machine_learning`, `research_methodology`, dan `dora_documentation`. Status verifikasi default adalah `UNVERIFIED`; indexing tidak mengubah status tersebut.

Contoh:

```python
from app.ai.knowledge_base import DoraKnowledgeBase, KnowledgeDocumentMetadata

knowledge_base = DoraKnowledgeBase()
document = knowledge_base.import_document(
    r"C:\Users\Researcher\Documents\statistics.md",
    KnowledgeDocumentMetadata(
        title="Descriptive Statistics Notes",
        author_or_org="Research Group",
        category="statistics",
        publication_year=2026,
        license="CC BY 4.0",
        verified=False,
    ),
)
```

Operasi service:

- `import_document(path, metadata)` menyalin, mengekstrak, membuat chunk, dan mengindeks dokumen.
- `list_documents(category=..., status=...)` mengembalikan daftar dan status indexing.
- `get_document(document_id)` mengambil metadata satu dokumen.
- `update_metadata(document_id, metadata)` memperbarui metadata dan field metadata yang ikut dicari.
- `reindex_document(document_id)` mengekstrak ulang salinan lokal yang dikelola DKB.
- `search(query, category=..., limit=...)` menjalankan keyword retrieval FTS5 dan mengembalikan chunk dengan DOI/URL, halaman, metadata, dan skor BM25.
- `delete_document(document_id)` menghapus chunk, metadata, serta salinan file lokal.

## Chunking dan retrieval

Nilai awal adalah 3.200 karakter per chunk dan overlap 480 karakter. Ukuran ini adalah pendekatan yang dapat dikonfigurasi, bukan hitungan tokenizer. Nilai optimal perlu diuji pada korpus Bahasa Indonesia dan model embedding yang dipilih.

Default retrieval memakai keyword SQLite FTS5. `embed_document()` dan `search_hybrid()` mendukung optional local Ollama embeddings; RAGPipeline menyusun context terbatas, citation dari chunk yang ditemukan, dan refusal tanpa sumber. Tidak ada Ollama di mesin audit sehingga embeddings/hybrid dan jawaban RAG berbasis model belum diverifikasi. Citation harus dibentuk hanya dari metadata dokumen yang ada; field kosong tidak boleh diisi dengan tebakan. Pengelolaan visual tersedia di Tanya DORA Studio → Knowledge Base.

## Uji lokal

```powershell
python -m pytest tests/test_knowledge_base.py -v
```

Tes saat implementasi memakai dokumen synthetic, menguji metadata, search/filter, re-index, delete, batas ukuran, kategori, ekstraksi HTML, dan penanganan dependency parser opsional.
