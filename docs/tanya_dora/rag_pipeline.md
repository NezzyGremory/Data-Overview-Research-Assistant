# Retrieval-Augmented Generation (RAG)

## Alur

1. `DoraKnowledgeBase.import_document()` menyimpan file lokal dan membuat chunk beserta metadata sumber.
2. `DoraKnowledgeBase.embed_document()` mengirim batch teks hanya ke Ollama lokal melalui `OllamaLLMManager.embed_texts()`.
3. Embedding per chunk disimpan sebagai float32 BLOB di SQLite dan dikaitkan dengan nama model embedding.
4. Pertanyaan diubah menjadi embedding lokal. `search_hybrid()` mengambil ranking keyword FTS5 dan semantic cosine, lalu menggabungkannya menggunakan reciprocal-rank fusion.
5. `RAGPipeline` memilih hasil, merakit konteks yang dibatasi ukurannya, dan memanggil model chat lokal.
6. Response menyimpan daftar sumber terpisah berisi metadata dan citation, agar UI dapat menampilkan bukti yang dipakai.

Ollama menyediakan `POST /api/embed` untuk teks tunggal atau daftar teks; client DORA memakai daftar batch dan mengatur `truncate: false` agar embedding gagal secara eksplisit bila input melewati batas model. Lihat [dokumentasi API embedding Ollama](https://docs.ollama.com/api/embed).

## Contoh penggunaan service

```python
from app.ai.knowledge_base import DoraKnowledgeBase
from app.ai.llm_manager import OllamaLLMManager
from app.ai.rag_pipeline import RAGPipeline

knowledge_base = DoraKnowledgeBase()
ollama = OllamaLLMManager()
ollama.select_model("qwen2.5:7b")

# Pilih model embedding lokal terpisah yang memang tersedia dan mendukung embed.
knowledge_base.embed_document(
    document_id,
    "embeddinggemma:latest",
    ollama,
)

assistant = RAGPipeline(
    knowledge_base,
    ollama,
    chat_model="qwen2.5:7b",
    embedding_model="embeddinggemma:latest",
)
response = assistant.answer("Apa temuan sumber tentang metode ini?")
```

Model chat dan model embedding harus sudah terpasang melalui tindakan pengguna. DORA tidak memanggil endpoint pull. Penggunaan keyword-only dapat dilakukan dengan `embedding_model=None`, tetapi semantic retrieval memerlukan vector untuk seluruh dokumen yang masuk ke filter pencarian.

## Batas implementasi saat ini

- Chunk default 3.200 karakter dengan overlap 480 karakter. Ini heuristik, bukan ukuran tokenizer; ukur ulang untuk korpus dan model sebenarnya.
- Hybrid retrieval menghitung cosine secara langsung di CPU dan dibatasi 10.000 chunk. Belum ada ANN index atau reranker.
- Jika vector model tidak lengkap atau dimensinya tidak cocok, hybrid retrieval berhenti dengan error agar dokumen terlewat tidak disembunyikan.
- Tidak ada sumber yang cocok menghasilkan pesan eksplisit bahwa DKB belum menemukan sumber; LLM tidak dipanggil pada kondisi ini.
- Citation terstruktur berasal dari metadata aktual. Jawaban model juga diarahkan menggunakan `[S#]`; UI tetap perlu menampilkan panel citation dari `response.sources`.
- RAG diuji menggunakan HTTP mock dan vector synthetic. Belum ada pengukuran relevansi, akurasi citation, latency, atau kualitas jawaban dengan model Ollama nyata.

## Pengujian

```powershell
python -m pytest tests/test_local_llm.py tests/test_knowledge_base.py tests/test_rag_pipeline.py -v
```
