# Arsitektur Tanya DORA

## Tujuan

Tanya DORA adalah asisten desktop lokal yang memisahkan inferensi bahasa, sumber pengetahuan, retrieval, riwayat, dan antarmuka. Angka analisis hanya boleh berasal dari engine DORA; prompt dan jawaban tidak dapat menggantikan perhitungan analyzer.

## Komponen yang tersedia

| Komponen | Modul | Status |
| --- | --- | --- |
| Ollama model inventory dan chat | `app/ai/llm_manager.py`, `model_registry.py` | Terpasang: daftar model, status, generate, streaming, cancellation, embeddings. |
| Dora Knowledge Base | `app/ai/knowledge_base.py` | Terpasang: metadata, import, extraction, chunking, FTS5, embedding SQLite, search, re-index, delete. |
| RAG | `app/ai/rag_pipeline.py` | Terpasang: retrieval keyword/hybrid, context terbatas, citation terstruktur, refusal tanpa sumber, conversation context. |
| Riwayat percakapan | `app/ai/conversation_store.py` | Terpasang: SQLite terpisah dari lifecycle proyek DORA. |
| Tanya DORA desktop | `app/ui/tanya_dora_page.py` | Terpasang: lazy-loaded PySide6 page, worker chat, streaming, stop, model/status, temperature, history, references. |
| Wiring aplikasi | `app/ui/main_window.py` | Route lazily loaded pada sidebar; handler dataset dan halaman lain tidak diubah. |
| DORA read-only tools dan online search opsional | `app/ai/tool_calling.py` | Tersedia di tool loop; online OpenAlex/Crossref hanya terdaftar jika checkbox aktif. |
| TDID builder dan snapshot | `app/ai/tdid_dataset.py` | Service import/validasi/split/versioning tersedia dan terhubung ke Dataset Manager. |
| SFT + LoRA/QLoRA | `app/ai/fine_tuning.py` | Engine, preflight, callback progress, workspace export, dan Fine-Tuning Studio terhubung; stack training opsional tidak terpasang di environment ini, belum ada run nyata. |
| Adapter registry dan TDEB | `app/ai/model_versioning.py`, `evaluation_benchmark.py` | Registry terhubung ke Model Manager dan jalur merge/import Ollama; TDEB terhubung ke AI Evaluation Studio. Belum ada merge atau hasil live benchmark. |

## Alur chat saat ini

```text
Pertanyaan
   ├─ optional embedding lokal → SQLite FTS5 + cosine hybrid
   └─ tanpa embedding → SQLite FTS5 keyword
           ↓
       RAGPipeline
           ↓
   local Ollama /api/chat stream
           ↓
   jawaban + metadata sumber → riwayat SQLite terpisah
```

Jika retrieval tidak menemukan sumber, pipeline mengembalikan pesan yang menyatakan sumber belum ditemukan dan tidak memanggil model chat. Riwayat percakapan dipakai sebagai konteks dialog, tetapi bukan bukti pengetahuan. Citation UI diambil dari hasil retrieval aktual.

## Penyimpanan

- DKB: `get_data_dir()/tanya_dora/knowledge/knowledge.db` beserta salinan dokumen di `sources/`.
- Percakapan: `get_data_dir()/tanya_dora/conversations/conversations.db`.
- Dua penyimpanan ini terpisah dari SQLite proyek dan tidak dimasukkan ke training set otomatis.
- Semua inferensi dan embedding pada fase ini hanya menuju loopback Ollama. Modul tidak mengunduh model.

## Management Studio dan batas integrasi

Tanya DORA memasang tool read-only untuk snapshot dataset/hasil analyzer, pencarian DKB, dan pencarian paper opsional. Tanya DORA Studio menyediakan enam tab untuk pengelolaan DKB, TDID, SFT, model/adapter, training history, dan TDEB. Adapter PEFT dijalankan melalui merge ke model penuh lokal sebelum import ke Ollama; tidak dimuat langsung sebagai adapter. Jalur conversion dan pilihan model masih menunggu validasi pada Ollama/model nyata.

Research gap/novelty belum mempunyai workflow dedicated; model harus menyebut hasil sebagai potensi, menampilkan paper yang benar-benar ditemukan, dan menyampaikan coverage terbatas. Lihat `research_intelligence.md`.

## Pengujian

Tes service berjalan tanpa Ollama menggunakan HTTP mock, vector synthetic, SQLite temporary, dokumen synthetic, dan callback benchmark yang terkontrol. UI Tanya DORA serta enam tab Studio diuji saat service lokal tidak tersedia. Fixture PDF dua halaman yang dibuat untuk audit berhasil diproses melalui PyMuPDF global Python dan memberikan citation page yang tersimpan; dokumen itu bukan paper penelitian independen. Tidak ada klaim kualitas jawaban nyata sebelum live inference/evaluation.

## Batas penggunaan versi

Model switch, aktivasi adapter hasil merge, deactivation, dan rollback memerlukan konfirmasi eksplisit. Fingerprint base model adalah fingerprint struktural (hash config + nama/ukuran bobot), bukan hash isi bobot. TDEB menghitung lexical proxy dan latency; factuality/reasoning/hallucination memerlukan human review. Lihat [completion audit](completion_audit.md) untuk hasil mesin dan status DoD.
