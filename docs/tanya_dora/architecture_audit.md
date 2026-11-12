# Audit arsitektur Tanya DORA

Tanggal: 2026-11-11 (Asia/Jakarta). Audit ini memeriksa source code, konfigurasi dependency, dan dokumentasi repository untuk menilai fondasi Tanya DORA. Audit tidak membaca `support.js`, file `.env`, database runtime, atau isi folder `data/`; tidak ada layanan Ollama yang dipanggil.

## Ringkasan

DORA sudah memiliki fondasi yang dapat dipakai ulang untuk analisis dataset lokal dan pencarian literatur. Tanya DORA belum tersedia sebagai asisten percakapan: belum ada runtime LLM lokal, knowledge base terindeks, RAG, riwayat percakapan, tool calling, dataset builder, pipeline fine-tuning, atau benchmark khusus Tanya DORA. Penjelasan lokal yang ada bersifat deterministik dan menggunakan hasil analisis yang sudah dihitung. Gemini merupakan opsi eksternal, bukan local LLM.

Repository sudah memiliki audit umum DORA di `docs/dora_comprehensive_audit.md` dan peta arsitektur di `docs/dora_architecture.md`. Dokumen ini menambahkan audit yang berfokus pada kebutuhan Tanya DORA dan menjadi baseline untuk fase implementasi.

## Fondasi yang dapat digunakan kembali

| Kebutuhan | Implementasi saat ini | Pemakaian untuk Tanya DORA |
| --- | --- | --- |
| Analisis dataset | `app/analyzer/` menyediakan loader, profiling, statistik, nilai kosong, duplikasi, outlier, korelasi, kualitas, dan fingerprint. | Tool read-only dapat membungkus hasil analisis yang sudah ada. Angka harus berasal dari hasil engine, bukan dihitung oleh LLM. |
| Penjelasan lokal | `app/ai/local_explainer.py` menyusun narasi deterministik dari profil dan statistik terhitung. | Pola sumber fakta lokal dapat dipertahankan; modul ini bukan model bahasa generatif. |
| Penjelasan eksternal | `app/ai/gemini_explainer.py` memakai Gemini opsional dengan fallback ke penjelasan lokal; `research_gap_explainer.py` juga dapat meminta penjelasan Gemini. | Pertahankan hanya sebagai opsi eksplisit. Jangan jadikan jalur inferensi utama atau mengirim percakapan/dokumen secara diam-diam. |
| Pencarian literatur | `app/research/search.py` mengorkestrasi OpenAlex, Crossref, Europe PMC, OpenAIRE, Semantic Scholar, dan Scopus opsional, lalu mendeduplikasi `Paper`. | Reuse adapter pencarian, hasil metadata, dan status sumber untuk pencarian akademik serta bukti research gap. Ringkasan/gap tetap harus membedakan bukti yang ada dari inferensi. |
| Penyimpanan proyek | `app/storage/database.py`, `project_repository.py`, dan `research_service.py` menyimpan proyek, workflow, eksperimen ML, paper, screening jurnal, dan versi dataset di SQLite. | Penyimpanan pengetahuan, dataset training, model, dan percakapan perlu skema terpisah atau migrasi teruji. Riwayat percakapan sebaiknya tidak ikut lifecycle pembersihan proyek. |
| Konteks DORA | Dokumentasi tersebar di `docs/`; feature, analyzer, ML, research, dan report memiliki implementasi nyata. | Dokumentasi dapat diindeks secara kurasi setelah metadata sumber dan lisensi ditentukan. Belum ada mekanisme indexing dokumen. |
| UI desktop | `app/ui/main_window.py` membangun page stack PySide6, routing, worker, dan orkestrasi. Halaman tambahan dibuat lazy. | Halaman Tanya DORA harus menjadi modul terpisah dan dipasang melalui wiring yang sudah ada. Hindari menambah orkestrasi AI besar ke `MainWindow`. |

## Komponen yang belum tersedia

- Tidak ditemukan Local LLM/Ollama manager, model registry/loader, atau inference settings. Paket Python `ollama`, `torch`, `transformers`, `peft`, dan `trl` tidak tersedia di `.venv` yang diperiksa. Pengecekan paket tidak membuktikan apakah aplikasi Ollama desktop/server terpasang atau sedang berjalan.
- `app/nlp/embeddings.py` berukuran 0 byte. NLP lain berisi ekstraksi kata kunci, deteksi domain, dan similarity fingerprint; tidak ada embedding semantik yang siap dipakai untuk RAG.
- Tidak ditemukan alur import dokumen pengetahuan, metadata dokumen, extraction/chunking, FTS/vector index, citation grounding, atau pengelolaan sumber DKB.
- Tidak ditemukan format TDID, validator/deduplicator dataset instruksi, data split, training runner, checkpoint/adapter registry, atau benchmark Tanya DORA.
- Tidak ditemukan riwayat percakapan/chat page atau kontrak tool calling untuk mengakses dataset aktif dan hasil analisis.
- `requirements.txt` memuat dependency aplikasi desktop saat ini, tetapi tidak memuat parser PDF/DOCX, model embedding lokal, vector store, atau stack fine-tuning. Fine-tuning sebaiknya menjadi extra/lingkungan terpisah agar instalasi desktop tetap ringan.

## Batas dan risiko integrasi

1. `MainWindow` merupakan shell besar yang memuat banyak handler dan worker. Engine AI harus tetap berada di modul terpisah, memakai service/adaptor, dan mengirim kerja berat ke worker; jangan memindahkan logika analisis ke LLM.
2. `Database.clear_session_data()` menghapus data proyek saat lifecycle sesi berjalan. Conversation history yang dimaksudkan untuk bertahan harus mempunyai storage dan kebijakan retensi yang tidak ikut terhapus bersama data proyek.
3. Schema SQLite dibuat di `Database._initialize()` dan sudah memiliki migrasi kolom kompatibilitas. Penambahan tabel perlu migration additive dan tes terhadap database lama, bukan menulis ulang database runtime.
4. DORA saat ini memproses dataset tabular. Dokumen, gambar, audio, dan percakapan perlu pipeline tersendiri; jangan menganggap `DatasetLoader` dapat mengindeks semuanya.
5. API akademik menyediakan metadata dan abstrak dari beberapa provider, tetapi tidak menjamin coverage seluruh literatur. Research gap harus dilabeli sebagai potensi dan ditautkan ke paper yang benar-benar ditemukan.
6. Kerja training membutuhkan storage dan hardware yang bervariasi. Model tidak boleh diunduh atau diaktifkan otomatis; hasil evaluasi harus menyimpan konfigurasi dan bukti eksekusi.
7. `design/html_export/` dan `design/tokens.md` yang disebut instruksi project tidak ditemukan pada checkout ini. Halaman Tanya DORA belum didesain atau diubah dalam audit ini.

## Kebutuhan dependency, storage, dan hardware

- **Inference:** implementasi awal dapat memakai HTTP API Ollama melalui `httpx` yang sudah menjadi dependency. Ollama harus berjalan lokal dan model harus sudah tersedia; aplikasi tidak memanggil endpoint download/pull otomatis.
- **Knowledge base:** simpan metadata/index di lokasi data pengguna yang berasal dari `app.core.config.get_data_dir()`. Simpan dokumen asli secara lokal, validasi ukuran/tipe file, catat lisensi dan status verifikasi, serta gunakan reference relatif. SQLite FTS5 dapat dievaluasi untuk retrieval keyword; embedding/vector index lokal baru dipilih setelah uji kualitas, ukuran, dan lisensi model.
- **Conversation history:** gunakan storage terpisah dari pembersihan sesi proyek dan jangan mengimpor percakapan ke DKB atau training set tanpa persetujuan jelas.
- **Training:** Transformers/PEFT/TRL/PyTorch dan kemungkinan CUDA/Unsloth belum tersedia di `.venv`. Tempatkan pipeline SFT/LoRA sebagai lingkungan opsional atau external training workspace; jangan menambahkan foundation-model training ke instalasi desktop utama.
- **Hardware:** kapasitas inference bergantung pada model, quantization, RAM/VRAM dan Ollama. Belum ada benchmark perangkat di audit ini; jangan menetapkan minimum resmi atau mengklaim model cocok sebelum pengukuran nyata.

## Tahapan implementasi yang disarankan

| Fase | Tujuan dan batas selesai |
| --- | --- |
| 1. Architecture Audit | Selesai melalui dokumen ini. Sebelum fitur berikutnya, tetapkan kontrak provider, penyimpanan, privasi, dan bukti evaluasi. |
| 2. Local LLM Foundation | Tambahkan klien Ollama lokal, daftar model yang sudah terpasang, pemilihan model, status koneksi, dan generate non-streaming. Tidak ada auto-download. |
| 3. Dora Knowledge Base | Definisikan metadata, import dokumen, validasi, extraction, chunking, delete/re-index, dan sumber/lisensi. |
| 4. RAG | Mulai dengan retrieval yang dapat dievaluasi; tambahkan semantic/hybrid setelah embedding lokal dan penyimpanan dipilih berdasarkan uji. Citation wajib berasal dari chunk yang diambil. |
| 5. Tanya DORA Assistant | Bangun satu halaman chat PySide6, worker/stream lifecycle, riwayat lokal, panel sumber, status model, dan pilihan Local/Hybrid Research. Integrasikan hanya service yang sudah tersedia. |
| 6. DORA Tools & Research | Tambahkan kontrak tool read-only untuk hasil analyzer, dan adapter pencarian paper yang memakai kembali OpenAlex/Crossref. Tindakan yang menulis data meminta konfirmasi. |
| 7. TDID Builder | Import/edit/validasi JSONL atau CSV, deteksi duplikasi, kelompok/sumber untuk split, status verifikasi, dan versioning. Tidak menggunakan percakapan otomatis. |
| 8. Fine-Tuning Studio | Buat konfigurasi/check pipeline SFT + LoRA/QLoRA di environment opsional; checkpoint, adapter, dan aktivasi melalui persetujuan pengguna. |
| 9. Evaluation Benchmark | Pisahkan test set; ukur factuality, citations, tool accuracy, latency, dan penggunaan memori dengan hasil nyata. |
| 10. Optimization & Release | Uji worker/UI, migration, privasi, packaging, compatibility, dan dokumentasi; klaim kesiapan hanya berdasarkan hasil yang tercatat. |

## Status pekerjaan

- Fase 1: audit source dan rencana integrasi selesai.
- Fase 2: fondasi Ollama lokal selesai; tidak ada live server test dan tidak ada auto-pull.
- Fase 3: DKB service lokal keyword/hybrid selesai; UI pengelolaan dokumen belum tersedia.
- Fase 4: retrieval, embedding lokal Ollama, context assembly, dan citation service selesai; kualitas retrieval nyata belum diukur.
- Fase 5: Tanya DORA chat page, streaming, history, dan source panel selesai; satu screenshot dicatat di bawah.
- Fase 6: tool calling read-only untuk DORA, DKB, dan OpenAlex/Crossref opt-in selesai; provider akademik belum diuji live.
- Fase 7: TDID service dan tes selesai; belum ada UI dataset builder dan dataset berkualitas tidak dibuat otomatis.
- Fase 8: SFT+LoRA/QLoRA engine, resource preflight, checkpoint callback, adapter manifest/versioning tersedia; tidak ada training run karena optional training dependencies tidak tersedia.
- Fase 9: TDEB dataset/run/review service tersedia; benchmark model nyata belum dijalankan.
- Fase 10: regresi penuh dan dokumentasi sedang diverifikasi; Definition of Done belum tercapai karena model live, adapter inference, dan UI studio belum tersedia.

## Implementasi Fase 2 — Local LLM Foundation

- `app/ai/model_registry.py` memetakan daftar model lokal, ukuran berkas, keluarga, ukuran parameter, dan kuantisasi.
- `app/ai/llm_manager.py` memakai endpoint lokal `/api/tags`, `/api/ps`, `/api/chat`, dan `/api/embed`; URL dibatasi ke alamat loopback. Pemilihan model hanya menerima inventori lokal dan tidak menjalankan download.
- Tidak ada request ke endpoint pull/download. Model harus diunduh dan tersedia di Ollama melalui tindakan pengguna sebelum dapat dipilih.
- API mengikuti dokumentasi resmi Ollama: [API documentation](https://docs.ollama.com/api).
- Tes mock mencakup daftar/pemilihan model, metadata ukuran, status server/model, generate, input invalid, model hilang, server gagal, dan tidak adanya auto-pull. Tidak dilakukan koneksi inference ke instalasi Ollama nyata.
- Status terkini: streaming/cancellation tersedia untuk chat di halaman Tanya DORA; prompt manager terpisah, setting persisten, resource check, dan benchmark model belum dibuat.

## Implementasi Fase 3 — Dora Knowledge Base

- `app/ai/knowledge_base.py` mengimpor TXT, Markdown, reStructuredText, HTML, PDF, dan DOCX; file disalin ke penyimpanan lokal yang dikelola DKB. Path tujuan dibentuk dari ID acak dan divalidasi tetap berada di direktori sumber.
- Metadata mencatat ID dokumen, judul, penulis/organisasi, kategori, tahun, URL, DOI, lisensi, versi, waktu indexing, status verifikasi, status indexing, jumlah chunk, ukuran sumber, dan hash SHA-256.
- Batas default satu file adalah 32 MiB. PDF dibatasi 5.000 halaman. Ekstraksi DOCX membatasi 10.000 entri archive dan 128 MiB ukuran ekspansi.
- Chunking saat ini memakai batas karakter 3.200 dengan overlap 480; ini heuristik yang dapat diubah, bukan ukuran token akurat. PDF menyimpan label halaman. SQLite FTS5 menjalankan keyword search dan filter kategori.
- PDF memakai PyMuPDF dan DOCX memakai python-docx hanya bila dipasang dari `requirements_ai.txt`. Kedua dependency tersebut tidak tersedia di environment saat audit; tes memastikan pesan instalasi yang jelas, tetapi extraction PDF/DOCX belum dijalankan pada dokumen nyata. Scanned PDF/OCR belum didukung.
- Tes gabungan Local LLM + DKB: 20 passed, 5 subtests passed. FTS5 diperiksa di SQLite in-memory; seluruh dokumen uji adalah synthetic dan tersimpan di temporary directory.
- Re-index background worker, evaluasi relevansi, dan UI DKB tetap menjadi pekerjaan fase berikutnya.

## Implementasi Fase 4 — Retrieval-Augmented Generation

- `OllamaLLMManager.embed_texts()` memanggil endpoint lokal `/api/embed`, mensyaratkan embedding model sudah terpasang, menolak input/keluaran invalid, dan meminta Ollama tidak memotong teks. API ini mengikuti [endpoint embeddings resmi Ollama](https://docs.ollama.com/api/embed).
- `DoraKnowledgeBase.embed_document()` membuat embedding secara batch dan menyimpan float32 vector secara lokal di SQLite per chunk dan versi nama model. Re-index atau penghapusan dokumen membuang vector lama.
- `search_hybrid()` menggabungkan FTS5 keyword rank dengan cosine similarity lokal menggunakan reciprocal-rank fusion. Implementasi exact search dibatasi 10.000 chunk untuk membatasi RAM/CPU; coverage vector harus lengkap untuk seluruh chunk kategori yang dicari, dan dimensinya harus cocok.
- `RAGPipeline` merakit konteks dengan batas karakter, meminta model lokal memakai penanda sumber `[S1]`, dan mengembalikan daftar sumber terpisah untuk UI. Tanpa hasil retrieval, pipeline mengembalikan pesan jujur tanpa meminta LLM mengisi jawaban dari ingatan.
- Jalur semantic/RAG belum diuji dengan server Ollama atau model embedding nyata; tes memakai respons HTTP dan vektor synthetic. Kualitas retrieval model nyata, citation faithfulness, dan latency belum diukur.
- Tes gabungan Local LLM + DKB + RAG: 28 passed, 5 subtests passed.
- Halaman Tanya DORA kini memakai streaming/cancellation dan menampilkan citation terstruktur; UI import/management DKB belum dibuat.

## Implementasi Fase 5 â€” Tanya DORA Assistant

- `app/ui/tanya_dora_page.py` menyediakan riwayat percakapan, transcript, input, send/stop, model selector dengan ukuran, status Ollama, temperature, pilihan embedding opsional, copy, regenerate, dan daftar/preview sumber.
- Chat dan model discovery berjalan di `QThread`; stream cancellation menutup response aktif. Saat DKB tidak menemukan sumber, UI menampilkan pesan RAG dan tidak meminta model menebak.
- `app/ai/conversation_store.py` menyimpan percakapan/sumber di SQLite terpisah dari database proyek dan DKB. Percakapan tidak masuk Knowledge Base atau training otomatis.
- `main_window.py` hanya menambahkan route/sidebar dan lazy page creation; handler dataset, analyzer, project storage, dan lifecycle sesi lama tidak diubah.
- Satu screenshot halaman diambil pada 1440×900: `screenshots/tanya_dora_page.png`. Tampilan diperiksa setelah capture.
- Tes service/UI Tanya DORA: 35 passed, 5 subtests passed pada set LLM + DKB + RAG + conversation + page. UI smoke memakai server unavailable; tidak ada live Ollama inference.
- Online Research checkbox tidak dicentang secara default. DORA analyzer tools tersedia dari fase 6; UI import DKB dan inference hardware panel belum tersedia.

## Update Implementasi Fase 6–9 — 2026-11-12

### Fase 6 — DORA Tool Integration

- `app/ai/tool_calling.py` memasang tools read-only untuk snapshot dataset aktif, hasil analyzer yang sudah tersedia, DKB, serta OpenAlex/Crossref bila Online Research diaktifkan.
- Tool schema membatasi argumen, ukuran respons, jumlah tool turn, dan tidak mengeksekusi kode model. Hasil kosong/error tidak dianggap evidence. Tidak ada tool untuk mengubah file/proyek.
- Tes tool/chat memakai HTTP mock dan konteks analyzer sintetis; tidak ada provider akademik live yang dipanggil.

### Fase 7 — TDID Dataset Builder

- `app/ai/tdid_dataset.py` mengimpor JSONL/CSV, menghasilkan ID lokal jika perlu, memvalidasi role/metadata/source, menghitung distribusi, mendeteksi exact duplicate/repeated prompt, split berdasarkan source group, ekspor, dan menyimpan snapshot SHA-256.
- Validasi split menolak ID duplikat, split kosong, record belum verified/invalid, exact duplicate lintas split, source group lintas split, serta versi final berisi record belum verified.
- Tes mencakup malformed import yang tidak boleh menggeser indeks validasi, grouped split, duplicate, snapshot integrity, dan final-verification gate. Tidak ada contoh TDID berkualitas yang dibuat secara massal.

### Fase 8 — Fine-Tuning Engine dan Model Versioning

- `app/ai/fine_tuning.py` menyediakan config validator, deteksi hardware, estimasi disk/memori, preflight dependency/model lokal/split, TRL SFTTrainer + PEFT LoRA, QLoRA 4-bit berbasis CUDA, log callback, cancellation, checkpoint, manifest, dan validasi output adapter.
- Model dimuat dengan `local_files_only=True` serta `trust_remote_code=False`. `requirements_training.txt` dan `requirements_qlora.txt` tetap optional. Environment ini tidak memiliki torch, transformers, datasets, PEFT, TRL, accelerate, atau bitsandbytes; tidak ada training yang dijalankan.
- `app/ai/model_versioning.py` registrasi adapter lokal, export/import ZIP, traversal/symlink/size checks, SHA-256 adapter, fingerprint struktural base model, manifest provenance, evaluasi, dan pilihan eksplisit. Pilihan adapter belum tersambung ke chat runtime Ollama.
- Tes mencakup validasi config, preflight missing dependency/CPU-only QLoRA, export-import adapter, konfirmasi pilihan, dan integrity checks. Checkpoint nyata belum dihasilkan karena trainer dependencies tidak tersedia.

### Fase 9 — Evaluation Benchmark

- `app/ai/evaluation_benchmark.py` menyimpan versi held-out terpisah dan mewajibkan leakage check terhadap TDID sebelum run. Runner menerima target/callback nyata; helper Ollama menggunakan model lokal terpasang. Output, latency, error, required-term/citation/tool coverage, serta snapshot RAM/VRAM disimpan di SQLite.
- Metrik lexical dilabeli proxy, bukan factual accuracy. Human review per kasus disimpan terpisah; tidak ada angka kualitas model yang dipalsukan.
- Tes memakai callback deterministik untuk menguji persistensi, leakage gate, scoring proxy, human review, dan error path. Tidak ada Ollama live atau base-vs-fine-tuned benchmark di environment ini.

## Fase 10 — Optimization & Release

- Targeted suite terbaru: 84 passed, 5 subtests passed pada Local LLM, DKB/RAG, conversation, tools, TDID, training preflight, model registry, benchmark, Tanya DORA UI, dan Vision UI.
- Full regression suite: 388 passed, 2 failed, 5 subtests passed. Dua kegagalan tetap pada `tests/test_sinta_batch_worker.py`: ekspektasi prefix cache `sinta-parser:v4|` padahal kode memakai v5, dan label `SINTA 2` padahal kode mengembalikan `Perlu Diperiksa Ulang`. Keduanya juga muncul pada full run sebelum pekerjaan Tanya DORA; tidak ada file SINTA yang diubah di sesi ini.
- Definition of Done belum tercapai pada audit awal: live Ollama belum diuji, training stack tidak ada, fine-tuned adapter belum bisa dipakai oleh runtime chat, dan halaman management/benchmark belum tersedia.

## Addendum integrasi — 2026-11-12

The earlier phase/status notes above are the historical architecture audit. Since then, the six-tab Tanya DORA Management Studio has been routed into `MainWindow`, and its Knowledge Base, TDID, Fine-Tuning, Model Manager, Training History, and AI Evaluation tabs call the existing local services. The chat page is linked to explicitly confirmed runtime model selection. A local PEFT-to-Ollama merge/import path, human-review UI, duplicate-document rejection, and verified TDID workspace export were added.

Automated tests and local generated-PDF/TDID plumbing checks pass as detailed in [completion_audit.md](completion_audit.md). Live Ollama inference, real-model RAG/embeddings, adapter conversion and activation, training, checkpoint reload, and five-condition TDEB remain unverified because this machine has no Ollama service/model or fine-tuning dependencies. Those DoD items remain outstanding.
