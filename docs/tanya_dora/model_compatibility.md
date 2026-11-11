# Kompatibilitas Model

## Saat ini

Runtime yang didukung pada implementasi awal adalah Ollama yang melayani HTTP lokal pada `127.0.0.1:11434` atau alamat loopback yang setara. Model chat harus muncul pada `/api/tags`. Model tidak dapat dipilih hanya berdasarkan nama atau diunduh dari Tanya DORA.

Registry menyimpan nama, ukuran berkas, tanggal modifikasi, family, parameter size, dan quantization level bila disediakan Ollama. Metadata tersebut deskriptif; belum ada tes yang memilih keluarga model terbaik.

## Chat dan embedding

- Chat membutuhkan model Ollama yang tersedia dan mendukung `/api/chat`.
- Hybrid retrieval memerlukan model embedding yang mendukung `/api/embed` dan vektor dengan dimensi konsisten.
- Keyword FTS5 dapat dipakai tanpa model embedding.
- Kompatibilitas konteks, bahasa Indonesia, tool calling, RAM, VRAM, dan lisensi belum menjadi gerbang otomatis.

## Keluarga model

Qwen, Gemma, Llama, dan Phi belum dibandingkan di mesin ini. Pilihan model perlu dinilai dengan dataset benchmark terpisah, mencatat model/tag, kuantisasi, hardware, latency, jawaban, dan penilaian manusia. Nama populer atau ukuran parameter bukan hasil evaluasi.

Tidak ada rentang versi resmi Ollama atau PySide6 yang ditetapkan oleh proyek. Jangan menafsirkan snapshot dependency lokal sebagai rentang dukungan resmi.
