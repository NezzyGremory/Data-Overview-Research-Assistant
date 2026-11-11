# Menyiapkan Local LLM

Tanya DORA memakai Ollama HTTP API lokal. Ollama harus sudah berjalan dan model harus sudah terpasang sebelum dipilih dari halaman Tanya DORA. DORA memeriksa inventori melalui `/api/tags`, status model berjalan melalui `/api/ps`, dan chat melalui `/api/chat`. Ollama mendokumentasikan endpoint tersebut pada [API reference](https://docs.ollama.com/api) dan [chat API](https://docs.ollama.com/api/chat).

1. Pasang dan jalankan Ollama secara manual sesuai dokumentasi resminya.
2. Unduh model melalui aplikasi/CLI Ollama dengan persetujuan pengguna.
3. Buka menu **Tanya DORA** dan tekan **Perbarui**.
4. Pilih model lokal dari daftar; ukuran berkas yang dilaporkan Ollama ditampilkan di dropdown.
5. Atur temperature (0–2) dan kirim pertanyaan.

Tidak ada proses download otomatis dari DORA. Jika Ollama tidak merespons, halaman tetap terbuka dan menampilkan status koneksi. Training hardware preflight tersedia terpisah di `fine_tuning.py`; halaman inference belum menampilkan estimasi RAM/VRAM untuk tiap model.

## Model embedding

Pilihan **Keyword saja** tidak memerlukan embedding. Untuk hybrid retrieval, pilih model yang memang mendukung endpoint embedding, lalu indeks dokumen dengan model embedding yang sama. Endpoint `/api/embed` dan penggunaan batch dijelaskan pada [Ollama embeddings API](https://docs.ollama.com/api/embed). DORA mengirim request hanya ke alamat loopback dan menonaktifkan truncation.

Chat model dan embedding model dapat berbeda. Daftar `/api/tags` tidak membuktikan bahwa setiap model mendukung embedding; jika model yang dipilih tidak cocok, operasi ditampilkan sebagai error dan dokumen tidak dianggap ter-embed.

## Parameter

Temperature awal UI adalah 0.2 dan berlaku untuk chat yang sedang dikirim. Nilai ini tidak menjamin factuality; grounding tetap bergantung pada chunk yang ditemukan dan jawaban harus ditinjau.
