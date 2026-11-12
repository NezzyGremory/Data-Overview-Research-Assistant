# Pemecahan Masalah Tanya DORA

## Ollama tidak tersedia

Periksa apakah layanan lokal Ollama berjalan dan merespons di `http://127.0.0.1:11434`. Tekan **Perbarui** di Tanya DORA. Aplikasi hanya memakai loopback dan tidak mengirim prompt ke server publik.

## Daftar model kosong

Pasang model secara manual melalui Ollama, kemudian refresh inventori. Tanya DORA tidak menjalankan `/api/pull` atau mengunduh model.

## DKB tidak menemukan sumber

Pastikan dokumen sudah diimpor dan status indexing-nya berhasil. Periksa kategori filter pada API DKB bila digunakan. Tanpa hasil retrieval, Tanya DORA menyatakan bahwa belum ada sumber dan tidak meminta LLM mengarang jawaban.

## Gagal membuat embedding

Model yang terpasang tidak selalu merupakan model embedding. Pilih model yang mendukung `/api/embed`. Jika sudah ada embedding lama, indeksnya harus dibuat ulang untuk model embedding yang dipilih sebelum hybrid search dapat digunakan.

## PDF/DOCX tidak dapat diindeks

Parser opsional memerlukan `PyMuPDF` dan `python-docx` dari `requirements_ai.txt`. PDF hasil scan belum memiliki OCR. Parser PDF telah diperiksa dengan fixture PDF dua halaman di global Python 3.12; parser tersebut tidak menggantikan uji terhadap dokumen penelitian independen. Package PDF/DOCX tetap tidak ada di project `.venv`.

## Batas yang diketahui

Tidak ada live Ollama inference, merge adapter, fine-tuning, atau benchmark kualitas yang sudah dijalankan pada mesin ini. Tanya DORA Studio menyediakan UI untuk dataset, training, model/adapter, history, dan evaluasi. Fine-tuning akan berhenti di preflight bila dependency, model lokal, split TDID, RAM/VRAM, atau ruang disk tidak memadai. Adapter perlu di-merge dan diimpor menjadi model penuh sebelum dapat dipakai oleh runtime chat. Lihat `ollama_verification.md`, `adapter_runtime.md`, `training_studio.md`, dan `evaluation_studio.md` untuk status serta batas rinci.

## Training dependencies tidak ada

Install PyTorch sesuai OS/hardware dan `requirements_training.txt` hanya pada environment training terpisah. Untuk QLoRA, pasang bitsandbytes yang cocok dengan versi CUDA/PyTorch mengikuti [panduan resmi](https://huggingface.co/docs/bitsandbytes/installation). Tanya DORA tidak memasang package atau mengunduh model secara otomatis.
