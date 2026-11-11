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

Parser opsional memerlukan `PyMuPDF` dan `python-docx` dari `requirements_ai.txt`. PDF hasil scan belum memiliki OCR. Tidak ada dokumen uji nyata atau OCR yang disertakan pada fase ini.

## Batas yang diketahui

Tidak ada live Ollama inference atau benchmark kualitas yang sudah dijalankan. TDID builder, fine-tuning engine, model version registry, dan TDEB service tersedia; UI khusus dataset/training/evaluasi belum tersedia. Fine-tuning akan berhenti di preflight bila dependency, model lokal, split TDID, RAM/VRAM, atau ruang disk tidak memadai. Adapter yang diregistrasi belum digunakan oleh runtime chat. Lihat `fine_tuning_guide.md`, `training_hardware.md`, `model_versioning.md`, dan `evaluation_benchmark.md` untuk batas rinci.

## Training dependencies tidak ada

Install PyTorch sesuai OS/hardware dan `requirements_training.txt` hanya pada environment training terpisah. Untuk QLoRA, pasang bitsandbytes yang cocok dengan versi CUDA/PyTorch mengikuti [panduan resmi](https://huggingface.co/docs/bitsandbytes/installation). Tanya DORA tidak memasang package atau mengunduh model secara otomatis.
