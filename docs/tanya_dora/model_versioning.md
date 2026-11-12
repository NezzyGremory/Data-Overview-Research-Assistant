# Model Versioning dan Adapter

`app/ai/model_versioning.py` menyimpan adapter lokal dan metadata provenance: base model path/fingerprint, hash adapter, TDID version, lisensi, training config, dan hasil evaluasi. SQLite menyimpan registry lokal; adapter disalin ke `get_data_dir()/tanya_dora/models/adapters/`.

## Operasi

- `register_adapter()` menerima folder yang memiliki `adapter_config.json` dan `adapter_model.*`.
- `verify_version()` menghitung ulang SHA-256 adapter dan fingerprint struktural base model.
- `export_adapter()` menghasilkan ZIP dengan adapter dan manifest.
- `import_adapter_archive()` memeriksa path traversal, symlink, batas jumlah/ukuran file, hash adapter, dan fingerprint base model sebelum registrasi.
- `record_evaluation()` menempelkan metrik hasil benchmark.
- `select_version(..., user_confirmed=True)` mencatat pilihan setelah konfirmasi eksplisit; adapter tidak diaktifkan oleh training.

Fingerprint base model saat ini merangkum hash `config.json` serta nama/ukuran file bobot, bukan hash isi setiap bobot. Ia mendeteksi perubahan konfigurasi dan susunan/ukuran file, tetapi tidak membuktikan identitas kriptografis semua bobot.

Pemilihan versi adapter di registry tidak dengan sendirinya mengaktifkan inference. Tanya DORA Studio memeriksa kompatibilitas, lalu dapat menggabungkan LoRA ke model penuh lokal dan meminta Ollama mengimpornya. Aktivasi model runtime, deactivation, dan rollback memerlukan konfirmasi eksplisit dan dicatat terpisah dari seleksi registry. Lihat [adapter runtime](adapter_runtime.md). Jalur merge/import sudah terintegrasi tetapi belum dijalankan karena Ollama CLI dan training dependencies tidak ada di environment saat ini.
