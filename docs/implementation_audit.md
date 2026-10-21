# Audit implementasi Dataset Research (DORA)

Tanggal audit: 2026-10-16 (Asia/Jakarta)

## Ringkasan

Aplikasi adalah desktop Python/PySide6. Analisis dataset berjalan dari aplikasi desktop dan modul layanan lokal. `MainWindow` menangani shell, navigasi, orkestrasi UI, dan beberapa worker; service analisis, riset, penyimpanan, dan laporan berada di package masing-masing. Backend telemetry FastAPI sudah dihapus karena tidak lagi digunakan.

Audit ini memetakan struktur kode dan sumber desain yang tersedia. Belum ada perubahan UI. Implementasi dibatasi satu frame/halaman per sesi sesuai `AGENTS.md`; frame target masih menunggu pilihan karena link Figma menunjuk ke halaman kanvas yang berisi beberapa layar.

## Kondisi repository

- Root: `C:\Users\Lenovo\Dataset-Research`.
- Branch: `main`.
- Commit terakhir: `94bda72 backup: sebelum implementasi UI Figma`.
- Status sesudah konfirmasi backup: dua file terlacak dalam `.pytest_tmp/` berstatus terhapus: `test_execution_order_cache_and0/graph.json` dan `test_export_python_script_exec0/run_pipeline.py`. Git juga memperingatkan direktori tersebut tidak dapat dibuka. Penghapusan ini sudah ada sebelum pekerjaan implementasi; tidak diubah oleh audit.
- Tes belum dijalankan pada tahap audit ini. Tes UI/fungsional dan screenshot akan dijalankan setelah frame target ditetapkan.
- Tidak ada file `.env`, database, konfigurasi runtime, atau folder data yang dibaca untuk memperoleh nilai rahasia.

## Entry point dan environment

- `run.py` memanggil `init_environment()` lalu `app.main.main()`.
- `app/main.py` mengatur high-DPI scaling, membuat `QApplication`, memuat `MainWindow`, lalu menjalankan event loop.
- Snapshot versi `.venv` pada audit ini mencakup Python 3.13.12, PySide6 6.11.2, pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, PyArrow 25.0.1, matplotlib 3.11.2, httpx 0.28.1, Jinja2 3.1.6, python-dotenv 1.2.3, pytest 9.1.1, Pillow 12.3.0, dan PyInstaller 6.22.3. `requirements.txt` tidak mengunci rentang dukungan resmi.

## UI dan navigasi aktif

Shell dan page stack aktif dibangun oleh `MainWindow` di `app/ui/main_window.py`. Sidebar mengubah halaman melalui key internal; alias berikut berbagi widget yang sama:

- `dashboard` → `DashboardPage` dari `app/ui/overview_page.py`.
- `dataset` dan `dataset_search` → shell dataset dengan tab internal `UploadPage`, `DatasetSearchPage`, dan `DatasetPreviewPage`.
- `analysis` → `AnalysisPage`.
- `editor_analysis` → `EditorAnalysisPage`.
- `node_editor` → `NodeEditorPage`, ditambahkan ketika `NODE_EDITOR_UI_ENABLED` aktif.
- `ml` → `MLIntelligencePage`.
- `research` dan `papers` → `PapersToolPage`.
- `insights`, `landscape`, dan `gap` → `ResearchInsightsPage`; landscape dan gap ditampilkan sebagai tab internal.
- `report` → `ResearchReportToolPage`; `about`, `copyright`, dan `help` punya halaman masing-masing.

Beberapa modul halaman berdiri sendiri dan tidak dipakai oleh page stack aktif yang diperiksa, termasuk `app/ui/dashboard.py`, `app/ui/analysis_page.py`, `app/ui/research_page.py`, `app/ui/ml_page.py`, `app/ui/upload_page.py`, dan `app/ui/Dataset preview.py`. `ResearchPage` juga dibuat di `MainWindow` namun disembunyikan; key navigasi `research` memakai `PapersToolPage`. Pemilihan target harus mengikuti wiring aktif, bukan nama file yang mirip.

## Design system dan aset

- Stylesheet utama adalah `app/ui/app.qss`, dimuat dari `MainWindow`.
- `app/ui/scale.py` menyediakan `ScaleManager`, skala teks/QSS, preset skala, dan helper ukuran px.
- Font lokal tersedia di `app/ui/assets/fonts/` (antara lain Inter, Plus Jakarta Sans, dan Intel One Mono); ikon SVG tersedia di `app/ui/assets/icons/`.
- `design/tokens.md` dan `design/html_export/` yang disebut di `AGENTS.md` tidak ada di checkout ini. Dokumentasi `docs/UI_PROGRESS.md` mencatat pekerjaan desain terdahulu; halaman dan node di sana berbeda dari halaman Figma yang kini ditautkan, sehingga dicatat sebagai konteks lama, bukan sumber visual terbaru.

## Lapisan fitur dan data

- `app/analyzer/`: pemuatan, profiling, statistik, fingerprint, nilai kosong, duplikasi, outlier, korelasi, kualitas, dan kesiapan ML.
- `app/ml/`: deteksi task/target, rekomendasi metode, intelligence, dan evaluasi. `app/ml/graph/` menyediakan engine, registry, project, exporter, serta dukungan graph/node.
- `app/research/` dan `app/nlp/`: query, pencarian metadata (termasuk OpenAlex dan Crossref), deduplikasi, ranking, tren, landscape, gap, verifikasi jurnal, kata kunci, similarity, dan embeddings.
- `app/ai/`: penjelasan Gemini opsional, penjelasan lokal, dan penjelasan research gap.
- `app/dataset_search/`: klien Hugging Face dan Kaggle.
- `app/storage/`: SQLite, repository project, service riset, versioning dataset, dan transformation trail.
- `app/reports/`: pembuatan laporan HTML; handler PDF juga terdapat di `MainWindow`.
- Worker berbasis `QObject`/`QThread` untuk analisis, ML, pencarian/riset, pemeriksaan jurnal, dan laporan terlihat di UI. Rincian interaksi setiap halaman belum diuji runtime dalam audit ini.

`app/storage/database.py` membuat tabel dengan `CREATE TABLE IF NOT EXISTS`, termasuk `projects`, `research_results`, `project_workflows`, `papers`, `dataset_versions`, `transformation_trail`, dan tabel screening jurnal. Ada penambahan kolom kompatibilitas dengan `ALTER TABLE`; perubahan skema tidak diperlukan untuk pekerjaan UI ini. Mode pengembangan menyimpan data di root `data/`, sedangkan aplikasi terpaket memakai direktori data pengguna melalui `app/core/config.py`.

## Pengujian dan risiko perubahan

- Suite `tests/` mencakup analyzer, ML, graph, riset, storage, laporan, dataset search, API, dialog, dan skala UI. Baseline belum dijalankan.
- `app/ui/main_window.py` berukuran besar dan memuat presentasi serta orkestrasi. Untuk implementasi halaman, edit bagian visual yang diperlukan saja; pertahankan handler, signal/slot, worker, dan alur data.
- Page class bernama sama terdapat di modul aktif dan modul lama/mandiri. Periksa import dan page stack sebelum mengubah kelas.
- Hasil desain Figma memuat aset sementara; aset yang dipakai harus tersedia secara lokal dan dicocokkan dengan aset proyek sebelum screenshot akhir.
- Tidak ada perubahan pada database, `.env`, `data/`, backend, atau logika data selama audit.

## Cara menjalankan

```powershell
python run.py
```

Perintah tes repository yang tercatat:

```powershell
python -m pytest tests/ -v
```
