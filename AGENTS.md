# Panduan Project Dataset Research

## Ringkasan Project
Aplikasi desktop untuk analisis dataset dan dukungan riset akademik. Analisis utama berjalan lokal; API FastAPI terpisah digunakan untuk telemetry opsional.

## Tech Stack
- Environment `.venv` yang diperiksa dari daftar package: Python 3.13.12, PySide6 6.11.2 (PySide6-Addons, PySide6-Essentials, dan shiboken6 juga 6.11.2).
- Aplikasi: pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, PyArrow 25.0.1, matplotlib 3.11.2, httpx 0.28.1, Jinja2 3.1.6, python-dotenv 1.2.3.
- API telemetry: FastAPI 0.143.0, Uvicorn 0.54.0, Pydantic 2.13.5, mysql-connector-python 9.7.0.
- Tes dan packaging: pytest 9.1.1, Pillow 12.3.0, PyInstaller 6.22.3. Penyimpanan desktop menggunakan SQLite melalui standard library Python.
- Semua versi package di atas cocok dengan daftar package `.venv` yang diperiksa. `requirements.txt` tidak mengunci versi; daftar ini bukan jaminan versi untuk instalasi lain.

## Struktur Folder
- `app/main.py`, `run.py`: entry point aplikasi.
- `app/ui/`: window, halaman, widget, stylesheet, dan aset UI.
- `app/analyzer/`: pemuatan, profiling, statistik, kualitas, dan diagnostik dataset.
- `app/ml/`, `app/nlp/`: deteksi task/rekomendasi ML dan pemrosesan bahasa.
- `app/research/`: pencarian, ranking, landscape, tren, dan kandidat research gap.
- `app/ai/`: penjelas berbasis layanan eksternal atau lokal.
- `app/dataset_search/`: klien pencarian/unduhan Hugging Face dan Kaggle.
- `app/storage/`: SQLite, repository project, versioning, dan service riset.
- `app/reports/`: pembentukan laporan dan template HTML.
- `app/core/`: konfigurasi, konstanta, exception, dan logger.
- `app/utils/`: folder tersedia, saat diperiksa belum berisi file.
- `backend/`: API FastAPI untuk telemetry; bukan backend analisis desktop.
- `evaluation/`: runner dan metrik evaluasi terpisah dari alur aplikasi.
- `tests/`: tes otomatis; `docs/`: panduan; `screenshots/`: gambar dokumentasi.
- `data/`: database dan data runtime lokal. `installer/`, spec, dan script build: packaging Windows.

## Cara Menjalankan Aplikasi
- Dari root project: `python run.py`.
- Tes: `python -m pytest tests/ -v`.
- Evaluasi cepat offline: `python -m evaluation.run --only all --offline --dev --quick`.
- API telemetry opsional: `python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000`.

## Aturan UI
- UI dibuat langsung dengan widget/layout Python; tidak ditemukan file Qt Designer `.ui`.
- Stylesheet utama: `app/ui/app.qss`, dimuat oleh `MainWindow._load_stylesheet()` di `app/ui/main_window.py`.
- Utamakan styling bersama di `app/ui/app.qss`; beberapa style inline masih ada, jadi jangan menambah inline style tanpa alasan.
- QSS tidak mendukung `box-shadow`. Untuk bayangan gunakan `QGraphicsDropShadowEffect` di Python.
- Aset UI yang ditemukan berada di `app/ui/assets/`.
- Kerjakan satu halaman per permintaan. Pertahankan signal/slot, method handler, dan alur data yang sudah ada.
- Kerjakan satu halaman per permintaan. Pertahankan signal/slot, method handler, dan alur data yang sudah ada.

## Aturan Backend
- Untuk permintaan UI, jangan mengedit file di `app/analyzer/`, `app/ml/`, `app/nlp/`, `app/research/`, `app/ai/`, `app/storage/`, `app/dataset_search/`, `app/reports/`, `backend/`, atau mengubah `app/telemetry.py`.
- `app/ui/main_window.py` memuat UI sekaligus orkestrasi, worker/thread, handler, dan operasi data. Edit bagian tampilan seperlunya; jangan mengubah handler, koneksi, worker, atau logika data untuk tugas visual.
- Jangan mengubah `.env`, `data/`, database, atau file konfigurasi runtime saat mengerjakan UI. Jalur penutupan window mengelola dan membersihkan data sesi; jangan mengubah lifecycle ini untuk pekerjaan tampilan.

## Cara Menyambungkan UI Baru ke Backend
- Hubungkan kontrol baru dengan pola Qt yang sudah dipakai, misalnya `button.clicked.connect(handler)`, dan teruskan aksi ke handler/halaman yang ada.
- Jangan menduplikasi atau memindahkan logika backend ke widget. Perubahan hanya pada presentasi/layout dan pertahankan alur data yang lama.

## Alur Kerja Desain Figma
- Gunakan Figma MCP untuk pekerjaan desain Figma.
- Kerjakan satu halaman per permintaan; implementasikan hanya halaman yang diminta.
- Setelah implementasi, jalankan aplikasi dan periksa tampilan serta interaksi kontrol.

## Konvensi Kode
- Class memakai PascalCase; fungsi dan variabel umumnya snake_case; indentasi Python 4 spasi.
- Komentar/docstring dan teks UI menggunakan campuran Bahasa Indonesia dan Inggris; ikuti bahasa/konteks file yang disentuh.

## Area Berisiko
- `MainWindow` mengatur navigasi antar halaman. Perubahan pada konstruksi page stack, callback, atau worker dapat memutus alur aplikasi.
- Jangan mengedit hasil build di `build/`, `dist/`, atau `hasil_compile/` secara manual.

## Perlu Konfirmasi
- Versi minimum/maksimum resmi PySide6 belum ditetapkan; `requirements.txt` tidak mengunci versi. Versi dalam Tech Stack adalah snapshot environment lokal. Tanyakan kepada pengguna sebelum menyatakan rentang dukungan resmi.
- Beberapa modul halaman berdiri sendiri di `app/ui/` tidak tampak terhubung ke page stack aktif `MainWindow`. Jika halaman yang dimaksud tidak jelas dari wiring aktif, tanyakan kepada pengguna sebelum memilih modul untuk diubah.