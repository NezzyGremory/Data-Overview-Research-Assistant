# Arsitektur DORA yang terpasang

## Jalur desktop

`run.py` menyiapkan environment, `app/main.py` membuat `QApplication`, dan `app/ui/main_window.py` membangun page stack PySide6. Halaman aktif meliputi Overview, Dataset (unggah, pencarian, pratinjau), Analysis, Analysis Editor, ML Intelligence, Academic Research, Insights/Gap, Research Report, dan Node Editor bila diaktifkan. Beberapa modul halaman mandiri di `app/ui/` adalah implementasi lain yang tidak otomatis dipakai oleh stack ini.

Dataset dibaca oleh `app/analyzer/loader.py` ke `current_dataset`. `DatasetAnalysisWorker` menjalankan profiler, statistik, missing, outlier, korelasi, covariance, asosiasi kategori dan fingerprint, lalu menyusun diagnosis kualitas serta kesiapan ML. Halaman Analysis memakai hasil hubungan tersebut bersama sampel DataFrame untuk membuat visualisasi EDA dengan batas kerja per plot. `MLIntelligenceEngine` menerima DataFrame dan fingerprint, mendeteksi task, memilih kandidat dan meminta `MLEvaluator` melatih estimator. Hasil bersama analysis, ML dan research menjadi masukan `ReportGenerator` untuk HTML/PDF.

Di halaman Analysis, `InferentialTestDialog` menerima pilihan uji dan kolom dari pengguna, lalu memanggil `InferentialStatisticsAnalyzer`. Hasil yang diterima disimpan di `analysis_result["inferential_statistics"]`, ditulis ke workflow proyek SQLite, ditampilkan kembali saat hasil analisis dirender, dan diteruskan ke template laporan. Jalur ini tidak memilih uji statistik otomatis.

Perhitungan numerik utama berlangsung lokal. Gemini di `app/ai/` adalah lapisan penjelasan; hasil numerik tidak boleh diambil dari jawaban generatif. Academic Research memakai klien metadata di `app/research/` termasuk OpenAlex/Crossref. Pencarian dataset memakai `app/dataset_search/` untuk Hugging Face/Kaggle. `backend/web_api.py` adalah jalur FastAPI terpisah; tidak diperlukan agar analisis desktop berjalan.

## Penyimpanan dan eksperimen

- `app/storage/database.py` menyimpan project, workflow, paper, versi dataset, jejak transformasi, dan `ml_experiments` ke SQLite utama. Tabel run dibuat dengan `CREATE TABLE IF NOT EXISTS` sehingga database lama dapat dibuka tanpa mengubah record lama. `ProjectRepository.record_ml_experiment` menambah satu snapshot per run; `list_ml_experiments` memberi data untuk perbandingan dan pengulangan di halaman ML.
- `app/ml/graph/project.py` menyediakan `ProjectManager` dengan schema SQLite lain untuk graph versions dan runs. Kelas ini belum dipakai oleh page Node Editor aktif. Page memakai `_run_history` dalam state sesi, dengan ekspor CSV dan pemulihan state. Run pada `ProjectManager` dapat menyimpan hash dataset, metrik, seed, durasi, versi library dan waktu, tetapi belum menjadi riwayat terpadu aplikasi.
- `app/ml/graph/engine.py` mengelola node, koneksi, validasi graph dan eksekusi. Jalur ini tidak identik dengan evaluator ML Intelligence.
- Run ML Intelligence kini terpisah dari state workflow terbaru. Riwayat Node Editor tetap memakai jalurnya sendiri; penyatuan lintas dua jalur belum dilakukan.

## Batas teknis

`app/ui/main_window.py` mencampur view, routing dan worker. Perubahan engine harus memakai kontrak hasil yang kompatibel dengan UI, report dan data yang sudah tersimpan. Model opsional pada registry memerlukan dependency tambahan; ketersediaan library saja belum membuktikan hasil ilmiah pada semua dataset. Dataset source tetap milik pengguna; pembersihan yang dibuat versi baru tidak boleh mengubah file asli.

Jalur supervised menyisihkan uji akhir sebelum cross-validation kandidat. Pilihan fitur dari dialog diteruskan ke `MLIntelligenceEngine`; `_FoldFeaturePreparation` dan `ColumnTransformer` belajar hanya pada train fold. Setelah pemilihan, model terbaik dilatih ulang pada data pengembangan dan diuji pada holdout yang belum dilihat. Report menampilkan protokol dan kedua jenis skor secara terpisah.
