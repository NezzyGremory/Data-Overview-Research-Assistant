# Audit komprehensif DORA

Tanggal: 2026-11-02. Audit ini memeriksa inventaris source `app/`, `backend/`, `evaluation/`, `tests/`, konfigurasi build, dan jalur implementasi utama. Status **Existing** berarti implementasi nyata ditemukan, bukan berarti semua metode tervalidasi. Status **Partial** berarti sebagian kemampuan atau pembuktian masih kurang. Tidak ada klaim akurasi pada dataset pengguna yang belum disediakan.

## Peta implementasi

| Modul | Status | Bukti implementasi | Kekurangan dan rekomendasi |
| --- | --- | --- | --- |
| Dataset Profiling | Existing | `app/analyzer/profiler.py`, `statistics.py`, `fingerprint.py` menghitung dimensi, tipe, missing, duplikat, statistik kolom, kandidat target | Profil agregat belum mencakup seluruh detail kolom; audit performa pada tabel sangat besar dan tipe nested masih perlu. |
| EDA | Partial | `correlations.py` menghitung Pearson; `outliers.py` memakai IQR; `app/ui/main_window.py` dan report membuat beberapa grafik | Kovarians, asosiasi kategori, grouped analysis, ECDF, Q-Q, dan banyak grafik yang diminta belum menjadi engine terstruktur. Tambahkan hanya bila tipe data dan tujuan analisis sesuai. |
| Statistical Analysis | Partial | `statistics.py` berisi statistik deskriptif; `inferential_statistics.py` menambahkan 16 uji yang dipilih pengguna dan melaporkan hipotesis, asumsi, p-value/keputusan serta CI/effect size yang didukung | Tidak ada pemilihan uji otomatis, post-hoc, power, atau koreksi multiple testing. Rancangan sampling dan asumsi independensi/pairing tetap harus diverifikasi pengguna. |
| Data Quality | Existing | `data_quality.py`, `missing_values.py`, `duplicates.py`, `outliers.py`, `ml_readiness.py` memberi diagnosis dan saran | Nilai non-finite ditambahkan pada sesi ini; validasi rentang berbasis domain, konsistensi tipe dan integritas relasi belum lengkap. Saran bukan bukti bahwa perbaikan sudah diterapkan. |
| Preprocessing | Partial | `cleaner.py` punya operasi pembersihan; `app/ml/evaluation.py` memakai `ColumnTransformer`/`Pipeline` untuk imputasi, encoding, scaling | Pilihan pipeline pengguna, seleksi fitur, PCA, dan parameterisasi belum menyatu di ML Intelligence. Keputusan fitur supervised yang semula melihat seluruh data dipindahkan ke training fold pada sesi ini. |
| Machine Learning | Existing | `app/ml/registry.py` memuat 49 spesifikasi model; `intelligence.py` dan `evaluation.py` menjalankan estimator melalui scikit-learn/opsional | Registri bukan bukti semua model berhasil pada semua dataset. Semi-supervised, reinforcement learning, CNN/RNN/LSTM/Transformer, dan reduksi dimensi sebagai task belum terintegrasi. Banyak parameter eksperimen belum tersedia di UI. |
| Evaluation | Partial | Klasifikasi memakai CV pada data pengembangan dan uji akhir holdout untuk kasus yang memenuhi syarat; regresi memakai CV/holdout; forecasting memakai chronological holdout; clustering tidak merangking metrik tidak terdefinisi | Split group/time untuk klasifikasi/regresi belum ada; belum semua algoritma atau metrik divalidasi di benchmark publik. Evaluasi anomaly tanpa label hanya deskriptif, bukan precision/recall. |
| Experiment Tracking | Partial | ML Intelligence menyimpan run ke tabel `ml_experiments` SQLite, opsi, fingerprint, hasil dan metadata; Node Editor tetap memakai jalurnya sendiri | `ProjectManager` di graph belum terhubung ke page aktif. Perbandingan eksperimen lebih luas dan reproduksi lingkungan identik belum dijamin. `register_dataset` memakai loader bersama. |
| Research Intelligence | Existing | `app/research/` memiliki OpenAlex, Crossref, ranking, landscape, gap, serta sumber tambahan; `app/dataset_search/` punya Hugging Face/Kaggle | Cakupan bergantung provider, snapshot dan label eksternal. Kandidat gap bukan klaim kebaruan terbukti. Protokol benchmark ada di `docs/EVALUATION_PROTOCOL.md`, tetapi tidak membuktikan seluruh fitur lulus benchmark. |
| Report | Existing | `app/reports/generator.py` dan `templates/report.html` menghasilkan ringkasan dataset, kualitas, ML, hasil inferensial yang disimpan, literatur dan keterbatasan | Perbandingan run belum menjadi bagian laporan. Hasil inferensial yang disertakan bergantung pada uji yang dipilih dan batas CI/effect size yang tersedia. |
| Desktop UI | Existing | `app/ui/main_window.py` memasang page stack aktif dan worker Qt; Node Editor di `app/ui/node_editor/page.py` | `main_window.py` menggabungkan presentasi dan orkestrasi, sehingga perubahan harus terarah. Modul halaman mandiri bernama sama tidak semuanya dipakai page stack aktif. |
| API dan build | Partial | `backend/web_api.py` memiliki endpoint profil, analisis, bersih, search dan catalog; `requirements.txt`, spec dan installer tersedia | API ini jalur terpisah dari analisis desktop. Dependensi di `requirements.txt` belum dikunci; matriks kompatibilitas/CI packaging belum dibuktikan. Jangan menganggap versi snapshot `.venv` sebagai rentang resmi. |

## Alur aktif dan batas data

- `run.py` → `app/main.py` → `MainWindow` → `DatasetAnalysisWorker` → analyzer lokal → state workspace. ML memakai `MLIntelligenceEngine` dan `MLEvaluator`; report memakai state analisis/ML/riset.
- Page aktif berasal dari `MainWindow`. `app/ui/ml_page.py`, `analysis_page.py`, `upload_page.py`, dan beberapa file lain berdiri sendiri; nama modul tidak cukup untuk menyimpulkan bahwa ia aktif.
- SQLite utama ada di `app/storage/database.py`; `ProjectManager` di `app/ml/graph/project.py` punya skema SQLite sendiri tetapi belum dipakai oleh page Node Editor aktif. Page menyimpan riwayat run dalam state sesi. Jangan menyatukan schema tanpa migrasi dan pemetaan ID yang jelas.
- Loader saat ini menerima tabel CSV, TSV, XLS/XLSX, JSON, JSONL dan Parquet. Gambar, audio, teks mentah, dan data streaming bukan dataset yang bisa otomatis diperlakukan sebagai tabel.
- Data pengguna lokal dan `.env` tidak dibaca dalam audit ini. Evaluasi ilmiah pada dataset pengguna masih menunggu dataset, definisi target, dan protokol pembanding.

## Temuan prioritas

1. **P1 — generalisasi model:** CV memilih model pada data pengembangan dan holdout terpisah tersedia untuk kasus yang memenuhi syarat. Tambahkan split group/time dan validasi kebocoran target turunan sebelum klaim generalisasi untuk studi terkelompok/temporal.
2. **P1 — profil/EDA/kualitas:** perlu kontrak hasil per kolom, aturan invalid value yang tidak mengarang batas domain, asosiasi kategori dan grafik sesuai tipe data. Nested JSON saat ini diprofil sebagai teks, belum dianalisis secara semantik.
3. **P1 — cakupan tracking:** ML Intelligence punya riwayat eksperimen SQLite; Node Editor memakai jalur terpisah. Perbandingan lintas-workflow dan reproduksi penuh lingkungan belum ada.
4. **P2 — statistik inferensial:** engine eksplisit sudah tersedia untuk 16 uji. Tambahkan koreksi multiple testing, post-hoc, ukuran power, dan validasi protokol sampling tanpa mengotomatisasi pemilihan metode berdasarkan dtype.
5. **P2 — algoritma lanjut:** tambahkan hanya jika kontrak data, evaluasi, batas komputasi, serta uji algoritma telah jelas. Library opsional tidak otomatis berarti fitur tersedia.

## Verifikasi yang diperlukan

Uji unit dan integrasi harus dijalankan pada dataset sintetis yang mempunyai hasil referensi; benchmark publik perlu sumber label yang dapat ditelusuri. Khusus clustering, pisahkan noise `-1` dari cluster, jangan membuat skor numerik untuk kasus satu cluster/all-noise, dan jangan menyebut banyaknya outlier sebagai kualitas deteksi tanpa ground truth. Detail status terkini dicatat di `dora_implementation_status.md` dan protokol di `dora_scientific_validation.md`.
