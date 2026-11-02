# Status pengembangan komprehensif DORA

Tanggal: 2026-11-02. Dokumen ini adalah catatan pekerjaan pada checkout saat ini, bukan klaim bahwa seluruh master prompt sudah selesai.

| Fase | Status | Bukti dan pekerjaan berikutnya |
| --- | --- | --- |
| 1. Audit dan arsitektur | Selesai untuk inventaris source dan jalur utama; audit empiris lanjutan perlu | `dora_comprehensive_audit.md`, `dora_architecture.md`, katalog metode/algoritma. Perlu audit runtime pada dataset nyata dan benchmark. |
| 2. Profiling, EDA, kualitas | Partial | Engine lama dipertahankan; diagnosis nilai numerik non-finite ditambahkan. Statistik deskriptif dan IQR mengecualikan nilai itu. Loader kini menolak benturan nama kolom dan mengubah sel nested menjadi teks JSON stabil; bukan analisis semantik isi nested. Analysis kini menghitung covariance finite, bias-corrected Cramér's V, dan menampilkan histogram, bar, box, violin, KDE, ECDF, normal Q-Q, scatter, grouped box, pair plot, correlation/covariance/association heatmap, serta missing. Validasi domain/rentang tetap belum tersedia; plot pair dibatasi 4 kolom dan 2.000 baris sampel. |
| 3. Statistik inferensial | Partial | `app/analyzer/inferential_statistics.py` kini menyediakan 16 uji yang dipilih pengguna; dialog Analysis menampilkan hipotesis/asumsi dan menyimpan hasil ke proyek serta laporan. Post-hoc, koreksi multiple testing, analisis power dan rancangan studi lanjutan belum tersedia. |
| 4. Preprocessing dan fitur | Partial | Pipeline imputasi/encoding/scaling ada; keputusan kolom supervised dipelajari dalam training fold CV. Pengguna kini bisa memilih kolom fitur pada dialog ML; salinan target dikeluarkan dan nilai tak hingga pada fitur menjadi missing di pipeline. Opsi scaler, imputer, PCA, seleksi fitur otomatis dan hyperparameter search belum tersedia di UI. |
| 5. Supervised/unsupervised ML | Partial | Registry dan evaluator lama dipertahankan. Clustering tanpa metrik valid tidak masuk ranking; noise tidak diperlakukan sebagai cluster. Kandidat label tidak dicentang otomatis sebagai fitur clustering pada dialog ML. Dukungan semua model opsional dan task lanjut belum dibuktikan. |
| 6. ML lanjut | Belum tersedia sebagai alur terpadu | Semi-supervised, reinforcement learning, deep learning khusus, dan reduksi dimensi sebagai task masih perlu desain dan uji. |
| 7. Evaluasi/tracking | Partial | Supervised tabular yang memenuhi syarat kini memakai CV pada data pengembangan lalu uji akhir terpisah; dataset kecil diberi alasan tidak tersedia. Dialog mengatur porsi uji dan fold. `ml_experiments` SQLite menyimpan tiap run dan UI menampilkan riwayat serta pengulangan bila fingerprint sama. Node Editor tetap terpisah; Group K-Fold, temporal supervised, tuning dan perbandingan eksperimen mendalam belum ada. `ProjectManager.register_dataset` memakai loader bersama. |
| 8. Integrasi research | Existing, perlu validasi lanjut | OpenAlex/Crossref, landscape, gap dan report dipertahankan; belum ada pembuktian benchmark baru pada sesi ini. |
| 9. Pengujian/optimasi/dokumentasi | Partial | Tes regresi clustering, kualitas numerik, holdout, pilihan manual, tracking SQLite, kebocoran target dan loader nested ditambahkan. Screenshot offscreen halaman ML dan dialog tersedia. Uji interaksi desktop langsung, packaging, benchmark dataset publik dan performa besar masih harus dinilai terpisah. |

## Perubahan kode sesi ini

- `app/ml/evaluation.py`: metrik internal clustering hanya dihitung bila ada sedikitnya dua cluster valid, setelah noise `-1` dikeluarkan. Kasus yang tidak layak mendapat `NOT_EVALUABLE` dalam diagnostik dan tidak dirangking. Ranking memakai Silhouette yang benar-benar dihitung; skor komposit buatan tidak lagi dihasilkan.
- `app/ml/evaluation.py`: `_FoldFeaturePreparation` memilih kolom dan memperluas tanggal dari training fold. Preprocessor memilih kolom secara dinamis setelah tahap ini; imputasi, encoding dan scaling tetap berada dalam pipeline CV.
- `app/ml/intelligence.py`: hasil clustering baru ditampilkan dengan Silhouette sebagai metrik utama; pembacaan `cluster_score` lama tetap kompatibel untuk hasil tersimpan.
- `app/ml/graph/project.py`: metadata file registrasi pada utility `ProjectManager` diperoleh lewat `DatasetLoader` yang sama dengan desktop. File yang tidak dapat dibaca sekarang gagal eksplisit sebelum record dataset tersimpan. Utility ini belum dipanggil page Node Editor aktif.
- `app/analyzer/data_quality.py`: mendeteksi `+Inf`/`-Inf` sebagai nilai invalid kritis tersendiri dari `NaN`, dengan jumlah, kolom terdampak, dampak dan opsi penanganan. Diagnosis ikut masuk laporan lewat alur lama.
- `app/analyzer/statistics.py` dan `outliers.py`: statistik numerik/IQR memakai nilai finite saja; `NaN` dan `±Inf` dicatat terpisah, dan simpangan baku atau skewness yang tidak terdefinisi dikembalikan sebagai `None`.
- `tests/test_clustering_validation.py` dan `tests/test_graph_project.py`: kasus all-noise, satu cluster, noise bersama cluster valid, format CSV/JSONL/Parquet dan file rusak.

## Pengujian yang sudah berjalan

- `tests/test_clustering_validation.py`, `tests/test_evaluation.py`, `tests/test_intelligence.py`: **8 lulus**; 26 peringatan deprecation pandas dari kode lama, tanpa kegagalan tes.
- `tests/test_graph_project.py`, `tests/test_loader.py`: **16 lulus**.
- `tests/test_data_quality.py`, `tests/test_outliers.py`, `tests/test_statistics.py`: **10 lulus** pada uji terarah; mencakup pembedaan non-finite dari missing, laporan, IQR, dan statistik deskriptif.
- `tests/test_fold_feature_preparation.py`: **2 lulus** pada uji terarah; memastikan keputusan fitur dari train saja dan `fit` terpisah pada setiap fold CV.
- Suite repository pertama: **240 lulus, 1 gagal** karena tes lama mencari heading report Inggris `<h2>` sementara template aktif memakai heading Indonesia `<h3>`. Tes disesuaikan dengan kontrak report aktif.
- Riwayat suite sebelum perubahan terbaru: **246 lulus, 0 gagal, 74 peringatan**, lalu **248 lulus, 0 gagal, 58 peringatan** setelah perbaikan fold. Suite final pada 2026-11-02 setelah fitur statistik inferensial: **287 lulus, 0 gagal, 61 peringatan** (`.venv\\Scripts\\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=.pytest_dora_inferential_final_tmp --disable-warnings`). Peringatan terutama deprecation pandas pada kode lama; bukan kegagalan tes.
- Verifikasi penuh final setelah Advanced EDA: **295 lulus, 0 gagal, 61 warning** dalam 61,93 detik (`.venv\\Scripts\\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=.pytest_dora_eda_final3_tmp`). Seluruh warning adalah `Pandas4Warning` deprecation yang sudah ada: 59 penggunaan `is_categorical_dtype` di `app/ml/target_detector.py:141` dan 2 pemilihan `select_dtypes(include=["object"])` di `app/ml/intelligence.py:479`. Tidak ada warning runtime/statistik dan tidak ada warning dari tes EDA.
- `compileall` dan `git diff --check` lulus. Pengujian semua algoritma opsional, dataset publik, UI aktif, installer dan benchmark TEST belum dinyatakan lulus.

## Lanjutan sesi ini

- `app/ml/evaluation.py`: pemilihan model supervised memakai CV pada data pengembangan. Pada data yang memenuhi syarat (sedikitnya 50 baris dan minimal 5 observasi per kelas), holdout akhir dipisahkan terlebih dahulu lalu model teratas dinilai sekali. Skor seleksi dan skor akhir disimpan terpisah; data kecil/kelas langka memberi status `NOT_AVAILABLE`. Fitur salinan persis target dikeluarkan dan nilai tak hingga pada fitur numerik menjadi missing dalam pipeline.
- `app/ml/intelligence.py` dan `app/ui/ml_run_options_dialog.py`: pilihan fitur, porsi uji dan jumlah fold diteruskan ke evaluator. Kandidat label tidak dicentang otomatis ketika task clustering dipilih. Konfigurasi aktual disimpan dalam hasil.
- `app/storage/database.py`, `project_repository.py`, dan halaman ML aktif: migrasi tambah tabel `ml_experiments` dengan `CREATE TABLE IF NOT EXISTS`; tiap run menyimpan fingerprint, opsi dan hasil. Halaman menampilkan lima run terakhir, skor seleksi/uji akhir, serta tombol ulangi dengan pemeriksaan fingerprint. Metadata mencatat durasi, versi library, seed dan parameter model terbaik. Riwayat mengikuti umur proyek.
- `app/reports/generator.py` dan template: label validasi mengikuti fold/metode aktual, menampilkan konfigurasi dan hasil uji akhir terpisah, serta mencatat fitur salinan target yang dibuang. Keterangan kekuatan/keterbatasan yang tidak ada tidak lagi diisi klaim generik.
- `app/analyzer/loader.py` dan `statistics.py`: tabular JSON/Parquet dengan isi sel objek/array dapat diprofil sebagai teks JSON; nama kolom yang berbenturan ditolak jelas; pandas StringDtype mendapatkan frekuensi kategori.
- `app/analyzer/inferential_statistics.py`: menambahkan Shapiro-Wilk, Anderson-Darling, Welch t, Mann-Whitney, paired t, Wilcoxon, ANOVA, Welch ANOVA, Kruskal-Wallis, Levene, Bartlett, chi-square, Fisher exact, Pearson, Spearman dan Kendall. Uji harus dipilih eksplisit; asumsi, hipotesis, jumlah observasi, keputusan, CI dan effect size ditampilkan jika tersedia. Anderson-Darling memberi nilai kritis, bukan p-value.
- `app/analyzer/relationships.py` dan halaman Analysis: covariance dihitung dari sampel hingga 20.000 baris dan 30 kolom numerik finite; asosiasi kategori memakai maksimal 20 kolom/kategori dan Cramér's V yang dikoreksi bias. Pemilih chart menyesuaikan tipe data dan menyediakan ECDF, normal Q-Q, violin, KDE, scatter, grouped box, pair plot, dan heatmap hubungan. Pair plot memakai maksimal empat kolom dan 2.000 baris sampel; plot tunggal memakai maksimal 20.000 baris. Matriks memakai mask untuk nilai tidak terdefinisi; tidak mengubahnya menjadi nol.
- `app/ui/main_window.py`: halaman Analysis kini punya dialog uji statistik dengan pemilihan jenis uji/kolom, validasi rancangan, ringkasan asumsi, dan hasil yang tersimpan di workflow analysis proyek aktif. Hasil lama ditampilkan kembali ketika Analysis dirender.
- `app/reports/generator.py` dan `templates/report.html`: laporan memuat statistik inferensial, hipotesis, asumsi, keputusan, nilai kritis, CI/effect size bila tersedia, dan batas interpretasi. P-value tidak dikoreksi untuk pengujian lain.
- `tests/test_inferential_statistics.py`, `tests/test_inferential_statistics_ui.py`, dan `tests/test_reports.py`: tes inferensial terarah membandingkan metode dengan SciPy serta memeriksa dialog, penyimpanan, dan laporan. Suite saat itu 287 lulus sebelum tes EDA ditambahkan.
- `tests/test_relationships.py`, `tests/test_analysis_visualization.py`, dan `tests/test_analysis_layout.py`: 13 tes terarah lulus, mencakup covariance/Cramér's V, render chart, filtering non-finite, sampling Q-Q, dan layout halaman.
- Screenshot offscreen: `screenshots/analysis_eda_final.png` menampilkan scatter dengan kontrol X/Y. Screenshot lain yang sudah tersedia: `screenshots/ml_intelligence_final.png`, `screenshots/ml_run_options_final.png`, `screenshots/analysis_inferential_final.png`, dan `screenshots/analysis_inferential_dialog_final.png`. Desain sumber `design/html_export/` dan `design/tokens.md` yang disebut AGENTS.md tidak ada pada checkout ini, sehingga UI mengikuti QSS aktif.

## Fitur lama yang dipertahankan dan batas saat ini

PySide6, SQLite, loader, Dataset Search, Analysis, ML Intelligence, Node Editor, Analysis Editor, OpenAlex/Crossref, Gemini opsional, Research Landscape/Gap dan report tetap pada arsitektur lama. Tidak ada dependency atau migrasi schema baru dalam sesi ini. Pembacaan metadata file melalui loader memuat tabel ke memori, sehingga batas ukuran loader masih berlaku. Format non-tabel memerlukan parser/kontrak analisis sendiri.

Prioritas berikutnya: koreksi multiple testing dan post-hoc, validasi group/time split serta kebocoran target tidak langsung, lalu lengkapi preprocessing dan metrik per task. Jangan mengklaim seluruh master prompt selesai sebelum dukungan algoritma lanjut, benchmark publik, dan UI lintas halaman benar-benar diuji.

## Audit final dan optimasi terukur — 2026-11-02

- Baseline sebelum perubahan: **295 tes lulus, 61 warning pandas, 0 gagal**. Warning berasal dari `is_categorical_dtype` (59) dan inferensi implicit `select_dtypes(include=["object"])` (2).
- `app/ml/target_detector.py` kini memakai `isinstance(dtype, pd.CategoricalDtype)`. `app/ml/intelligence.py` menghitung object/string/category/bool secara eksplisit, termasuk `StringDtype`; tidak ada suppression warning.
- Default loader CSV membaca langsung ke DataFrame. Parameter chunk eksplisit masih tersedia dan regresi memeriksa kedua hasil sama.
- Alur unduh-dan-analisis memakai callback setelah page Analysis lazy dibuat; tes urutan callback kini tersedia.
- Full suite final: **301 passed, 0 failed, 0 warnings dalam 62,94 detik**. `compileall` dan `git diff --check` lulus.
- Median loader pada CSV sintetis 120.000 × 9: chunked 3,512 dtk; langsung 3,332 dtk (sekitar 5,1% lebih cepat pada mesin benchmark). Puncak RSS median 30,36 MB vs 30,64 MB; tidak ada penghematan RAM yang terukur.
- Startup source/offscreen: 8,51 dtk / 294,63 MB baseline dan 8,08 dtk / 294,59–295,15 MB sesudah. Tidak ada optimasi startup atau idle RAM yang dapat diklaim.
- Detail temuan dan batas verifikasi ada di `docs/final_audit_report.md`, `docs/performance_optimization_report.md`, dan `docs/regression_test_report.md`.
- Smoke test PyInstaller mendapati child mode `--dora-editor-pip` gagal karena `sys.stdout` bernilai `None` di aplikasi windowed. `app/main.py` kini memulihkan stdio dari pipe OS atau fallback aman; tes unit lulus. Build akhir berhasil dan EXE mengembalikan `pip 25.3` dengan exit code 0.
- Distribusi `dist/DORA` berukuran **335.546.836 byte** (1.612 file). PyInstaller tetap mengeluarkan warning hidden import `scipy.special._cdflib` yang belum divalidasi pada semua alur SciPy packaged.

Fitur, pengujian jaringan live, evaluasi benchmark publik, split group/time, post-hoc, multiple-testing correction, dan UI semua halaman belum dinyatakan selesai hanya dari suite ini.
