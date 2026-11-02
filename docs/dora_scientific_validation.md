# Validasi ilmiah DORA

## Status bukti

`tests/` berisi uji untuk loader, analyzer, kualitas, evaluator, graph, storage, report, riset dan UI. `evaluation/` mempunyai runner dan protokol benchmark terpisah; lihat `EVALUATION_PROTOCOL.md`. Kehadiran tes tidak berarti semua metode dan kombinasi dataset telah tervalidasi. Dataset nyata pengguna dan label Colab belum tersedia untuk perbandingan 1:1.

## Aturan evaluasi

- Laporkan `FAILED`/`NOT_EVALUABLE` bila metrik tidak terdefinisi; jangan mengisi 0 atau angka sentinel lalu merangkingnya.
- Data untuk belajar imputasi, encoding, scaling, seleksi fitur dan hyperparameter harus dibatasi pada train fold. Pada jalur supervised, keputusan fitur dan preprocessing kini berada di pipeline CV. Untuk dataset yang cukup, holdout akhir dipisahkan sebelum CV kandidat dan hanya model teratas yang diuji di sana. Skor CV adalah dasar pemilihan; skor holdout adalah estimasi akhir, bukan bukti performa pada populasi lain.
- Fitur yang identik dengan target dikeluarkan sebelum training. Ini menangkap salinan persis, tetapi belum semua bentuk target leakage atau hubungan sebab akibat. `+Inf`/`-Inf` pada fitur numerik menjadi missing di dalam pipeline fold; nilai target regresi yang tidak finite dikeluarkan dari analisis.
- Clustering internal: minimal dua cluster dan jumlah baris berlabel cluster lebih besar dari jumlah cluster. Label noise `-1` tidak dihitung sebagai cluster. Silhouette, Calinski-Harabasz dan Davies-Bouldin tidak membuktikan cluster cocok dengan label domain.
- Statistik deskriptif dan IQR hanya memakai nilai numerik finite; `±Inf` dicatat sebagai masalah kualitas, terpisah dari `NaN`. Metrik yang tidak terdefinisi dikembalikan sebagai tidak tersedia, bukan `NaN` atau angka pengganti.
- Deteksi anomali tanpa ground truth hanya bisa melaporkan jumlah/proporsi prediksi dan distribusi skor, bukan precision, recall atau AUC.
- EDA hubungan memakai covariance sampel berpasangan dan nilai numerik finite. Cramér's V yang dikoreksi bias hanya dipakai untuk kolom kategori/string/bool dengan kardinalitas terbatas; ini ukuran asosiasi deskriptif, bukan uji hipotesis atau bukti sebab-akibat. Heatmap menutupi nilai tidak terdefinisi alih-alih menggambarkannya sebagai nol.
- Covariance memakai sampel sistematis maksimum 20.000 baris dan 30 kolom numerik pertama; asosiasi kategori memakai sampel 20.000 baris, maksimum 20 kolom dan 20 level per kolom. Plot baris besar memakai maksimum 20.000 baris; pair plot memakai maksimum 2.000 baris dan empat kolom. Plot sampel membantu eksplorasi, tetapi tidak menggantikan inferensi berbasis rancangan sampling. KDE mengasumsikan estimator kernel Gaussian; Q-Q hanya pemeriksaan visual terhadap distribusi normal.
- Forecasting harus mempertahankan urutan waktu, menyatakan horizon, frekuensi, penanganan timestamp ganda, baseline, dan kebijakan nilai nol untuk MAPE.
- Uji inferensial perlu hipotesis dan desain sampling dari pengguna. Laporkan asumsi, ukuran sampel, p-value, CI/effect size bila tersedia, serta multiple testing bila menguji banyak hipotesis.

## Uji inferensial yang tersedia

Pengguna memilih satu dari 16 metode di halaman Analysis: Shapiro-Wilk dan Anderson-Darling untuk normalitas; Welch independent t-test dan Mann-Whitney untuk dua kelompok; paired t-test dan Wilcoxon signed-rank untuk pasangan; one-way ANOVA, Welch ANOVA, Kruskal-Wallis, Levene dan Bartlett untuk kelompok/varians; chi-square dan Fisher exact untuk asosiasi kategori; serta Pearson, Spearman dan Kendall untuk korelasi. Pemilihan metode tidak dilakukan hanya dari dtype.

Hasil mencatat variabel, jumlah observasi yang dipakai, hipotesis, asumsi pengguna/diagnostik, statistik, keputusan dan p-value bila tersedia. Interval tersedia untuk Welch t, paired t dan Pearson. Ukuran efek hanya dihitung untuk metode yang sudah memiliki rumus yang diimplementasikan; nilai yang tidak tersedia tidak diisi angka pengganti. Anderson-Darling memakai nilai kritis 5% dan tidak menghasilkan p-value. P-value belum dikoreksi untuk banyak hipotesis; untuk banyak uji, koreksi dan protokol analisis harus dilakukan di luar hasil ini. Dialog tidak memvalidasi mekanisme sampling, independensi, matching pasangan, pemilihan hipotesis sebelum melihat data, atau makna praktis efek.

## Rencana pembuktian berikutnya

1. Tambah dataset sintetis untuk all-noise, satu cluster, data campuran, missing, duplikat, imbalance, fitur konstan, dan indeks nonunik.
2. Uji estimator pada fixture kecil per keluarga model; tandai skip untuk dependency opsional yang belum dipasang.
3. Lanjutkan uji kebocoran pada fitur turunan target yang tidak identik, data berkelompok, dan observasi temporal. Holdout independen sudah ada untuk supervised tabular yang memenuhi syarat; Group K-Fold dan split temporal untuk klasifikasi/regresi belum ada.
4. Uji benchmark dataset publik dengan sumber target/task dan hash; hasil yang tidak tersedia tetap dilaporkan sebagai `TIDAK DIJALANKAN`.
5. Uji UI aktif dan report terhadap hasil gagal/tidak layak, termasuk screenshot pada halaman yang diubah.

Tes terarah inferensial membandingkan hasil yang didukung dengan SciPy dan memeriksa rancangan yang invalid, dialog, penyimpanan hasil dan report. Tes EDA mencakup covariance, asosiasi kategori, render chart, dan filtering nilai non-finite. Suite penuh setelah EDA: 295 lulus dan 61 warning deprecation pandas, tanpa warning statistik/runtime; rincian ada di `dora_implementation_status.md`. Tes terarah lainnya mencakup holdout klasifikasi/regresi, pilihan fitur/fold dari dialog ML, tracking SQLite, nilai non-finite, salinan target, dan loader nested JSON. Dataset pengguna serta benchmark publik belum diuji 1:1.
