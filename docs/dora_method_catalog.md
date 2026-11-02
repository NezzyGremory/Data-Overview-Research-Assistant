# Katalog metode analisis yang terpasang

Katalog ini membedakan **dijalankan oleh kode produksi**, **rekomendasi saja**, dan **belum ada**. Menu atau deskripsi metode tidak dianggap bukti implementasi.

| Area | Metode saat ini | Status dan batas |
| --- | --- | --- |
| Profil tabel | Dimensi, jumlah tipe kolom, memori, missing, duplikat; statistik numerik per kolom dan frekuensi kategori | Dijalankan oleh `profiler.py` dan `statistics.py`. Statistik deskriptif mengecualikan `±Inf` serta mencatat jumlahnya; data nested masih perlu penanganan eksplisit. |
| Fingerprint/target | Tipe semantik, kardinalitas, kandidat target dan sinyal identifier | Heuristik di `fingerprint.py`/`target_detector.py`; target final harus dikonfirmasi pengguna. |
| Korelasi EDA | Pearson untuk pasangan kolom numerik | Dijalankan oleh `correlations.py`. Tidak boleh ditafsirkan sebagai kausalitas. Dialog inferensial juga menyediakan Spearman/Kendall dan p-value terpisah. |
| Outlier | IQR 1,5 kali rentang kuartil pada kolom numerik finite | Dijalankan oleh `outliers.py`; `±Inf` ditangani sebagai nilai invalid, bukan outlier statistik. Outlier statistik belum tentu kesalahan data. |
| Kualitas data | Missing, duplikat, nilai numerik non-finite, outlier, identifier, kolom konstan, ketidakseimbangan kandidat target | Diagnosis bersyarat di `data_quality.py`; tidak otomatis memodifikasi data. Validasi nilai/rentang berbasis domain belum lengkap. |
| Pembersihan | Normalisasi placeholder, imputasi sederhana, drop missing, ekspor dan versi | Dijalankan oleh `cleaner.py`/`version_manager.py`. Pilihan penanganan harus mempertimbangkan mekanisme missingness. |
| Kesiapan ML | Distribusi, class balance, risiko leakage, rekomendasi split/pipeline/fitur | `ml_readiness.py` memberi saran; bukan pipeline eksperimen yang sudah dijalankan. |
| Inferensial | Shapiro-Wilk, Anderson-Darling, Welch t, Mann-Whitney, paired t, Wilcoxon, one-way ANOVA, Welch ANOVA, Kruskal-Wallis, Levene, Bartlett, chi-square, Fisher exact, Pearson, Spearman, Kendall | Dijalankan lewat pilihan eksplisit pengguna di `app/analyzer/inferential_statistics.py`; dialog Analysis menampilkan asumsi dan report menyimpan hasil. CI/effect size hanya tersedia untuk subset metode. Post-hoc, power, koreksi multiple testing, dan verifikasi rancangan sampling belum tersedia. |
| EDA lanjut | Sample covariance finite; bias-corrected Cramér's V; histogram, bar, box, violin, KDE, ECDF, normal Q-Q, scatter, grouped box, pair plot, correlation/covariance/association heatmap, missing values | Dijalankan pada halaman Analysis. Covariance memakai hingga 20.000 baris dan 30 kolom numerik, mempertahankan satuan kuadrat dan sensitif terhadap skala; Cramér's V membatasi 20 kolom/kategori dan merupakan deskripsi asosiasi, bukan uji signifikansi atau kausalitas. Plot dibatasi dengan sampling untuk merespons dataset besar; pair plot memakai maksimal empat kolom numerik dan 2.000 baris. Validasi domain dan rentang nilai belum tersedia. |

Metode yang belum tersedia tidak boleh ditampilkan sebagai hasil empiris. Uji inferensial tidak dipilih otomatis; pengguna harus mencocokkan uji dengan pertanyaan, sampling, kelompok, dan pairing. Hasil mencatat statistik, asumsi, p-value, interval ketidakpastian dan effect size bila tersedia; p-value belum dikoreksi untuk banyak pengujian.
