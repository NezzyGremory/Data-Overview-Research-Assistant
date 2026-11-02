# Panduan singkat analisis dataset DORA

## Memuat dataset

Pada aplikasi desktop, pilih file tabel CSV, TSV, XLS, XLSX, JSON, JSONL, atau Parquet, atau unduh file dari Dataset Search. DORA membaca file tersebut menjadi satu tabel. File gambar, audio dan teks mentah memerlukan alur analisis lain. Periksa jumlah baris, kolom, tipe data, missing, dan baris duplikat sebelum menafsirkan hasil.

Sel JSON/Parquet yang berisi objek atau array disimpan sebagai teks JSON stabil untuk profil tabel; isi struktur tersebut belum dianalisis sebagai fitur numerik tersendiri. Nama kolom kosong atau nama yang berbenturan setelah spasi/BOM dibersihkan ditolak dengan pesan jelas supaya hasil tidak tertukar.

## Memilih task ML

Buka **ML Intelligence** lalu **Jalankan ML**. Saran task dari sistem adalah estimasi berdasarkan struktur data. Pilih sendiri task dan kolom target yang sesuai pertanyaan riset; dataset yang sama bisa dipakai untuk klasifikasi maupun clustering. Untuk klasifikasi biner, target harus berisi tepat dua kelas; untuk multikelas minimal tiga kelas. Clustering tidak memakai target. Pilihan algoritma otomatis hanya memilih kandidat algoritma untuk task yang telah dipilih.

Di **Opsi ML**, atur persentase data uji akhir (10–40%), jumlah fold validasi (2–10), lalu buka tab **Fitur** untuk mencentang kolom yang digunakan. Kolom target tidak tersedia sebagai fitur; kandidat label pada clustering tidak dicentang otomatis. Data asli tidak diubah oleh pilihan ini. Metode yang membutuhkan paket opsional akan menampilkan kebutuhan paketnya.

## Membaca hasil

Perhatikan protokol validasi dan metrik utama pada hasil. Skor CV mengukur performa pada fold yang digunakan untuk perbandingan kandidat dan belum otomatis menjadi hasil test independen. Pada clustering, Silhouette hanya tersedia bila sedikitnya dua cluster dapat dibandingkan; label noise dikeluarkan dari perhitungan. Jika semua titik menjadi noise atau hanya ada satu cluster, metode tersebut tidak dirangking. Banyaknya anomali yang ditandai tanpa label acuan bukan ukuran precision atau recall.

Untuk klasifikasi dan regresi dengan sedikitnya 50 baris serta kelas yang cukup, DORA menyisihkan data uji sebelum membandingkan model, lalu menguji hanya model teratas sekali pada data itu. Lihat **Pemilihan model** dan **Uji akhir terpisah** sebagai dua angka dengan tujuan berbeda. Jika data terlalu kecil, kelas langka, atau split tidak layak, uji akhir diberi status tidak tersedia beserta alasannya. Fitur yang identik dengan target dikeluarkan otomatis agar nilai metrik tidak terangkat oleh salinan label. Nilai tak hingga pada fitur numerik diperlakukan sebagai missing di dalam pipeline training.

**Riwayat eksperimen** pada halaman ML menyimpan setiap run ke SQLite proyek, termasuk pilihan task, target, fitur, algoritma, split, fold, seed, metrik, durasi, versi library, dan parameter model terbaik. Tombol **Ulangi** menjalankan pilihan yang tersimpan bila fingerprint dataset saat ini sama. Hasil dapat berbeda jika versi library atau data berubah; menjalankan ulang bukan jaminan bit per bit identik. Riwayat terikat pada proyek dan ikut dihapus bila proyek dihapus.

## Eksplorasi visual di Analysis

Jalankan **Jalankan Analysis**, lalu pilih jenis plot pada kartu **Data Visualization**. Daftar kolom menyesuaikan tipe chart: pilih satu kolom untuk histogram, box, violin, KDE, ECDF, atau Q-Q; pilih X dan Y untuk scatter; pilih kolom nilai dan kelompok untuk grouped box. Pair plot memakai maksimal empat kolom numerik bervariasi. Heatmap correlation/covariance memakai kolom numerik, sedangkan **Categorical Association** memakai kolom kategori yang sesuai.

Plot satu variabel/scatter memakai sampel maksimum 20.000 baris dan pair plot maksimum 2.000 baris, agar tetap responsif. Bar chart menampilkan hingga 20 kategori dengan hitungan terbesar. Covariance mempertahankan satuan kuadrat dan sensitif terhadap skala. Cramér's V merangkum kekuatan asosiasi kategori; nilai tersebut bukan p-value, bukti kausal, atau aturan untuk memilih task ML. Pair plot/Q-Q membantu eksplorasi visual, bukan pengganti validasi statistik.

## Uji statistik inferensial

Di halaman **Analysis**, pilih **Uji Statistik**, tentukan metode dan kolom yang sesuai rancangan penelitian, lalu tekan **Hitung Uji**. Dialog hanya menampilkan kolom numerik untuk pengukuran numerik; Anda tetap menentukan sendiri variabel kelompok atau pasangan. Contohnya, Welch t/Mann-Whitney memerlukan dua kelompok, sedangkan ANOVA/Welch ANOVA/Kruskal-Wallis memerlukan tiga kelompok atau lebih. Uji berpasangan menganggap setiap baris di dua kolom benar-benar pasangan yang sama.

Baca H0/H1, jumlah observasi, asumsi, p-value dan ukuran efek sebelum memilih **Simpan ke Analysis**. `p < 0,05` berarti hasil memberi bukti terhadap H0 menurut asumsi uji; `p ≥ 0,05` bukan bukti bahwa H0 benar. P-value tidak menunjukkan besarnya dampak praktis dan tidak membuktikan sebab-akibat. Interval kepercayaan dan ukuran efek hanya muncul untuk metode yang sudah didukung. Anderson-Darling memakai nilai kritis dan tidak menampilkan p-value. Jika menguji banyak hipotesis, koreksi multiple testing belum otomatis diterapkan.

Laporan menyebut asal pemilihan task dan target serta keterbatasan hasil yang tersedia. Temuan research gap adalah kandidat untuk ditinjau terhadap literatur yang lebih luas. Untuk membandingkan hasil DORA dengan Colab, gunakan file data, definisi target, preprocessing, split/seed, algoritma, parameter dan metrik yang sama.
