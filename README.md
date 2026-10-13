# DORA — Data Overview & Research Assistant

DORA adalah aplikasi desktop untuk membantu peneliti memahami dataset, mencoba analisis machine learning, dan meninjau literatur yang berkaitan. Analisis dataset berjalan lokal di komputer. DORA membantu proses riset; hasil dan rekomendasinya tetap perlu ditinjau oleh peneliti.

Versi project saat ini: **v5.0.0**.

Nama tim pada tampilan aplikasi: **Metriks Studio**.

## Galeri aplikasi

Gambar di bawah menampilkan antarmuka DORA saat ini. Isi dataset dan hasil pencarian pada screenshot adalah **data demo lokal untuk dokumentasi**, bukan hasil pencarian internet atau data pengguna.

### Memuat dan meninjau dataset

Dataset dapat dipilih lewat tombol unggah atau dengan mengeklik/meletakkan file CSV pada area unggah. Setelah konfirmasi, halaman Dataset menyediakan pratinjau tabel dan tab kualitas data.

![Halaman unggah dataset](screenshots/readme/dataset-upload.png)

![Pratinjau dataset, ringkasan kualitas, dan kolom](screenshots/readme/dataset-preview.png)

Tab analisis dataset lainnya:

![Ringkasan Data Quality](screenshots/readme/dataset-data-quality.png)

![Tipe dan peran kolom](screenshots/readme/dataset-columns.png)

![Pemeriksaan Missing Values](screenshots/readme/dataset-missing-values.png)

![Pemeriksaan Outliers](screenshots/readme/dataset-outliers.png)

![Matriks Correlation](screenshots/readme/dataset-correlation.png)

### Dataset Search

Cari dataset publik berdasarkan kata kunci dari Kaggle, Hugging Face, atau keduanya lewat pilihan **All providers**. Screenshot hasil pencarian memakai rekaman demo.

![Pencarian dataset](screenshots/readme/dataset-search.png)

### Analisis dan eksperimen

Analysis merangkum struktur, statistik, kualitas data, nilai kosong, duplikasi, outlier, korelasi, dan grafik. **Editor Analysis** menyediakan tahap analisis dengan template kode Python yang bisa diedit dan dijalankan, output teks/grafik, notebook `.ipynb`, serta pemasangan paket Python dari PyPI.

![Analysis dataset](screenshots/readme/analysis.png)

![Editor Analysis](screenshots/readme/editor-analysis.png)

![Rekomendasi ML Intelligence](screenshots/readme/ml-intelligence.png)

### Ringkasan alur kerja

Overview mengumpulkan status dataset, kualitas, rekomendasi ML, kemajuan pipeline, dan pintasan ke laporan.

![Overview](screenshots/readme/overview.png)

### Literatur dan keluaran riset

Academic Research mencari serta memberi peringkat paper dari provider akademik yang tersedia. Papers menyediakan penyaringan artikel dan pemeriksaan status jurnal. SINTA adalah akreditasi nasional; indeks lain seperti DOAJ dan Scopus ditampilkan terpisah karena status indeks bukan peringkat SINTA.

![Academic Research](screenshots/readme/academic-research.png)

![Daftar dan pemeriksaan paper](screenshots/readme/papers.png)

![Research Landscape](screenshots/readme/research-landscape.png)

![Kandidat Research Gap](screenshots/readme/research-gap.png)

![Pratinjau Research Report](screenshots/readme/research-report.png)

### Panduan dan informasi aplikasi

![Halaman Help](screenshots/readme/help.png)

![Halaman About](screenshots/readme/about.png)

![Animasi credits Metriks Studio](screenshots/readme/copyright-credits.png)

## Fitur

### Dataset

- Unggah CSV dengan pemilih file, klik area unggah, atau drag-and-drop.
- Cari dataset publik melalui Kaggle dan Hugging Face; hasil bisa diunduh ke workspace DORA.
- Lihat pratinjau baris dan identifikasi tipe kolom.
- Tinjau Data Quality, Columns, Missing Values, Outliers, dan Correlation.
- Gunakan tindakan pembersihan data dan lanjutkan dataset ke tahap analisis.

### Analysis

- Profil ukuran dataset, tipe kolom, dan statistik deskriptif.
- Pemeriksaan nilai kosong, baris duplikat, outlier, dan korelasi.
- Ringkasan kualitas data, fingerprint dataset, serta visualisasi yang sesuai.
- Hasil analisis menjadi konteks untuk ML Intelligence dan pencarian literatur.

### Editor Analysis

Editor bergaya notebook ini menyediakan tahapan seperti Data Profiling, Data Cleaning, Exploratory Data Analysis, Feature Engineering, pemodelan, dan evaluasi.

- Pilih kelompok dan template, lalu edit kode Python sebelum dijalankan.
- Jalankan satu langkah atau seluruh rangkaian; hasil dapat berupa teks maupun grafik.
- Template mencakup beberapa metode klasifikasi, regresi, clustering, dan anomaly detection, termasuk metode ensemble.
- Pasang paket PyPI untuk workspace editor menggunakan perintah seperti `pip install seaborn`.
- Buka dan simpan notebook `.ipynb`.

Kode editor dijalankan secara lokal dengan akses ke lingkungan komputer dan **tidak diisolasi dalam sandbox**. Jalankan hanya kode yang kamu pahami dan percaya. Template adalah titik awal eksperimen, bukan jaminan bahwa kode cocok untuk semua dataset.

### ML Intelligence

- Memperkirakan jenis tugas berdasarkan karakteristik dataset dan target yang dipilih.
- Menampilkan metode kandidat beserta hasil evaluasi yang tersedia.
- Rekomendasi merupakan bahan pertimbangan; validasi model tetap menjadi tanggung jawab peneliti.

### Academic Research dan Papers

- Membentuk query dari konteks dataset dan mengumpulkan metadata publikasi.
- Menggabungkan dan menghapus duplikasi hasil, lalu menampilkan sinyal relevansi.
- Meninjau judul, penulis, tahun, jurnal/venue, abstrak, DOI, ISSN, sumber, dan status pemeriksaan jurnal bila tersedia.
- Memeriksa status SINTA secara terpisah dari indeks jurnal lain. Status dan masa berlaku akreditasi perlu cocok dengan tahun terbit artikel.
- Pemeriksaan indeks lain mendukung DOAJ; pemeriksaan Scopus memerlukan API key dan entitlement Elsevier. Hasilnya dapat berubah dan merupakan snapshot saat diperiksa.
- UCL Discovery dibuka sebagai pencarian eksternal dan hasilnya tidak otomatis dicampur ke daftar ranking.

Provider pencarian akademik yang digunakan aplikasi mencakup OpenAlex, Crossref, Semantic Scholar, Europe PMC, dan OpenAIRE. Scopus opsional bergantung pada konfigurasi serta hak akses API Elsevier. Pencarian akademik memerlukan koneksi internet.

### Landscape, Research Gap, dan Report

- **Research Landscape** merangkum metode, venue, dan pola publikasi dari paper yang terkumpul.
- **Research Gap** menampilkan kandidat celah dari literatur yang ditemukan. Kandidat bukan bukti kebaruan; periksa artikel asli dan publikasi terbaru.
- **Research Report** menyusun pratinjau gabungan dataset, kualitas data, ML, literatur, tren, dan kandidat gap serta dapat diekspor ke PDF setelah tahapan yang diperlukan tersedia.

### Overview, Help, dan Credits

- Overview menunjukkan status pipeline dan ringkasan hasil yang sudah tersedia.
- Help memuat panduan penggunaan dan pintasan ke halaman aplikasi.
- Tombol Copyright menampilkan animasi end credits; informasinya dapat diedit di `app/ui/data/credits.json`.

## Alur penggunaan singkat

1. Buka **Dataset** untuk memilih CSV, atau cari dataset publik di **Dataset Search**.
2. Tinjau tabel dan kualitas data pada halaman Dataset.
3. Jalankan **Analysis** dan periksa temuan sebelum membersihkan atau memodelkan data.
4. Gunakan **ML Intelligence** untuk meninjau task dan kandidat metode.
5. Gunakan **Editor Analysis** untuk mengubah dan menjalankan template kode secara manual.
6. Jalankan **Academic Research**, lalu tinjau paper, landscape, serta kandidat research gap.
7. Buka **Research Report** untuk melihat ringkasan dan mengekspor PDF jika semua tahap siap.

## Persyaratan dan teknologi

Snapshot lingkungan pengembangan yang digunakan saat ini:

- Python 3.13.12
- PySide6 6.11.2
- pandas, NumPy, SciPy, scikit-learn, PyArrow, dan matplotlib
- httpx untuk koneksi layanan pencarian
- SQLite lokal untuk penyimpanan aplikasi
- FastAPI dan MySQL hanya untuk API telemetry opsional

Versi di `requirements.txt` tidak dikunci; snapshot di atas bukan pernyataan rentang versi resmi.

## Menjalankan dari source

Di PowerShell pada folder project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```

Jika PowerShell memblokir aktivasi environment, jalankan Python di environment tersebut langsung:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

## Konfigurasi opsional

Salin `.env.example` menjadi `.env` hanya jika perlu mengatur layanan tambahan. Jangan commit `.env` atau menaruh kredensial di kode.

- `GEMINI_API_KEY`: penjelasan opsional berbasis Gemini pada fitur yang mendukungnya.
- `SEMANTIC_SCHOLAR_API_KEY`: opsional untuk batas pemakaian layanan yang lebih tinggi.
- `ELSEVIER_API_KEY`: diperlukan untuk pemeriksaan Scopus dan tetap bergantung pada entitlement akun Elsevier.
- `TELEMETRY_API_URL`: kosong secara default untuk menonaktifkan pengiriman telemetry.

Telemetry hanya mengirim event penggunaan yang dibatasi; konten dataset, hasil analisis, input pengguna, dan API key tidak disertakan dalam event.

## Struktur project

```text
app/
  ai/             Penjelasan research gap lokal dan layanan opsional
  analyzer/       Profiling, statistik, kualitas, dan diagnostik dataset
  core/           Konfigurasi, konstanta, exception, dan logger
  dataset_search/ Pencarian dan unduhan Kaggle/Hugging Face
  ml/             Deteksi task dan rekomendasi/evaluasi ML
  nlp/            Ekstraksi kata kunci, kemiripan teks, dan domain
  research/       Pencarian paper, ranking, indeks, landscape, dan gap
  reports/        Pratinjau dan ekspor laporan
  storage/        Penyimpanan lokal SQLite dan versi dataset
  ui/             Halaman PySide6, kernel editor, QSS, dan aset
  utils/          Utilitas bersama
backend/          API FastAPI opsional untuk telemetry
data/             Data runtime lokal; diabaikan oleh Git
docs/             Panduan project
screenshots/      Gambar dokumentasi UI
tests/            Tes otomatis
tools/             Utilitas pengembangan dan dokumentasi
run.py             Entry point aplikasi
```

## Pemeriksaan project

```powershell
python -m pytest tests/ -v
python -m evaluation.run --only all --offline --dev --quick
```

Runner evaluasi cepat bekerja offline. Evaluasi benchmark final memerlukan dataset dan label yang disiapkan sesuai [protokol evaluasi](docs/EVALUATION_PROTOCOL.md).

## Dokumentasi lain

- [Panduan GitHub Release DORA](docs/GITHUB_RELEASE_DORA.md) — langkah rilis v5.0.0, pemeriksaan versi/tag, dan deskripsi siap-tempel.
- [Panduan Windows Installer](docs/WINDOWS_INSTALLER.md)
- [Protokol Evaluasi](docs/EVALUATION_PROTOCOL.md)
- [Pengaturan Usage Telemetry](docs/USAGE_TELEMETRY.md)
- [Catatan UI](docs/UI_PROGRESS.md)

## Tim dan lisensi

DORA dikembangkan oleh **Metriks Studio** untuk kebutuhan pembelajaran, riset, dan portofolio. Lihat [LICENSE](LICENSE) serta lisensi masing-masing dependency dan layanan data sebelum mendistribusikan aplikasi.
