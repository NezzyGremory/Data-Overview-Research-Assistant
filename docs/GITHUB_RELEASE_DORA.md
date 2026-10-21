# GitHub Release DORA v5.0.0

## Sebelum rilis

Tag `v4.0.0` sudah ada di GitHub. Jangan memindahkan atau memakai ulang tag tersebut. Build saat ini ditargetkan sebagai rilis `v5.0.0` dengan nama aplikasi DORA.

Pastikan versi berikut sama-sama `5.0.0`:

- `APP_VERSION` pada `build_windows_installer.bat`
- `AppVersion` pada `installer/DatasetResearch.iss`

Build harus menghasilkan `hasil_compile/DORA-v5.0.0-Setup.exe`.

## Langkah membuat release

1. Jalankan tes yang relevan dan build ulang installer dengan `build_windows_installer.bat`.
2. Uji installer dan pastikan nama/version asset adalah `DORA-v5.0.0-Setup.exe`.
3. Commit perubahan source, konfigurasi build, dan dokumentasi yang memang akan dirilis. Jangan commit `.env`, `data/config.json`, database/data pengguna, `.venv`, atau folder payload hasil build.
4. Push commit rilis ke branch `main` di GitHub.
5. Buka repository **Dataset-Research** di GitHub, pilih **Releases**, lalu **Draft a new release**.
6. Pada **Choose a tag**, masukkan `v5.0.0` dan buat tag dari branch `main` yang sudah berisi commit rilis.
7. Isi judul `DORA v5.0.0`, lalu tempel deskripsi siap-pakai di bawah.
8. Unggah hanya `hasil_compile/DORA-v5.0.0-Setup.exe` sebagai asset.
9. Periksa tag, judul, deskripsi, dan asset, lalu pilih **Publish release**.

Jika tag `v5.0.0` sudah ada, pastikan tag tersebut menunjuk commit yang benar. Jangan force-push atau memindahkan tag release yang sudah dipublikasikan.

## Deskripsi release siap-tempel

```markdown
## DORA v5.0.0

DORA adalah aplikasi desktop untuk membantu peneliti meninjau dataset, menjalankan analisis lokal, mencoba workflow machine learning, dan mengeksplorasi literatur akademik.

### Sorotan

- Analisis dataset lokal untuk profiling, kualitas data, nilai kosong, duplikasi, outlier, korelasi, dan visualisasi.
- Pencarian dataset dari Kaggle dan Hugging Face.
- ML Intelligence untuk membantu meninjau task dan kandidat metode.
- Editor Analysis bergaya notebook dengan template Python yang dapat diedit dan dijalankan, output teks/grafik, serta dukungan notebook `.ipynb`.
- Pencarian paper akademik, Research Landscape, kandidat Research Gap, dan ekspor laporan.
- Installer Windows per pengguna dengan shortcut Start Menu/Desktop dan uninstaller.
- Aplikasi dan setup menggunakan nama DORA; data lokal lama tetap menggunakan lokasi `%LOCALAPPDATA%\Dataset Research`.

### Instalasi Windows

1. Unduh `DORA-v5.0.0-Setup.exe` dari bagian **Assets**.
2. Jalankan installer. Instalasi berlaku untuk pengguna Windows saat ini dan tidak memerlukan hak administrator.
3. Buka DORA dari shortcut Desktop atau Start Menu.

Persyaratan: Windows 10/11 64-bit. Koneksi internet diperlukan untuk pencarian dataset dan literatur. Kredensial API bersifat opsional untuk layanan yang mendukungnya.

### Catatan keamanan

Installer ini belum ditandatangani dengan sertifikat code-signing. Windows SmartScreen dapat menampilkan peringatan pada installer yang belum dikenal. Unduh hanya dari release resmi repository ini dan periksa file sebelum menjalankannya.

Kode yang dijalankan melalui Editor Analysis berjalan dengan akses lokal pengguna dan tidak diisolasi dalam sandbox. Jalankan hanya kode yang dipercaya.
```

## Catatan distribusi

- Unggah installer Inno Setup saja; jangan unggah folder `payload`, `.env`, `data/config.json`, database, dataset pengguna, atau `.venv`.
- Tidak ada cara packaging yang dapat menjamin Windows Defender atau SmartScreen tidak memberi peringatan. Authenticode signing dengan sertifikat tepercaya dapat membantu mengidentifikasi penerbit; sertakan checksum SHA-256 jika diperlukan.
- Sesuaikan sorotan dengan perubahan final yang benar-benar masuk ke commit release.
