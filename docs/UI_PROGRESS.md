# Kemajuan Penyesuaian UI Figma

Sumber desain: DORA, Page 1 (`oLQhZyWW2vIPM9FmqvYewD`). Ukuran frame utama: **1920 × 1080 px**.

## Inventaris Figma

| Halaman / kondisi | Node ID | Frame | Padanan aktif di aplikasi | Status |
|---|---:|---:|---|---|
| Dataset, belum ada dataset | `2:4` | 1920 × 1080 | `UploadPage` dalam `datasets_page` | selesai |
| Dataset, dataset terisi | `24:42` | 1920 × 1080 | `DatasetPreviewPage` | selesai |
| Popup Upload Dataset | `11:101` | 1920 × 1080 | `UploadDatasetDialog` dengan overlay | selesai |
| Analysis | `31:406` | 1920 × 1080 | `AnalysisPage` | selesai |
| Dataset Search | `41:221` | 1920 × 1080 | `DatasetSearchPage` | selesai |
| Research Landscape | `54:843` | 1920x1080 | `ResearchLandscapeToolPage` di `ResearchInsightsPage` | selesai |
| Overview, versi terbaru | `144:275` | 1920 x 1080 | `DashboardPage` | selesai |
| About | `141:373` | 1920 x 1080 | `AboutPage` | selesai |
| Overview lama/duplikat, tidak dipakai | `104:192` | 1920 × 1080 | duplikat `DashboardPage` | dilewati — versi lama |

Komponen lepas pada canvas: `Drop duplicates` (`24:95`, 172 × 38 px), `Full missing values` (`31:390`, 204 × 38 px), `Remove column` (`31:395`, 182 × 38 px), dan ikon informasi (`144:520`, 16 × 16 px). Tiga chip cleaning termasuk bagian Dataset terisi.

Gaya dasar yang dibaca dari Figma: kanvas `#F1F7FF`; biru `#006BB8`, `#0071C2`, dan `#0D99FF`; font Inter dan Plus Jakarta Sans, dengan Intel One Mono untuk sidik jari dataset; kartu radius 15 px, tombol kecil 8 px dan chip atas 20 px. Frame yang tersedia tidak menunjukkan variasi hover, pressed, focus, loading, atau error tersendiri.

## Status Pekerjaan

| Area | Status | Catatan |
|---|---|---|
| Fondasi token, shell, skala, dan screenshot | selesai | Shell/sidebar dan topbar dicocokkan ulang ke posisi node Figma; token serta utilitas screenshot tersedia. `Editor Analysis` kini ditampilkan di sidebar sesuai permintaan pengguna; callback internal lama dipakai. |
| Dataset kosong dan terisi | selesai | Dua kondisi disetel ke frame 1920×1080; layout kosong turun menjadi satu kolom pada jendela lebih kecil. Sinyal lama dipertahankan. Screenshot dibandingkan dengan frame Figma. |
| Popup Upload Dataset | selesai | Dialog tepat 1027×757 px pada x=447, y=162; ukuran field dan tombol dicocokkan dari metadata. Switch menampilkan knob putih. Layout ringkas dipakai saat tinggi layar di bawah ukuran desain. Screenshot dibandingkan dengan Figma. |
| Analysis | selesai | Banner, tombol, area visualisasi, kartu fingerprint, dan kartu kualitas dicocokkan ke koordinat node `31:406`. Subtitle responsif dan detail hasil tetap tersedia di bawah ringkasan. Screenshot 1920×1080 dibandingkan dengan Figma; 1366×768 diperiksa agar teks membungkus tanpa menumpuk. |
| Dataset Search | selesai | Judul, deskripsi, panel pencarian, dua kolom kartu hasil, metadata, file, dan tombol dicocokkan ke koordinat node `41:221`. Kontrol pencarian dan aksi download tetap memakai handler lama. Screenshot 1920×1080 dibandingkan dengan Figma; 1366×768 diperiksa untuk susunan responsif. |
| Overview | selesai | Implementasi frame `144:275`; judul, temuan, pipeline, dan tiga kartu mengikuti frame. Tombol laporan dan detail tetap memakai handler lama. Screenshot 1920x1080 dibandingkan dengan Figma; 1366x768 diperiksa untuk susunan responsif. |
| Research Landscape | selesai | Tombol `Refresh Landscape`, ringkasan 3 kartu, distribusi metode, dan venue publikasi dicocokkan ke node `54:843`. Tab internal disembunyikan; navigasi sidebar ke Research Gap tetap aktif. Screenshot 1920x1080 dibandingkan dengan frame Figma; 1366x768 diperiksa dan konten bertumpuk/bergulir tanpa tumpang tindih. |
| About | selesai | Deskripsi aplikasi dipertahankan. Kartu ML Intelligence di halaman About dihapus. About, Copyright, dan Help berada di kiri bawah dengan ukuran lebih ringkas; nama tim di topbar dan halaman Copyright menjadi `Metriks Studio`. Revisi ini tidak mengambil screenshot sesuai instruksi pengguna. |
| ML Intelligence | dilewati | Tidak ada frame halaman di Figma. |
| Academic Research | dilewati | Tidak ada frame halaman di Figma. |
| Papers | diperiksa | Tombol dan filter SINTA yang ada tetap tersambung. Pembaruan tabel dibatasi agar tabel tidak dirender ulang untuk setiap hasil. Daftar artikel tetap berasal dari Academic Research; verifier memakai ISSN/nama jurnal, masa berlaku per tahun, dan DOI artikel jika tercantum di profil resmi. Kode pemeriksa di `app/research/` tidak diubah. |
| Research Gap | dilewati | Tidak ada frame halaman di Figma. |
| Research Report | dilewati | Tidak ada frame halaman di Figma. |
| Help dan Copyright | dilewati | Kontrol tampil di shell Figma, tetapi tidak ada frame isi/tujuan. |
| Editor Analysis | ditampilkan | Ditambahkan ke kelompok Workspace dengan ikon hitam; key `editor_analysis` dan callback lama tetap digunakan. |

## Perubahan Nama yang Terlihat

| Teks aktif di aplikasi | Teks Figma | Lokasi |
|---|---|---|
| `Datasets` | `Dataset` | Judul halaman `datasets_page` |
| `Dashboard` (judul halaman internal) | `Overview` | Judul `DashboardPage` bila tampil |
| `Editor Analysis` | `Editor Analysis` | Ditampilkan atas permintaan pengguna; kunci internal tetap `editor_analysis` |
| `Metriks v5.0.0` | `Metriks Studio` | Identitas tim terbaru yang diminta pengguna; tampil di topbar dan halaman Copyright |

Nama menu utama lainnya sudah sejalan dengan Figma. Navigasi dan backend memakai kunci internal seperti `dataset`, `analysis`, `dashboard`, dan `research`; perubahan label tidak akan mengubah kunci tersebut.

## Catatan Implementasi

- Font Figma sudah tersedia di `app/ui/assets/fonts/`: Inter, Plus Jakarta Sans, Intel One Mono.
- Ikon SVG yang sudah ada di `app/ui/assets/icons/` akan dipakai ulang setelah dicocokkan dengan slot Figma.
- `tools/screenshot_ui.py` menangkap 8 keadaan halaman pada 1920×1080 dan menerima ukuran lain.
- Bukti Dataset: `screenshots/app_dataset-empty-final.png`, `screenshots/app_dataset-loaded-final.png`, dan uji responsif `screenshots/app_dataset-empty-1366x768.png`.
- Bukti popup: `screenshots/app-popup-final.png` dan uji responsif `screenshots/app-popup-1366x768.png`.
- Bukti Analysis: `screenshots/app-analysis-final.png` dan uji responsif `screenshots/app-analysis-1366x768.png`.
- Bukti Dataset Search: `screenshots/app-dataset-search-final.png` dan uji responsif `screenshots/app-dataset-search-1366x768.png`.
- Bukti Research Landscape: `screenshots/app-landscape-final.png` dan uji responsif `screenshots/app-landscape-1366x768.png`.
- Bukti Overview: `screenshots/app-overview-final.png` dan `screenshots/app-overview-1366x768.png`.
- Bukti About: `screenshots/app-about-final.png` dan `screenshots/app-about-1366x768.png`.
- Pada 1366×768, kartu di halaman Dataset kosong ditumpuk agar tidak terpotong horizontal; pada 1920×1080 susunannya mengikuti frame.
- Logika backend, worker, koneksi signal/slot, data runtime, dan database tidak boleh diubah.

## Credits Roll

- Overlay Copyright memakai scrim navy transparan di atas halaman aktif, font Georgia untuk nuansa film, dan data editable dari `app/ui/data/credits.json`; GUI yang dipakai dicatat sebagai PySide6.
- Implementasi ada di `app/ui/views/credits_overlay.py` dan style khususnya di `app/ui/styles/credits.qss`; aset logo transparan memakai `app/ui/assets/about-brand-mark.png`.
- Capture awal, tengah, dan akhir: `screenshots/credits-start.png`, `screenshots/credits-middle.png`, dan `screenshots/credits-end.png`; generator ada di `tools/screenshot_credits.py`.
- Pemeriksaan interaksi mencakup buka berulang, resize, Escape, klik tutup, penyelesaian otomatis, serta pembersihan animasi.
