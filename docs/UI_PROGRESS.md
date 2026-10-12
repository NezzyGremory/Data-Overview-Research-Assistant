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
| Overview, versi terbaru | `144:275` | 1920 × 1080 | `DashboardPage` | belum |
| About | `141:373` | 1920 × 1080 | `AboutPage` | belum |
| Overview lama/duplikat, tidak dipakai | `104:192` | 1920 × 1080 | duplikat `DashboardPage` | dilewati — versi lama |

Komponen lepas pada canvas: `Drop duplicates` (`24:95`, 172 × 38 px), `Full missing values` (`31:390`, 204 × 38 px), `Remove column` (`31:395`, 182 × 38 px), dan ikon informasi (`144:520`, 16 × 16 px). Tiga chip cleaning termasuk bagian Dataset terisi.

Gaya dasar yang dibaca dari Figma: kanvas `#F1F7FF`; biru `#006BB8`, `#0071C2`, dan `#0D99FF`; font Inter dan Plus Jakarta Sans, dengan Intel One Mono untuk sidik jari dataset; kartu radius 15 px, tombol kecil 8 px dan chip atas 20 px. Frame yang tersedia tidak menunjukkan variasi hover, pressed, focus, loading, atau error tersendiri.

## Status Pekerjaan

| Area | Status | Catatan |
|---|---|---|
| Fondasi token, shell, skala, dan screenshot | selesai | Shell/sidebar dan topbar dicocokkan ulang ke posisi node Figma; token serta utilitas screenshot tersedia. Sidebar Figma berisi 10 menu. `Editor Analysis` disembunyikan, wiring internal tetap ada. |
| Dataset kosong dan terisi | selesai | Dua kondisi disetel ke frame 1920×1080; layout kosong turun menjadi satu kolom pada jendela lebih kecil. Sinyal lama dipertahankan. Screenshot dibandingkan dengan frame Figma. |
| Popup Upload Dataset | selesai | Dialog tepat 1027×757 px pada x=447, y=162; ukuran field dan tombol dicocokkan dari metadata. Switch menampilkan knob putih. Layout ringkas dipakai saat tinggi layar di bawah ukuran desain. Screenshot dibandingkan dengan Figma. |
| Analysis | selesai | Banner, tombol, area visualisasi, kartu fingerprint, dan kartu kualitas dicocokkan ke koordinat node `31:406`. Subtitle responsif dan detail hasil tetap tersedia di bawah ringkasan. Screenshot 1920×1080 dibandingkan dengan Figma; 1366×768 diperiksa agar teks membungkus tanpa menumpuk. |
| Dataset Search | selesai | Judul, deskripsi, panel pencarian, dua kolom kartu hasil, metadata, file, dan tombol dicocokkan ke koordinat node `41:221`. Kontrol pencarian dan aksi download tetap memakai handler lama. Screenshot 1920×1080 dibandingkan dengan Figma; 1366×768 diperiksa untuk susunan responsif. |
| Overview | belum | Memakai frame terbaru `144:275`; frame `104:192` diabaikan. |
| Research Landscape | selesai | Tombol `Refresh Landscape`, ringkasan 3 kartu, distribusi metode, dan venue publikasi dicocokkan ke node `54:843`. Tab internal disembunyikan; navigasi sidebar ke Research Gap tetap aktif. Screenshot 1920x1080 dibandingkan dengan frame Figma; 1366x768 diperiksa dan konten bertumpuk/bergulir tanpa tumpang tindih. |
| About | belum | Ada frame Figma. |
| ML Intelligence | dilewati | Tidak ada frame halaman di Figma. |
| Academic Research | dilewati | Tidak ada frame halaman di Figma. |
| Papers | dilewati | Tidak ada frame halaman di Figma. |
| Research Gap | dilewati | Tidak ada frame halaman di Figma. |
| Research Report | dilewati | Tidak ada frame halaman di Figma. |
| Help dan Copyright | dilewati | Kontrol tampil di shell Figma, tetapi tidak ada frame isi/tujuan. |
| Editor Analysis | dilewati | Tidak ada frame Figma; sesuai arahan, disembunyikan dari sidebar tanpa menghapus kode. |

## Perubahan Nama yang Terlihat

| Teks aktif di aplikasi | Teks Figma | Lokasi |
|---|---|---|
| `Datasets` | `Dataset` | Judul halaman `datasets_page` |
| `Dashboard` (judul halaman internal) | `Overview` | Judul `DashboardPage` bila tampil |
| `Editor Analysis` | tidak ada | Disembunyikan dari sidebar; kunci `editor_analysis` tetap dipertahankan |
| `2026 [nama tim] v.` | `2026 UIN HIJAU v5` | Identitas tim dari preferensi yang sudah diberikan pengguna |

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
- Pada 1366×768, kartu di halaman Dataset kosong ditumpuk agar tidak terpotong horizontal; pada 1920×1080 susunannya mengikuti frame.
- Logika backend, worker, koneksi signal/slot, data runtime, dan database tidak boleh diubah.
