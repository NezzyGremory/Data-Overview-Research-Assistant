# Pemetaan UI Figma ke PySide6

Tanggal inspeksi: 2026-10-16 (Asia/Jakarta)

## Sumber dan cakupan

File Figma DORA dapat dibaca lewat MCP menggunakan fileKey `oLQhZyWW2vIPM9FmqvYewD`. Key pada URL yang ditempel berakhir satu karakter lebih awal; key lengkap ini tercatat di `docs/UI_PROGRESS.md`. Node URL `223:277` adalah **Page 4**, bukan satu frame layar. Metadata Page 4 mencantumkan dua belas frame. Karena aturan proyek meminta satu halaman per sesi, frame implementasi harus dipilih sebelum kode UI diubah.

`design/html_export/` dan `design/tokens.md` tidak tersedia. Karena itu inspeksi desain menggunakan Figma MCP; file lokal yang hilang tidak digantikan dengan tebakan.

## Frame pada Page 4

| Frame Figma | Ukuran (px) | Padanan page stack aktif | Catatan |
|---|---:|---|---|
| `10. Dataset Analysis` (`309:6185`) | 1920 × 1024 | `AnalysisPage` di `app/ui/main_window.py` | Selesai; screenshot `screenshots/app-analysis-figma-309-6185.png`; handler analyzer yang ada dipertahankan. |
| `2. Cari Dataset Publik` (`309:2753`) | 1440 × 992 | `DatasetSearchPage` dalam shell dataset | Selesai; screenshot `screenshots/app-dataset-search-309-2753.png`; pencarian dan download memakai handler yang ada. |
| `4. Dataset Preview` (`309:3164`) | 1440 × 1024 | `DatasetPreviewPage` | Selesai; screenshot `screenshots/app-preview-309-3164.png`; tabel preview memakai dataset aktif. |
| `5. Dataset Preview - Kualitas Data` (`309:3854`) | 1440 × 1024 | `DatasetPreviewPage` | Selesai sebagai tampilan kualitas pada halaman preview; memakai hasil analyzer yang tersedia. |
| `7. Dataset Preview - Missing Value` (`309:5063`) | 1440 × 1024 | `DatasetPreviewPage` | Selesai sebagai tampilan missing value pada halaman preview; memakai hasil analyzer yang tersedia. |
| `3. Upload Dataset` (`309:2155`) | 1440 × 1024 | `UploadPage` dalam shell dataset | Selesai; screenshot `screenshots/app-upload-loaded-309-2155.png`; jalur upload tetap memakai handler lama. |
| `11. Editor Analysis` (`309:6509`) | 1440 × 1024 | `EditorAnalysisPage` | Selesai; screenshot `screenshots/app-editor-analysis-309-6509.png`; notebook, eksekusi kode, dan output yang ada dipertahankan. |
| `12 · ML Intelligence` (`309:6661`) | 1536 × 965 | `MLIntelligencePage` aktif di `app/ui/main_window.py` | Selesai; screenshot `screenshots/app-ml-309-6661.png`; metrik berasal dari evaluasi model lokal. |
| `1. Overview` (`309:1100`) | 1440 × 1024 | `DashboardPage` dari `app/ui/overview_page.py`, dengan tampilan `Page4OverviewEmptyPage` | Selesai; kondisi kosong mengikuti frame. Kondisi dataset aktif memakai panel ringkasan dengan art style dan komponen yang sama. Kedua kondisi tetap pada Overview. |
| `6. Dataset Preview - Informasi Kolom` (`309:4607`) | 1440 × 1024 | `DatasetPreviewPage` | Selesai sebagai tampilan informasi kolom pada halaman preview. |
| `8. Dataset Preview - Deteksi Outliner` (`309:5438`) | 1440 × 1024 | `DatasetPreviewPage` | Selesai sebagai tampilan deteksi outlier pada halaman preview; memakai hasil analyzer yang tersedia. |
| `9. Dataset Preview - Korelasi` (`309:5649`) | 1536 × 1024 | `DatasetPreviewPage` | Selesai sebagai tampilan korelasi pada halaman preview; memakai hasil analyzer yang tersedia. |

Frame penelitian, research gap, landscape, dan report tidak tampak di daftar Page 4 yang diperiksa. Jika scope nantinya mencakupnya, cari frame yang sesuai pada halaman Figma lain sebelum memetakan.

## Navigasi aktif dan tampilan tanpa frame khusus

Semua callback di Overview dan sidebar diarahkan ke halaman aktif pada `MainWindow.pages`; tampilan lama Dashboard tidak lagi dipilih ketika dataset dimuat.

| Kontrol / tujuan | Halaman aktif | Perilaku |
|---|---|---|
| Cari Dataset | `DatasetSearchPage` (`dataset_search`) | Membuka pencarian provider. |
| Upload Dataset | `UploadPage` (`dataset`) | Membuka upload; jika dataset tersedia, menu Dataset mengarah ke preview. |
| Jelajahi Data | `DatasetPreviewPage` (`dataset`) | Membuka preview data aktif. |
| Analisis Data | `AnalysisPage` (`analysis`) | Membuka analisis untuk dataset aktif atau empty state yang tersedia. |
| ML Intelligence | `MLIntelligencePage` (`ml`) | Membuka halaman rekomendasi/evaluasi ML aktif. |
| Hasilkan Insight | `ResearchReportToolPage` (`report`) | Membuka laporan atau empty state sesuai data yang tersedia. |
| Dukungan Riset / Academic Research | `PapersToolPage` (`research`) | Membuka halaman literatur aktif. `ResearchPage` tetap dipakai sebagai service/orchestrator internal dan tidak ditampilkan sebagai halaman. |
| Research Gap / Landscape | Tab `ResearchGapToolPage` / `ResearchLandscapeToolPage` di `ResearchInsightsPage` (`gap` / `landscape`) | Kedua rute memakai host halaman insights yang sama. |
| Editor Analysis | `EditorAnalysisPage` (`editor_analysis`) | Membuka notebook aktif. |
| About / Help | `AboutPage` / `HelpPage` | Membuka halaman informasi dan panduan. |

Halaman aktif tanpa frame khusus di daftar Page 4 memakai token, latar kanvas, kartu putih berbatas biru muda, tipografi Inter, serta tombol biru/outline yang sama di `app/ui/styles/theme.qss`. Ikon SVG dirender pada resolusi perangkat; ikon berwarna memakai aset vektor lama yang disesuaikan tint-nya.

## Detail terverifikasi: `10. Dataset Analysis` (`309:6185`)

- Kanvas 1920 × 1024 px, latar `#F5F9FF`, font Inter.
- Sidebar selebar 237 px; area konten dimulai sekitar x=280 px.
- Header berisi judul/deskripsi dan tombol `Run Analysis`.
- Kartu dataset berada di x=280, y=152 dengan ukuran 1580 × 92 px.
- Kartu visualisasi berada di x=284, y=283 dengan ukuran 1028 × 596 px. Kontrol jenis grafik dan kolom berada di bagian atas; histogram mengisi bagian tengah/bawah.
- Kartu fingerprint berada di x=1332, y=272 dengan ukuran 456 × 172 px.
- Kartu temuan kualitas berada di x=1332, y=468 dengan ukuran 456 × 452 px dan memuat temuan missing value, outlier, serta duplikasi.
- Variabel yang dikembalikan MCP: `Big Stone` / `color/azure/12` = `#0F1B2D`.
- Screenshot Figma diperoleh melalui MCP. `get_design_context` juga mengembalikan susunan teks, dimensi, warna, serta referensi SVG/PNG sementara.
- Prototype states/interaksi (hover, focus, loading, error) tidak terverifikasi dari frame statis ini. Perilaku tombol dan dropdown harus memakai handler aplikasi yang sudah ada.

## Langkah setelah frame dipilih

1. Ambil konteks desain dan aset untuk frame yang dipilih.
2. Cocokkan komponen dan aset dengan halaman aktif serta stylesheet/tokens yang tersedia.
3. Implementasikan frame itu saja dan pertahankan alur data existing.
4. Jalankan aplikasi, periksa kontrol yang terlihat, simpan satu screenshot aplikasi untuk halaman tersebut, lalu bandingkan pada ukuran jendela yang setara.
