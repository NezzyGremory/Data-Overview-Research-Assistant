# Matriks cakupan fitur DORA

Tanggal audit: 2026-10-16 (Asia/Jakarta)

Status di bawah berdasarkan keberadaan source dan wiring yang diperiksa. Status ini bukan klaim keberhasilan runtime; baseline tes dan interaksi UI belum dijalankan.

| Fitur | Frontend yang ditemukan | Backend/service yang ditemukan | Kategori source-level | Catatan |
|---|---|---|---|---|
| Impor dataset dan preview | `UploadPage`, tab `DatasetPreviewPage` | `app/analyzer/loader.py` dan profiler | Frontend dan service tersedia | Dipasang pada shell dataset di page stack. |
| Statistik, fingerprint, dan kualitas data | `AnalysisPage` | `app/analyzer/statistics.py`, `fingerprint.py`, `data_quality.py`, profiler, missing values, duplicates, outliers, correlations | Frontend dan service tersedia | Analisis dikoordinasi dari `MainWindow` dan worker. |
| Pencarian dan unduh dataset publik | `DatasetSearchPage` | Klien `app/dataset_search/huggingface.py` dan `kaggle.py` | Frontend dan service tersedia | Memerlukan akses jaringan/kredensial sesuai provider. Belum diuji dalam sesi ini. |
| Cleaning dan versioning dataset | Dialog/halaman dataset; worker di `MainWindow` | `app/analyzer/cleaner.py`, `app/storage/version_manager.py`, `research_service.py` | Frontend dan service tersedia | Skema/database tidak disentuh. |
| Kesiapan ML, pemilihan task, rekomendasi, evaluasi | `MLIntelligencePage`, `NodeEditorPage` | `app/ml/` dan `app/ml/graph/` | Frontend dan service tersedia | Perilaku model tidak diuji dalam audit. |
| Pencarian dan deduplikasi literatur | `PapersToolPage` (key `research`/`papers`) | `app/research/` (termasuk OpenAlex, Crossref, deduplikasi, ranking) | Frontend dan service tersedia | `ResearchPage` terpisah dibuat tetapi disembunyikan; key aktif memakai `PapersToolPage`. |
| Research landscape, tren, research gap | `ResearchInsightsPage` dengan tab landscape/gap | `app/research/landscape.py`, `trend_analyzer.py`, `gap_analyzer.py`, `intelligence.py` | Frontend dan service tersedia | Frame terkait tidak tercantum pada Page 4 yang dibagikan. |
| Verifikasi jurnal/akreditasi | Kontrol screening pada halaman Papers | `sinta_verifier.py`, `accreditation_verifier.py`, `journal_index_verifier.py` | Frontend dan service tersedia | Belum divalidasi terhadap layanan eksternal pada sesi ini. |
| Penjelasan berbantuan AI | Hasil penjelasan pada halaman terkait | `app/ai/gemini_explainer.py`, `local_explainer.py`, `research_gap_explainer.py` | Frontend dan service tersedia | Gemini opsional; kode juga menyediakan jalur lokal. Tidak ada kredensial yang dibaca. |
| Laporan HTML/PDF | `ResearchReportToolPage` dan handler laporan | `app/reports/generator.py`; worker laporan di `MainWindow` | Frontend dan service tersedia | Build/ekspor belum diuji. |
| Penyimpanan project dan hasil riset | Tidak ada halaman Projects terpisah dalam page stack aktif | `app/storage/database.py`, `project_repository.py`, `research_service.py`, `version_manager.py` | Service tersedia; frontend khusus belum terpetakan | Sebagian operasi dipakai dari alur halaman aktif. |
| Telemetry API opsional | Integrasi desktop melalui `app/telemetry.py` | `backend/api.py` dan `backend/graph_api.py` | Client dan API tersedia | API terpisah dari analisis dataset lokal. Tidak dijalankan. |
| Pengaturan aplikasi | Kontrol skala tersedia pada infrastruktur UI | `app/core/config.py`, `app/ui/scale.py` | Parsial | Tidak ada page Settings tersendiri pada daftar aktif yang diperiksa. |

## Batas audit

- Penilaian hanya memastikan source, modul, dan page stack yang ditemukan; fitur belum dinilai lengkap secara end-to-end.
- Tidak ada fitur yang ditandai selesai secara runtime sebelum pengujian relevan dijalankan.
- Implementasi visual sesi ini harus menargetkan satu frame Figma yang dipilih. Page 4 memuat beberapa layar dan tidak menentukan satu target tunggal.

## Pemetaan halaman aktif setelah perapian Overview

- Overview (`DashboardPage`) tetap memakai tampilan bergaya Figma baik saat dataset kosong maupun terisi. Saat terisi, ringkasan menerima dataframe dan hasil yang telah dihitung aplikasi.
- Langkah Overview menuju halaman aktif: upload dan preview Dataset, Analysis, ML Intelligence, serta Research Report. Klik langkah diuji melalui smoke check UI.
- Halaman literatur aktif adalah `PapersToolPage` pada key `research`; `ResearchPage` tersembunyi tetap menjadi orchestrator internal. Research Gap dan Landscape memakai `ResearchInsightsPage` dengan tab yang sesuai.
- Halaman aktif tanpa frame spesifik menggunakan gaya bersama di `app/ui/styles/theme.qss`. Ini adalah penyelarasan visual, bukan verifikasi desain terhadap frame Figma.
- Status baris fitur di atas tetap menunjukkan ketersediaan source/service saat audit, bukan hasil uji end-to-end. Tidak ada perubahan service backend atau logika data dalam perapian UI Overview.
