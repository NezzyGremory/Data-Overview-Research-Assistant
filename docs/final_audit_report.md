# DORA Final Audit Report

Tanggal audit: 2026-11-02  
Ruang lingkup: checkout lokal desktop DORA dan source yang tersedia. Dataset pengguna, `.env`, dan database runtime tidak dibuka.

## Kondisi awal

- Instruksi root `AGENTS.md`, status implementasi, arsitektur, audit komprehensif, validasi ilmiah, dependensi, konfigurasi pytest, dan PyInstaller spec sudah diperiksa sebelum perubahan source.
- Skill `codebase-memory` tidak ditemukan pada lokasi filesystem yang terdaftar dan tidak ada tool graph yang tersedia. Penelusuran dilakukan langsung memakai `rg` dan pembacaan source.
- Checkout sudah memiliki banyak perubahan: 37 file tracked berubah, sejumlah file baru, serta direktori pytest sementara. Perubahan itu sudah ada sebelum audit; tidak ada yang di-reset atau dihapus.
- Tidak ada nested `AGENTS.md` lain di dalam repo.
- Struktur aplikasi yang diperiksa: PySide6 desktop dengan page stack `MainWindow`; analyzer, ML, research, report, dan storage lokal terpisah; FastAPI hanya telemetry/web API. Sebagian modul UI tidak berada di page stack aktif.

## Baseline regresi

Perintah:

```text
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=.pytest_dora_audit_baseline_tmp
```

Hasil: **295 passed, 0 failed, 61 warnings dalam 81,73 detik**.

Rincian warning:

- 59 `Pandas4Warning` dari pemakaian `pd.api.types.is_categorical_dtype` di `app/ml/target_detector.py:141`.
- 2 `Pandas4Warning` dari `select_dtypes(include=["object", ...])` di `app/ml/intelligence.py:479`; pandas 3 memasukkan dtype string secara implisit dan memperingatkan perubahan perilaku mendatang.
- Tidak ada warning runtime atau statistik lain yang tercatat pada baseline.

## Temuan menurut kategori

### Critical Bug

- Tidak ada kegagalan kritis yang direproduksi oleh suite baseline.

### Functional Bug

- Jalur unduh-dan-analisis pada `DatasetSearchPage` saat ini memanggil `open_analysis_page(on_ready=...)`; callback menerima instance halaman setelah dibuat. Ini menghindari pemanggilan `run_analysis` pada referensi halaman yang masih `None`. Belum ada regresi khusus untuk urutan callback lazy-page tersebut; tes akan ditambahkan.
- Suite lulus, tetapi startup dan integrasi jaringan eksternal belum tercakup oleh tes suite. Keduanya tidak boleh dinyatakan sudah lolos end-to-end.

### Scientific/Methodological Issue

- Batas yang telah dicatat tetap berlaku: supervised CV belum menyediakan group split atau time split; pencegahan leakage menangkap salinan target identik dan belum semua fitur turunan/relasi target; statistik inferensial belum mengoreksi multiple testing; benchmark belum membandingkan dengan dataset publik atau hasil Colab yang ditentukan.
- Ini batas kapabilitas/protokol yang diketahui, bukan kegagalan baru yang dapat diperbaiki tanpa rancangan evaluasi dan data pembanding yang sesuai.

### Performance Issue

- Startup desktop source terukur sekitar **8,51 dtk** sampai window terlihat pada satu probe terisolasi. Proses menggunakan **294,63 MB RSS** setelah tampil dan **295,11 MB** setelah 1,49 dtk idle. Probe memakai Qt offscreen, direktori data dan QSettings sementara.
- Pembacaan CSV default mengumpulkan semua chunk ke list lalu menggabungkannya menjadi DataFrame penuh. Untuk keluaran yang tetap harus berupa satu DataFrame, jalur ini menambah kerja gabung. Pada file benchmark yang sama, satu pembacaan chunked memerlukan 3,547 dtk; pembacaan tanpa chunk 3,344 dtk. Selisih satu pengukuran ini perlu dibaca sebagai indikasi, bukan klaim peningkatan final.
- Worker analisis menjalankan hingga delapan analyzer bersamaan. Ini menjaga UI tetap berada di thread utama, tetapi penggunaan CPU/RAM puncak untuk dataset besar belum dibatasi atau diukur pada perangkat berspesifikasi rendah.

### Memory Issue

- Loader harus mengembalikan tabel lengkap ke pemanggil. Membaca dengan chunk saat ini tetap menyimpan semua chunk dan kemudian membuat hasil gabungan, sehingga tidak memberi streaming atau batas memori.
- Pengukuran proses menunjukkan puncak tambahan sekitar **31,69 MB** selama load CSV, **13,95 MB** saat EDA, **1,16 MB** saat satu model dilatih, dan **0,04 MB** saat membuat laporan pada benchmark ini. Ini ukuran proses benchmark, bukan batas universal.
- Tidak ditemukan bukti kebocoran objek Qt pada baseline suite; interaksi tutup-buka seluruh dialog belum diuji sebagai siklus desktop penuh.

### UI/UX Responsiveness

- Analisis, ML, pencarian dataset, riset, dan pembuatan report memiliki worker/thread yang terpisah dari GUI. Analisis menampilkan status loading, tetapi belum menyediakan pembatalan yang aman atau progress per tahap.
- Halaman Analysis dibuat lazy; pembuatan halaman pertama pada probe selesai dalam 0,215 dtk setelah permintaan ketika dependensi chart sudah terimpor.

### Dependency/Packaging Issue

- `requirements.txt` tidak mengunci versi; versi di `AGENTS.md` merupakan snapshot environment lokal, bukan rentang dukungan resmi.
- Artefak yang sudah ada berukuran sekitar **710.667.278 byte** di `dist/` dan **991.344.760 byte** di `build/`. Folder-folder itu hanya diukur; belum dibuat ulang dalam audit baseline.
- Build spec mengumpulkan `pip` dan submodulnya karena editor notebook memasang paket runtime. Pengurangan dependensi perlu mempertahankan alur instalasi tersebut dan diuji pada hasil build.
- Smoke test EXE pertama menemukan bug nyata pada mode child `--dora-editor-pip`: Python windowed membiarkan `sys.stdout`/`sys.stderr` bernilai `None`, lalu `pip` berhenti dengan `'NoneType' object has no attribute 'write'`. Perbaikan dibuat pada `app/main.py`; suite menguji pemulihan stream dan EXE sedang dibangun ulang untuk uji langsung.

### Minor Improvement

- Warning pandas adalah perbaikan kompatibilitas yang terlokalisasi dan dapat dihapus tanpa suppression.
- Benchmark yang tersedia di sesi ini bersifat sintetis dan deterministik; tidak ada file dataset publik atau perangkat kelas rendah yang ditentukan.

## Benchmark awal

Dataset sintetis deterministik: **120.000 baris × 9 kolom**, CSV **15.721.569 byte**, seed 20261102. ML memakai sampel tetap 5.000 baris, logistic regression, 3-fold CV, `n_jobs=1`. Report dibuat dari hasil benchmark yang sama.

| Tahap | Wall time | CPU time proses | Puncak RSS tambahan |
| --- | ---: | ---: | ---: |
| Startup window source/offscreen | 8,51 dtk | — | — |
| Idle window | 1,49 dtk sampling | 0,078 dtk | — |
| Load CSV | 3,547 dtk | 3,297 dtk | 31,69 MB |
| EDA profile/statistics/covariance/association | 0,391 dtk | 0,391 dtk | 13,95 MB |
| Logistic regression 3-fold | 0,432 dtk | 0,359 dtk | 1,16 MB |
| HTML report | 0,070 dtk | 0,047 dtk | 0,04 MB |
| Distribusi build yang sudah ada | — | — | 710,67 MB |

RSS saat window idle: **294,63–295,11 MB**. Pengukuran load eksplisit `chunksize=None` pada file yang sama: **3,344 dtk** dan puncak tambahan **32,06 MB**. Ini baseline; hasil sesudah perubahan akan dicatat terpisah.

## Status verifikasi awal fitur

| Area | Bukti yang tersedia | Status audit awal |
| --- | --- | --- |
| Import CSV/TSV/XLS/XLSX/JSON/JSONL/Parquet | Loader bersama; tes CSV, TSV, XLSX, JSON, JSONL, dan Parquet | Format bertes unit-level lulus; jalur `.xls` belum dapat diuji karena checkout tidak memiliki fixture dan writer `xlwt` tidak terpasang. Belum semua format diuji dengan file publik besar. |
| Profiling, kualitas, statistik, EDA | Analyzer dan tes analyzer/visualisasi | Unit-level lulus |
| Inferential statistics | Tes SciPy, dialog, penyimpanan dan report | Unit-level lulus; koreksi multiple testing belum tersedia |
| ML, split, tracking | Tes evaluator, holdout, fold, fitur manual, SQLite | Unit-level lulus; group/time split belum tersedia |
| Node Editor dan Analysis Editor | Source, page tests/graph tests | Tes komponen; bukan interaksi semua halaman desktop |
| OpenAlex, Crossref, Gemini | Client/worker source tersedia | Tidak ada panggilan jaringan live pada audit baseline |
| SQLite project restore dan HTML report | Tes repository/database/report | Unit-level lulus; belum diuji siklus UI buka-tutup penuh |
| Build/installer | Spec dan installer source tersedia | Ukuran artefak lama diukur; build belum dijalankan ulang |

## Tindakan setelah audit awal

1. Hilangkan dua pemakaian API pandas deprecated dengan pemeriksaan dtype eksplisit dan tes kompatibilitas.
2. Gunakan parsing CSV non-chunk sebagai default karena API tetap mengembalikan DataFrame utuh dan baseline menunjukkan jalur itu lebih cepat; pertahankan opsi `chunksize` eksplisit.
3. Tambahkan tes urutan callback Analysis lazy-page untuk alur unduh-dan-analisis.
4. Ulangi benchmark dan regresi. Jangan menyatakan peningkatan RAM kecuali data sesudah membuktikannya.

## Hasil tindakan audit

- Kedua sumber `Pandas4Warning` diperbaiki tanpa suppression. Tes dtype object/string/category/bool ditambahkan.
- Default CSV kini langsung membaca ke DataFrame; argumen `chunksize` eksplisit tetap didukung dan menghasilkan tabel yang identik.
- Regresi urutan callback halaman Analysis lazy ditambahkan untuk alur unduh-dan-analisis.
- Suite final setelah seluruh regresi: **301 passed, 0 failed, 0 warnings dalam 62,94 detik**. `compileall` dan `git diff --check` lulus.
- Benchmark median berulang di file sama: load chunked **3,512 dtk**, langsung **3,332 dtk** (sekitar 5,1% lebih cepat pada mesin ini); puncak RSS masing-masing **30,36 MB** dan **30,64 MB**, jadi tidak ada penghematan RAM yang terukur.
- Probe startup sesudah perubahan: **8,076 dtk**, idle RSS **294,59–295,15 MB**. Perubahan source ini tidak menargetkan startup; selisih dari baseline berada dalam variasi pengukuran dan tidak diklaim sebagai peningkatan.
- PyInstaller build akhir berhasil. Uji child CLI pada EXE hasil rebuild menjalankan `--dora-editor-pip --version`, menerima output `pip 25.3`, stderr kosong, exit code 0.
- Folder aplikasi hasil build `dist/DORA`: **335.546.836 byte** (1.612 file). Artefak lama lain tetap ada di `dist/`; total direktori `dist/` menjadi 1.046.214.114 byte. Ukuran total lama 710.667.278 byte bukan folder aplikasi yang sebanding, sehingga tidak diklaim sebagai pengurangan ukuran.
- PyInstaller menampilkan warning `Hidden import "scipy.special._cdflib" not found!` tetapi build berhasil. Warning ini tidak disuppress; jalur SciPy lengkap di EXE belum diuji satu per satu.

Lihat `performance_optimization_report.md`, `regression_test_report.md`, dan `dora_implementation_status.md` untuk status akhir serta keterbatasan yang belum selesai.

