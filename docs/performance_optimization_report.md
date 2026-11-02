# DORA Performance Optimization Report

Tanggal: 2026-11-02. Pengukuran dilakukan pada Windows 10, Python 3.13.12, pandas 3.0.5, NumPy 2.5.3, scikit-learn 1.9.1, satu mesin lokal, Qt offscreen untuk probe startup.

## Metode

- Dataset deterministik dibuat dengan NumPy seed `20261102`: 120.000 baris, 9 kolom, CSV 15.721.569 byte. Fitur dan target sama pada pengukuran sebelum/sesudah.
- Load menggunakan `DatasetLoader`; EDA mengukur `DatasetProfiler`, `DatasetStatistics`, covariance, dan categorical association. ML memakai sampel tetap 5.000 baris, logistic regression, 3-fold CV, `n_jobs=1`. Report HTML memakai hasil dari run tersebut.
- Wall time diukur dengan `time.perf_counter`; CPU time menggunakan `time.process_time`; working set proses diambil lewat `GetProcessMemoryInfo` Windows dengan sampling 10 ms. RSS adalah penggunaan proses benchmark, bukan batas universal.
- Kecepatan loader dibandingkan tiga pengulangan chunked dan tiga pengulangan langsung pada file yang sama. Kesetaraan DataFrame diperiksa untuk setiap run.
- Probe window memakai direktori data dan QSettings sementara. Database pengguna dan `.env` tidak disentuh.

## Sebelum dan sesudah

| Parameter | Sebelum | Sesudah | Kesimpulan |
| --- | ---: | ---: | --- |
| Startup sampai window tampil | 8,51 dtk | 8,08 dtk | Tidak ada perubahan startup yang ditargetkan; selisih tidak diklaim sebagai optimasi. |
| Idle RSS | 294,63–295,11 MB | 294,59–295,15 MB | Tidak berubah terukur. |
| Idle CPU selama ~1,5 dtk | 0,078 dtk | 0,047 dtk | Sampel singkat, tidak cukup untuk menyimpulkan penghematan. |
| Load CSV, median tiga run | 3,512 dtk chunked | 3,332 dtk langsung | Sekitar 5,1% lebih cepat pada workload ini; hasil DataFrame sama. |
| Puncak RSS tambahan saat load, median tiga run | 30,36 MB | 30,64 MB | Tidak ada penghematan; selisih 0,28 MB berada dalam noise pengukuran. |
| EDA | 0,391 dtk | 0,378 dtk | Tidak ada perubahan EDA; satu run per kondisi, jangan ditafsirkan sebagai peningkatan. |
| ML logistic regression 3-fold | 0,432 dtk | 0,433 dtk | Praktis sama; pipeline dan hasil tidak diubah. |
| Report HTML | 0,070 dtk | 0,056 dtk | Tidak ada perubahan generator; selisih satu run terlalu kecil untuk klaim. |
| Peak RSS tambahan EDA / ML / report | 13,95 / 1,16 / 0,04 MB | 15,07 / 1,30 / 0,00 MB | Tidak menunjukkan penghematan memori. |
| Distribusi build DORA | Tidak dicatat per folder saat baseline; seluruh `dist/` lama 710.667.278 byte | `dist/DORA`: 335.546.836 byte, 1.612 file | Baseline seluruh `dist/` memuat artefak lama lain, jadi ukuran ini tidak dapat dibandingkan langsung. |
| Cache build lokal | Seluruh `build/` lama 991.344.760 byte | `build/Dataset Research`: 145.282.330 byte | Target subfolder vs seluruh cache lama; bukan perbandingan apples-to-apples dan bukan installer. |

Satu run workload lengkap menghasilkan load 3,547 dtk sebelum dan 3,393 dtk sesudah; tabel loader memakai median pengulangan sebagai angka utama. Perbedaan hasil statistik/ML tidak diharapkan karena algoritme dan input yang sama, dan DataFrame hasil loader dibandingkan identik. Skor ML pada dataset sintetis ini hanya menguji jalur kerja, bukan kualitas model ilmiah.

## Perubahan yang diterapkan

- `DatasetLoader.load()` dan `load_csv()` sekarang memakai `chunksize=None` sebagai default. Hasil tetap berupa DataFrame lengkap, sehingga parser pandas tidak membuat list chunk dan DataFrame gabungan sebagai jalur default. Pemanggil masih dapat memilih `chunksize` eksplisit.
- Warning dtype pandas diganti dengan pemeriksaan dtype eksplisit. Ini kompatibilitas dan kebersihan warning, bukan optimasi runtime.

## Tidak dioptimalkan atau belum terukur

- Puncak RAM dataset besar tetap perlu ditinjau: normalisasi missing dan analyzer membuat salinan sementara. Benchmark kecil ini tidak memberi bukti bahwa perubahan pada area tersebut aman atau menguntungkan.
- Worker analisis dapat menjalankan hingga delapan analyzer secara bersamaan. Tidak ada perubahan jumlah worker karena belum ada hasil perbandingan low-spec dan menurunkan paralelisme tanpa data berisiko memperlambat perangkat lain.
- Startup dan idle RSS masih tinggi. Matplotlib/SciPy, pandas, Qt, dan komponen report dimuat dalam proses UI. Memindahkan impor ke thread UI atau memecah lazy initialization tanpa uji responsivitas berisiko menukar startup lebih cepat dengan freeze saat halaman Analysis dibuka.
- Build package mengumpulkan `pip` untuk mempertahankan instalasi paket runtime pada Notebook Editor. Dependensi tidak dihapus hanya berdasarkan ukuran.
- Build PyInstaller berhasil dan child CLI `--dora-editor-pip --version` pada EXE final keluar code 0 dengan output `pip 25.3`. PyInstaller tetap melaporkan hidden import `scipy.special._cdflib` tidak ditemukan; tes scientific methods dijalankan di source environment, bukan keseluruhan secara packaged.
- Benchmark publik, variasi ukuran dataset, mesin berspesifikasi rendah, dan perbandingan sistem CPU masih belum tersedia.

