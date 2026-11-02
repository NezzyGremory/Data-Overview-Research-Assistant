# DORA Regression Test Report

Tanggal: 2026-11-02.

## Ringkasan hasil

| Pemeriksaan | Sebelum perubahan | Sesudah perubahan |
| --- | --- | --- |
| Full suite | 295 passed, 0 failed, 61 warning, 81,73 dtk | 301 passed, 0 failed, 0 warning, 62,94 dtk |
| Compile source/tests | — | `python -m compileall -q app tests`: lulus |
| Whitespace/conflict markers | — | `git diff --check`: lulus; Git menampilkan peringatan normalisasi line ending LF/CRLF pada file lama |
| Dataset load | CSV/TSV/XLS/XLSX/JSON/JSONL/Parquet memiliki klaim dukungan | CSV/TSV/XLSX/JSON/JSONL/Parquet dites; `.xls` belum punya fixture valid karena writer `xlwt` tidak tersedia. Default dan `chunksize=2` menghasilkan DataFrame identik |
| Dtype kategori | 61 warning pandas ada pada baseline | object, pandas StringDtype, category, bool terhitung eksplisit; warning hilang |
| Callback auto-analysis | Jalur source sudah menunggu page siap, tanpa tes urutan tersendiri | Tes lazy page membuktikan `run_analysis` dijalankan pada halaman yang telah dibuat |
| Build aplikasi | Artefak lokal tersedia | PyInstaller build berhasil; child CLI final `--dora-editor-pip --version` exit 0 dan menghasilkan `pip 25.3` |

## Tes baru

- `tests/test_loader.py`: membandingkan pembacaan default dengan chunk eksplisit pada isi CSV yang sama.
- `tests/test_intelligence.py`: memastikan kategori object, StringDtype, category, dan bool dihitung eksplisit.
- `tests/test_task_detector.py`: memastikan target categorical tetap terdeteksi tanpa API pandas deprecated.
- `tests/test_dataset_search_analysis_flow.py`: memastikan analisis hasil unduhan menunggu pembuatan halaman Analysis lazy dan tidak memanggil method pada `None`.
- `tests/test_packaged_cli.py`: memastikan mode `--dora-editor-pip` tetap mempunyai stdio yang dapat ditulis saat berjalan di PyInstaller `console=False`.

## Batas verifikasi

- Tidak ada dataset pengguna/Colab yang disediakan untuk membandingkan task klasifikasi versus clustering. Unit test menguji pemilihan task dan override yang ada, tetapi belum membuktikan hasil pada dataset tertentu.
- `.xls` tetap menjadi satu format yang diklaim loader tetapi tidak dapat diuji penuh dalam environment ini karena tidak ada fixture dan paket writer `xlwt` tidak terpasang; `xlrd` untuk membaca tersedia.
- OpenAlex, Crossref, Hugging Face, Kaggle, dan Gemini tidak dipanggil ke internet pada regresi ini; kredensial dan jaringan tidak diasumsikan tersedia.
- Tes komponen UI ada untuk preview, Analysis, dialog statistik, opsi ML, dan Node Editor, tetapi tidak dilakukan siklus manual semua halaman/dialog dari aplikasi desktop.
- Build pertama berhasil. Smoke test mode CLI menemukan pesan `'NoneType' object has no attribute 'write'` saat pip menulis output dari windowed EXE; `app/main.py` sudah diperbaiki dan build baru sedang diuji.
- SQLite, penyimpanan eksperimen, restore unit-level, dan pembuatan report diuji oleh suite. Pembukaan proyek pengguna nyata tidak dilakukan.
- Build final selesai; mode CLI pip pada EXE berhasil dijalankan. Hidden import SciPy `_cdflib` tetap menjadi warning PyInstaller. Belum dilakukan instalasi paket runtime sungguhan melalui editor.
- Distribusi `dist/DORA` berukuran 335.546.836 byte. Installer Inno Setup tidak dibuat.

