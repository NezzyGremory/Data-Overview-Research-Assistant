# Pengelolaan Tanya DORA Instruction Dataset (TDID)

## Cakupan

`app/ai/tdid_dataset.py` menyediakan builder layanan untuk import JSONL/CSV, validasi, tambah/edit/hapus record, statistik kategori dan kesulitan, pemeriksaan duplikasi, grouped split, export, serta snapshot versi immutable di SQLite. Ini belum menjadi halaman Dataset Builder di UI.

Tidak ada percakapan DORA yang dimasukkan ke TDID secara otomatis. Data percakapan dan TDID disimpan terpisah. Salin contoh secara manual hanya setelah persetujuan dan anonimisasi yang sesuai.

## Alur yang disarankan

1. Import berkas JSONL atau CSV dan perbaiki semua error import/validasi.
2. Periksa kategori, tingkat kesulitan, instruction type, sumber, dan status verifikasi.
3. Periksa exact duplicate dan prompt yang berulang. Exact duplicate tidak dapat disimpan sebagai versi.
4. Verifikasi jawaban ilmiah oleh peninjau dan isi `source` atau `source_url`. Status verified bukan hasil pemeriksaan otomatis.
5. Buat split deterministik; default 80/10/10 mengelompokkan record menurut source/source_id/source_group dan menahan prompt identik pada partisi yang sama.
6. Simpan versi `draft` untuk pengerjaan atau `final` hanya jika semua record sudah verified.
7. Simpan split bersama versi jika akan dipakai oleh Fine-Tuning Engine. Test split TDID tidak pernah dimasukkan ke trainer.

Split mengecualikan record invalid, belum verified, dan duplikasi tambahan. Jika kurang dari tiga independent source group, split ditolak. Semua kategori pada prompt dapat dihitung, tetapi target 300–600 per kategori hanyalah sasaran kurasi; repository ini tidak mengisi contoh untuk mengejar jumlah.

| Kategori | Sasaran ilustratif |
| --- | ---: |
| Data Science | 500 |
| Statistics | 400 |
| Machine Learning | 600 |
| Deep Learning | 500 |
| Computer Vision | 400 |
| Programming | 600 |
| Research Methodology | 500 |
| Research Gap | 400 |
| DORA Knowledge | 300 |
| Response Behavior | 300 |

Sasaran total brief sekitar 4.500 contoh terkurasi, mulai dari 500–1.000 verified examples bila tersedia; angka tersebut bukan jumlah record saat ini dan builder tidak membuat data sintetis otomatis.

## Format dan batas ukuran

Lihat [dataset_format.md](dataset_format.md) dan [template JSONL](examples/tdid_template.jsonl). Importer membatasi dataset ke 100.000 record dan isi pesan ke 40.000 karakter. PDF/DOCX atau percakapan privat bukan input TDID otomatis.

## Penyimpanan

Default: `get_data_dir()/tanya_dora/tdid/`. Metadata versi berada di `tdid.db`; salinan JSONL ber-hash SHA-256 ada di `versions/`. Path snapshot dibatasi ke folder tersebut dan hash diverifikasi saat load. Jangan mengedit snapshot versi secara manual.

## Status implementasi

Service dan unit test tersedia. UI builder, proses review multi-user, mekanisme anonimisasi percakapan, serta anotasi kontradiksi semantik belum tersedia.
