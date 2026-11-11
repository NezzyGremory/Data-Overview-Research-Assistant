# Environment dan Hardware Training

`detect_training_hardware()` melaporkan logical CPU, RAM total/tersedia, PyTorch/CUDA, perangkat GPU, VRAM, dan status Ollama jika manager diberikan. Preflight mengukur ruang disk, ukuran bobot lokal dan estimasi kebutuhan memori. Nilai estimasi LoRA/QLoRA bersifat heuristik; batch, panjang sequence, arsitektur, optimizer, serta kernel aktual dapat mengubah pemakaian.

## Mode

| Mode | Status saat ini | Catatan |
| --- | --- | --- |
| Inference Mode | Ollama lokal | Tidak memerlukan stack training Python; model harus dipasang pengguna. |
| Lightweight Training | Engine service tersedia | SFT+LoRA memakai folder model lokal. QLoRA 4-bit di engine ini memerlukan CUDA + bitsandbytes. Dependency belum ada di environment ini, jadi belum ada run nyata. |
| External Training Workspace | Pertukaran file tersedia | Export TDID JSONL dan file konfigurasi; salin manual ke workstation yang disetujui. Kembalikan adapter sebagai ZIP dari `ModelVersionStore`, lalu import dengan base model lokal yang fingerprint-nya cocok. |

Jangan menjalankan training jika preflight gagal. Jika ruang/memori tidak cukup, pilih base model lebih kecil, sequence/batch lebih rendah, gunakan mesin eksternal, atau lanjutkan inference Ollama tanpa fine-tuning. RAM dan VRAM tidak otomatis disisihkan dari proses lain.

Untuk QLoRA, bitsandbytes menyediakan dukungan platform/hardware yang berubah menurut versi. Periksa [panduan instalasi bitsandbytes](https://huggingface.co/docs/bitsandbytes/installation) dan cocokkan PyTorch/CUDA sebelum instalasi. Repo ini tidak menyatakan sebuah model atau konfigurasi cocok sebelum uji hardware nyata.

## Batas monitor

Callback training melaporkan step, epoch, loss/logging yang disediakan Trainer, learning rate, durasi, available RAM/VRAM, dan folder checkpoint. Kode ini belum memiliki progress page; panggil `run()` pada worker, bukan UI thread. Pengukuran RAM/VRAM adalah snapshot sistem, bukan pemakaian eksklusif proses model.
