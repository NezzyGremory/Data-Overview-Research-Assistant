# Panduan Fine-Tuning Lokal

`app/ai/fine_tuning.py` menyediakan engine SFT dengan LoRA dan QLoRA opsional. Engine ini tidak mengunduh model atau kode model. Base model wajib berupa folder lokal, tokenizer harus memiliki chat template, dan dataset wajib berupa TDID versi dengan train/validation/test split tersimpan.

## Setup terpisah

Training stack tidak dimasukkan ke dependency desktop. Pasang PyTorch yang cocok dengan OS/GPU terlebih dahulu, lalu gunakan `requirements_training.txt`. QLoRA memerlukan CUDA dan `requirements_qlora.txt`; cocokkan versi PyTorch/CUDA/bitsandbytes dengan dokumentasi resmi sebelum memasang. Tidak ada dependency training yang terpasang pada environment project saat fase ini, jadi SFT belum dijalankan di komputer ini.

TRL menyediakan `SFTTrainer` dan konfigurasi untuk dataset conversational; PEFT menyediakan adapter LoRA yang disimpan terpisah dari bobot foundation model. [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) dan [PEFT quicktour](https://huggingface.co/docs/peft/quicktour) menjelaskan API tersebut.

## Konfigurasi

Lihat [training_config.example.json](examples/training_config.example.json). Engine memvalidasi epoch, learning rate, batch/accumulation, sequence length, rank/alpha/dropout, quantization, warmup, weight decay, checkpoint/eval step, seed, dan `target_modules`. Nama target LoRA bergantung pada arsitektur base model; periksa model lokal dan jangan menganggap daftar default cocok untuk semua Qwen/Gemma/Llama/Phi.

Langkah Python:

```python
from app.ai.fine_tuning import FineTuningConfig, FineTuningEngine
from app.ai.tdid_dataset import TDIDVersionStore

store = TDIDVersionStore()  # atau berikan storage_dir lokal
engine = FineTuningEngine(store)
config = FineTuningConfig(
    base_model_path=r"C:\models\instruct-model",
    dataset_version_id="ID-VERSI-TDID",
    output_dir=r"D:\training\tanya-dora",
)
preflight = engine.preflight(config)
if preflight.ready:
    result = engine.run(config)
```

`preflight` mengembalikan dependency yang hilang, CUDA/RAM/VRAM, disk, file model, validitas split, dan estimasi memori. Estimasi hanyalah perkiraan dan bukan jaminan model tidak akan kehabisan memori. Engine berhenti sebelum trainer jika preflight memblokir.

`run()` menulis checkpoint berkala dan `training_manifest.json`. Error menyimpan status gagal dan lokasi checkpoint; cancellation menghentikan langkah trainer dengan callback dan menyimpan hasil lokal. Adapter dianggap selesai hanya jika `adapter_config.json` dan bobot adapter benar-benar ada. Adapter tidak diaktifkan otomatis.

## Keamanan dan batasan

Model dimuat dengan `local_files_only=True` dan `trust_remote_code=False`. Tidak ada percakapan sebagai dataset otomatis. Validation set dipakai untuk `evaluate()`; test set tidak dikirim ke trainer. Saat ini engine tidak menyediakan Fine-Tuning Studio UI, inference adapter runtime untuk Tanya DORA, perbandingan otomatis base-vs-adapter, atau hasil training aktual. Gunakan Mode External Workspace di [training_hardware.md](training_hardware.md) untuk mesin lain.
