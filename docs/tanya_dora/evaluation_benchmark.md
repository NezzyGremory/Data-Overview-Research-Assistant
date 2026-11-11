# Tanya Dora Evaluation Benchmark (TDEB)

`app/ai/evaluation_benchmark.py` mengelola snapshot test JSONL terpisah, memeriksa prompt dan source group terhadap TDID, menjalankan target yang diberikan pengguna, menyimpan output/error/latency dan snapshot RAM/VRAM, serta menyimpan human review di SQLite lokal.

## Alur

1. Buat kasus TDEB terpisah dari training. Jangan salin case ke TDID.
2. Panggil `TDEBBenchmarkStore.save_version(..., training_dataset=tdid_builder)`. Save ditolak bila prompt atau source group bertabrakan. Tanpa builder pembanding, versi ditandai belum diperiksa dan runner menolak menjalankannya.
3. Buat `BenchmarkTarget`; `BenchmarkTarget.ollama(manager, model)` menjalankan model lokal nyata. Untuk RAG/tools, berikan callback yang benar-benar memanggil pipeline tersebut dan mengembalikan `BenchmarkOutput`/respons dengan sources dan tool results.
4. Jalankan tiap target terhadap versi dataset yang sama, lalu `compare_runs()`.
5. Tambahkan human review per kasus untuk faktual, reasoning, kualitas Indonesia, coding/math, citation faithfulness, hallucination, dan kriteria lain.

## Interpretasi

Engine mencatat latency per kasus, error, exact-match bila reference answer diberikan, required-term recall, citation ID recall, dan expected-tool coverage. Metrik lexical itu proxy sederhana dan **bukan** factual accuracy, reasoning quality, atau hallucination rate. `human_rated_quality` tetap `null` pada summary otomatis; nilai yang diberikan peninjau tersimpan terpisah. RAM/VRAM adalah snapshot sistem sebelum/sesudah seluruh run, bukan peak usage proses.

Tidak ada live Ollama, base-vs-fine-tuned inference, atau angka kualitas nyata yang dijalankan di fase ini. Target `mode` hanyalah label; pastikan callback target benar-benar menjalankan kondisi Local, RAG, adapter, atau tools yang disebut.
