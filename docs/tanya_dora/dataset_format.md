# Format Dataset TDID

Satu baris JSONL berisi satu contoh percakapan. Contoh lengkap yang belum diverifikasi tersedia pada [tdid_template.jsonl](examples/tdid_template.jsonl); nilai `verified: false` mencegahnya masuk ke split training.

## Skema record

```json
{
  "record_id": "id-unik",
  "messages": [
    {"role": "system", "content": "Anda adalah Tanya DORA."},
    {"role": "user", "content": "Pertanyaan."},
    {"role": "assistant", "content": "Jawaban."}
  ],
  "metadata": {
    "category": "statistics",
    "difficulty": "beginner",
    "instruction_type": "knowledge",
    "verified": false,
    "source": "",
    "source_url": "",
    "source_group": ""
  }
}
```

`record_id` akan dibuat saat record masuk jika tidak disediakan. Role yang dikenali: `system`, `user`, `assistant`, `tool`; percakapan harus punya user dan assistant, serta berakhir dengan assistant. Verified record harus mempunyai `source` atau `source_url`.

## Nilai kategori dan label

- `category`: `data_science`, `statistics`, `machine_learning`, `deep_learning`, `computer_vision`, `programming`, `research_methodology`, `research_gap`, `dora_knowledge`, `response_behavior`.
- `difficulty`: `beginner`, `intermediate`, `advanced`.
- `instruction_type`: `knowledge`, `reasoning`, `coding`, `research`, `dora_context`, `behavioral`, `uncertainty`, `tool_calling`.
- `verified`: JSON boolean, bukan teks.
- `source_group` atau `source_id`: identifier yang sama untuk contoh dari sumber/kelompok yang harus berada pada split yang sama.

Kategori tambahan dari brief awal seperti Mathematics, NLP, Software Engineering, dan Scientific Writing saat ini dapat dicatat di teks/sumber, tetapi taxonomy validator belum mengizinkan kategori baru. Ubah taxonomy dengan tes jika format perlu dikembangkan.

## CSV

CSV dapat memetakan kolom JSON `messages`, atau kolom pasangan `user` dan `assistant`; `system` opsional. Kolom metadata dapat dipetakan dengan `field_map`, misalnya `{"user":"question","assistant":"answer","category":"topic","verified":"reviewed"}`. String verified yang diterima adalah `true`, `1`, `yes`, atau `verified`; lainnya dianggap belum verified.

Importer tidak mengubah data menjadi format trainer tanpa validasi. Fine-Tuning Engine menggunakan format percakapan TRL dari split yang disimpan.
