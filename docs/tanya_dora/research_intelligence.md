# Academic Research Intelligence

Tool Tanya DORA menggunakan kembali `AcademicSearchEngine` DORA, yang menggabungkan provider OpenAlex, Crossref, Europe PMC, OpenAIRE, Semantic Scholar dan Scopus opsional, lalu deduplikasi hasil. Dari Tanya DORA, search provider hanya dipanggil bila pengguna mengaktifkan Online Research. Detail hasil tool dan paper yang ditemukan disimpan sebagai sumber percakapan.

Hasil provider adalah bukti pencarian yang tersedia, bukan cakupan lengkap literatur. Query, limit, status provider, metadata paper, dan abstrak dapat dibahas; identifikasi metode/dataset/limitasi adalah interpretasi abstrak dan harus diberi label demikian. Citation tidak boleh dibuat bila DOI/URL tidak diberikan hasil sumber.

Research gap harus disebut *potential/underexplored research direction*, disertai himpunan paper yang ditemukan, strategi query, limit pencarian, rentang tanggal/provider bila tersedia, dan keterbatasannya. Ketiadaan hasil pada query terbatas bukan bukti bahwa topik belum pernah diteliti. Novelty perlu dibandingkan berdasarkan metode, data, objek, dan kontribusi, serta dinilai peneliti; AI tidak menjaminnya.

Belum ada pipeline UI khusus untuk perbandingan paper, peta topik, abstrak terstruktur, atau review gap. Penggunaan research tool memakai client pencarian yang ada dan berjalan di worker chat, bukan inferensi cloud.
