# DORA Tool Calling

`app/ai/tool_calling.py` menyediakan registry schema fungsi dan eksekusi argument yang dibatasi. Model hanya dapat meminta nama tool terdaftar. Sistem tidak mengeksekusi kode atau shell yang dihasilkan model; argumen divalidasi menurut schema JSON dan output dibatasi ukurannya.

Tool sesi Tanya DORA mencakup dataset aktif tanpa membaca baris, hasil analyzer DORA yang sudah dihitung, search DKB, serta pencarian paper lewat provider riset existing bila checkbox **Online Research** dicentang. Tool analysis read-only. Tool akademik hanya menambah sumber online; inferensi tetap di Ollama lokal. Kegagalan provider tidak menjadi bukti untuk jawaban.

Chat tool loop dibatasi jumlah putaran/panggilan, mengirim output tool kembali sebagai pesan tool, merekam hasil untuk referensi, dan mengembalikan jawaban tanpa bukti bila tool tidak memberi evidence. Tool dengan `requires_confirmation=True` tidak dijalankan oleh loop ini; interaksi konfirmasi UI untuk tindakan write belum disediakan. Registry Tanya DORA hanya memasang fungsi read-only.

Tes memakai konteks dataset dan respons model sintetis. Belum ada penggunaan Ollama live untuk membuktikan perilaku tool pada model tertentu.
