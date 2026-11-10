"""Rule-based, explainable model-family guidance for vision datasets."""

from __future__ import annotations

import os
import ctypes
from typing import Any


IMAGE_TASKS = (
    ("classification", "Image classification"),
    ("object_detection", "Object detection"),
    ("segmentation", "Image segmentation"),
    ("image_similarity", "Image similarity"),
)
VIDEO_TASKS = (
    ("video_classification", "Video classification"),
    ("action_recognition", "Action recognition"),
    ("video_similarity", "Video similarity"),
)


def detect_hardware() -> dict[str, Any]:
    """Report only hardware facts that can be checked without adding packages."""
    gpu_available = None
    gpu_note = "GPU availability is unknown because PyTorch is not installed."
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
        gpu_note = "CUDA GPU detected." if gpu_available else "PyTorch did not detect a CUDA GPU."
    except Exception:
        pass
    ram_total_gb = ram_available_gb = None
    if os.name == "nt":
        try:
            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                ram_total_gb = round(status.ullTotalPhys / (1024 ** 3), 2)
                ram_available_gb = round(status.ullAvailPhys / (1024 ** 3), 2)
        except Exception:
            pass
    else:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            ram_total_gb = round(page_size * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3), 2)
            ram_available_gb = round(page_size * os.sysconf("SC_AVPHYS_PAGES") / (1024 ** 3), 2)
        except (AttributeError, OSError, ValueError):
            pass
    return {
        "cpu_cores": os.cpu_count(),
        "gpu_available": gpu_available,
        "gpu_note": gpu_note,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_available_gb,
    }


def _recommendation(
    *,
    name: str,
    task: str,
    reason: str,
    pros: list[str],
    cons: list[str],
    compute: str,
    complexity: str,
    suitable: str,
    limitations: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "task": task,
        "reason": reason,
        "advantages": pros,
        "disadvantages": cons,
        "computational_requirements": compute,
        "complexity": complexity,
        "suitable_dataset_conditions": suitable,
        "limitations": limitations,
    }


def _label_warning(analysis: dict[str, Any]) -> str | None:
    if analysis.get("has_labels"):
        return None
    return (
        "Label tidak terdeteksi. Rekomendasi supervised hanya ditampilkan sebagai langkah setelah anotasi; "
        "dataset ini belum siap untuk klasifikasi supervised."
    )


def _image_recommendations(task: str, analysis: dict[str, Any], hardware: dict[str, Any]):
    sample_count = int(analysis.get("total_samples") or 0)
    class_distribution = analysis.get("class_distribution") or {}
    class_count = int(analysis.get("class_count") or len(class_distribution))
    min_class_count = min(class_distribution.values()) if class_distribution else 0
    imbalance = float(analysis.get("class_imbalance_ratio") or 1.0)
    has_labels = bool(analysis.get("has_labels"))
    gpu = hardware.get("gpu_available") is True
    small_data = sample_count < 2_000 or (class_count and min_class_count < 100)
    imbalance_note = (
        f"Rasio jumlah kelas terbesar/terkecil sekitar {imbalance:.1f}:1; gunakan stratified split, "
        "class weights atau sampling yang terkontrol."
        if class_count > 1 and imbalance >= 3
        else "Distribusi kelas belum menunjukkan ketimpangan besar dari pemeriksaan sederhana ini."
    )

    if task == "image_similarity":
        primary = _recommendation(
            name="Image embeddings + nearest-neighbor search",
            task="Image similarity",
            reason="Tugas kemiripan tidak membutuhkan label kelas. Embedding dari backbone vision pralatih dapat dipakai untuk pencarian tetangga, pengelompokan awal, atau audit duplikasi.",
            pros=["Dapat dimulai tanpa label", "Menggunakan ekstraksi fitur yang dapat di-cache", "Cocok untuk eksplorasi kemiripan visual"],
            cons=["Hasil bergantung pada domain pralatih", "Threshold kemiripan perlu divalidasi manusia"],
            compute="CPU dapat digunakan untuk batch kecil; GPU mempercepat ekstraksi embedding. Simpan embedding bertahap, bukan semua gambar di RAM.",
            complexity="Sedang",
            suitable="Koleksi gambar tanpa label atau tugas pencarian visual yang memiliki contoh kueri.",
            limitations="Kemiripan embedding bukan bukti bahwa dua gambar mempunyai kelas atau makna yang sama.",
        )
        alternatives = [
            _recommendation(
                name="Perceptual hash / local feature matching",
                task="Image similarity",
                reason="Hash perceptual ringan berguna untuk menemukan salinan identik atau perubahan visual kecil sebelum eksperimen embedding.",
                pros=["Tidak memerlukan GPU", "Mudah diaudit dan cepat untuk pemeriksaan awal"],
                cons=["Kurang tahan terhadap crop, rotasi, atau perubahan besar", "Bukan pengganti semantic similarity"],
                compute="CPU rendah; memori kecil.",
                complexity="Rendah",
                suitable="Pemeriksaan duplikasi near-identical pada kumpulan gambar.",
                limitations="Collision dan false positive mungkin terjadi; tinjau kandidat secara visual.",
            )
        ]
        return primary, alternatives, None

    if task in {"object_detection", "segmentation"}:
        annotation = "bounding box" if task == "object_detection" else "pixel-level mask"
        task_name = "Object detection" if task == "object_detection" else "Image segmentation"
        primary = _recommendation(
            name=f"Anotasi {annotation} lalu fine-tuning model pralatih",
            task=task_name,
            reason=f"Struktur folder hanya memberi label gambar. Tugas ini memerlukan anotasi {annotation} per objek/area; label kelas folder tidak cukup sebagai target.",
            pros=["Target sesuai dengan kebutuhan lokasi objek atau area", "Backbone pralatih dapat mengurangi kebutuhan data dibanding training dari nol"],
            cons=["Anotasi memerlukan waktu dan pedoman konsisten", "Evaluasi harus memperhitungkan objek kecil, batas mask, dan variasi antar anotator"],
            compute="GPU disarankan untuk fine-tuning dan resolusi besar. CPU dapat dipakai untuk menyiapkan anotasi dan uji pipeline kecil.",
            complexity="Tinggi",
            suitable=f"Dataset yang memiliki atau akan diberi anotasi {annotation}; split harus dipisah berdasarkan sumber/pasien bila relevan.",
            limitations="Belum ada pemeriksaan berkas anotasi di tahap ini. Rekomendasi model belum merupakan hasil training atau evaluasi.",
        )
        alternatives = [
            _recommendation(
                name="Image similarity untuk audit dan pengelompokan awal",
                task=task_name,
                reason="Sebelum anotasi, embedding dapat membantu menemukan variasi visual dan contoh representatif untuk memberi label.",
                pros=["Tidak membutuhkan anotasi objek", "Membantu merencanakan sampling anotasi"],
                cons=["Tidak menghasilkan bounding box atau mask", "Kelompok embedding perlu ditinjau manusia"],
                compute="CPU untuk eksplorasi kecil; GPU opsional untuk embedding banyak gambar.",
                complexity="Sedang",
                suitable="Dataset yang belum memiliki anotasi lokasi objek atau mask.",
                limitations="Ini hanya langkah eksplorasi dan tidak menyelesaikan tugas supervised.",
            )
        ]
        return primary, alternatives, _label_warning(analysis) if not has_labels else "Label folder/CSV yang ada belum menyediakan anotasi bounding box atau mask."

    if has_labels and class_count < 2:
        primary = _recommendation(
            name="Lengkapi minimal dua kelas sebelum classification",
            task="Image classification",
            reason="Hanya satu kelas terlabel yang ditemukan. Model klasifikasi tidak dapat membedakan kategori yang belum tersedia.",
            pros=["Menghindari rekomendasi training yang tidak dapat dievaluasi", "Menunjukkan kebutuhan dataset secara eksplisit"],
            cons=["Memerlukan anotasi atau sumber kelas tambahan", "Split evaluasi belum dapat dibuat dengan satu kelas"],
            compute="Belum perlu GPU sampai label dan variasi kelas tersedia.",
            complexity="Rendah untuk audit dataset",
            suitable="Dataset satu kelas yang sedang disiapkan untuk tugas klasifikasi multi-kelas.",
            limitations="Tidak ada model classification yang direkomendasikan untuk kondisi satu kelas ini.",
        )
        return primary, [], "Klasifikasi supervised memerlukan setidaknya dua kelas dan sampel yang cukup pada tiap split."

    if not has_labels:
        primary = _recommendation(
            name="Anotasi kelas sebelum supervised classification",
            task="Image classification",
            reason="Tidak ada label kelas yang terdeteksi dari struktur folder atau CSV; model supervised belum memiliki target yang dapat dipelajari.",
            pros=["Mencegah asumsi label yang tidak tersedia", "Label dapat dimulai pada subset yang representatif"],
            cons=["Memerlukan pedoman anotasi dan pemeriksaan konsistensi", "Kelas langka perlu disampel secara sengaja"],
            compute="CPU cukup untuk anotasi dan inspeksi. Kebutuhan training baru dapat ditentukan setelah label dan jumlah per kelas diketahui.",
            complexity="Sedang",
            suitable="Dataset gambar tanpa label atau dengan metadata label yang belum dapat dipetakan.",
            limitations="Kelas tidak dapat disimpulkan hanya dari tampilan dataset; model belum dilatih.",
        )
        alternative = _recommendation(
            name="Image embeddings untuk clustering eksploratif",
            task="Exploratory grouping (unsupervised)",
            reason="Embedding pralatih dapat membantu memeriksa apakah ada kelompok visual yang konsisten sebelum anotasi.",
            pros=["Tidak membutuhkan label", "Membantu pemilihan contoh untuk anotasi"],
            cons=["Cluster tidak otomatis menjadi label ilmiah", "Pilihan metric dan backbone memengaruhi hasil"],
            compute="CPU untuk sampel kecil; GPU opsional untuk embedding seluruh dataset.",
            complexity="Sedang",
            suitable="Dataset belum berlabel dengan variasi visual yang ingin dieksplorasi.",
            limitations="Hasil clustering bukan klasifikasi tervalidasi dan tidak menunjukkan kualitas model.",
        )
        return primary, [alternative], _label_warning(analysis)

    backbone_reason = (
        f"Terdapat {sample_count:,} gambar valid dan {class_count:,} kelas berlabel. "
        + (f"Data per kelas masih kecil (minimum {min_class_count:,}); transfer learning mengurangi kebutuhan belajar fitur dari nol." if small_data else "Ukuran data memungkinkan pembandingan beberapa backbone dengan split yang benar.")
        + " " + imbalance_note
    )
    primary = _recommendation(
        name="Transfer learning dengan ResNet atau EfficientNet",
        task="Image classification",
        reason=backbone_reason,
        pros=["Baseline kuat untuk data kecil-menengah", "Backbone pralatih mengurangi biaya dibanding CNN dari nol", "Mudah dibandingkan dengan metrik per kelas"],
        cons=["Preprocessing dan domain pralatih memengaruhi hasil", "Ketimpangan label dapat membuat akurasi agregat menyesatkan"],
        compute="GPU " + ("tersedia menurut PyTorch; gunakan batch sesuai memori." if gpu else "belum terkonfirmasi; mulai dari batch kecil pada CPU atau periksa GPU/CUDA sebelum fine-tuning."),
        complexity="Sedang",
        suitable="Label kelas valid, split stratified, dan minimal beberapa contoh per kelas. Mulai dengan resolusi yang tersedia lalu ukur biaya resize.",
        limitations="Ini rekomendasi pendekatan, bukan skor eksperimen. Laporkan holdout terpisah, macro-F1, confusion matrix, dan variasi seed.",
    )
    alternatives = [
        _recommendation(
            name="MobileNetV3 transfer learning",
            task="Image classification",
            reason="Alternatif ringan bila deployment atau memori terbatas.",
            pros=["Komputasi dan ukuran model lebih rendah", "Cocok sebagai baseline edge"],
            cons=["Dapat tertinggal pada tugas visual kompleks", "Tetap memerlukan validasi pada domain target"],
            compute="Rendah-menengah; dapat diuji di CPU dengan batch kecil.",
            complexity="Rendah-menengah",
            suitable="Dataset berlabel dan kebutuhan inferensi hemat sumber daya.",
            limitations="Model ringkas tidak menjamin akurasi atau latensi pada perangkat tertentu.",
        ),
    ]
    if sample_count >= 2_000 and gpu and min_class_count >= 20:
        alternatives.append(_recommendation(
            name="Vision Transformer (ViT) fine-tuning",
            task="Image classification",
            reason="Pertimbangkan bila tersedia backbone pralatih yang relevan, jumlah data cukup, dan resource komputasi memadai.",
            pros=["Dapat menangkap konteks visual global", "Ekosistem backbone pralatih luas"],
            cons=["Fine-tuning dan resolusi tinggi dapat mahal", "Lebih sensitif terhadap jumlah dan kualitas data dibanding baseline ringan"],
            compute="GPU dengan memori cukup disarankan; CPU hanya realistis untuk eksperimen sangat kecil.",
            complexity="Tinggi",
            suitable="Data berlabel cukup, GPU terverifikasi, dan alasan tugas mendukung konteks global.",
            limitations="Tidak otomatis lebih baik dari CNN; lakukan perbandingan pada protokol split yang sama.",
        ))
    return primary, alternatives, None


def _video_recommendations(task: str, analysis: dict[str, Any], hardware: dict[str, Any]):
    sample_count = int(analysis.get("total_samples") or 0)
    distribution = analysis.get("class_distribution") or {}
    class_count = int(analysis.get("class_count") or len(distribution))
    minimum_class = min(distribution.values()) if distribution else 0
    has_labels = bool(analysis.get("has_labels"))
    gpu = hardware.get("gpu_available") is True
    imbalance = float(analysis.get("class_imbalance_ratio") or 1.0)
    imbalance_note = f"Periksa ketimpangan kelas sekitar {imbalance:.1f}:1 dan gunakan macro-F1." if class_count > 1 and imbalance >= 3 else "Laporkan metrik per kelas selain skor agregat."

    if task == "video_similarity":
        primary = _recommendation(
            name="Sampled frame embeddings + temporal aggregation",
            task="Video similarity",
            reason="Kemiripan dapat dieksplorasi tanpa label dengan mengambil sejumlah frame terkontrol dan membandingkan embedding atau signature temporal.",
            pros=["Tidak memerlukan label kelas", "Biaya bisa dibatasi lewat jumlah frame sampling"],
            cons=["Sampling jarang dapat melewatkan aksi singkat", "Urutan/kecepatan video memengaruhi kemiripan"],
            compute="CPU untuk sampling ringan; GPU opsional bila mengekstrak embedding backbone besar.",
            complexity="Sedang",
            suitable="Kumpulan video tanpa label untuk deduplikasi atau pencarian contoh mirip.",
            limitations="Kemiripan frame bukan bukti bahwa dua video menampilkan aksi yang sama.",
        )
        return primary, [], None

    if not has_labels:
        primary = _recommendation(
            name="Anotasi clip dan label sebelum supervised learning",
            task="Action recognition" if task == "action_recognition" else "Video classification",
            reason="Label kelas video tidak terdeteksi. Klasifikasi/action recognition supervised membutuhkan target label; action recognition juga perlu definisi clip dan rentang waktu yang konsisten.",
            pros=["Menetapkan target yang dapat dievaluasi", "Memungkinkan split berdasarkan subjek/sesi untuk mengurangi leakage"],
            cons=["Anotasi video lebih mahal dari gambar", "Batas clip dan definisi aksi harus konsisten"],
            compute="CPU cukup untuk membuat manifest dan anotasi. Kebutuhan GPU ditentukan setelah panjang clip dan resolusi dipilih.",
            complexity="Sedang-tinggi",
            suitable="Dataset video yang belum memiliki label atau metadata kelas yang belum dapat dipetakan.",
            limitations="Belum ada pelatihan model. Jumlah frame dan metadata codec bukan label aksi.",
        )
        alternative = _recommendation(
            name="Video similarity dari sampled frames",
            task="Exploratory grouping (unsupervised)",
            reason="Sampling frame dapat membantu mengelompokkan video yang tampak serupa sebelum anotasi.",
            pros=["Dapat dimulai tanpa label", "Jumlah frame per video dapat dibatasi"],
            cons=["Aksi singkat mungkin terlewat", "Cluster perlu ditafsirkan manusia"],
            compute="CPU ringan untuk signature; GPU opsional untuk embedding.",
            complexity="Sedang",
            suitable="Video tanpa label yang perlu diperiksa atau diprioritaskan untuk anotasi.",
            limitations="Pengelompokan tidak menggantikan label video yang tervalidasi.",
        )
        return primary, [alternative], _label_warning(analysis)

    if class_count < 2:
        primary = _recommendation(
            name="Lengkapi label video sebelum classification",
            task="Action recognition" if task == "action_recognition" else "Video classification",
            reason="Hanya satu kelas video yang ditemukan. Model supervised memerlukan kategori yang dapat dibedakan.",
            pros=["Mencegah evaluasi yang tidak bermakna", "Menjelaskan syarat dataset sebelum training"],
            cons=["Memerlukan kelas tambahan atau koreksi label", "Split evaluasi belum dapat dibuat"],
            compute="Belum perlu GPU sampai label dan variasi kelas tersedia.",
            complexity="Rendah untuk audit dataset",
            suitable="Dataset satu kelas yang sedang disiapkan untuk classification/action recognition.",
            limitations="Tidak ada model supervised yang direkomendasikan untuk kondisi satu kelas ini.",
        )
        return primary, [], "Video classification/action recognition memerlukan setidaknya dua kelas untuk evaluasi komparatif."

    task_name = "Action recognition" if task == "action_recognition" else "Video classification"
    note = f"Ada {sample_count:,} video valid dengan {class_count:,} kelas. "
    note += f"Jumlah minimum per kelas {minimum_class:,}. " if class_count else ""
    note += imbalance_note
    primary = _recommendation(
        name="Frame-based CNN baseline",
        task=task_name,
        reason=note + " Pendekatan ini memproses beberapa frame yang disampling dan menjadi baseline biaya lebih rendah sebelum model temporal.",
        pros=["Biaya komputasi dapat dikontrol dengan jumlah frame", "Memanfaatkan backbone gambar pralatih", "Mudah untuk baseline video pendek"],
        cons=["Hubungan temporal antar-frame terbatas", "Aksi singkat dapat terlewat oleh sampling"],
        compute="CPU dapat menguji pipeline kecil; GPU disarankan untuk fine-tuning dataset penuh.",
        complexity="Menengah",
        suitable="Label video tersedia dan frame sampling cukup mewakili aksi/isi yang diprediksi.",
        limitations="Sampling frame adalah pendekatan, bukan hasil training. Pisahkan data berdasarkan subjek, kamera, atau sesi bila berpotensi berbagi konten.",
    )
    alternatives = [
        _recommendation(
            name="CNN + LSTM",
            task=task_name,
            reason="Gunakan bila urutan fitur antar-frame diperkirakan membawa informasi penting dan clip berlabel tersedia.",
            pros=["Merepresentasikan urutan temporal", "Dapat memakai encoder gambar pralatih"],
            cons=["Training lebih kompleks daripada frame-based baseline", "Panjang sequence dan sampling memengaruhi hasil"],
            compute="GPU disarankan; CPU cocok untuk prototipe kecil dan sequence pendek.",
            complexity="Tinggi",
            suitable="Video berlabel dengan urutan frame yang relevan dan panjang clip terkontrol.",
            limitations="Tidak menjamin menangkap aksi berdurasi panjang; bandingkan dengan baseline frame-based.",
        ),
        _recommendation(
            name="3D CNN",
            task=task_name,
            reason="Dapat dipertimbangkan bila gerak lokal ruang-waktu penting dan tersedia clip berlabel yang cukup.",
            pros=["Belajar fitur spasial-temporal bersama", "Pilihan baseline untuk klip aksi"],
            cons=["Memori dan waktu komputasi tinggi", "Memerlukan ukuran clip dan preprocessing konsisten"],
            compute="GPU dengan memori memadai sangat disarankan; CPU biasanya lambat untuk training.",
            complexity="Tinggi",
            suitable="Dataset berlabel dengan banyak clip dan kebutuhan temporal yang jelas.",
            limitations="Jumlah video saja tidak cukup untuk menyimpulkan kelayakan; durasi, resolusi, split, dan GPU perlu diverifikasi.",
        ),
    ]
    if sample_count >= 500 and gpu and minimum_class >= 20:
        alternatives.append(_recommendation(
            name="Video Transformer pralatih",
            task=task_name,
            reason="Pertimbangkan untuk tugas temporal kompleks hanya bila backbone relevan, data berlabel, dan komputasi tersedia.",
            pros=["Dapat memodelkan konteks lintas frame", "Backbone pralatih dapat mengurangi kebutuhan data"],
            cons=["Mahal dalam memori/komputasi", "Fine-tuning dan sampling perlu disetel hati-hati"],
            compute="GPU dengan memori besar sangat disarankan; CPU hanya untuk inspeksi pipeline.",
            complexity="Sangat tinggi",
            suitable="Label clip berkualitas, protokol split bebas leakage, dan GPU/CUDA terverifikasi.",
            limitations="Tidak direkomendasikan sebagai titik awal pada dataset kecil; hasil harus dibandingkan pada split tetap.",
        ))
    return primary, alternatives, None


def recommend_approaches(
    dataset_type: str,
    analysis: dict[str, Any],
    task: str,
    *,
    hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return structured guidance with evidence, compute needs, and limits."""
    hardware_info = dict(hardware or detect_hardware())
    if dataset_type == "image":
        allowed = {key for key, _label in IMAGE_TASKS}
        if task not in allowed:
            task = "classification"
        primary, alternatives, warning = _image_recommendations(task, analysis, hardware_info)
    elif dataset_type == "video":
        allowed = {key for key, _label in VIDEO_TASKS}
        if task not in allowed:
            task = "video_classification"
        primary, alternatives, warning = _video_recommendations(task, analysis, hardware_info)
    else:
        raise ValueError("dataset_type harus image atau video.")
    warnings = [warning] if warning else []
    if analysis.get("invalid_files"):
        warnings.append(f"{analysis['invalid_files']} file tidak dapat dibaca; rekomendasi memakai file valid yang tersisa.")
    if analysis.get("invalid_metadata_count"):
        warnings.append(f"{analysis['invalid_metadata_count']} video mempunyai metadata frame/FPS/resolusi yang tidak lengkap.")
    sample_count = int(analysis.get("total_samples") or 0)
    if sample_count >= 50_000:
        warnings.append("Dataset berukuran besar; analisis membaca satu media per langkah. Untuk mengecek keterwakilan lebih cepat, mulai dari subfolder sampel yang representatif.")
    ram_available = hardware_info.get("ram_available_gb")
    if ram_available is not None and float(ram_available) < 4:
        warnings.append(f"RAM tersedia sekitar {float(ram_available):.1f} GiB. Gunakan subset kecil dan sampling video 1–2 frame per video sebelum eksperimen model.")
    dimensions = analysis.get("dimensions") or {}
    if max(int(dimensions.get("width_max") or 0), int(dimensions.get("height_max") or 0)) >= 4096:
        warnings.append("Ada media beresolusi sangat tinggi; turunkan resolusi pada pipeline training dan uji satu file dulu untuk membatasi kebutuhan RAM/GPU.")
    return {
        "dataset_type": dataset_type,
        "task": task,
        "labels_available": bool(analysis.get("has_labels")),
        "hardware": hardware_info,
        "primary": primary,
        "alternatives": alternatives,
        "warnings": warnings,
        "is_experiment_result": False,
        "note": "Rekomendasi berbasis metadata dataset; tidak ada model yang dilatih atau dievaluasi.",
    }
