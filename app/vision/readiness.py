"""Evidence-based image task and experiment readiness summaries."""

from __future__ import annotations

from typing import Any


IMAGE_TASK_LABELS = {
    "classification": "Image Classification",
    "object_detection": "Object Detection",
    "segmentation": "Image Segmentation",
    "image_similarity": "Image Similarity",
    "image_clustering": "Image Clustering",
}


def _decimal_display(value: Any, places: int = 2) -> str:
    try:
        return f"{float(value):.{places}f}".replace(".", ",")
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _label_evidence(analysis: dict[str, Any]) -> str:
    distribution = analysis.get("class_distribution") or {}
    labeled = int(analysis.get("labeled_count") if analysis.get("labeled_count") is not None else sum(distribution.values()))
    total = int(analysis.get("total_samples") or 0)
    unlabeled = int(analysis.get("unlabeled_count") or max(0, total - labeled))
    source = analysis.get("label_source_description") or "Label tidak terdeteksi."
    return f"{labeled:,}/{total:,} gambar berlabel; {unlabeled:,} belum berlabel. {source}"


def assess_image_tasks(analysis: dict[str, Any]) -> list[dict[str, str]]:
    """Describe task prerequisites without inferring labels or annotations."""
    sample_count = int(analysis.get("total_samples") or 0)
    class_count = int(analysis.get("class_count") or 0)
    labeled_count = int(analysis.get("labeled_count") if analysis.get("labeled_count") is not None else sum((analysis.get("class_distribution") or {}).values()))
    unlabeled_count = int(analysis.get("unlabeled_count") or max(0, sample_count - labeled_count))
    class_distribution = analysis.get("class_distribution") or {}
    min_class = min(class_distribution.values()) if class_distribution else 0
    source_note = str(analysis.get("label_source_description") or "Sumber label belum terdeteksi.")

    if class_count == 0:
        classification_status = "Belum siap supervised"
        classification_evidence = "Label kelas tidak terdeteksi. " + source_note
    elif class_count < 2:
        classification_status = "Belum siap multi-kelas"
        classification_evidence = f"Hanya {class_count} kelas terdeteksi; classification memerlukan setidaknya dua kelas yang tervalidasi. {source_note}"
    elif unlabeled_count:
        classification_status = "Perlu pemeriksaan label"
        classification_evidence = f"{class_count} kelas terdeteksi, dengan {unlabeled_count:,} gambar valid tanpa label. Tinjau cakupan dan arti label sebelum split. {source_note}"
    elif min_class < 2:
        classification_status = "Perlu lebih banyak sampel per kelas"
        classification_evidence = f"Ada {class_count} kelas, tetapi setidaknya satu kelas memiliki kurang dari dua gambar. Split dan validitas label belum diuji. {source_note}"
    else:
        classification_status = "Dapat dipertimbangkan untuk pilot"
        classification_evidence = f"{class_count} kelas dan {labeled_count:,} gambar berlabel; minimum {min_class:,} gambar per kelas. Split, kebocoran, dan validitas semantik belum diuji. {source_note}"

    if sample_count >= 2:
        unsupervised_status = "Dapat dieksplorasi"
        unsupervised_evidence = f"{sample_count:,} gambar valid tersedia; label tidak wajib. Cluster/kemiripan perlu divalidasi secara visual dan domain."
    elif sample_count == 1:
        unsupervised_status = "Data terlalu sedikit untuk perbandingan"
        unsupervised_evidence = "Satu gambar valid terdeteksi; belum ada pasangan untuk perbandingan atau kelompok."
    else:
        unsupervised_status = "Tidak ada gambar valid"
        unsupervised_evidence = "Tambahkan gambar yang dapat dibaca sebelum eksplorasi."

    annotation_note = str(
        (analysis.get("annotation_readiness") or {}).get(
            "note",
            "Berkas bounding box dan mask tidak diperiksa oleh pemindai ini; status anotasi belum diketahui.",
        )
    )
    return [
        {
            "task": "Image Classification",
            "status": classification_status,
            "evidence": classification_evidence,
        },
        {
            "task": "Object Detection",
            "status": "Anotasi belum diverifikasi",
            "evidence": "Memerlukan bounding box per objek. " + annotation_note,
        },
        {
            "task": "Image Segmentation",
            "status": "Anotasi belum diverifikasi",
            "evidence": "Memerlukan mask/poligon per piksel atau area. " + annotation_note,
        },
        {
            "task": "Image Similarity",
            "status": unsupervised_status,
            "evidence": unsupervised_evidence,
        },
        {
            "task": "Image Clustering",
            "status": unsupervised_status,
            "evidence": unsupervised_evidence + " Cluster bukan label ground-truth.",
        },
    ]


def assess_image_readiness(
    analysis: dict[str, Any],
    selected_task: str,
) -> dict[str, Any]:
    """Return transparent readiness checks; intentionally no composite score."""
    valid_count = int(analysis.get("total_samples") or 0)
    invalid_count = int(analysis.get("invalid_files") or 0)
    labeled_count = int(analysis.get("labeled_count") or 0)
    unlabeled_count = int(analysis.get("unlabeled_count") or max(0, valid_count - labeled_count))
    class_distribution = analysis.get("class_distribution") or {}
    class_count = int(analysis.get("class_count") or len(class_distribution))
    min_class = min(class_distribution.values()) if class_distribution else 0
    max_min_ratio = analysis.get("class_imbalance_ratio")
    coefficient = analysis.get("class_count_coefficient_of_variation")
    source_note = str(analysis.get("label_source_description") or "Label tidak terdeteksi.")
    formats = analysis.get("formats") or {}
    modes = analysis.get("image_modes") or {}
    aspect_buckets = analysis.get("aspect_ratio_buckets") or {}
    resolution_buckets = analysis.get("resolution_buckets") or {}
    near_uniform = int(analysis.get("near_uniform_count") or 0)
    sharpness = (analysis.get("visual_characteristics") or {}).get("mean_laplacian_variance_64px")

    if labeled_count == 0:
        label_status = "Belum siap untuk supervised"
    elif class_count < 2:
        label_status = "Perlu kelas tambahan atau koreksi target"
    elif unlabeled_count:
        label_status = "Perlu audit cakupan label"
    else:
        label_status = "Label terdeteksi; validasi semantik tetap diperlukan"

    if valid_count:
        quantity_evidence = (
            f"{valid_count:,} gambar valid; {class_count:,} kelas; minimum {min_class:,} gambar per kelas. "
            "Kecukupan statistik tidak diberi skor karena bergantung pada task, variasi target, dan rancangan evaluasi."
        )
        quantity_status = "Jumlah terukur; kecukupan task-specific belum disimpulkan"
    else:
        quantity_status = "Belum ada data valid"
        quantity_evidence = "Tidak ada gambar valid untuk dinilai."

    if class_count < 2 or max_min_ratio is None:
        balance_status = "Tidak dapat dihitung"
        balance_evidence = "Rasio kelas memerlukan setidaknya dua kelas berlabel."
    else:
        ratio_value = float(max_min_ratio)
        balance_status = "Perlu tinjau distribusi" if ratio_value > 3 else "Di bawah ambang triase 3:1"
        balance_evidence = (
            f"Rasio maksimum/minimum {_decimal_display(ratio_value)}:1; CV populasi {_decimal_display(coefficient, 4) if coefficient is not None else 'tidak tersedia'}. "
            "Ambang 3:1 adalah triase sederhana, bukan batas universal; periksa semua jumlah kelas."
        )

    quality_flags = []
    if invalid_count:
        quality_flags.append(f"{invalid_count:,} file tidak terbaca")
    if near_uniform:
        quality_flags.append(f"{near_uniform:,} thumbnail hampir seragam")
    quality_status = "Perlu tinjau sampel" if quality_flags else "Tidak ada indikator awal yang melewati pemeriksaan sederhana"
    quality_evidence = "; ".join(quality_flags) if quality_flags else "Tidak ditemukan file rusak atau thumbnail hampir seragam."
    characteristics = analysis.get("visual_characteristics") or {}
    quality_evidence += (
        f" Brightness luminans mean {_decimal_display(characteristics.get('mean_brightness_0_255'))}/255, "
        f"contrast mean {_decimal_display(characteristics.get('mean_contrast_luminance_std_0_255'))}/255, "
        f"sharpness Laplacian thumbnail {_decimal_display(sharpness, 3) if sharpness is not None else 'belum tersedia'}; "
        "indikator sederhana tanpa ambang blur universal."
    )

    consistency_flags = []
    if len(formats) > 1:
        consistency_flags.append(f"{len(formats)} format")
    if len(modes) > 1:
        consistency_flags.append(f"{len(modes)} mode/channel")
    if len(aspect_buckets) > 1:
        consistency_flags.append(f"{len(aspect_buckets)} kelompok aspect ratio")
    if len(resolution_buckets) > 1:
        consistency_flags.append(f"{len(resolution_buckets)} kelompok resolusi")
    consistency_status = "Perlu rencana preprocessing" if consistency_flags else "Tidak ada variasi kategori yang terdeteksi"
    consistency_evidence = ", ".join(consistency_flags) if consistency_flags else "Format, mode, aspect ratio, dan kelompok resolusi konsisten pada ringkasan ini."

    task = selected_task if selected_task in IMAGE_TASK_LABELS else "classification"
    task_name = IMAGE_TASK_LABELS[task]
    if task == "classification":
        task_ready = class_count >= 2 and labeled_count > 0 and unlabeled_count == 0 and min_class >= 2
        experiment_status = "Syarat dasar terdeteksi; perlu split dan validasi" if task_ready else "Belum siap untuk eksperimen classification"
        experiment_evidence = (
            f"Task {task_name}: minimal dua kelas dan dua gambar per kelas terdeteksi, tanpa gambar valid yang belum berlabel. "
            "Split independen, leakage, dan validitas label belum diperiksa."
            if task_ready
            else f"Task {task_name}: perlu >=2 kelas berlabel, >=2 gambar per kelas, dan cakupan label penuh untuk pemeriksaan dasar. {source_note}"
        )
    elif task in {"object_detection", "segmentation"}:
        experiment_status = "Belum dapat dinyatakan siap"
        experiment_evidence = "File anotasi tidak diperiksa; verifikasi schema, cakupan objek/mask, dan split sebelum memilih model."
    elif task in {"image_similarity", "image_clustering"}:
        experiment_status = "Eksplorasi awal dapat dimulai" if valid_count >= 2 else "Belum cukup pasangan gambar"
        experiment_evidence = "Label tidak wajib, tetapi pilihan embedding/metric dan hasil kelompok perlu validasi domain serta pemeriksaan sampel."
    else:
        experiment_status = "Task belum dikenali"
        experiment_evidence = "Pilih salah satu task image yang tersedia."

    return {
        "has_composite_score": False,
        "selected_task": task_name,
        "items": [
            {"name": "Label readiness", "status": label_status, "evidence": _label_evidence(analysis)},
            {"name": "Data quantity", "status": quantity_status, "evidence": quantity_evidence},
            {"name": "Class balance", "status": balance_status, "evidence": balance_evidence},
            {"name": "Image quality", "status": quality_status, "evidence": quality_evidence},
            {"name": "Dataset consistency", "status": consistency_status, "evidence": consistency_evidence},
            {"name": "Experiment readiness", "status": experiment_status, "evidence": experiment_evidence},
        ],
        "task_assessments": assess_image_tasks(analysis),
        "method_note": "Tidak ada skor gabungan. Setiap kesimpulan ditautkan ke hitungan, aturan triase, atau status pemeriksaan yang belum tersedia.",
    }
