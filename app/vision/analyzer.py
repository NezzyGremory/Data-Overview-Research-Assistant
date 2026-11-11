"""Memory-conscious local analysis for image and video dataset collections."""

from __future__ import annotations

import csv
import io
import itertools
import math
import os
import random
import statistics
import shutil
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, ImageOps, ImageStat


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_IMAGE_PIXELS = 25_000_000
MAX_SAMPLE_REFERENCES = 48
MAX_STATISTICS_SAMPLE = 20_000
IMAGE_BATCH_SIZE = 25
MAX_ERROR_LOGS = 100
MAX_ZIP_MEMBER_BYTES = 16 * 1024 * 1024 * 1024
SPLIT_DIRECTORIES = {"train", "training", "val", "valid", "validation", "test", "testing"}
GENERIC_DIRECTORIES = SPLIT_DIRECTORIES | {"image", "images", "video", "videos", "dataset", "data"}

ProgressCallback = Callable[[int, str], None]


class AnalysisCancelledError(Exception):
    """Raised when the caller requests cancellation between bounded work units."""


class VisionDependencyError(RuntimeError):
    """Raised when an optional native media dependency is missing."""


@dataclass(frozen=True)
class _MediaRef:
    name: str
    size_bytes: int
    relative_path: str
    member: str | None = None


def _check_cancel(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelledError("Analisis dibatalkan oleh pengguna.")


def _emit_progress(progress: ProgressCallback | None, percent: int, message: str) -> None:
    if progress is not None:
        progress(max(0, min(100, int(percent))), str(message))


def _is_zip_path(path: Path) -> bool:
    return path.is_file() and path.suffix.casefold() == ".zip"


def _iter_folder_media(root: Path, extensions: set[str], cancel_event=None) -> Iterable[_MediaRef]:
    for current, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            name for name in dirs
            if not (Path(current) / name).is_symlink()
        )
        for name in sorted(names, key=str.casefold):
            _check_cancel(cancel_event)
            file_path = Path(current) / name
            if file_path.suffix.casefold() not in extensions or file_path.is_symlink():
                continue
            try:
                size = file_path.stat().st_size
                relative = file_path.relative_to(root).as_posix()
            except OSError:
                size = 0
                relative = file_path.name
            yield _MediaRef(str(file_path), size, relative)


def _measure_folder_media(root: Path, extensions: set[str], cancel_event=None) -> tuple[int, int]:
    count = size_bytes = 0
    for ref in _iter_folder_media(root, extensions, cancel_event):
        count += 1
        size_bytes += ref.size_bytes
    return count, size_bytes


def _zip_media_infos(archive: zipfile.ZipFile, extensions: set[str]):
    return [
        info for info in archive.infolist()
        if not info.is_dir()
        and Path(info.filename).suffix.casefold() in extensions
        and not PurePosixPath(info.filename).is_absolute()
        and ".." not in PurePosixPath(info.filename).parts
    ]


def _iter_archive_media(infos) -> Iterable[_MediaRef]:
    for info in infos:
        yield _MediaRef(
            info.filename,
            int(info.file_size),
            PurePosixPath(info.filename).as_posix(),
            info.filename,
        )


def _iter_batched_refs(references: Iterable[_MediaRef], batch_size: int = IMAGE_BATCH_SIZE):
    """Enumerate lightweight media references in bounded batches, without decoding a batch into RAM."""
    iterator = iter(references)
    while True:
        batch = list(itertools.islice(iterator, max(1, int(batch_size))))
        if not batch:
            return
        yield from batch


def _source_kind(path: str | Path) -> tuple[Path, str]:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {source}")
    if source.is_dir():
        return source.resolve(), "folder"
    if _is_zip_path(source):
        return source.resolve(), "zip"
    raise ValueError("Pilih folder dataset atau file ZIP.")


def _folder_label(relative_path: str) -> str | None:
    parts = PurePosixPath(relative_path).parts[:-1]
    if not parts:
        return None
    folded = [part.casefold() for part in parts]
    if folded[0] in GENERIC_DIRECTORIES and len(parts) > 1:
        parts = parts[1:]
        folded = folded[1:]
    if not parts:
        return None
    if folded[0] in SPLIT_DIRECTORIES and len(parts) > 1:
        return parts[1]
    return parts[0]


def _normalize_metadata_key(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").lstrip("./").casefold()


def _metadata_columns(fieldnames: list[str] | None) -> tuple[str | None, str | None]:
    normalized = {
        str(name).strip().casefold().replace(" ", "_"): name
        for name in (fieldnames or [])
        if name is not None
    }
    path_candidates = (
        "image_path", "video_path", "clip_path", "filepath", "file_path",
        "filename", "file_name", "image", "video", "clip", "path", "file",
        "image_id", "video_id",
    )
    label_candidates = ("label", "class", "category", "target", "class_name")
    path_column = next((normalized[key] for key in path_candidates if key in normalized), None)
    label_column = next((normalized[key] for key in label_candidates if key in normalized), None)
    return path_column, label_column


def _read_metadata_labels(stream) -> tuple[dict[str, str], dict[str, str], str | None]:
    text_stream = io.TextIOWrapper(stream, encoding="utf-8-sig", errors="replace", newline="")
    try:
        reader = csv.DictReader(text_stream)
        path_column, label_column = _metadata_columns(reader.fieldnames)
        if not path_column or not label_column:
            return {}, {}, None
        relative: dict[str, str] = {}
        basename: dict[str, str | None] = {}
        for row_index, row in enumerate(reader):
            if row_index >= 1_000_000:
                break
            raw_path = row.get(path_column)
            label = str(row.get(label_column) or "").strip()
            if not raw_path or not label:
                continue
            key = _normalize_metadata_key(raw_path)
            if key:
                relative[key] = label
            name = PurePosixPath(key).name
            if name:
                previous = basename.get(name)
                basename[name] = label if previous in (None, label) else None
        return relative, {
            key: value for key, value in basename.items() if value is not None
        }, label_column
    finally:
        try:
            text_stream.close()
        except Exception:
            pass


def _find_folder_metadata(root: Path) -> tuple[dict[str, str], dict[str, str], str | None, list[str]]:
    for path in sorted(root.glob("*.csv"), key=lambda item: item.name.casefold())[:8]:
        try:
            with path.open("rb") as stream:
                by_path, by_name, label_column = _read_metadata_labels(stream)
            if label_column:
                return by_path, by_name, label_column, [path.name]
        except OSError:
            continue
    return {}, {}, None, []


def _find_archive_metadata(archive: zipfile.ZipFile):
    for info in sorted(
        (entry for entry in archive.infolist() if not entry.is_dir() and entry.filename.casefold().endswith(".csv")),
        key=lambda item: item.filename.casefold(),
    )[:8]:
        try:
            with archive.open(info, "r") as stream:
                by_path, by_name, label_column = _read_metadata_labels(stream)
            if label_column:
                return by_path, by_name, label_column, [info.filename]
        except (OSError, zipfile.BadZipFile, RuntimeError):
            continue
    return {}, {}, None, []


def _metadata_label(ref: _MediaRef, by_path: dict[str, str], by_name: dict[str, str]) -> str | None:
    key = _normalize_metadata_key(ref.relative_path)
    return by_path.get(key) or by_name.get(PurePosixPath(key).name)


def _label_source(metadata_count: int, folder_count: int) -> str | None:
    if metadata_count and folder_count:
        return "metadata_csv+folder_structure"
    if metadata_count:
        return "metadata_csv"
    if folder_count:
        return "folder_structure"
    return None


def _split_name(relative_path: str) -> str | None:
    parts = {part.casefold() for part in PurePosixPath(relative_path).parts[:-1]}
    return next((name for name in SPLIT_DIRECTORIES if name in parts), None)


def _record_error(errors: list[dict[str, str]], ref: _MediaRef, error: Exception | str) -> None:
    if len(errors) >= MAX_ERROR_LOGS:
        return
    detail = str(error)[:240]
    errors.append({"file": ref.relative_path, "error": detail})


def _reservoir_add(
    samples: list[Any],
    reference: Any,
    seen: int,
    rng: random.Random,
    limit: int = MAX_SAMPLE_REFERENCES,
) -> None:
    if len(samples) < limit:
        samples.append(reference)
        return
    index = rng.randrange(seen)
    if index < limit:
        samples[index] = reference


def _dimension_bucket(value: int) -> str:
    if value < 256:
        return "under_256"
    if value < 512:
        return "256_to_511"
    if value < 1024:
        return "512_to_1023"
    return "1024_or_more"


def _file_size_bucket(value: int) -> str:
    if value < 64 * 1024:
        return "under_64_kb"
    if value < 256 * 1024:
        return "64_to_255_kb"
    if value < 1024 * 1024:
        return "256_to_1023_kb"
    if value < 4 * 1024 * 1024:
        return "1_to_3_mb"
    return "4_mb_or_more"


def _image_quality_metrics(image: Image.Image) -> tuple[float, float, float, bool]:
    """Return thumbnail luminance mean/std and Laplacian variance proxies."""
    gray = ImageOps.grayscale(image)
    gray_pixels = np.asarray(gray, dtype=np.float32)
    brightness = float(gray_pixels.mean()) if gray_pixels.size else 0.0
    contrast = float(gray_pixels.std()) if gray_pixels.size else 0.0
    if min(gray_pixels.shape, default=0) >= 3:
        center = gray_pixels[1:-1, 1:-1]
        laplacian = (
            gray_pixels[:-2, 1:-1]
            + gray_pixels[2:, 1:-1]
            + gray_pixels[1:-1, :-2]
            + gray_pixels[1:-1, 2:]
            - 4 * center
        )
        laplacian_variance = float(laplacian.var())
    else:
        laplacian_variance = 0.0

    color_pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    channel_spread = color_pixels.max(axis=2) - color_pixels.min(axis=2)
    grayscale_content = bool(np.mean(channel_spread <= 2) >= 0.99)
    return brightness, contrast, laplacian_variance, grayscale_content


def _dhash(image: Image.Image) -> int:
    pixels = image.resize((9, 8), Image.Resampling.LANCZOS).convert("L").tobytes()
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def _aspect_bucket(ratio: float) -> str:
    if ratio < 0.75:
        return "portrait"
    if ratio < 1.33:
        return "near_square"
    if ratio < 1.78:
        return "landscape"
    return "wide"


def _resolution_bucket(width: int, height: int) -> str:
    longest = max(width, height)
    if longest < 256:
        return "under_256"
    if longest < 512:
        return "256_to_511"
    if longest < 1024:
        return "512_to_1023"
    return "1024_or_more"


def _open_image(ref: _MediaRef, archive: zipfile.ZipFile | None):
    if ref.member is not None:
        if archive is None:
            raise RuntimeError("Arsip ZIP tidak tersedia.")
        return archive.open(ref.member, "r")
    return open(ref.name, "rb")


def analyze_image_dataset(
    path: str | Path,
    *,
    progress: ProgressCallback | None = None,
    cancel_event=None,
) -> dict[str, Any]:
    """Inspect a folder or ZIP without retaining decoded images in memory."""
    source, source_type = _source_kind(path)
    errors: list[dict[str, str]] = []
    samples: list[dict[str, Any]] = []
    rng = random.Random(2026)
    labels: Counter[str] = Counter()
    metadata_label_count = folder_label_count = 0
    formats: Counter[str] = Counter()
    aspects: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    width_buckets: Counter[str] = Counter()
    height_buckets: Counter[str] = Counter()
    file_size_buckets: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    band_counts: Counter[str] = Counter()
    hashes: set[int] = set()
    duplicate_count = 0
    unreadable_count = 0
    near_uniform_count = 0
    total_size_bytes = 0
    media_size_bytes = 0
    valid_file_size_sum = 0
    file_size_min = file_size_max = None
    valid_count = 0
    width_sum = height_sum = 0
    resolution_pixel_sum = 0
    resolution_pixel_min = resolution_pixel_max = None
    width_min = height_min = None
    width_max = height_max = 0
    ratio_min = ratio_max = None
    ratio_sum = 0.0
    brightness_sum = contrast_sum = sharpness_sum = 0.0
    brightness_min = contrast_min = sharpness_min = None
    brightness_max = contrast_max = sharpness_max = 0.0
    grayscale_content_count = 0
    statistics_samples: list[tuple[int, int, float, int, float, float, float, int]] = []
    statistics_rng = random.Random(20261110)

    archive = zipfile.ZipFile(source, "r") if source_type == "zip" else None
    try:
        infos = _zip_media_infos(archive, IMAGE_EXTENSIONS) if archive is not None else None
        if infos is not None:
            by_path, by_name, label_column, metadata_files = _find_archive_metadata(archive)
            refs = _iter_archive_media(infos)
            total_candidates = len(infos)
            media_size_bytes = sum(int(info.file_size) for info in infos)
            total_size_bytes = source.stat().st_size
        else:
            by_path, by_name, label_column, metadata_files = _find_folder_metadata(source)
            total_candidates, total_size_bytes = _measure_folder_media(source, IMAGE_EXTENSIONS, cancel_event)
            media_size_bytes = total_size_bytes
            refs = _iter_folder_media(source, IMAGE_EXTENSIONS, cancel_event)
        if total_candidates == 0:
            raise ValueError("Tidak ditemukan file gambar yang didukung pada dataset.")

        if source_type == "folder":
            for metadata_name in metadata_files:
                try:
                    total_size_bytes += (source / metadata_name).stat().st_size
                except OSError:
                    pass

        _emit_progress(progress, 2, "Memeriksa daftar file gambar")
        for processed, ref in enumerate(_iter_batched_refs(refs), start=1):
            _check_cancel(cancel_event)
            if source_type == "zip" and ref.size_bytes > MAX_ZIP_MEMBER_BYTES:
                unreadable_count += 1
                _record_error(errors, ref, "Ukuran file di dalam ZIP melewati batas 16 GiB.")
                continue
            metadata_label = _metadata_label(ref, by_path, by_name)
            label = metadata_label or _folder_label(ref.relative_path)
            try:
                with _open_image(ref, archive) as stream:
                    with Image.open(stream) as image:
                        width, height = image.size
                        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                            raise ValueError(f"Dimensi tidak aman untuk diproses: {width} × {height}.")
                        image.load()
                        image_format = (image.format or Path(ref.relative_path).suffix.lstrip(".")).upper()
                        image_mode = image.mode
                        channel_count = len(image.getbands())
                        image.thumbnail((64, 64), Image.Resampling.BILINEAR)
                        brightness, contrast, sharpness, is_grayscale = _image_quality_metrics(image)
                        small_image = image.copy()
                        small_image.thumbnail((24, 24), Image.Resampling.BILINEAR)
                        small_gray = ImageOps.grayscale(small_image)
                        deviation = float(ImageStat.Stat(small_gray).stddev[0])
                        image_hash = _dhash(small_image.convert("RGB"))
                valid_count += 1
                if label:
                    labels[label] += 1
                    if metadata_label:
                        metadata_label_count += 1
                    else:
                        folder_label_count += 1
                formats[image_format] += 1
                mode_counts[image_mode] += 1
                band_counts[str(channel_count)] += 1
                grayscale_content_count += int(is_grayscale)
                ratio = width / height
                aspects[_aspect_bucket(ratio)] += 1
                resolutions[_resolution_bucket(width, height)] += 1
                width_buckets[_dimension_bucket(width)] += 1
                height_buckets[_dimension_bucket(height)] += 1
                file_size_buckets[_file_size_bucket(ref.size_bytes)] += 1
                duplicate_count += int(image_hash in hashes)
                hashes.add(image_hash)
                near_uniform_count += int(deviation < 1.5)
                width_sum += width
                height_sum += height
                resolution_pixels = width * height
                resolution_pixel_sum += resolution_pixels
                resolution_pixel_min = resolution_pixels if resolution_pixel_min is None else min(resolution_pixel_min, resolution_pixels)
                resolution_pixel_max = max(resolution_pixel_max or 0, resolution_pixels)
                width_min = width if width_min is None else min(width_min, width)
                height_min = height if height_min is None else min(height_min, height)
                width_max = max(width_max, width)
                height_max = max(height_max, height)
                ratio_sum += ratio
                ratio_min = ratio if ratio_min is None else min(ratio_min, ratio)
                ratio_max = ratio if ratio_max is None else max(ratio_max, ratio)
                valid_file_size_sum += ref.size_bytes
                file_size_min = ref.size_bytes if file_size_min is None else min(file_size_min, ref.size_bytes)
                file_size_max = max(file_size_max or 0, ref.size_bytes)
                brightness_sum += brightness
                contrast_sum += contrast
                sharpness_sum += sharpness
                brightness_min = brightness if brightness_min is None else min(brightness_min, brightness)
                brightness_max = max(brightness_max, brightness)
                contrast_min = contrast if contrast_min is None else min(contrast_min, contrast)
                contrast_max = max(contrast_max, contrast)
                sharpness_min = sharpness if sharpness_min is None else min(sharpness_min, sharpness)
                sharpness_max = max(sharpness_max, sharpness)
                _reservoir_add(
                    statistics_samples,
                    (width, height, ratio, ref.size_bytes, brightness, contrast, sharpness, resolution_pixels),
                    valid_count,
                    statistics_rng,
                    limit=MAX_STATISTICS_SAMPLE,
                )
                _reservoir_add(
                    samples,
                    {"name": ref.relative_path, "path": ref.name, "member": ref.member, "label": label, "kind": "image"},
                    valid_count,
                    rng,
                )
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                unreadable_count += 1
                _record_error(errors, ref, error)
            if processed == total_candidates or processed % 25 == 0:
                percent = 5 + round(85 * processed / max(total_candidates, 1))
                _emit_progress(progress, percent, f"Memeriksa gambar {processed:,} dari {total_candidates:,}")

        _check_cancel(cancel_event)
    finally:
        if archive is not None:
            archive.close()

    if valid_count == 0:
        raise ValueError("Tidak ada gambar yang dapat dibaca. Periksa format dan isi dataset.")
    ordered_labels = dict(sorted(labels.items(), key=lambda item: item[0].casefold()))
    class_sizes = list(ordered_labels.values())
    imbalance_ratio = max(class_sizes) / min(class_sizes) if class_sizes else None
    labeled_count = sum(class_sizes)
    width_samples = [sample[0] for sample in statistics_samples]
    height_samples = [sample[1] for sample in statistics_samples]
    ratio_samples = [sample[2] for sample in statistics_samples]
    file_size_samples = [sample[3] for sample in statistics_samples]
    brightness_samples = [sample[4] for sample in statistics_samples]
    contrast_samples = [sample[5] for sample in statistics_samples]
    sharpness_samples = [sample[6] for sample in statistics_samples]
    resolution_pixel_samples = [sample[7] for sample in statistics_samples]
    class_mean = sum(class_sizes) / len(class_sizes) if class_sizes else 0.0
    class_count_cv = (
        math.sqrt(sum((count - class_mean) ** 2 for count in class_sizes) / len(class_sizes)) / class_mean
        if class_sizes and class_mean
        else None
    )
    label_source = _label_source(metadata_label_count, folder_label_count)
    label_source_description = {
        "metadata_csv": "Label diambil dari CSV yang cocok ke file media.",
        "folder_structure": "Nama folder dipakai sebagai label sementara; validasi bahwa folder memang mewakili kelas target.",
        "metadata_csv+folder_structure": "Label berasal dari CSV bila cocok, lalu folder dipakai sebagai fallback; validasi sumber setiap label.",
    }.get(label_source, "Label belum terdeteksi dari CSV atau struktur folder.")
    result = {
        "dataset_type": "image",
        "source_path": str(source),
        "source_kind": source_type,
        "processing_batch_size": IMAGE_BATCH_SIZE,
        "total_files": total_candidates,
        "total_samples": valid_count,
        "class_count": len(ordered_labels),
        "has_labels": bool(ordered_labels),
        "labeled_count": labeled_count,
        "label_source": label_source,
        "label_source_description": label_source_description,
        "annotation_readiness": {
            "bounding_box": "not_checked",
            "segmentation_mask": "not_checked",
            "note": "Berkas YOLO/COCO dan mask tidak diperiksa oleh pemindai gambar; konfirmasi anotasi secara terpisah sebelum memilih task detection atau segmentation.",
        },
        "label_column": label_column,
        "metadata_files": metadata_files,
        "class_distribution": ordered_labels,
        "unlabeled_count": max(0, valid_count - labeled_count),
        "class_imbalance_ratio": imbalance_ratio,
        "class_count_coefficient_of_variation": round(class_count_cv, 4) if class_count_cv is not None else None,
        "class_balance_method": "Rasio kelas maksimum/minimum dan CV populasi dari jumlah sampel per kelas; rasio 3:1 hanya ambang triase, bukan aturan universal.",
        "source_size_bytes": total_size_bytes,
        "media_size_bytes": media_size_bytes,
        "file_size_bytes": {
            "min": file_size_min,
            "mean": round(valid_file_size_sum / valid_count, 2),
            "median": round(statistics.median(file_size_samples), 2),
            "max": file_size_max,
            "statistics_sample_size": len(statistics_samples),
            "median_is_sampled": valid_count > len(statistics_samples),
        },
        "formats": dict(sorted(formats.items())),
        "image_modes": dict(sorted(mode_counts.items())),
        "channel_counts": dict(sorted(band_counts.items(), key=lambda item: int(item[0]))),
        "grayscale_content_count": grayscale_content_count,
        "color_content_count": valid_count - grayscale_content_count,
        "invalid_files": unreadable_count,
        "duplicate_count": duplicate_count,
        "near_uniform_count": near_uniform_count,
        "resolution_buckets": dict(resolutions),
        "width_distribution": dict(sorted(width_buckets.items())),
        "height_distribution": dict(sorted(height_buckets.items())),
        "file_size_distribution": dict(sorted(file_size_buckets.items())),
        "aspect_ratio_buckets": dict(aspects),
        "dimensions": {
            "width_min": width_min,
            "width_mean": round(width_sum / valid_count, 2),
            "width_median": round(statistics.median(width_samples), 2),
            "width_max": width_max,
            "height_min": height_min,
            "height_mean": round(height_sum / valid_count, 2),
            "height_median": round(statistics.median(height_samples), 2),
            "height_max": height_max,
            "resolution_pixels_min": resolution_pixel_min,
            "resolution_pixels_mean": round(resolution_pixel_sum / valid_count, 2),
            "resolution_pixels_median": round(statistics.median(resolution_pixel_samples), 2),
            "resolution_pixels_max": resolution_pixel_max,
            "median_is_sampled": valid_count > len(statistics_samples),
            "statistics_sample_size": len(statistics_samples),
        },
        "aspect_ratio": {
            "min": round(ratio_min, 4),
            "mean": round(ratio_sum / valid_count, 4),
            "median": round(statistics.median(ratio_samples), 4),
            "max": round(ratio_max, 4),
            "median_is_sampled": valid_count > len(statistics_samples),
        },
        "visual_characteristics": {
            "mean_brightness_0_255": round(brightness_sum / valid_count, 2),
            "brightness_min_0_255": round(float(brightness_min), 2),
            "brightness_median_0_255": round(statistics.median(brightness_samples), 2),
            "brightness_max_0_255": round(brightness_max, 2),
            "mean_contrast_luminance_std_0_255": round(contrast_sum / valid_count, 2),
            "contrast_min_luminance_std_0_255": round(float(contrast_min), 2),
            "contrast_median_luminance_std_0_255": round(statistics.median(contrast_samples), 2),
            "contrast_max_luminance_std_0_255": round(contrast_max, 2),
            "mean_laplacian_variance_64px": round(sharpness_sum / valid_count, 3),
            "laplacian_variance_min_64px": round(float(sharpness_min), 3),
            "laplacian_variance_median_64px": round(statistics.median(sharpness_samples), 3),
            "laplacian_variance_max_64px": round(sharpness_max, 3),
            "statistics_sample_size": len(statistics_samples),
            "near_uniform_images": near_uniform_count,
            "grayscale_content_count": grayscale_content_count,
            "grayscale_content_rule": "Konten disebut grayscale bila setidaknya 99% piksel thumbnail 64x64 memiliki rentang kanal RGB <= 2.",
            "sharpness_method": "Varians Laplacian 4-neighbor pada luminans thumbnail maksimal 64x64; nilai bergantung pada konten dan resize, tanpa ambang blur universal.",
            "brightness_contrast_method": "Rata-rata dan simpangan baku luminans pada thumbnail maksimal 64x64, skala 0-255.",
            "noise_estimation": "Tidak dihitung: estimasi noise umum memerlukan asumsi sensor/jenis gambar atau flat-region yang tervalidasi.",
            "note": "Brightness, contrast, dan sharpness merupakan indikator thumbnail 64×64; bukan penilaian kualitas visual menyeluruh.",
        },
        "statistics_method": {
            "exact_count": valid_count,
            "median_sample_size": len(statistics_samples),
            "median_is_sampled": valid_count > len(statistics_samples),
            "median_description": "Median exact untuk dataset sampai 20.000 gambar; di atasnya memakai reservoir deterministik maksimal 20.000 gambar. Mean, min, max, dan histogram dihitung streaming dari semua file valid.",
        },
        "sample_references": samples,
        "error_log": errors,
        "analysis_notes": [
            "Duplikat diperkirakan dari kesamaan 64-bit difference hash; hasil perlu ditinjau sebelum menghapus file.",
            "Distribusi label hanya menggunakan folder atau metadata CSV yang terdeteksi.",
            "Thumbnail dimuat terpisah per halaman agar seluruh gambar tidak disimpan di RAM.",
        ],
    }
    _emit_progress(progress, 95, "Menyusun ringkasan gambar")
    return result


def _load_cv2():
    try:
        import cv2
    except ImportError as error:
        raise VisionDependencyError(
            "Analisis video memerlukan OpenCV. Pasang dependency desktop dengan `python -m pip install opencv-python-headless`."
        ) from error
    return cv2


def _video_temp_path(archive: zipfile.ZipFile, ref: _MediaRef, cancel_event=None) -> str:
    if ref.member is None:
        return ref.name
    if ref.size_bytes > MAX_ZIP_MEMBER_BYTES:
        raise ValueError("Ukuran file video di dalam ZIP melewati batas 16 GiB.")
    temporary_directory = tempfile.gettempdir()
    free_bytes = shutil.disk_usage(temporary_directory).free
    if ref.size_bytes + 32 * 1024 * 1024 > free_bytes:
        raise OSError("Ruang disk sementara tidak cukup untuk membuka video dari ZIP.")
    suffix = Path(ref.relative_path).suffix or ".video"
    handle = tempfile.NamedTemporaryFile(prefix="dora_vision_", suffix=suffix, delete=False)
    temp_path = handle.name
    try:
        with handle, archive.open(ref.member, "r") as source:
            while True:
                _check_cancel(cancel_event)
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        return temp_path
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _frame_hash(frame, cv2) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pixels = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).reshape(-1)
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def _sample_video(cap, frame_count: int, cv2, cancel_event=None, sample_count: int = 4):
    if frame_count > 0:
        target_count = min(sample_count, frame_count)
        denominator = max(target_count - 1, 1)
        indices = sorted({int(round(index * (frame_count - 1) / denominator)) for index in range(target_count)})
    else:
        indices = [0]
    frames = []
    hashes = []
    for frame_index in indices:
        _check_cancel(cancel_event)
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        hashes.append(_frame_hash(frame, cv2))
        frames.append(frame_index)
    return frames, tuple(hashes)


def _capture_number(capture, property_id) -> float:
    try:
        value = float(capture.get(property_id))
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


def analyze_video_dataset(
    path: str | Path,
    *,
    progress: ProgressCallback | None = None,
    cancel_event=None,
    frame_sample_count: int = 4,
) -> dict[str, Any]:
    """Read bounded metadata and a few frames from each video using OpenCV."""
    cv2 = _load_cv2()
    frame_sample_count = max(1, min(8, int(frame_sample_count)))
    source, source_type = _source_kind(path)
    errors: list[dict[str, str]] = []
    samples: list[dict[str, Any]] = []
    rng = random.Random(2026)
    labels: Counter[str] = Counter()
    metadata_label_count = folder_label_count = 0
    frame_count_buckets: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    fingerprints: set[tuple[int, ...]] = set()
    split_fingerprints: dict[tuple[int, ...], set[str]] = {}
    duplicate_count = invalid_count = invalid_metadata_count = 0
    total_size_bytes = 0
    valid_count = frame_samples = 0
    duration_sum = fps_sum = 0.0
    duration_min = fps_min = None
    duration_max = fps_max = 0.0
    width_min = height_min = None
    width_max = height_max = 0

    archive = zipfile.ZipFile(source, "r") if source_type == "zip" else None
    try:
        infos = _zip_media_infos(archive, VIDEO_EXTENSIONS) if archive is not None else None
        if infos is not None:
            by_path, by_name, label_column, metadata_files = _find_archive_metadata(archive)
            refs = _iter_archive_media(infos)
            total_candidates = len(infos)
            total_size_bytes = source.stat().st_size
        else:
            by_path, by_name, label_column, metadata_files = _find_folder_metadata(source)
            total_candidates, total_size_bytes = _measure_folder_media(source, VIDEO_EXTENSIONS, cancel_event)
            refs = _iter_folder_media(source, VIDEO_EXTENSIONS, cancel_event)
            for metadata_name in metadata_files:
                try:
                    total_size_bytes += (source / metadata_name).stat().st_size
                except OSError:
                    pass
        if total_candidates == 0:
            raise ValueError("Tidak ditemukan file video yang didukung pada dataset.")

        _emit_progress(progress, 2, "Membaca metadata video")
        for processed, ref in enumerate(refs, start=1):
            _check_cancel(cancel_event)
            metadata_label = _metadata_label(ref, by_path, by_name)
            label = metadata_label or _folder_label(ref.relative_path)
            temp_path = None
            capture = None
            try:
                if ref.size_bytes <= 0:
                    raise ValueError("File video kosong.")
                if ref.member is not None:
                    if archive is None:
                        raise RuntimeError("Arsip ZIP tidak tersedia.")
                    temp_path = _video_temp_path(archive, ref, cancel_event)
                    video_path = temp_path
                else:
                    video_path = ref.name
                capture = cv2.VideoCapture(video_path)
                if not capture.isOpened():
                    raise ValueError("OpenCV tidak dapat membuka video.")
                reported_frames = int(round(_capture_number(capture, cv2.CAP_PROP_FRAME_COUNT)))
                fps = _capture_number(capture, cv2.CAP_PROP_FPS)
                width = int(round(_capture_number(capture, cv2.CAP_PROP_FRAME_WIDTH)))
                height = int(round(_capture_number(capture, cv2.CAP_PROP_FRAME_HEIGHT)))
                decoded_frames, signature = _sample_video(
                    capture, reported_frames, cv2, cancel_event, sample_count=frame_sample_count
                )
                if not decoded_frames:
                    raise ValueError("Tidak ada frame yang dapat dibaca dari video.")
                valid_count += 1
                duration = reported_frames / fps if fps > 0 and reported_frames > 0 else None
                metadata_invalid = fps <= 0 or width <= 0 or height <= 0 or reported_frames <= 0
                invalid_metadata_count += int(metadata_invalid)
                if label:
                    labels[label] += 1
                    if metadata_label:
                        metadata_label_count += 1
                    else:
                        folder_label_count += 1
                formats[Path(ref.relative_path).suffix.lstrip(".").upper()] += 1
                if reported_frames <= 0:
                    frame_count_buckets["unknown"] += 1
                elif reported_frames < 100:
                    frame_count_buckets["under_100"] += 1
                elif reported_frames < 500:
                    frame_count_buckets["100_to_499"] += 1
                elif reported_frames < 2_000:
                    frame_count_buckets["500_to_1999"] += 1
                else:
                    frame_count_buckets["2000_or_more"] += 1
                if width > 0 and height > 0:
                    resolutions[_resolution_bucket(width, height)] += 1
                    width_min = width if width_min is None else min(width_min, width)
                    height_min = height if height_min is None else min(height_min, height)
                    width_max = max(width_max, width)
                    height_max = max(height_max, height)
                if fps > 0:
                    fps_sum += fps
                    fps_min = fps if fps_min is None else min(fps_min, fps)
                    fps_max = max(fps_max, fps)
                if duration is not None:
                    duration_sum += duration
                    duration_min = duration if duration_min is None else min(duration_min, duration)
                    duration_max = max(duration_max, duration)
                duplicate_count += int(bool(signature) and signature in fingerprints)
                if signature:
                    fingerprints.add(signature)
                    split = _split_name(ref.relative_path)
                    if split:
                        split_fingerprints.setdefault(signature, set()).add(split)
                for frame_index in decoded_frames:
                    frame_samples += 1
                    _reservoir_add(
                        samples,
                        {"name": ref.relative_path, "path": ref.name, "size_bytes": ref.size_bytes, "member": ref.member, "label": label, "kind": "video", "frame_index": frame_index},
                        frame_samples,
                        rng,
                    )
            except AnalysisCancelledError:
                raise
            except Exception as error:
                invalid_count += 1
                _record_error(errors, ref, error)
            finally:
                if capture is not None:
                    capture.release()
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
            if processed == total_candidates or processed % 5 == 0:
                percent = 5 + round(85 * processed / max(total_candidates, 1))
                _emit_progress(progress, percent, f"Memeriksa video {processed:,} dari {total_candidates:,}")
        _check_cancel(cancel_event)
    finally:
        if archive is not None:
            archive.close()

    if valid_count == 0:
        raise ValueError("Tidak ada video yang dapat dibaca. Periksa codec dan isi dataset.")
    ordered_labels = dict(sorted(labels.items(), key=lambda item: item[0].casefold()))
    class_sizes = list(ordered_labels.values())
    imbalance_ratio = max(class_sizes) / min(class_sizes) if class_sizes else None
    result = {
        "dataset_type": "video",
        "source_path": str(source),
        "source_kind": source_type,
        "total_files": total_candidates,
        "total_samples": valid_count,
        "class_count": len(ordered_labels),
        "has_labels": bool(ordered_labels),
        "label_source": _label_source(metadata_label_count, folder_label_count),
        "label_column": label_column,
        "metadata_files": metadata_files,
        "class_distribution": ordered_labels,
        "unlabeled_count": max(0, valid_count - sum(class_sizes)),
        "class_imbalance_ratio": imbalance_ratio,
        "source_size_bytes": total_size_bytes,
        "formats": dict(sorted(formats.items())),
        "invalid_files": invalid_count,
        "invalid_metadata_count": invalid_metadata_count,
        "possible_duplicate_count": duplicate_count,
        "frame_samples_read": frame_samples,
        "frame_count_buckets": dict(frame_count_buckets),
        "possible_split_leakage_signatures": sum(
            1 for splits in split_fingerprints.values() if len(splits) > 1
        ),
        "resolution_buckets": dict(resolutions),
        "dimensions": {
            "width_min": width_min,
            "width_max": width_max,
            "height_min": height_min,
            "height_max": height_max,
        },
        "fps": {
            "mean": round(fps_sum / valid_count, 3),
            "min": round(fps_min, 3) if fps_min is not None else None,
            "max": round(fps_max, 3) if fps_max else None,
        },
        "duration_seconds": {
            "mean": round(duration_sum / valid_count, 3),
            "min": round(duration_min, 3) if duration_min is not None else None,
            "max": round(duration_max, 3) if duration_max else None,
        },
        "sample_references": samples,
        "error_log": errors,
        "analysis_notes": [
            "Jumlah frame berasal dari metadata codec dan dapat bersifat perkiraan.",
            "Duplikasi memakai signature perceptual dari maksimal empat frame; ini hanya indikasi dan perlu ditinjau.",
            "Hanya beberapa frame per video yang dibaca untuk menjaga penggunaan memori dan waktu analisis.",
            f"Sampling dibatasi maksimal {frame_sample_count} frame per video.",
        ],
    }
    _emit_progress(progress, 95, "Menyusun ringkasan video")
    return result


def _render_image_thumbnail(source: Path, reference: dict[str, Any], archive: zipfile.ZipFile | None) -> bytes:
    if reference.get("member"):
        if archive is None:
            raise RuntimeError("Arsip ZIP tidak tersedia.")
        stream = archive.open(reference["member"], "r")
    else:
        file_path = Path(reference.get("path") or source / reference["name"])
        stream = open(file_path, "rb")
    with stream, Image.open(stream) as image:
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise ValueError(f"Dimensi tidak aman untuk pratinjau: {width} × {height}.")
        image.thumbnail((192, 144), Image.Resampling.LANCZOS)
        rgb = image.convert("RGB")
        output = io.BytesIO()
        rgb.save(output, format="JPEG", quality=82, optimize=True)
        return output.getvalue()


def _render_video_thumbnail(source: Path, reference: dict[str, Any], archive: zipfile.ZipFile | None, cancel_event=None) -> bytes:
    cv2 = _load_cv2()
    temp_path = None
    if reference.get("member"):
        if archive is None:
            raise RuntimeError("Arsip ZIP tidak tersedia.")
        ref = _MediaRef(
            reference["member"],
            int(reference.get("size_bytes", 0)),
            reference["name"],
            reference["member"],
        )
        temp_path = _video_temp_path(archive, ref, cancel_event)
        video_path = temp_path
    else:
        video_path = str(reference.get("path") or source / reference["name"])
    capture = None
    try:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError("Video tidak dapat dibuka untuk pratinjau.")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(reference.get("frame_index", 0)))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError("Frame pratinjau tidak tersedia.")
        height, width = frame.shape[:2]
        scale = min(192 / max(width, 1), 144 / max(height, 1), 1.0)
        resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise ValueError("Frame tidak dapat dikodekan untuk pratinjau.")
        return encoded.tobytes()
    finally:
        if capture is not None:
            capture.release()
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def create_preview_thumbnails(
    path: str | Path,
    dataset_type: str,
    references: list[dict[str, Any]],
    *,
    cancel_event=None,
) -> list[dict[str, Any]]:
    """Decode only the requested preview page and return bounded JPEG bytes."""
    source, source_type = _source_kind(path)
    archive = zipfile.ZipFile(source, "r") if source_type == "zip" else None
    output = []
    try:
        for index, reference in enumerate(references):
            _check_cancel(cancel_event)
            try:
                if dataset_type == "image":
                    data = _render_image_thumbnail(source, reference, archive)
                elif dataset_type == "video":
                    data = _render_video_thumbnail(source, reference, archive, cancel_event)
                else:
                    raise ValueError(f"Jenis dataset tidak dikenal: {dataset_type}")
                output.append({"reference": reference, "data": data, "error": None})
            except AnalysisCancelledError:
                raise
            except Exception as error:
                output.append({"reference": reference, "data": None, "error": str(error)[:200]})
    finally:
        if archive is not None:
            archive.close()
    return output
