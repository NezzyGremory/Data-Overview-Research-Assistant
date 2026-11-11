"""CPU-friendly pretrained image recognition and a small supervised ML baseline."""

from __future__ import annotations

import hashlib
import io
import math
import random
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

from app.vision.analyzer import (
    AnalysisCancelledError,
    IMAGE_EXTENSIONS,
    MAX_IMAGE_PIXELS,
    _MediaRef,
    _find_archive_metadata,
    _find_folder_metadata,
    _folder_label,
    _iter_archive_media,
    _iter_folder_media,
    _metadata_label,
    _open_image,
    _source_kind,
    _zip_media_infos,
)


MODEL_NAME = "MobileNetV2 ImageNet-1K"
MODEL_VERSION = "mobilenetv2-12.onnx"
MODEL_URL = (
    "https://media.githubusercontent.com/media/onnx/models/main/validated/"
    "vision/classification/mobilenet/model/mobilenetv2-12.onnx"
)
LABELS_NAME = "imagenet-synset.txt"
LABELS_URL = "https://raw.githubusercontent.com/onnx/models/main/validated/vision/classification/synset.txt"
MODEL_SHA256 = "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5"
MODEL_BYTES = 13_300_000
INPUT_SIZE = 224
TOP_K = 5
MAX_TRAINING_FEATURES = 5_000
MAX_TRAINING_CLASSES = 50
MIN_TRAINING_SAMPLES_PER_CLASS = 5
ProgressCallback = Callable[[int, str], None]


def model_bundle_paths(cache_dir: str | Path) -> tuple[Path, Path]:
    target = Path(cache_dir)
    return target / MODEL_VERSION, target / LABELS_NAME


def model_bundle_available(cache_dir: str | Path) -> bool:
    model_path, labels_path = model_bundle_paths(cache_dir)
    if not model_path.is_file() or not labels_path.is_file():
        return False
    if MODEL_SHA256:
        return _sha256(model_path) == MODEL_SHA256
    return model_path.stat().st_size >= MODEL_BYTES and len(_read_labels(labels_path)) == 1000


def _check_cancel(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelledError("Pengenalan gambar dibatalkan.")


def _emit_progress(progress: ProgressCallback | None, percent: int, message: str) -> None:
    if progress:
        progress(max(0, min(100, int(percent))), str(message))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_labels(path: Path) -> list[str]:
    labels = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        value = line.strip()
        if not value:
            continue
        parts = value.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            labels.append(parts[1].strip())
        elif len(parts) == 2 and len(parts[0]) == 9 and parts[0].startswith("n") and parts[0][1:].isdigit():
            labels.append(parts[1].strip())
        else:
            labels.append(value)
    return labels


def _download_file(
    url: str,
    destination: Path,
    *,
    progress: ProgressCallback | None,
    cancel_event,
    progress_start: int,
    progress_span: int,
    description: str,
    max_bytes: int,
) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DORA-Vision/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as target:
            total = int(response.headers.get("Content-Length") or 0)
            if total > max_bytes:
                raise ValueError(f"Ukuran {description} melebihi batas yang diizinkan.")
            received = 0
            while True:
                _check_cancel(cancel_event)
                block = response.read(1024 * 1024)
                if not block:
                    break
                received += len(block)
                if received > max_bytes:
                    raise ValueError(f"Ukuran {description} melebihi batas yang diizinkan.")
                target.write(block)
                if total:
                    fraction = min(1.0, received / total)
                    percent = progress_start + int(progress_span * fraction)
                    _emit_progress(progress, percent, f"Mengunduh {description}: {received / (1024 * 1024):.1f} MiB")
        _check_cancel(cancel_event)
        if temporary.stat().st_size == 0:
            raise ValueError(f"Berkas {description} kosong.")
        temporary.replace(destination)
    except AnalysisCancelledError:
        temporary.unlink(missing_ok=True)
        raise
    except (OSError, urllib.error.URLError, TimeoutError, ValueError):
        temporary.unlink(missing_ok=True)
        raise


def ensure_model_bundle(
    cache_dir: str | Path,
    *,
    progress: ProgressCallback | None = None,
    cancel_event=None,
) -> tuple[Path, Path]:
    """Download the published ONNX model and ImageNet labels into the local model cache."""
    model_path, labels_path = model_bundle_paths(cache_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file():
        _download_file(
            MODEL_URL,
            model_path,
            progress=progress,
            cancel_event=cancel_event,
            progress_start=0,
            progress_span=65,
            description="model MobileNetV2 (sekitar 13,3 MiB)",
            max_bytes=25 * 1024 * 1024,
        )
    if MODEL_SHA256 and _sha256(model_path) != MODEL_SHA256:
        model_path.unlink(missing_ok=True)
        raise ValueError("Checksum model MobileNetV2 tidak sesuai; model dihapus dan tidak digunakan.")
    if model_path.stat().st_size < MODEL_BYTES:
        model_path.unlink(missing_ok=True)
        raise ValueError("Unduhan model MobileNetV2 tidak lengkap.")

    if not labels_path.is_file() or len(_read_labels(labels_path)) != 1000:
        _download_file(
            LABELS_URL,
            labels_path,
            progress=progress,
            cancel_event=cancel_event,
            progress_start=65,
            progress_span=5,
            description="label ImageNet",
            max_bytes=1024 * 1024,
        )
    labels = _read_labels(labels_path)
    if len(labels) != 1000:
        labels_path.unlink(missing_ok=True)
        raise ValueError(f"Daftar label ImageNet tidak lengkap: ditemukan {len(labels)} dari 1000.")
    _emit_progress(progress, 70, "Model siap; mulai pengenalan lokal")
    return model_path, labels_path


def _make_blob(image: Image.Image) -> np.ndarray:
    width, height = image.size
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"Dimensi gambar tidak aman untuk inferensi: {width} x {height}.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = 256.0 / min(width, height)
    resized = image.resize(
        (max(INPUT_SIZE, round(width * scale)), max(INPUT_SIZE, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    left = (resized.width - INPUT_SIZE) // 2
    top = (resized.height - INPUT_SIZE) // 2
    cropped = resized.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    pixels = np.asarray(cropped, dtype=np.float32) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    normalized = (pixels - mean) / std
    return np.transpose(normalized, (2, 0, 1))[None, ...]


def _softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    shifted = values - np.max(values)
    exponents = np.exp(shifted)
    denominator = float(exponents.sum())
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Model menghasilkan skor yang tidak dapat dinormalisasi.")
    return exponents / denominator


def _reference_key(path: str | Path | None, member: str | None) -> str:
    if member:
        return f"zip:{member.casefold()}"
    try:
        return f"file:{Path(path or '').resolve()}".casefold()
    except (OSError, RuntimeError):
        return f"file:{path}".casefold()


def _bounded_training_features(
    counts: dict[str, int],
) -> tuple[bool, int, str | None]:
    labels = list(counts)
    if len(labels) < 2:
        return False, 0, "Training ML memerlukan setidaknya dua label kelas yang terpetakan."
    if len(labels) > MAX_TRAINING_CLASSES:
        return False, 0, f"Training ML dilewati karena jumlah kelas ({len(labels)}) melewati batas aman {MAX_TRAINING_CLASSES}."
    minimum = min((int(counts[label]) for label in labels), default=0)
    if minimum < MIN_TRAINING_SAMPLES_PER_CLASS:
        return False, 0, (
            "Training ML dilewati: setiap kelas memerlukan minimal lima gambar berlabel "
            "agar split train/validasi berstrata dapat dibuat."
        )
    return True, max(1, min(1000, MAX_TRAINING_FEATURES // len(labels))), None


def _train_custom_classifier(
    samples: dict[str, list[tuple[np.ndarray, str]]],
    *,
    cancel_event=None,
    model_directory: str | Path | None = None,
    dataset_path: str | Path | None = None,
    label_source: str | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    _check_cancel(cancel_event)
    from sklearn.base import clone
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    features = []
    target = []
    for label, items in samples.items():
        for vector, _name in items:
            features.append(vector)
            target.append(label)
    matrix = np.asarray(features, dtype=np.float32)
    labels = np.asarray(target, dtype=object)
    indexes = np.arange(len(labels))
    train_indexes, test_indexes = train_test_split(
        indexes,
        test_size=0.2,
        random_state=2026,
        stratify=labels,
    )

    candidates = [
        (
            "SGD Logistic Classifier",
            make_pipeline(
                StandardScaler(),
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    max_iter=500,
                    tol=1e-3,
                    random_state=2026,
                    n_jobs=1,
                ),
            ),
        ),
        (
            "Linear SVM",
            make_pipeline(
                StandardScaler(),
                LinearSVC(
                    class_weight="balanced",
                    random_state=2026,
                    dual="auto",
                    max_iter=3000,
                ),
            ),
        ),
        (
            "Extra Trees",
            ExtraTreesClassifier(
                n_estimators=80,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=1,
                random_state=2026,
            ),
        ),
    ]
    training_matrix = matrix[train_indexes]
    training_labels = labels[train_indexes]
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
    comparison = []
    total_folds = len(candidates) * folds.get_n_splits()
    completed_folds = 0
    for model_name, candidate in candidates:
        fold_scores = []
        for fold_train, fold_validation in folds.split(training_matrix, training_labels):
            _check_cancel(cancel_event)
            fitted = clone(candidate)
            fitted.fit(training_matrix[fold_train], training_labels[fold_train])
            fold_prediction = fitted.predict(training_matrix[fold_validation])
            fold_scores.append(
                float(f1_score(training_labels[fold_validation], fold_prediction, average="macro", zero_division=0))
            )
            completed_folds += 1
            _emit_progress(
                progress,
                95 + int(4 * completed_folds / total_folds),
                f"Membandingkan model: {model_name} ({completed_folds}/{total_folds})",
            )
        comparison.append(
            {
                "model": model_name,
                "mean_macro_f1": float(np.mean(fold_scores)),
                "std_macro_f1": float(np.std(fold_scores)),
                "fold_scores": fold_scores,
            }
        )

    comparison.sort(key=lambda item: (-item["mean_macro_f1"], item["model"]))
    selected_name = comparison[0]["model"]
    selected_template = next(candidate for name, candidate in candidates if name == selected_name)
    estimator = clone(selected_template)
    estimator.fit(training_matrix, training_labels)
    _check_cancel(cancel_event)
    predicted = estimator.predict(matrix[test_indexes])
    actual = labels[test_indexes]
    class_names = [str(label) for label in np.unique(labels)]
    precision, recall, class_f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=class_names,
        zero_division=0,
    )
    per_class_metrics = [
        {
            "class": class_name,
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1": float(class_score),
            "support": int(class_support),
        }
        for class_name, class_precision, class_recall, class_score, class_support in zip(
            class_names, precision, recall, class_f1, support
        )
    ]
    final_estimator = clone(selected_template)
    final_estimator.fit(matrix, labels)

    model_path = None
    artifact_note = None
    if model_directory is not None:
        try:
            import joblib

            signature = hashlib.sha256()
            signature.update(str(dataset_path or "").encode("utf-8", errors="replace"))
            signature.update(str(label_source or "").encode("utf-8", errors="replace"))
            signature.update("\x1f".join(str(value) for value in labels).encode("utf-8"))
            signature.update(matrix.tobytes(order="C"))
            artifact_directory = Path(model_directory) / "trained_image_models"
            artifact_directory.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_directory / f"{signature.hexdigest()[:24]}.joblib"
            temporary_path = artifact_path.with_suffix(".joblib.part")
            joblib.dump(
                {
                    "estimator": final_estimator,
                    "model": selected_name,
                    "feature_source": MODEL_NAME,
                    "feature_count": int(matrix.shape[1]),
                    "classes": class_names,
                    "label_source": label_source,
                    "dataset_path": str(dataset_path or ""),
                    "training_samples": int(len(labels)),
                    "random_state": 2026,
                },
                temporary_path,
                compress=3,
            )
            temporary_path.replace(artifact_path)
            model_path = str(artifact_path)
        except (OSError, ValueError) as error:
            if "temporary_path" in locals():
                temporary_path.unlink(missing_ok=True)
            artifact_note = f"Model berhasil dievaluasi, tetapi berkas model tidak tersimpan: {error}"

    return {
        "status": "evaluated",
        "method": "model selection on frozen MobileNetV2 ImageNet scores",
        "selected_model": selected_name,
        "candidate_models": comparison,
        "split": "80/20 stratified holdout",
        "feature_count": int(matrix.shape[1]),
        "training_samples": int(len(train_indexes)),
        "validation_samples": int(len(test_indexes)),
        "class_count": int(len(np.unique(labels))),
        "classes": class_names,
        "class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=class_names).tolist(),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "model_path": model_path,
        "artifact_note": artifact_note,
        "note": (
            "Seleksi kandidat memakai macro-F1 3-fold pada bagian data latih; metrik akhir memakai holdout berstrata 20% yang terpisah. "
            "Dataset kecil atau banyak duplikat dapat menghasilkan estimasi yang tidak stabil atau terlalu optimistis. Label folder/CSV harus diperiksa sebagai ground truth. "
            "Bobot MobileNetV2 tetap dibekukan; hanya classifier ML terpilih yang dilatih. Model tersimpan memerlukan fitur skor dari versi MobileNetV2 yang sama."
        ),
    }


def recognize_image_dataset(
    path: str | Path,
    *,
    model_path: str | Path,
    labels_path: str | Path,
    model_directory: str | Path | None = None,
    sample_references: list[dict[str, Any]] | None = None,
    known_label_counts: dict[str, int] | None = None,
    label_source: str | None = None,
    progress: ProgressCallback | None = None,
    cancel_event=None,
) -> dict[str, Any]:
    """Run pretrained ImageNet inference for every image, retaining only bounded summaries."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV DNN tidak tersedia untuk menjalankan model vision.") from error

    source, source_type = _source_kind(path)
    imagenet_labels = _read_labels(Path(labels_path))
    if len(imagenet_labels) != 1000:
        raise ValueError("Model memerlukan tepat 1000 label ImageNet.")
    net = cv2.dnn.readNetFromONNX(str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # OpenCV DNN defaults to its CPU target; setting it explicitly emits an
    # unsupported-target warning in recent OpenCV graph-engine builds.
    try:
        cv2.setNumThreads(max(1, min(4, int(cv2.getNumberOfCPUs()))))
    except (AttributeError, TypeError, ValueError):
        pass

    archive = zipfile.ZipFile(source, "r") if source_type == "zip" else None
    if archive is not None:
        refs = _iter_archive_media(_zip_media_infos(archive, IMAGE_EXTENSIONS))
        total_files = sum(
            1 for info in archive.infolist()
            if not info.is_dir() and Path(info.filename).suffix.casefold() in IMAGE_EXTENSIONS
        )
        by_path, by_name, _label_column, _metadata_files = _find_archive_metadata(archive)
    else:
        refs = _iter_folder_media(source, IMAGE_EXTENSIONS, cancel_event)
        total_files = sum(
            1 for _ in _iter_folder_media(source, IMAGE_EXTENSIONS, cancel_event)
        )
        by_path, by_name, _label_column, _metadata_files = _find_folder_metadata(source)

    desired = {
        _reference_key(reference.get("path"), reference.get("member")): reference
        for reference in (sample_references or [])
    }
    sampled_predictions: dict[str, dict[str, Any]] = {}
    predicted_counts: Counter[str] = Counter()
    confidence_sum = 0.0
    processed = failed = seen = 0
    label_counts = known_label_counts or {}
    train_enabled, per_class_budget, train_note = _bounded_training_features(label_counts)
    training_rng = random.Random(20261110)
    training_seen: Counter[str] = Counter()
    training_samples: dict[str, list[tuple[np.ndarray, str]]] = {label: [] for label in label_counts} if train_enabled else {}

    try:
        if archive is None:
            iterator = ((ref, None) for ref in refs)
        else:
            iterator = ((ref, archive) for ref in refs)
        for ref, ref_archive in iterator:
            _check_cancel(cancel_event)
            seen += 1
            try:
                with _open_image(ref, ref_archive) as stream:
                    with Image.open(stream) as image:
                        blob = _make_blob(image)
            except (OSError, EOFError, SyntaxError, ValueError, zipfile.BadZipFile, Image.DecompressionBombError):
                failed += 1
                continue

            # Keep model/runtime errors visible. Only image decoding and per-file
            # input validation count as unreadable data; a broken model must fail.
            net.setInput(blob)
            scores = np.asarray(net.forward()).reshape(-1)
            if scores.size != 1000:
                raise ValueError(f"Output model berisi {scores.size} kelas, bukan 1000.")
            probabilities = _softmax(scores)
            top_indices = np.argsort(probabilities)[-TOP_K:][::-1]
            top_predictions = [
                {"label": imagenet_labels[int(index)], "score": float(probabilities[index])}
                for index in top_indices
            ]
            top_label = top_predictions[0]["label"]
            predicted_counts[top_label] += 1
            confidence_sum += top_predictions[0]["score"]
            processed += 1

            sample_key = _reference_key(ref.name, ref.member)
            if sample_key in desired:
                reference = desired[sample_key]
                sampled_predictions[sample_key] = {
                    "name": reference.get("name") or ref.relative_path,
                    "path": reference.get("path") or ref.name,
                    "member": ref.member,
                    "folder_csv_label": reference.get("label"),
                    "predictions": top_predictions,
                }

            target_label = _metadata_label(ref, by_path, by_name) or _folder_label(ref.relative_path)
            if train_enabled and target_label in training_samples:
                budget = per_class_budget
                current = training_samples[target_label]
                training_seen[target_label] += 1
                item = (scores.astype(np.float32, copy=True), ref.relative_path)
                if len(current) < budget:
                    current.append(item)
                else:
                    replacement = training_rng.randrange(training_seen[target_label])
                    if replacement < budget:
                        current[replacement] = item
            if seen % 10 == 0 or seen == total_files:
                fraction = seen / max(1, total_files)
                percent = 70 + int(24 * fraction)
                _emit_progress(progress, percent, f"Mengenali gambar {seen:,} dari {total_files:,}")

        if processed == 0:
            raise ValueError("Tidak ada gambar valid yang dapat dikenali oleh model.")

        custom_ml = {"status": "not_trained", "note": train_note}
        if train_enabled:
            if any(len(training_samples[label]) < MIN_TRAINING_SAMPLES_PER_CLASS for label in training_samples):
                custom_ml = {
                    "status": "not_trained",
                    "note": "Training ML dilewati karena sebagian kelas tidak memiliki cukup gambar valid untuk split holdout.",
                }
            else:
                _emit_progress(progress, 96, "Menguji klasifier ML pada split holdout berstrata")
                try:
                    custom_ml = _train_custom_classifier(
                        training_samples,
                        cancel_event=cancel_event,
                        model_directory=model_directory,
                        dataset_path=source,
                        label_source=label_source,
                        progress=progress,
                    )
                except ValueError as error:
                    custom_ml = {"status": "not_trained", "note": f"Training ML dilewati: {error}"}
        if custom_ml.get("status") == "evaluated":
            custom_ml["available_labeled_images"] = int(sum(training_seen.values()))
            custom_ml["sampled_labeled_images"] = int(sum(len(items) for items in training_samples.values()))

        _emit_progress(progress, 100, "Pengenalan gambar selesai")
        return {
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "model_task": "single-label ImageNet-1K image classification",
            "dataset_path": str(source),
            "label_source": label_source,
            "total_files": total_files,
            "classified_count": processed,
            "unreadable_count": failed,
            "mean_top1_score": confidence_sum / processed,
            "predicted_class_distribution": dict(predicted_counts.most_common()),
            "sample_predictions": list(sampled_predictions.values()),
            "custom_ml": custom_ml,
            "method_note": (
                "MobileNetV2 ImageNet-1K memprediksi satu kelas umum per gambar. Skor softmax bukan confidence yang terkalibrasi, "
                "dan kelas ImageNet dapat tidak sesuai dengan domain dataset. Pengenalan ini bukan object detection atau segmentation."
            ),
            "license": "Apache-2.0 (ONNX Model Zoo MobileNetV2)",
        }
    finally:
        if archive is not None:
            archive.close()
