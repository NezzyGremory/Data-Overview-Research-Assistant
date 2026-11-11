from __future__ import annotations

import csv
import io
import shutil
import threading
import time
import zipfile

import pytest
from PIL import Image, ImageDraw

from app.storage import Database, ProjectRepository
from app.vision.analyzer import (
    AnalysisCancelledError,
    VisionDependencyError,
    analyze_image_dataset,
    analyze_video_dataset,
    create_preview_thumbnails,
)
from app.vision.recommendations import recommend_approaches
from app.vision.readiness import assess_image_readiness, assess_image_tasks
from app.vision.storage import VisionAnalysisRepository


def _make_image(path, color=(32, 100, 180), size=(64, 48), pattern=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    if pattern:
        draw = ImageDraw.Draw(image)
        draw.rectangle((7, 6, 25, 30), fill=(220, 50, 70))
        draw.line((0, 0, size[0] - 1, size[1] - 1), fill=(255, 230, 20), width=3)
    image.save(path)
    return path


def _make_mp4(path, frame_count=8):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build tidak menyediakan encoder MP4 mp4v.")
    for index in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = 30 + index * 10
        cv2.rectangle(frame, (4 + index, 8), (24 + index, 30), (20, 200, 120), -1)
        writer.write(frame)
    writer.release()
    return path


def test_image_analysis_reads_jpg_png_webp_tiff_classes_and_corrupt_files(tmp_path):
    root = tmp_path / "images"
    first = _make_image(root / "cats" / "one.jpg")
    shutil.copy2(first, root / "cats" / "same.jpg")
    _make_image(root / "dogs" / "two.png", color=(20, 160, 90))
    _make_image(root / "dogs" / "three.webp", color=(130, 40, 140))
    _make_image(root / "birds" / "four.tiff", color=(150, 120, 40))
    (root / "birds" / "broken.png").write_bytes(b"not an image")

    result = analyze_image_dataset(root)

    assert result["total_samples"] == 5
    assert result["class_count"] == 3
    assert result["class_distribution"] == {"birds": 1, "cats": 2, "dogs": 2}
    assert result["invalid_files"] == 1
    assert result["duplicate_count"] >= 1
    assert {"JPEG", "PNG", "WEBP", "TIFF"}.issubset(result["formats"])
    assert result["dimensions"]["width_min"] == 64
    assert result["sample_references"]


def test_image_analysis_reads_csv_labels_for_unstructured_folder(tmp_path):
    root = tmp_path / "flat"
    root.mkdir()
    _make_image(root / "one.png")
    _make_image(root / "two.jpg", color=(10, 200, 60))
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "label"])
        writer.writeheader()
        writer.writerow({"filename": "one.png", "label": "indoor"})
        writer.writerow({"filename": "two.jpg", "label": "outdoor"})

    result = analyze_image_dataset(root)

    assert result["label_source"] == "metadata_csv"
    assert result["class_distribution"] == {"indoor": 1, "outdoor": 1}
    assert result["metadata_files"] == ["metadata.csv"]


def test_image_overview_statistics_quality_and_class_imbalance(tmp_path):
    root = tmp_path / "mixed-images"
    class_a = root / "class-a"
    class_b = root / "class-b"
    class_a.mkdir(parents=True)
    class_b.mkdir(parents=True)
    grayscale = Image.new("L", (20, 10), 110)
    grayscale.save(class_a / "gray.png")
    for name, size in (("two.jpg", (32, 16)), ("three.jpg", (40, 20)), ("four.jpg", (64, 32))):
        _make_image(class_a / name, size=size)
    _make_image(class_b / "only.png", size=(16, 16))
    _make_image(root / "unlabeled.png", size=(15, 25))

    result = analyze_image_dataset(root)

    assert result["total_files"] == 6
    assert result["total_samples"] == 6
    assert result["processing_batch_size"] == 25
    assert result["labeled_count"] == 5
    assert result["unlabeled_count"] == 1
    assert result["class_count"] == 2
    assert result["class_distribution"] == {"class-a": 4, "class-b": 1}
    assert result["class_imbalance_ratio"] == pytest.approx(4.0)
    assert result["class_count_coefficient_of_variation"] == pytest.approx(0.6)
    assert result["label_source"] == "folder_structure"
    assert "sementara" in result["label_source_description"]
    assert result["dimensions"]["width_min"] == 15
    assert result["dimensions"]["width_max"] == 64
    assert result["dimensions"]["width_mean"] == pytest.approx(round((20 + 32 + 40 + 64 + 16 + 15) / 6, 2))
    assert result["dimensions"]["resolution_pixels_min"] == 20 * 10
    assert result["dimensions"]["resolution_pixels_max"] == 64 * 32
    assert result["file_size_bytes"]["min"] > 0
    assert result["file_size_bytes"]["max"] >= result["file_size_bytes"]["min"]
    assert result["channel_counts"] == {"1": 1, "3": 5}
    assert result["image_modes"]["L"] == 1
    assert result["grayscale_content_count"] >= 1
    assert "mean_laplacian_variance_64px" in result["visual_characteristics"]
    assert "brightness_contrast_method" in result["visual_characteristics"]
    assert "Tidak dihitung" in result["visual_characteristics"]["noise_estimation"]
    assert result["width_distribution"]
    assert result["height_distribution"]
    assert result["aspect_ratio_buckets"]
    assert result["file_size_distribution"]
    assert set(result["formats"]) == {"JPEG", "PNG"}


def test_large_image_statistics_use_bounded_deterministic_median_sample(monkeypatch, tmp_path):
    import app.vision.analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "MAX_STATISTICS_SAMPLE", 2)
    root = tmp_path / "many-images" / "class-a"
    sizes = [(12, 10), (24, 20), (36, 30), (48, 40), (60, 50)]
    for index, size in enumerate(sizes):
        _make_image(root / f"{index}.png", size=size)

    result = analyze_image_dataset(root.parent)

    assert result["total_samples"] == len(sizes)
    assert result["statistics_method"]["median_sample_size"] == 2
    assert result["statistics_method"]["median_is_sampled"] is True
    assert result["dimensions"]["width_min"] == 12
    assert result["dimensions"]["width_max"] == 60
    assert result["dimensions"]["width_mean"] == pytest.approx(36)
    assert result["file_size_bytes"]["median_is_sampled"] is True


def test_image_task_assessments_and_model_guidance_explain_prerequisites():
    analysis = {
        "dataset_type": "image",
        "total_samples": 40,
        "total_files": 40,
        "labeled_count": 40,
        "unlabeled_count": 0,
        "has_labels": True,
        "class_count": 2,
        "class_distribution": {"a": 20, "b": 20},
        "label_source_description": "Folder dipakai sebagai label sementara; validasi semantik.",
        "annotation_readiness": {"note": "Anotasi YOLO/COCO/mask tidak diperiksa."},
        "dimensions": {"width_max": 2048, "height_max": 1024},
    }
    tasks = {item["task"]: item for item in assess_image_tasks(analysis)}
    readiness = assess_image_readiness(analysis, "classification")
    classification = recommend_approaches(
        "image", analysis, "classification", hardware={"gpu_available": False}
    )
    clustering = recommend_approaches(
        "image", analysis, "image_clustering", hardware={"gpu_available": False}
    )
    detection = recommend_approaches(
        "image", analysis, "object_detection", hardware={"gpu_available": False}
    )
    segmentation = recommend_approaches(
        "image", analysis, "segmentation", hardware={"gpu_available": False}
    )
    low_ram = recommend_approaches(
        "image", analysis, "classification",
        hardware={"gpu_available": False, "ram_available_gb": 2.5},
    )

    assert "sementara" in tasks["Image Classification"]["evidence"]
    assert tasks["Object Detection"]["status"] == "Anotasi belum diverifikasi"
    assert "tidak diperiksa" in tasks["Object Detection"]["evidence"]
    assert tasks["Image Segmentation"]["status"] == "Anotasi belum diverifikasi"
    assert tasks["Image Similarity"]["status"] == "Dapat dieksplorasi"
    assert tasks["Image Clustering"]["status"] == "Dapat dieksplorasi"
    assert readiness["has_composite_score"] is False
    assert len(readiness["items"]) == 6
    assert "ResNet atau EfficientNet" in classification["primary"]["name"]
    assert any("MobileNet" in item["name"] for item in classification["alternatives"])
    assert "2,048" in classification["primary"]["computational_requirements"]
    assert "MiniBatchKMeans" in clustering["primary"]["name"]
    assert "Faster R-CNN" in detection["primary"]["name"]
    assert "U-Net" in segmentation["primary"]["name"]
    assert any("2,5 GiB" in warning for warning in low_ram["warnings"])


def test_image_classifier_preprocessing_and_softmax_are_well_formed():
    import numpy as np

    from app.vision.image_classifier import _make_blob, _softmax

    image = Image.new("RGB", (480, 320), (220, 90, 30))
    blob = _make_blob(image)
    probabilities = _softmax(np.array([0.0, 2.0, -1.0], dtype=np.float32))

    assert blob.shape == (1, 3, 224, 224)
    assert np.isfinite(blob).all()
    assert probabilities.sum() == pytest.approx(1.0)
    assert int(np.argmax(probabilities)) == 1


def test_image_classifier_runs_top5_over_folder_and_maps_sample_previews(monkeypatch, tmp_path):
    import cv2
    import numpy as np

    from app.vision.image_classifier import recognize_image_dataset

    root = tmp_path / "image-classification"
    _make_image(root / "cats" / "cat.jpg")
    _make_image(root / "dogs" / "dog.jpg", color=(40, 190, 80))
    analysis = analyze_image_dataset(root)
    labels_path = tmp_path / "synset.txt"
    labels_path.write_text(
        "\n".join(f"{index + 1} class-{index}" for index in range(1000)),
        encoding="utf-8",
    )

    class FakeNet:
        def setPreferableBackend(self, _backend):
            pass

        def setPreferableTarget(self, _target):
            pass

        def setInput(self, _blob):
            pass

        def forward(self):
            scores = np.zeros(1000, dtype=np.float32)
            scores[11] = 7.0
            scores[29] = 4.0
            return scores[None, :]

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda _path: FakeNet())
    result = recognize_image_dataset(
        root,
        model_path=tmp_path / "model.onnx",
        labels_path=labels_path,
        sample_references=analysis["sample_references"],
        known_label_counts=analysis["class_distribution"],
        label_source=analysis["label_source"],
    )

    assert result["model"] == "MobileNetV2 ImageNet-1K"
    assert result["classified_count"] == 2
    assert result["predicted_class_distribution"] == {"class-11": 2}
    assert len(result["sample_predictions"]) == 2
    assert all(item["predictions"][0]["label"] == "class-11" for item in result["sample_predictions"])
    assert result["custom_ml"]["status"] == "not_trained"


def test_image_classifier_reports_model_runtime_failure_instead_of_counting_bad_images(monkeypatch, tmp_path):
    import cv2

    from app.vision.image_classifier import recognize_image_dataset

    root = tmp_path / "runtime-error" / "class-a"
    _make_image(root / "image.png")
    labels_path = tmp_path / "synset.txt"
    labels_path.write_text("\n".join(f"class-{index}" for index in range(1000)), encoding="utf-8")

    class BrokenNet:
        def setPreferableBackend(self, _backend):
            pass

        def setPreferableTarget(self, _target):
            pass

        def setInput(self, _blob):
            pass

        def forward(self):
            raise RuntimeError("model runtime failure")

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda _path: BrokenNet())
    with pytest.raises(RuntimeError, match="model runtime failure"):
        recognize_image_dataset(
            root.parent,
            model_path=tmp_path / "model.onnx",
            labels_path=labels_path,
        )


def test_image_classifier_compares_models_and_evaluates_a_stratified_holdout(tmp_path):
    from pathlib import Path

    import numpy as np

    from app.vision.image_classifier import _train_custom_classifier

    samples = {
        "class-a": [(np.concatenate((np.array([-10.0]), np.zeros(7))).astype(np.float32), f"a-{i}") for i in range(10)],
        "class-b": [(np.concatenate((np.array([10.0]), np.zeros(7))).astype(np.float32), f"b-{i}") for i in range(10)],
    }
    result = _train_custom_classifier(
        samples,
        model_directory=tmp_path / "models",
        dataset_path=tmp_path / "dataset",
        label_source="folder",
    )

    assert result["status"] == "evaluated"
    assert result["split"] == "80/20 stratified holdout"
    assert result["selected_model"] in {"SGD Logistic Classifier", "Linear SVM", "Extra Trees"}
    assert {item["model"] for item in result["candidate_models"]} == {
        "SGD Logistic Classifier", "Linear SVM", "Extra Trees"
    }
    assert len(result["candidate_models"]) == 3
    assert result["training_samples"] == 16
    assert result["validation_samples"] == 4
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["weighted_f1"] == pytest.approx(1.0)
    assert {item["class"] for item in result["class_metrics"]} == {"class-a", "class-b"}
    assert all(item["f1"] == pytest.approx(1.0) for item in result["class_metrics"])
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["model_path"] and Path(result["model_path"]).is_file()

    import joblib
    bundle = joblib.load(result["model_path"])
    assert bundle["model"] == result["selected_model"]
    assert bundle["classes"] == ["class-a", "class-b"]


_QT_TEST_APP = None


def test_image_sample_gallery_keeps_pagination_and_bounded_thumbnail_cache():
    import io
    from PySide6.QtWidgets import QApplication

    from app.ui.vision_intelligence_page import _VisionTab

    class PageStub:
        def start_analysis(self, _tab):
            pass

        def start_recognition(self, _tab):
            pass

        def cancel_active_operation(self):
            pass

        def update_recommendations(self, _tab):
            pass

        def load_previews(self, _tab, _references):
            pass

    global _QT_TEST_APP
    _QT_TEST_APP = QApplication.instance() or QApplication([])
    tab = _VisionTab(PageStub(), "image")
    assert tab.view_stack.currentWidget() is tab.upload_page
    tab.show_results_page()
    assert tab.view_stack.currentWidget() is tab.results_page
    tab.show_upload_page()
    assert tab.view_stack.currentWidget() is tab.upload_page
    tab.sample_refs = [
        {"name": f"image-{index}.png", "path": f"/tmp/image-{index}.png", "kind": "image"}
        for index in range(25)
    ]
    tab.render_preview_page()
    assert len(tab.selected_references()) == tab.PAGE_SIZE == 12
    assert tab.next_button.isEnabled()
    tab._change_preview_page(1)
    assert len(tab.selected_references()) == 12
    tab._change_preview_page(1)
    assert len(tab.selected_references()) == 1
    assert not tab.next_button.isEnabled()

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (51, 102, 153)).save(buffer, format="JPEG")
    jpeg = buffer.getvalue()
    previews = [
        {"reference": reference, "data": jpeg}
        for reference in tab.sample_refs[:25]
    ]
    tab.add_previews(previews)
    assert len(tab._thumbnail_cache) == 24
    tab.close()


def test_image_analysis_zip_classes_corrupt_and_preview(tmp_path):
    root = tmp_path / "source"
    _make_image(root / "class-a" / "a.png")
    _make_image(root / "class-b" / "b.webp", color=(90, 50, 200))
    archive_path = tmp_path / "images.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for media in root.rglob("*"):
            if media.is_file():
                archive.write(media, media.relative_to(root).as_posix())
        archive.writestr("class-b/broken.jpg", b"invalid")

    result = analyze_image_dataset(archive_path)
    previews = create_preview_thumbnails(
        archive_path, "image", result["sample_references"][:1]
    )

    assert result["total_samples"] == 2
    assert result["class_distribution"] == {"class-a": 1, "class-b": 1}
    assert result["invalid_files"] == 1
    assert previews[0]["data"].startswith(b"\xff\xd8")


def test_image_analysis_empty_dataset_and_cancellation(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="Tidak ditemukan"):
        analyze_image_dataset(empty)

    _make_image(empty / "one.png")
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(AnalysisCancelledError):
        analyze_image_dataset(empty, cancel_event=cancelled)


def test_recommendations_respect_missing_labels_and_task_requirements():
    unlabelled = {"total_samples": 8, "class_count": 0, "has_labels": False}
    classification = recommend_approaches(
        "image", unlabelled, "classification", hardware={"gpu_available": False}
    )
    detection = recommend_approaches(
        "image", unlabelled, "object_detection", hardware={"gpu_available": False}
    )
    similarity = recommend_approaches(
        "image", unlabelled, "image_similarity", hardware={"gpu_available": False}
    )

    assert "Anotasi kelas" in classification["primary"]["name"]
    assert "Anotasi bounding box" in detection["primary"]["name"]
    assert similarity["labels_available"] is False
    required_fields = {
        "name", "task", "reason", "advantages", "disadvantages",
        "computational_requirements", "complexity", "suitable_dataset_conditions", "limitations",
    }
    assert required_fields.issubset(classification["primary"])
    assert classification["is_experiment_result"] is False


def test_recommendations_do_not_suggest_supervised_model_for_one_class():
    result = recommend_approaches(
        "image",
        {"total_samples": 8, "class_count": 1, "class_distribution": {"only": 8}, "has_labels": True},
        "classification",
        hardware={"gpu_available": False},
    )
    assert result["alternatives"] == []
    assert "Lengkapi minimal dua kelas" in result["primary"]["name"]


def test_video_analysis_metadata_sampling_corrupt_and_preview(tmp_path):
    cv2 = pytest.importorskip("cv2")
    root = tmp_path / "videos" / "walking"
    valid = _make_mp4(root / "clip.mp4", frame_count=8)
    (root / "broken.mp4").write_bytes(b"not a video")
    (root / "empty.avi").write_bytes(b"")

    result = analyze_video_dataset(valid.parents[1], frame_sample_count=2)
    preview = create_preview_thumbnails(
        valid.parents[1], "video", result["sample_references"][:1]
    )

    assert result["total_samples"] == 1
    assert result["class_distribution"] == {"walking": 1}
    assert result["invalid_files"] == 2
    assert result["fps"]["mean"] == pytest.approx(5.0, abs=0.2)
    assert result["dimensions"]["width_max"] == 64
    assert result["frame_samples_read"] <= 2
    assert preview[0]["data"].startswith(b"\xff\xd8")
    assert cv2 is not None


def test_video_zip_sampling_and_empty_video_dataset(tmp_path):
    pytest.importorskip("cv2")
    root = tmp_path / "clips"
    _make_mp4(root / "running" / "short.mp4", frame_count=4)
    archive_path = tmp_path / "clips.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for media in root.rglob("*"):
            if media.is_file():
                archive.write(media, media.relative_to(root).as_posix())
    result = analyze_video_dataset(archive_path, frame_sample_count=1)
    assert result["total_samples"] == 1
    assert result["class_distribution"] == {"running": 1}
    assert result["frame_samples_read"] == 1

    empty = tmp_path / "empty-videos"
    empty.mkdir()
    with pytest.raises(ValueError, match="Tidak ditemukan"):
        analyze_video_dataset(empty)


def test_video_analysis_flags_identical_sampled_signature_across_standard_splits(tmp_path):
    pytest.importorskip("cv2")
    train_clip = _make_mp4(tmp_path / "dataset" / "train" / "clip.mp4", frame_count=8)
    test_clip = tmp_path / "dataset" / "test" / "clip.mp4"
    test_clip.parent.mkdir(parents=True)
    shutil.copy2(train_clip, test_clip)

    result = analyze_video_dataset(tmp_path / "dataset", frame_sample_count=2)

    assert result["possible_split_leakage_signatures"] == 1


def test_vision_analysis_sqlite_persistence_and_project_reopening(tmp_path):
    database = Database(tmp_path / "dora.sqlite")
    projects = ProjectRepository(database)
    project_id = projects.create_project(
        "Vision project", dataset_name="images", dataset_path=str(tmp_path / "images")
    )
    repository = VisionAnalysisRepository(database)
    result = {
        "total_samples": 2,
        "sample_references": [{"name": "cat/a.jpg", "path": str(tmp_path / "images" / "cat" / "a.jpg")}],
        "error_log": [],
    }
    recommendations = {"primary": {"name": "Transfer learning"}, "alternatives": []}
    dataset_path = tmp_path / "images"
    dataset_path.mkdir()
    analysis_id = repository.save(
        project_id=project_id,
        dataset_path=dataset_path,
        dataset_type="image",
        dataset_metadata={"total_samples": 2},
        analysis_config={"task": "classification"},
        results=result,
        recommendations=recommendations,
    )

    reopened = VisionAnalysisRepository(Database(tmp_path / "dora.sqlite"))
    by_project = reopened.get_latest_for_project(project_id, "image")
    by_path = reopened.get_latest_by_path(dataset_path, "image")
    assert by_project["id"] == analysis_id
    assert by_project["results"] == result
    assert by_path["recommendations"] == recommendations
    assert by_path["dataset_metadata"] == {"total_samples": 2}

    recognition = {"model": "MobileNetV2 ImageNet-1K", "classified_count": 2}
    reopened.update_results(analysis_id, {**result, "recognition": recognition})
    assert reopened.get_latest_by_path(dataset_path, "image")["results"]["recognition"] == recognition


def test_qthread_worker_keeps_event_loop_responsive_and_cancels(monkeypatch, tmp_path):
    from PySide6.QtCore import QCoreApplication, QEventLoop, QThread, QTimer
    import app.ui.vision_intelligence_page as page_module

    app = QCoreApplication.instance() or QCoreApplication([])
    main_thread_id = threading.get_ident()
    worker_thread_ids = []

    def slow_analysis(path, *, progress, cancel_event):
        worker_thread_ids.append(threading.get_ident())
        progress(50, "working")
        time.sleep(0.08)
        return {"total_samples": 2, "class_count": 2, "has_labels": True, "class_distribution": {"a": 1, "b": 1}}

    monkeypatch.setattr(page_module, "analyze_image_dataset", slow_analysis)
    monkeypatch.setattr(
        page_module,
        "recommend_approaches",
        lambda *args, **kwargs: {"primary": {}, "alternatives": [], "is_experiment_result": False},
    )
    from app.ui.vision_intelligence_page import _VisionWorker

    thread = QThread()
    worker = _VisionWorker(
        "analysis",
        {"path": str(tmp_path), "dataset_type": "image", "task": "classification"},
        threading.Event(),
    )
    worker.moveToThread(thread)
    loop = QEventLoop()
    heartbeats = []
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: heartbeats.append(time.monotonic()))
    worker.completed.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.started.connect(worker.run)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(loop.quit)
    thread.start()
    timer.start()
    QTimer.singleShot(1500, loop.quit)
    loop.exec()
    timer.stop()
    thread.wait(2000)

    assert worker_thread_ids and worker_thread_ids[0] != main_thread_id
    assert len(heartbeats) >= 3


def test_qthread_worker_emits_cancelled_when_requested(monkeypatch, tmp_path):
    from PySide6.QtCore import QCoreApplication, QEventLoop, QThread, QTimer
    import app.ui.vision_intelligence_page as page_module

    QCoreApplication.instance() or QCoreApplication([])
    cancel_event = threading.Event()

    def waiting_analysis(path, *, progress, cancel_event):
        while not cancel_event.is_set():
            time.sleep(0.005)
        raise AnalysisCancelledError("cancelled")

    monkeypatch.setattr(page_module, "analyze_image_dataset", waiting_analysis)
    from app.ui.vision_intelligence_page import _VisionWorker

    thread = QThread()
    worker = _VisionWorker(
        "analysis",
        {"path": str(tmp_path), "dataset_type": "image", "task": "classification"},
        cancel_event,
    )
    worker.moveToThread(thread)
    loop = QEventLoop()
    cancelled = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.cancelled.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.started.connect(worker.run)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(loop.quit)
    thread.start()
    QTimer.singleShot(20, cancel_event.set)
    QTimer.singleShot(1500, loop.quit)
    loop.exec()
    thread.wait(2000)
    assert cancelled == [True]

