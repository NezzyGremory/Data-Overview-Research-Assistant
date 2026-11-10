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

