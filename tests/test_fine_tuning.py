from __future__ import annotations

from pathlib import Path
import json
import zipfile

import pytest

from app.ai.fine_tuning import (
    FineTuningConfig,
    FineTuningEngine,
    TrainingHardware,
    _records_for_training,
)
from app.ai.tdid_dataset import TDIDDatasetBuilder, TDIDVersionStore


def _record(index: int):
    return {
        "record_id": f"training-{index}",
        "messages": [
            {"role": "user", "content": f"Question {index}?"},
            {"role": "assistant", "content": f"Answer {index}."},
        ],
        "metadata": {
            "category": "statistics",
            "difficulty": "beginner",
            "instruction_type": "knowledge",
            "verified": True,
            "source": f"Source {index}",
        },
    }


def _hardware(**overrides):
    values = {
        "cpu_count": 8,
        "total_ram_bytes": 64 * 1024**3,
        "available_ram_bytes": 48 * 1024**3,
        "torch_available": True,
        "torch_version": "test",
        "cuda_available": False,
        "cuda_device_name": None,
        "vram_total_bytes": None,
        "vram_available_bytes": None,
        "ollama_available": None,
        "ollama_detail": None,
    }
    values.update(overrides)
    return TrainingHardware(**values)


def _saved_split(store: TDIDVersionStore):
    builder = TDIDDatasetBuilder([_record(index) for index in range(6)])
    return store.save_version("tdid-v1", builder, split=builder.split(seed=12))


def test_training_config_validates_conservative_lora_parameters():
    config = FineTuningConfig(
        base_model_path="model",
        dataset_version_id="version",
        output_dir="output",
    )
    assert config.validate() == ()
    assert FineTuningConfig(
        base_model_path="model",
        dataset_version_id="version",
        output_dir="output",
        lora_dropout=1,
        quantization="8bit",
    ).validate()


def test_split_selection_uses_only_training_and_validation_record_ids(tmp_path: Path):
    store = TDIDVersionStore(tmp_path / "tdid")
    version = _saved_split(store)
    builder = store.load_version(version.version_id)
    training, validation = _records_for_training(builder, version.split_manifest)
    test_ids = set(version.split_manifest["record_ids"]["test"])
    selected_ids = {record["record_id"] for record in training + validation}

    assert selected_ids.isdisjoint(test_ids)
    assert selected_ids == set(version.split_manifest["record_ids"]["train"] + version.split_manifest["record_ids"]["validation"])


def test_preflight_reports_missing_optional_training_dependencies_without_downloads(tmp_path: Path):
    store = TDIDVersionStore(tmp_path / "tdid")
    version = _saved_split(store)
    model = tmp_path / "local-model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"local weights")
    output = tmp_path / "training-output"
    output.mkdir()
    engine = FineTuningEngine(store, hardware_probe=lambda: _hardware())

    report = engine.preflight(
        FineTuningConfig(str(model), version.version_id, str(output)),
        hardware=_hardware(),
    )

    assert report.estimated_weight_bytes == len(b"local weights")
    assert report.missing_dependencies
    assert not report.ready
    assert any("packages are missing" in blocker for blocker in report.blockers)


def test_qlora_preflight_requires_cuda_and_never_starts_on_cpu(tmp_path: Path):
    store = TDIDVersionStore(tmp_path / "tdid")
    version = _saved_split(store)
    model = tmp_path / "local-model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"local weights")
    output = tmp_path / "training-output"
    output.mkdir()
    engine = FineTuningEngine(store, hardware_probe=lambda: _hardware())
    config = FineTuningConfig(str(model), version.version_id, str(output), quantization="4bit")

    report = engine.preflight(config, hardware=_hardware())
    assert not report.ready
    assert any("requires a CUDA-capable GPU" in blocker for blocker in report.blockers)
    with pytest.raises(RuntimeError, match="preflight failed"):
        engine.run(config)


def test_preflight_rejects_network_or_invalid_model_paths(tmp_path: Path):
    store = TDIDVersionStore(tmp_path / "tdid")
    version = _saved_split(store)
    output = tmp_path / "training-output"
    output.mkdir()
    engine = FineTuningEngine(store, hardware_probe=lambda: _hardware())

    report = engine.preflight(
        FineTuningConfig("org/model-name", version.version_id, str(output)),
        hardware=_hardware(),
    )
    assert not report.ready
    assert any("network downloads are disabled" in blocker for blocker in report.blockers)


def test_workspace_export_contains_verified_persisted_splits(tmp_path: Path):
    store = TDIDVersionStore(tmp_path / "tdid")
    version = _saved_split(store)
    engine = FineTuningEngine(store, hardware_probe=lambda: _hardware())
    output = engine.export_workspace(version.version_id, tmp_path / "portable-workspace.zip")

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("workspace_manifest.json"))
        train = archive.read("dataset/train.jsonl").decode("utf-8").splitlines()
        validation = archive.read("dataset/validation.jsonl").decode("utf-8").splitlines()
        test = archive.read("dataset/test.jsonl").decode("utf-8").splitlines()
    assert {"dataset/train.jsonl", "dataset/validation.jsonl", "dataset/test.jsonl", "README.md"} <= names
    assert manifest["verified_only"] is True
    assert manifest["record_counts"] == {"train": len(train), "validation": len(validation), "test": len(test)}
    assert all(json.loads(line)["metadata"]["verified"] is True for line in train + validation + test)

