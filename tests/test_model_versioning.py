from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from app.ai.model_versioning import ModelVersionError, ModelVersionStore


def _model_dirs(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    base = root / "base-model"
    base.mkdir()
    (base / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"base weights")
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base), "r": 8, "lora_alpha": 16}),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    return base, adapter


def test_register_verify_and_explicitly_select_adapter(tmp_path: Path):
    base, adapter = _model_dirs(tmp_path)
    store = ModelVersionStore(tmp_path / "managed")

    version = store.register_adapter(
        "Tanya DORA LoRA v1",
        adapter,
        base,
        dataset_version_id="tdid-v1",
        license_name="Apache-2.0",
        training_config={"epochs": 1},
        evaluation_metrics={"eval_loss": 0.4},
    )
    assert store.verify_version(version.version_id).adapter_sha256 == version.adapter_sha256
    assert version.evaluation_metrics["eval_loss"] == 0.4
    with pytest.raises(ModelVersionError, match="confirmation"):
        store.select_version(version.version_id)

    selected = store.select_version(version.version_id, user_confirmed=True)
    assert selected.selected
    with pytest.raises(ModelVersionError, match="Clear the selected"):
        store.delete_version(version.version_id)

    assert store.clear_selection(user_confirmed=True)
    assert store.delete_version(version.version_id)


def test_adapter_export_import_round_trip_checks_base_compatibility(tmp_path: Path):
    base, adapter = _model_dirs(tmp_path)
    store = ModelVersionStore(tmp_path / "managed")
    version = store.register_adapter("source", adapter, base)
    archive = store.export_adapter(version.version_id, tmp_path / "adapter-export.zip")

    imported = store.import_adapter_archive(archive, base, name="imported")
    assert imported.name == "imported"
    assert store.verify_version(imported.version_id).adapter_sha256 == imported.adapter_sha256

    other_base, _other_adapter = _model_dirs(tmp_path / "other")
    (other_base / "config.json").write_text('{"model_type":"other"}', encoding="utf-8")
    with pytest.raises(ModelVersionError, match="different base model"):
        store.import_adapter_archive(archive, other_base)


def test_integrity_detects_adapter_and_base_changes(tmp_path: Path):
    base, adapter = _model_dirs(tmp_path)
    store = ModelVersionStore(tmp_path / "managed")
    version = store.register_adapter("protected", adapter, base)
    managed_adapter = store.storage_dir / version.adapter_path
    (managed_adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ModelVersionError, match="SHA-256"):
        store.verify_version(version.version_id)

    clean = store.register_adapter("second", adapter, base)
    (base / "config.json").write_text('{"model_type":"changed"}', encoding="utf-8")
    with pytest.raises(ModelVersionError, match="base model"):
        store.verify_version(clean.version_id)


def test_import_rejects_zip_path_traversal(tmp_path: Path):
    base, _adapter = _model_dirs(tmp_path)
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "bad")
    store = ModelVersionStore(tmp_path / "managed")

    with pytest.raises(ModelVersionError, match="unsafe path"):
        store.import_adapter_archive(archive, base)
    assert not (tmp_path / "escape.txt").exists()


def test_active_runtime_switch_requires_confirmation_and_can_rollback(tmp_path: Path):
    store = ModelVersionStore(tmp_path / "models")
    with pytest.raises(ModelVersionError, match="confirmation"):
        store.set_active_runtime_model("qwen2.5:7b")

    first = store.set_active_runtime_model("qwen2.5:7b", user_confirmed=True)
    second = store.set_active_runtime_model("llama3.2:3b", user_confirmed=True)
    assert store.get_active_runtime_model() == second
    assert store.list_activation_history()[0]["previous"]["model_name"] == "qwen2.5:7b"

    rolled_back = store.rollback_active_runtime_model(user_confirmed=True)
    assert rolled_back["model_name"] == first["model_name"]
    assert store.get_active_runtime_model()["model_name"] == "qwen2.5:7b"


def test_clear_runtime_selection_is_explicit_and_recorded(tmp_path: Path):
    store = ModelVersionStore(tmp_path / "models")
    store.set_active_runtime_model("gemma3:4b", user_confirmed=True)
    with pytest.raises(ModelVersionError, match="confirmation"):
        store.clear_active_runtime_model()
    assert store.clear_active_runtime_model(user_confirmed=True)
    assert store.get_active_runtime_model() is None
    assert store.list_activation_history()[0]["new"] is None

