from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ai.adapter_runtime import AdapterRuntimeError, AdapterRuntimeService
from app.ai.model_versioning import ModelVersionStore


def _registered_adapter(root: Path, *, model_type: str = "llama", peft_type: str = "LORA"):
    base = root / "base"
    base.mkdir(parents=True)
    (base / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"local base model weights")
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({
            "base_model_name_or_path": str(base),
            "peft_type": peft_type,
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
        }),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    store = ModelVersionStore(root / "registry")
    version = store.register_adapter("adapter test", adapter, base, training_config={"quantization": "none"})
    return store, version


def test_compatibility_report_checks_peft_base_format_and_local_runtime(tmp_path: Path):
    store, version = _registered_adapter(tmp_path)
    report = AdapterRuntimeService(store).inspect(version.version_id)

    assert report.compatible
    assert report.model_type == "llama"
    assert report.peft_type == "LORA"
    assert report.adapter_format == "safetensors"
    assert report.ollama_import_supported
    assert report.runtime_ready == (not report.missing_dependencies)


def test_unsupported_ollama_family_is_not_marked_importable(tmp_path: Path):
    store, version = _registered_adapter(tmp_path, model_type="unsupported_arch")
    report = AdapterRuntimeService(store).inspect(version.version_id)
    assert report.compatible
    assert not report.ollama_import_supported


def test_non_lora_adapter_is_rejected_before_runtime_activation(tmp_path: Path):
    store, version = _registered_adapter(tmp_path, peft_type="IA3")
    report = AdapterRuntimeService(store).inspect(version.version_id)
    assert not report.compatible
    assert any("not LoRA" in issue for issue in report.issues)


def test_model_creation_requires_explicit_confirmation(tmp_path: Path):
    store, version = _registered_adapter(tmp_path)
    with pytest.raises(AdapterRuntimeError, match="confirmation"):
        AdapterRuntimeService(store).merge_and_import_to_ollama(version.version_id)
