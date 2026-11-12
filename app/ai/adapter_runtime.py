"""Compatibility checks and local PEFT-to-Ollama model conversion.

Adapters are never selected merely because they are present in the registry.
The conversion path merges a verified LoRA adapter into its local base model,
then asks the installed Ollama CLI to import the resulting Safetensors model.
It does not pull a base model or replace Tanya DORA's active model.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from app.ai.model_versioning import ManagedModelVersion, ModelVersionError, ModelVersionStore


class AdapterRuntimeError(RuntimeError):
    """Raised when an adapter cannot safely be checked, merged, or imported."""


@dataclass(frozen=True, slots=True)
class AdapterCompatibilityReport:
    version_id: str
    base_model_path: str
    model_type: str
    peft_type: str
    adapter_format: str
    training_quantization: str
    compatible: bool
    ollama_import_supported: bool
    runtime_ready: bool
    missing_dependencies: tuple[str, ...]
    issues: tuple[str, ...]
    warnings: tuple[str, ...]


class AdapterRuntimeService:
    """Validate a registry adapter and optionally merge/import it into Ollama."""

    OLLAMA_SAFETENSORS_MODEL_TYPES = frozenset({"llama", "mistral", "mixtral", "gemma", "gemma2", "phi3"})
    REQUIRED_MERGE_PACKAGES = ("torch", "transformers", "peft", "accelerate")

    def __init__(self, version_store: ModelVersionStore) -> None:
        self.version_store = version_store

    def inspect(self, version_id: str) -> AdapterCompatibilityReport:
        version = self.version_store.verify_version(version_id)
        adapter_dir = self.version_store._resolve_adapter_path(version.adapter_path)
        base_dir = Path(version.base_model_path)
        adapter_config = _read_object(adapter_dir / "adapter_config.json")
        model_config = _read_object(base_dir / "config.json")
        issues: list[str] = []
        warnings: list[str] = []

        model_type = str(model_config.get("model_type") or "").strip().casefold()
        peft_type = str(adapter_config.get("peft_type") or "").strip().upper()
        task_type = str(adapter_config.get("task_type") or "").strip().upper()
        safe_weights = tuple(adapter_dir.glob("adapter_model.safetensors"))
        binary_weights = tuple(adapter_dir.glob("adapter_model.bin"))
        adapter_format = "safetensors" if safe_weights else "pytorch-bin" if binary_weights else "unknown"
        training_quantization = str(version.training_config.get("quantization") or "none").casefold()

        if peft_type != "LORA":
            issues.append(f"Adapter type {peft_type or 'unknown'} is not LoRA.")
        if task_type and task_type != "CAUSAL_LM":
            issues.append(f"Adapter task type {task_type} is not supported for causal chat inference.")
        elif not task_type:
            warnings.append("Adapter metadata omits task_type; the local PEFT merge remains the final compatibility check.")
        if not model_type:
            issues.append("Base model config.json does not declare model_type.")
        if not adapter_config.get("target_modules"):
            issues.append("Adapter metadata does not declare target_modules.")
        else:
            warnings.append(
                "target_modules are declared in adapter metadata; PEFT validates them against instantiated base layers during merge."
            )
        if adapter_format == "unknown":
            issues.append("Adapter weights must use adapter_model.safetensors or adapter_model.bin.")
        if model_config.get("quantization_config"):
            issues.append("The registered base model is already quantized; merge compatibility is not guaranteed.")
        configured_base = str(adapter_config.get("base_model_name_or_path") or "").strip()
        if configured_base:
            try:
                configured_path = Path(configured_base).expanduser()
                if configured_path.exists() and configured_path.resolve() != base_dir.resolve():
                    issues.append("Adapter metadata points to a different local base model directory.")
                elif not configured_path.exists() and configured_base.casefold() != base_dir.name.casefold():
                    warnings.append(
                        "Adapter base_model_name_or_path is not a local path; model identity is checked using the registered local base fingerprint and metadata."
                    )
            except (OSError, RuntimeError):
                warnings.append("Adapter base-model path could not be normalized; review the provenance before activation.")
        if training_quantization not in {"none", "4bit"}:
            issues.append(f"Unknown training quantization mode: {training_quantization}.")
        if training_quantization == "4bit":
            warnings.append(
                "This adapter was trained with 4-bit QLoRA. Conversion will reload the local base in full precision; verify the merged model before activation."
            )
        if adapter_format == "pytorch-bin":
            warnings.append("The adapter uses PyTorch .bin weights; prefer Safetensors for portable imports.")

        missing = tuple(name for name in self.REQUIRED_MERGE_PACKAGES if importlib.util.find_spec(name) is None)
        if missing:
            warnings.append("Local merge runtime packages are missing: " + ", ".join(missing))
        ollama_cli_available = shutil.which("ollama") is not None
        if not ollama_cli_available:
            warnings.append("Ollama CLI is not available on PATH; the merged model cannot be imported yet.")
        return AdapterCompatibilityReport(
            version_id=version.version_id,
            base_model_path=str(base_dir),
            model_type=model_type,
            peft_type=peft_type,
            adapter_format=adapter_format,
            training_quantization=training_quantization,
            compatible=not issues,
            ollama_import_supported=model_type in self.OLLAMA_SAFETENSORS_MODEL_TYPES,
            runtime_ready=not missing and ollama_cli_available,
            missing_dependencies=missing,
            issues=tuple(issues),
            warnings=tuple(warnings),
        )

    def merge_and_import_to_ollama(
        self,
        version_id: str,
        *,
        model_tag: str | None = None,
        user_confirmed: bool = False,
        timeout_seconds: int = 7200,
    ) -> tuple[str, Path]:
        """Merge a verified adapter and import the full local model into Ollama.

        ``user_confirmed`` is required because this may use substantial local
        storage and creates a model entry in the Ollama installation.
        """
        if not user_confirmed:
            raise AdapterRuntimeError("Explicit user confirmation is required before creating an Ollama model.")
        report = self.inspect(version_id)
        if not report.compatible:
            raise AdapterRuntimeError("Adapter compatibility check failed: " + " ".join(report.issues))
        if not report.ollama_import_supported:
            raise AdapterRuntimeError(
                f"Ollama Safetensors import is not enabled for model_type '{report.model_type}'. "
                "Use a compatible Transformers runtime or convert the merged model with a supported GGUF tool."
            )
        if not report.runtime_ready:
            if report.missing_dependencies:
                raise AdapterRuntimeError(
                    "Install the optional local merge dependencies first: " + ", ".join(report.missing_dependencies)
                )
            raise AdapterRuntimeError("Ollama CLI was not found. Install/start Ollama before importing this model.")
        ollama = shutil.which("ollama")
        if not ollama:
            raise AdapterRuntimeError("Ollama CLI was not found. Install/start Ollama before importing this model.")

        version = self.version_store.verify_version(version_id)
        adapter_dir = self.version_store._resolve_adapter_path(version.adapter_path)
        base_dir = Path(version.base_model_path)
        merged_root = self.version_store.storage_dir / "merged"
        merged_root.mkdir(parents=True, exist_ok=True)
        safe_version = re.sub(r"[^a-zA-Z0-9_-]+", "-", version.version_id)[:48]
        destination = merged_root / f"{safe_version}-{__import__('uuid').uuid4().hex[:8]}"
        destination.mkdir(parents=False, exist_ok=False)
        _ensure_merge_disk_space(base_dir, adapter_dir, destination.parent)

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            base_model = AutoModelForCausalLM.from_pretrained(
                str(base_dir),
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            )
            peft_model = PeftModel.from_pretrained(
                base_model,
                str(adapter_dir),
                is_trainable=False,
                local_files_only=True,
            )
            merged_model = peft_model.merge_and_unload(safe_merge=True)
            merged_model.save_pretrained(destination, safe_serialization=True)
            tokenizer = AutoTokenizer.from_pretrained(
                str(base_dir), local_files_only=True, trust_remote_code=False
            )
            tokenizer.save_pretrained(destination)
            del merged_model, peft_model, base_model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as error:
            raise AdapterRuntimeError(
                f"Adapter merge failed. The previous active model remains selected; partial output is at {destination}: {error}"
            ) from error

        tag = _normalize_model_tag(model_tag or f"tanya-dora-{safe_version}")
        modelfile = destination / "Modelfile"
        modelfile.write_text(f'FROM "{destination.as_posix()}"\n', encoding="utf-8")
        try:
            result = subprocess.run(
                [ollama, "create", tag, "-f", str(modelfile)],
                cwd=str(destination),
                capture_output=True,
                text=True,
                timeout=max(60, int(timeout_seconds)),
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise AdapterRuntimeError(
                f"Ollama import timed out. The merged files remain at {destination}; the active model was not changed."
            ) from error
        except OSError as error:
            raise AdapterRuntimeError(f"Ollama import could not start: {error}") from error
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "Ollama returned no error detail.").strip()[-1200:]
            raise AdapterRuntimeError(
                f"Ollama could not import the merged model. Merged files remain at {destination}: {detail}"
            )
        return tag, destination


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise AdapterRuntimeError(f"Model metadata could not be read from {path.name}: {error}") from error
    if not isinstance(value, Mapping):
        raise AdapterRuntimeError(f"Model metadata in {path.name} must be a JSON object.")
    return dict(value)


def _normalize_model_tag(value: str) -> str:
    normalized = str(value or "").strip().casefold()
    normalized = re.sub(r"[^a-z0-9._:/-]+", "-", normalized).strip("-.:/")
    if not normalized or len(normalized) > 120:
        raise ValueError("Ollama model tag must contain 1-120 supported characters.")
    return normalized


def _ensure_merge_disk_space(base_dir: Path, adapter_dir: Path, output_parent: Path) -> None:
    base_bytes = sum(path.stat().st_size for path in base_dir.rglob("*") if path.is_file())
    adapter_bytes = sum(path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file())
    required_bytes = int(base_bytes * 1.15 + adapter_bytes + 1024**3)
    free_bytes = shutil.disk_usage(output_parent).free
    if free_bytes < required_bytes:
        raise AdapterRuntimeError(
            f"Merged model needs approximately {required_bytes / 1024**3:.1f} GiB free; "
            f"only {free_bytes / 1024**3:.1f} GiB is available. No model was activated."
        )
