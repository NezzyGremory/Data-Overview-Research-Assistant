"""Optional, local-only supervised fine-tuning with LoRA/QLoRA.

Heavy Hugging Face dependencies are imported only when a training job starts.
The engine never downloads a model, executes remote model code, or activates a
new adapter automatically.
"""

from __future__ import annotations

import ctypes
import importlib.util
import inspect
import json
import os
import shutil
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

from app.ai.tdid_dataset import TDIDDatasetBuilder, TDIDError, TDIDVersionStore


class FineTuningError(RuntimeError):
    """Raised when a fine-tuning job cannot safely start or finish."""


@dataclass(frozen=True, slots=True)
class FineTuningConfig:
    base_model_path: str
    dataset_version_id: str
    output_dir: str
    epochs: float = 1.0
    learning_rate: float = 2e-4
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_sequence_length: int = 1024
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    quantization: str = "none"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    checkpoint_steps: int = 50
    evaluation_steps: int = 50
    random_seed: int = 42
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not str(self.base_model_path).strip():
            issues.append("Choose a local base model directory.")
        if not str(self.dataset_version_id).strip():
            issues.append("Choose a saved TDID version.")
        if not str(self.output_dir).strip():
            issues.append("Choose an output directory.")
        try:
            epochs = float(self.epochs)
            learning_rate = float(self.learning_rate)
            dropout = float(self.lora_dropout)
            warmup = float(self.warmup_ratio)
            weight_decay = float(self.weight_decay)
        except (TypeError, ValueError, OverflowError):
            issues.append("Numeric training settings must contain valid numbers.")
            epochs = learning_rate = dropout = warmup = weight_decay = 0.0
        if not 0 < epochs <= 100:
            issues.append("Epochs must be greater than 0 and at most 100.")
        if not 0 < learning_rate <= 1:
            issues.append("Learning rate must be greater than 0 and at most 1.")
        for name, value in (
            ("per-device batch size", self.per_device_batch_size),
            ("gradient accumulation steps", self.gradient_accumulation_steps),
            ("maximum sequence length", self.max_sequence_length),
            ("LoRA rank", self.lora_rank),
            ("LoRA alpha", self.lora_alpha),
            ("checkpoint interval", self.checkpoint_steps),
            ("evaluation interval", self.evaluation_steps),
        ):
            try:
                positive = int(value) > 0
            except (TypeError, ValueError, OverflowError):
                positive = False
            if not positive:
                issues.append(f"{name.capitalize()} must be positive.")
        if not 0 <= dropout < 1:
            issues.append("LoRA dropout must be at least 0 and less than 1.")
        if not 0 <= warmup < 1:
            issues.append("Warmup ratio must be at least 0 and less than 1.")
        if not 0 <= weight_decay <= 1:
            issues.append("Weight decay must be between 0 and 1.")
        if str(self.quantization).casefold() not in {"none", "4bit"}:
            issues.append("Quantization must be 'none' or '4bit'.")
        if not self.target_modules or any(not str(item).strip() for item in self.target_modules):
            issues.append("Specify at least one model-specific LoRA target module.")
        return tuple(issues)

    def normalized(self) -> "FineTuningConfig":
        return FineTuningConfig(
            **{
                **asdict(self),
                "base_model_path": str(Path(self.base_model_path).expanduser().resolve()),
                "output_dir": str(Path(self.output_dir).expanduser().resolve()),
                "quantization": str(self.quantization).casefold(),
                "target_modules": tuple(str(item).strip() for item in self.target_modules),
            }
        )


@dataclass(frozen=True, slots=True)
class TrainingHardware:
    cpu_count: int
    total_ram_bytes: int | None
    available_ram_bytes: int | None
    torch_available: bool
    torch_version: str | None
    cuda_available: bool
    cuda_device_name: str | None
    vram_total_bytes: int | None
    vram_available_bytes: int | None
    ollama_available: bool | None
    ollama_detail: str | None


@dataclass(frozen=True, slots=True)
class FineTuningPreflight:
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    missing_dependencies: tuple[str, ...]
    hardware: TrainingHardware
    estimated_weight_bytes: int | None
    estimated_memory_bytes: int | None
    available_disk_bytes: int | None


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    status: str
    step: int
    total_steps: int | None
    epoch: float | None
    train_loss: float | None
    validation_loss: float | None
    learning_rate: float | None
    elapsed_seconds: float
    available_ram_bytes: int | None
    available_vram_bytes: int | None
    checkpoint_dir: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class FineTuningResult:
    status: str
    job_id: str
    job_dir: str
    adapter_dir: str | None
    metrics: Mapping[str, Any]
    manifest_path: str


def detect_training_hardware(ollama_manager: Any | None = None) -> TrainingHardware:
    total_ram, available_ram = _memory_status()
    torch_available = importlib.util.find_spec("torch") is not None
    torch_version = None
    cuda_available = False
    cuda_name = None
    vram_total = vram_available = None
    if torch_available:
        try:
            import torch

            torch_version = str(torch.__version__)
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                device = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(device)
                cuda_name = str(properties.name)
                vram_total = int(properties.total_memory)
                vram_available = int(torch.cuda.mem_get_info(device)[0])
        except Exception:
            cuda_available = False
    ollama_available = None
    ollama_detail = None
    if ollama_manager is not None:
        try:
            status = ollama_manager.get_status()
            ollama_available = bool(status.available)
            ollama_detail = str(status.detail)
        except Exception as error:
            ollama_available = False
            ollama_detail = str(error)
    return TrainingHardware(
        cpu_count=max(1, os.cpu_count() or 1),
        total_ram_bytes=total_ram,
        available_ram_bytes=available_ram,
        torch_available=torch_available,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_device_name=cuda_name,
        vram_total_bytes=vram_total,
        vram_available_bytes=vram_available,
        ollama_available=ollama_available,
        ollama_detail=ollama_detail,
    )


class FineTuningEngine:
    """Preflight and run local SFT against a saved, split TDID version."""

    REQUIRED_PACKAGES = ("torch", "transformers", "datasets", "peft", "trl", "accelerate")

    def __init__(
        self,
        dataset_store: TDIDVersionStore,
        *,
        hardware_probe: Callable[[], TrainingHardware] = detect_training_hardware,
    ) -> None:
        self.dataset_store = dataset_store
        self.hardware_probe = hardware_probe

    def preflight(
        self,
        config: FineTuningConfig,
        *,
        hardware: TrainingHardware | None = None,
    ) -> FineTuningPreflight:
        blockers = list(config.validate())
        warnings: list[str] = []
        normalized = config.normalized() if not blockers else config
        model_dir = Path(normalized.base_model_path).expanduser()
        output_dir = Path(normalized.output_dir).expanduser()
        weight_bytes: int | None = None
        memory_estimate: int | None = None
        disk_available: int | None = None
        missing = [name for name in self.REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]
        if normalized.quantization == "4bit" and importlib.util.find_spec("bitsandbytes") is None:
            missing.append("bitsandbytes")
        if not model_dir.is_dir():
            blockers.append("Base model must be an existing local directory; network downloads are disabled.")
        elif output_dir == model_dir or model_dir in output_dir.parents:
            blockers.append("Training output must be outside the base model directory.")
        elif output_dir in model_dir.parents:
            warnings.append("Output directory is a parent of the base model; choose a separate location to avoid confusion.")
        else:
            weight_bytes = _local_model_weight_bytes(model_dir)
            if not weight_bytes:
                blockers.append("No local .safetensors or .bin model weights were found.")
        try:
            version = self.dataset_store.get_version(normalized.dataset_version_id)
            builder = self.dataset_store.load_version(version.version_id)
            if builder.validate():
                blockers.append("Selected TDID snapshot has validation issues.")
            split_ids = version.split_manifest.get("record_ids")
            if not isinstance(split_ids, dict) or not all(
                isinstance(split_ids.get(key), list) and split_ids[key]
                for key in ("train", "validation", "test")
            ):
                blockers.append("Save a TDID version with non-empty train, validation and test splits first.")
            else:
                known_ids = {str(record.get("record_id") or "") for record in builder.records}
                all_ids = [str(item) for key in ("train", "validation", "test") for item in split_ids[key]]
                if len(set(all_ids)) != len(all_ids) or not set(all_ids).issubset(known_ids):
                    blockers.append("TDID split manifest is inconsistent or contains cross-split duplicates.")
                else:
                    by_id = {str(record.get("record_id") or ""): record for record in builder.records}
                    selected = [by_id[item] for item in all_ids]
                    if any(
                        not isinstance(record.get("metadata"), Mapping)
                        or record["metadata"].get("verified") is not True
                        for record in selected
                    ):
                        blockers.append("TDID split contains an unverified example.")
                    if version.split_manifest.get("source_grouped"):
                        split_groups = []
                        for split_name in ("train", "validation", "test"):
                            split_groups.append(
                                {
                                    _training_source_group(by_id[str(item)])
                                    for item in split_ids[split_name]
                                    if _training_source_group(by_id[str(item)])
                                }
                            )
                        if any(
                            split_groups[left] & split_groups[right]
                            for left in range(3)
                            for right in range(left + 1, 3)
                        ):
                            blockers.append("TDID source groups cross the saved train/validation/test splits.")
        except (TDIDError, OSError, ValueError) as error:
            blockers.append(f"TDID version cannot be loaded: {error}")

        resource = hardware or self.hardware_probe()
        if missing:
            blockers.append("Optional training packages are missing: " + ", ".join(missing))
        if normalized.quantization == "4bit":
            if not resource.cuda_available:
                blockers.append("QLoRA 4-bit mode requires a CUDA-capable GPU in this training engine.")
            if resource.cuda_available and "bitsandbytes" in missing:
                blockers.append("QLoRA requires the optional bitsandbytes package.")
        if weight_bytes:
            memory_estimate = (
                int(weight_bytes * 0.35) + 3 * 1024**3
                if normalized.quantization == "4bit"
                else int(weight_bytes * 2.5) + 2 * 1024**3
            )
            available_memory = resource.vram_available_bytes if resource.cuda_available else resource.available_ram_bytes
            if available_memory is not None and available_memory < memory_estimate:
                target = "VRAM" if resource.cuda_available else "RAM"
                blockers.append(
                    f"Estimated training memory is {memory_estimate / 1024**3:.1f} GiB, above available {target} "
                    f"({available_memory / 1024**3:.1f} GiB). Choose a smaller model or sequence/batch settings."
                )
            else:
                warnings.append("Memory estimate is approximate; actual demand depends on model architecture and sequence lengths.")
            output_parent = output_dir if output_dir.exists() else output_dir.parent
            if output_parent.exists():
                try:
                    disk_available = shutil.disk_usage(output_parent).free
                    disk_estimate = weight_bytes * 2 + 1024**3
                    if disk_available < disk_estimate:
                        blockers.append("Free disk space is below the estimated checkpoint and adapter requirement.")
                except OSError:
                    warnings.append("Free disk space could not be measured.")
            else:
                blockers.append("Training output parent directory does not exist.")
        if not resource.torch_available:
            warnings.append("PyTorch is not available; inference may still use Ollama independently.")
        return FineTuningPreflight(
            ready=not blockers,
            blockers=tuple(dict.fromkeys(blockers)),
            warnings=tuple(dict.fromkeys(warnings)),
            missing_dependencies=tuple(dict.fromkeys(missing)),
            hardware=resource,
            estimated_weight_bytes=weight_bytes,
            estimated_memory_bytes=memory_estimate,
            available_disk_bytes=disk_available,
        )

    def run(
        self,
        config: FineTuningConfig,
        *,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[TrainingProgress], None] | None = None,
    ) -> FineTuningResult:
        normalized = config.normalized()
        preflight = self.preflight(normalized)
        if not preflight.ready:
            raise FineTuningError("Training preflight failed: " + " ".join(preflight.blockers))
        cancel = cancel_event or threading.Event()
        version = self.dataset_store.get_version(normalized.dataset_version_id)
        builder = self.dataset_store.load_version(version.version_id)
        train_records, validation_records = _records_for_training(builder, version.split_manifest)
        if not train_records or not validation_records:
            raise FineTuningError("Training and validation splits must both contain records.")

        job_id = uuid4().hex
        job_dir = Path(normalized.output_dir) / f"tdid-{job_id}"
        checkpoint_dir = job_dir / "checkpoints"
        adapter_dir = job_dir / "adapter"
        job_dir.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        manifest_path = job_dir / "training_manifest.json"
        manifest: dict[str, Any] = {
            "job_id": job_id,
            "status": "starting",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "base_model_path": normalized.base_model_path,
            "dataset_version_id": version.version_id,
            "dataset_sha256": version.sha256,
            "train_record_ids": [str(item.get("record_id") or "") for item in train_records],
            "validation_record_ids": [str(item.get("record_id") or "") for item in validation_records],
            "config": asdict(normalized),
            "hardware": _hardware_dict(preflight.hardware),
            "preflight_warnings": list(preflight.warnings),
            "progress_log_path": str(job_dir / "training_progress.jsonl"),
        }
        _write_manifest(manifest_path, manifest)
        try:
            result = self._train_local(
                normalized,
                train_records,
                validation_records,
                checkpoint_dir=checkpoint_dir,
                adapter_dir=adapter_dir,
                cancel_event=cancel,
                started=started,
                progress_callback=progress_callback,
            )
            status, metrics = result
            manifest.update(
                {
                    "status": status,
                    "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "metrics": metrics,
                    "adapter_dir": str(adapter_dir) if adapter_dir.is_dir() else None,
                }
            )
            _write_manifest(manifest_path, manifest)
            return FineTuningResult(
                status=status,
                job_id=job_id,
                job_dir=str(job_dir),
                adapter_dir=str(adapter_dir) if adapter_dir.is_dir() else None,
                metrics=metrics,
                manifest_path=str(manifest_path),
            )
        except Exception as error:
            manifest.update(
                {
                    "status": "failed",
                    "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "error": f"{type(error).__name__}: {error}",
                    "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir.exists() else None,
                    "adapter_dir": str(adapter_dir) if adapter_dir.exists() else None,
                }
            )
            _write_manifest(manifest_path, manifest)
            if isinstance(error, FineTuningError):
                raise
            raise FineTuningError(f"Training failed; saved checkpoints remain at {checkpoint_dir}: {error}") from error

    def export_workspace(
        self,
        dataset_version_id: str,
        destination: str | Path,
        *,
        config: FineTuningConfig | None = None,
    ) -> Path:
        """Package a verified, split TDID workspace for a compatible machine."""
        version = self.dataset_store.get_version(dataset_version_id)
        builder = self.dataset_store.load_version(version.version_id)
        issues = builder.validate()
        if issues:
            raise FineTuningError("Cannot export a TDID workspace with validation issues.")
        split_manifest = version.split_manifest
        split_ids = split_manifest.get("record_ids")
        if not isinstance(split_ids, Mapping) or not all(
            isinstance(split_ids.get(name), list) and split_ids[name]
            for name in ("train", "validation", "test")
        ):
            raise FineTuningError("The selected TDID version must contain train, validation and test splits.")
        records_by_id = {str(item.get("record_id") or ""): item for item in builder.records}
        exported_splits: dict[str, list[dict[str, Any]]] = {}
        for split_name in ("train", "validation", "test"):
            ids = [str(item) for item in split_ids[split_name]]
            if any(item not in records_by_id for item in ids):
                raise FineTuningError(f"The saved {split_name} split references an unknown record.")
            records = [records_by_id[item] for item in ids]
            if any(
                not isinstance(item.get("metadata"), Mapping)
                or item["metadata"].get("verified") is not True
                for item in records
            ):
                raise FineTuningError(f"The saved {split_name} split contains unverified examples.")
            exported_splits[split_name] = records

        target = Path(destination).expanduser().resolve()
        if target.suffix.casefold() != ".zip":
            raise ValueError("Training workspace export must use the .zip extension.")
        target.parent.mkdir(parents=True, exist_ok=True)
        selected_config = asdict(config.normalized()) if config is not None else {
            "base_model_path": "SET_A_LOCAL_MODEL_PATH",
            "dataset_version_id": version.version_id,
            "output_dir": "SET_A_LOCAL_OUTPUT_DIRECTORY",
            "quantization": "none",
        }
        manifest = {
            "format": "tanya-dora-training-workspace-v1",
            "dataset_version_id": version.version_id,
            "dataset_sha256": version.sha256,
            "split_manifest": dict(split_manifest),
            "record_counts": {key: len(value) for key, value in exported_splits.items()},
            "verified_only": True,
            "training_config": selected_config,
        }
        readme = """# Tanya DORA training workspace\n\nThis archive contains a verified TDID snapshot and the original split assignment. It does not contain a base model, Python environment, checkpoints, or an adapter.\n\n1. Use a compatible local Python/PyTorch environment and install the repository's optional training requirements.\n2. Install the correct PyTorch build for the target CPU/CUDA system before installing bitsandbytes.\n3. Set `base_model_path` and `output_dir` in `training_config.json` to local directories on that machine.\n4. Keep `train.jsonl`, `validation.jsonl`, and `test.jsonl` separate; do not move records across splits.\n5. Re-run TDID validation and the trainer preflight before starting. The app does not download model weights or execute remote model code.\n\nA small dataset is a pipeline check only; it does not make Tanya DORA an expert model.\n"""
        temporary = target.with_suffix(".tmp")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("README.md", readme)
                archive.writestr("training_config.json", json.dumps(selected_config, ensure_ascii=False, indent=2))
                archive.writestr("workspace_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
                for split_name, records in exported_splits.items():
                    content = "".join(
                        json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
                        for item in records
                    )
                    archive.writestr(f"dataset/{split_name}.jsonl", content)
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return target

    def _train_local(
        self,
        config: FineTuningConfig,
        train_records: list[dict[str, Any]],
        validation_records: list[dict[str, Any]],
        *,
        checkpoint_dir: Path,
        adapter_dir: Path,
        cancel_event: threading.Event,
        started: float,
        progress_callback: Callable[[TrainingProgress], None] | None,
    ) -> tuple[str, Mapping[str, Any]]:
        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                TrainerCallback,
            )
            from trl import SFTConfig, SFTTrainer
        except ImportError as error:
            raise FineTuningError(f"Training dependency is unavailable: {error}") from error

        base_path = Path(config.base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            str(base_path), local_files_only=True, trust_remote_code=False
        )
        if not getattr(tokenizer, "chat_template", None):
            raise FineTuningError("The local tokenizer has no chat template for conversational SFT.")
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
            "torch_dtype": "auto",
        }
        if config.quantization == "4bit":
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        elif torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(str(base_path), **model_kwargs)
        if config.quantization == "4bit":
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)

        train_dataset = Dataset.from_list([{"messages": item["messages"]} for item in train_records])
        validation_dataset = Dataset.from_list([{"messages": item["messages"]} for item in validation_records])
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        total_steps = None
        last_logs: dict[str, Any] = {}

        class _TrainingCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                nonlocal last_logs
                last_logs = dict(logs or {})
                if progress_callback:
                    hardware = detect_training_hardware()
                    progress = TrainingProgress(
                        status="training",
                        step=int(state.global_step),
                        total_steps=int(state.max_steps) if state.max_steps else total_steps,
                        epoch=float(state.epoch) if state.epoch is not None else None,
                        train_loss=_optional_float(last_logs.get("loss")),
                        validation_loss=_optional_float(last_logs.get("eval_loss")),
                        learning_rate=_optional_float(last_logs.get("learning_rate")),
                        elapsed_seconds=max(0.0, time.monotonic() - started),
                        available_ram_bytes=hardware.available_ram_bytes,
                        available_vram_bytes=hardware.vram_available_bytes,
                        checkpoint_dir=str(checkpoint_dir),
                    )
                    progress_log = checkpoint_dir.parent / "training_progress.jsonl"
                    try:
                        with progress_log.open("a", encoding="utf-8", newline="\n") as handle:
                            handle.write(json.dumps(asdict(progress), ensure_ascii=False, allow_nan=False) + "\n")
                    except OSError:
                        pass
                    progress_callback(progress)
                return control

            def on_step_end(self, args, state, control, **kwargs):
                if cancel_event.is_set():
                    control.should_training_stop = True
                return control

        sft_values: dict[str, Any] = {
            "output_dir": str(checkpoint_dir),
            "num_train_epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "per_device_train_batch_size": config.per_device_batch_size,
            "per_device_eval_batch_size": config.per_device_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "warmup_ratio": config.warmup_ratio,
            "weight_decay": config.weight_decay,
            "logging_steps": 1,
            "save_strategy": "steps",
            "save_steps": config.checkpoint_steps,
            "save_total_limit": 3,
            "eval_steps": config.evaluation_steps,
            "seed": config.random_seed,
            "report_to": "none",
            "load_best_model_at_end": False,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported() and config.quantization == "4bit"),
            "fp16": bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported() and config.quantization == "4bit"),
        }
        sft_parameters = inspect.signature(SFTConfig).parameters
        if "eval_strategy" in sft_parameters:
            sft_values["eval_strategy"] = "steps"
        elif "evaluation_strategy" in sft_parameters:
            sft_values["evaluation_strategy"] = "steps"
        if "max_length" in sft_parameters:
            sft_values["max_length"] = config.max_sequence_length
        elif "max_seq_length" in sft_parameters:
            sft_values["max_seq_length"] = config.max_sequence_length
        if "dataset_kwargs" in sft_parameters:
            sft_values["dataset_kwargs"] = {"skip_prepare_dataset": False}
        args = SFTConfig(**sft_values)
        trainer_parameters = inspect.signature(SFTTrainer).parameters
        trainer_values: dict[str, Any] = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "eval_dataset": validation_dataset,
            "peft_config": lora_config,
            "callbacks": [_TrainingCallback()],
        }
        if "processing_class" in trainer_parameters:
            trainer_values["processing_class"] = tokenizer
        elif "tokenizer" in trainer_parameters:
            trainer_values["tokenizer"] = tokenizer
        trainer = SFTTrainer(**trainer_values)
        train_output = trainer.train()
        if cancel_event.is_set():
            trainer.save_state()
            status = "cancelled"
        else:
            status = "completed"
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        adapter_config = adapter_dir / "adapter_config.json"
        adapter_weights = tuple(adapter_dir.glob("adapter_model.*"))
        if not adapter_config.is_file() or not adapter_weights:
            raise FineTuningError("Trainer returned without producing a verifiable LoRA adapter.")
        metrics = dict(getattr(train_output, "metrics", {}) or {})
        try:
            evaluation = trainer.evaluate()
        except Exception as error:
            raise FineTuningError(
                f"Training checkpoint and adapter were saved, but validation evaluation failed: {error}"
            ) from error
        if "eval_loss" not in evaluation:
            raise FineTuningError(
                "Training checkpoint and adapter were saved, but validation returned no eval_loss."
            )
        metrics["validation"] = evaluation
        metrics["elapsed_seconds"] = max(0.0, time.monotonic() - started)
        return status, metrics


def _records_for_training(
    builder: TDIDDatasetBuilder,
    split_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_ids = split_manifest.get("record_ids")
    if not isinstance(split_ids, Mapping):
        raise FineTuningError("TDID version is missing its split manifest.")
    by_id = {str(record.get("record_id") or ""): record for record in builder.records}
    try:
        train = [by_id[str(record_id)] for record_id in split_ids["train"]]
        validation = [by_id[str(record_id)] for record_id in split_ids["validation"]]
    except (KeyError, TypeError) as error:
        raise FineTuningError("TDID split refers to a missing record.") from error
    return train, validation


def _local_model_weight_bytes(model_dir: Path) -> int:
    supported = {".safetensors", ".bin"}
    total = 0
    try:
        for path in model_dir.rglob("*"):
            if path.is_file() and path.suffix.casefold() in supported:
                total += path.stat().st_size
    except OSError:
        return 0
    return total


def _training_source_group(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return ""
    value = metadata.get("source_group") or metadata.get("source_id") or metadata.get("source") or metadata.get("source_url")
    return " ".join(str(value or "").casefold().split())


def _memory_status() -> tuple[int | None, int | None]:
    if os.name == "nt":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys), int(status.ullAvailPhys)
        except (AttributeError, OSError):
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return total_pages * page_size, available_pages * page_size
    except (AttributeError, OSError, ValueError):
        return None, None


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(dict(manifest), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _hardware_dict(hardware: TrainingHardware) -> dict[str, Any]:
    return asdict(hardware)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
