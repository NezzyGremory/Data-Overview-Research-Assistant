"""Registry for models already installed in a local Ollama instance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class LocalModel:
    """Small, display-safe subset of Ollama's local model metadata."""

    name: str
    size_bytes: int
    modified_at: str | None = None
    family: str | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LocalModel":
        name = str(payload.get("model") or payload.get("name") or "").strip()
        if not name:
            raise ValueError("Ollama model metadata is missing its model name.")

        try:
            size_bytes = max(0, int(payload.get("size") or 0))
        except (TypeError, ValueError, OverflowError):
            size_bytes = 0

        details = payload.get("details")
        if not isinstance(details, Mapping):
            details = {}

        return cls(
            name=name,
            size_bytes=size_bytes,
            modified_at=_optional_text(payload.get("modified_at")),
            family=_optional_text(details.get("family")),
            parameter_size=_optional_text(details.get("parameter_size")),
            quantization_level=_optional_text(details.get("quantization_level")),
        )

    @property
    def size_label(self) -> str:
        if self.size_bytes <= 0:
            return "Unknown size"

        value = float(self.size_bytes)
        units = ("B", "KiB", "MiB", "GiB", "TiB")
        unit_index = 0
        while value >= 1024 and unit_index < len(units) - 1:
            value /= 1024
            unit_index += 1
        return f"{value:.1f} {units[unit_index]}"


class ModelRegistry:
    """Keeps the latest inventory returned by Ollama's local tags endpoint."""

    def __init__(self) -> None:
        self._models: dict[str, LocalModel] = {}

    def replace_from_payload(self, payload: Mapping[str, Any]) -> tuple[LocalModel, ...]:
        entries = payload.get("models")
        if not isinstance(entries, list):
            raise ValueError("Ollama model inventory has an invalid format.")

        models: dict[str, LocalModel] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            try:
                model = LocalModel.from_payload(entry)
            except ValueError:
                continue
            models[model.name] = model

        self._models = models
        return self.list_models()

    def list_models(self) -> tuple[LocalModel, ...]:
        return tuple(self._models.values())

    def get(self, name: str) -> LocalModel | None:
        return self._models.get(str(name or "").strip())


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
