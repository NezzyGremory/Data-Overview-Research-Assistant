"""Local manifest, integrity, import/export and explicit selection for adapters."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from uuid import uuid4


class ModelVersionError(RuntimeError):
    """Raised when a local model version is invalid or incompatible."""


@dataclass(frozen=True, slots=True)
class ManagedModelVersion:
    version_id: str
    name: str
    created_at: str
    base_model_path: str
    base_model_fingerprint: str
    adapter_path: str
    adapter_sha256: str
    dataset_version_id: str | None
    license_name: str
    training_config: dict[str, Any]
    evaluation_metrics: dict[str, Any]
    selected: bool


class ModelVersionStore:
    """Store local LoRA adapters and their provenance without activating them."""

    MAX_ADAPTER_BYTES = 50 * 1024**3
    MAX_ARCHIVE_FILES = 2_000

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        if storage_dir is None:
            from app.core.config import get_data_dir

            storage_dir = get_data_dir() / "tanya_dora" / "models"
        self.storage_dir = Path(storage_dir).expanduser().resolve()
        self.adapter_dir = self.storage_dir / "adapters"
        self.staging_dir = self.storage_dir / "staging"
        self.db_path = self.storage_dir / "model_versions.db"
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    base_model_path TEXT NOT NULL,
                    base_model_fingerprint TEXT NOT NULL,
                    adapter_path TEXT NOT NULL,
                    adapter_sha256 TEXT NOT NULL,
                    dataset_version_id TEXT,
                    license_name TEXT NOT NULL,
                    training_config_json TEXT NOT NULL,
                    evaluation_metrics_json TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS model_settings (
                    setting_key TEXT PRIMARY KEY,
                    setting_value TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS model_activation_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    changed_at TEXT NOT NULL,
                    previous_selection_json TEXT,
                    new_selection_json TEXT
                )"""
            )

    def register_adapter(
        self,
        name: str,
        adapter_path: str | Path,
        base_model_path: str | Path,
        *,
        dataset_version_id: str | None = None,
        license_name: str = "Not recorded",
        training_config: Mapping[str, Any] | None = None,
        evaluation_metrics: Mapping[str, Any] | None = None,
    ) -> ManagedModelVersion:
        normalized_name = " ".join(str(name or "").split())
        if not normalized_name or len(normalized_name) > 100:
            raise ValueError("Model version name must contain 1-100 characters.")
        source_adapter = Path(adapter_path).expanduser().resolve()
        base_path = Path(base_model_path).expanduser().resolve()
        self._validate_adapter_directory(source_adapter)
        base_fingerprint = _base_model_fingerprint(base_path)
        adapter_config = _read_json(source_adapter / "adapter_config.json")
        if not isinstance(adapter_config, Mapping):
            raise ModelVersionError("Adapter configuration must be a JSON object.")
        version_id = uuid4().hex
        target = self.adapter_dir / version_id
        staging = self.staging_dir / f"register-{version_id}"
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            shutil.copytree(source_adapter, staging, symlinks=False)
            # The name is reserved for the transport manifest written by ZIP exports.
            (staging / "tanya_dora_model.json").unlink(missing_ok=True)
            self._validate_adapter_directory(staging)
            digest = _directory_sha256(staging)
            staging.replace(target)
            relative_path = target.relative_to(self.storage_dir).as_posix()
            with self._connection() as connection:
                connection.execute(
                    """INSERT INTO model_versions
                       (version_id, name, created_at, base_model_path, base_model_fingerprint,
                        adapter_path, adapter_sha256, dataset_version_id, license_name,
                        training_config_json, evaluation_metrics_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        version_id,
                        normalized_name,
                        created_at,
                        str(base_path),
                        base_fingerprint,
                        relative_path,
                        digest,
                        str(dataset_version_id) if dataset_version_id else None,
                        " ".join(str(license_name or "Not recorded").split())[:200],
                        _safe_json(dict(training_config or {})),
                        _safe_json(dict(evaluation_metrics or {})),
                    ),
                )
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            raise
        return self.get_version(version_id)

    def list_versions(self) -> tuple[ManagedModelVersion, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM model_versions ORDER BY created_at DESC, version_id"
            ).fetchall()
            active_row = connection.execute(
                "SELECT setting_value FROM model_settings WHERE setting_key = 'selected_version'"
            ).fetchone()
        selected = str(active_row["setting_value"]) if active_row else None
        return tuple(_version_from_row(row, row["version_id"] == selected) for row in rows)

    def get_version(self, version_id: str) -> ManagedModelVersion:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM model_versions WHERE version_id = ?", (str(version_id),)
            ).fetchone()
            active_row = connection.execute(
                "SELECT setting_value FROM model_settings WHERE setting_key = 'selected_version'"
            ).fetchone()
        if row is None:
            raise ModelVersionError("Model version was not found.")
        selected = str(active_row["setting_value"]) if active_row else None
        return _version_from_row(row, row["version_id"] == selected)

    def verify_version(self, version_id: str) -> ManagedModelVersion:
        version = self.get_version(version_id)
        adapter_path = self._resolve_adapter_path(version.adapter_path)
        self._validate_adapter_directory(adapter_path)
        if _directory_sha256(adapter_path) != version.adapter_sha256:
            raise ModelVersionError("Adapter files do not match the registered SHA-256 manifest.")
        if _base_model_fingerprint(Path(version.base_model_path)) != version.base_model_fingerprint:
            raise ModelVersionError("The base model files or configuration have changed since registration.")
        return version

    def select_version(self, version_id: str, *, user_confirmed: bool = False) -> ManagedModelVersion:
        if not user_confirmed:
            raise ModelVersionError("Explicit user confirmation is required to select a model version.")
        version = self.verify_version(version_id)
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO model_settings(setting_key, setting_value) VALUES ('selected_version', ?)",
                (version.version_id,),
            )
        return self.get_version(version_id)

    def clear_selection(self, *, user_confirmed: bool = False) -> bool:
        if not user_confirmed:
            raise ModelVersionError("Explicit user confirmation is required to clear the selection.")
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM model_settings WHERE setting_key = 'selected_version'"
            )
        return cursor.rowcount > 0

    def get_active_runtime_model(self) -> dict[str, Any] | None:
        """Return the model explicitly activated for Tanya DORA chat, if any."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT setting_value FROM model_settings WHERE setting_key = 'active_runtime_json'"
            ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(str(row["setting_value"]))
        except (TypeError, ValueError):
            return None
        return dict(value) if isinstance(value, Mapping) else None

    def set_active_runtime_model(
        self,
        model_name: str,
        *,
        model_kind: str = "ollama",
        version_id: str | None = None,
        user_confirmed: bool = False,
    ) -> dict[str, Any]:
        """Persist an explicit runtime switch and retain the previous selection."""
        if not user_confirmed:
            raise ModelVersionError("Explicit user confirmation is required to switch the active chat model.")
        normalized_name = str(model_name or "").strip()
        normalized_kind = str(model_kind or "").strip().casefold()
        if not normalized_name or len(normalized_name) > 200:
            raise ValueError("Runtime model name must contain 1-200 characters.")
        if normalized_kind not in {"ollama", "ollama_adapter"}:
            raise ValueError("Runtime model kind must be 'ollama' or 'ollama_adapter'.")
        normalized_version = str(version_id) if version_id else None
        if normalized_kind == "ollama_adapter" and not normalized_version:
            raise ValueError("An adapter-backed Ollama model must reference its registered adapter version.")
        if normalized_version:
            self.verify_version(normalized_version)
        selection = {
            "model_name": normalized_name,
            "model_kind": normalized_kind,
            "version_id": normalized_version,
            "changed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        previous = self.get_active_runtime_model()
        if previous and all(previous.get(key) == selection[key] for key in ("model_name", "model_kind", "version_id")):
            return previous
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO model_activation_history(changed_at, previous_selection_json, new_selection_json) "
                "VALUES (?, ?, ?)",
                (
                    selection["changed_at"],
                    _safe_json(previous) if previous else None,
                    _safe_json(selection),
                ),
            )
            connection.execute(
                "INSERT OR REPLACE INTO model_settings(setting_key, setting_value) "
                "VALUES ('active_runtime_json', ?)",
                (_safe_json(selection),),
            )
        return selection

    def clear_active_runtime_model(self, *, user_confirmed: bool = False) -> bool:
        """Deactivate the explicit runtime choice; the chat page then uses its local selection."""
        if not user_confirmed:
            raise ModelVersionError("Explicit user confirmation is required to deactivate the active chat model.")
        previous = self.get_active_runtime_model()
        if previous is None:
            return False
        changed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO model_activation_history(changed_at, previous_selection_json, new_selection_json) "
                "VALUES (?, ?, NULL)",
                (changed_at, _safe_json(previous)),
            )
            connection.execute(
                "DELETE FROM model_settings WHERE setting_key = 'active_runtime_json'"
            )
        return True

    def list_activation_history(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        safe_limit = max(1, min(int(limit), 500))
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT history_id, changed_at, previous_selection_json, new_selection_json "
                "FROM model_activation_history ORDER BY history_id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return tuple(
            {
                "history_id": int(row["history_id"]),
                "changed_at": str(row["changed_at"]),
                "previous": _json_object(row["previous_selection_json"] or "{}") or None,
                "new": _json_object(row["new_selection_json"] or "{}") or None,
            }
            for row in rows
        )

    def rollback_active_runtime_model(self, *, user_confirmed: bool = False) -> dict[str, Any]:
        """Restore the previous explicit model as a new, auditable activation event."""
        if not user_confirmed:
            raise ModelVersionError("Explicit user confirmation is required to roll back the active chat model.")
        history = self.list_activation_history(limit=1)
        if not history or not history[0].get("previous"):
            raise ModelVersionError("There is no previous runtime model to restore.")
        previous = history[0]["previous"]
        return self.set_active_runtime_model(
            str(previous.get("model_name") or ""),
            model_kind=str(previous.get("model_kind") or "ollama"),
            version_id=str(previous.get("version_id")) if previous.get("version_id") else None,
            user_confirmed=True,
        )

    def record_evaluation(
        self,
        version_id: str,
        metrics: Mapping[str, Any],
    ) -> ManagedModelVersion:
        self.verify_version(version_id)
        with self._connection() as connection:
            connection.execute(
                "UPDATE model_versions SET evaluation_metrics_json = ? WHERE version_id = ?",
                (_safe_json(dict(metrics)), str(version_id)),
            )
        return self.get_version(version_id)

    def export_adapter(self, version_id: str, destination: str | Path) -> Path:
        version = self.verify_version(version_id)
        target = Path(destination).expanduser().resolve()
        if target.suffix.casefold() != ".zip":
            raise ValueError("Model export destination must use the .zip extension.")
        target.parent.mkdir(parents=True, exist_ok=True)
        adapter_path = self._resolve_adapter_path(version.adapter_path)
        manifest = {
            "format": "tanya-dora-adapter-v1",
            "version_id": version.version_id,
            "name": version.name,
            "base_model_path": version.base_model_path,
            "base_model_fingerprint": version.base_model_fingerprint,
            "adapter_sha256": version.adapter_sha256,
            "dataset_version_id": version.dataset_version_id,
            "license_name": version.license_name,
            "training_config": version.training_config,
            "evaluation_metrics": version.evaluation_metrics,
        }
        temporary = target.with_suffix(".tmp")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("tanya_dora_model.json", _safe_json(manifest, indent=2))
                for path in sorted(adapter_path.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(adapter_path).as_posix())
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return target

    def import_adapter_archive(
        self,
        archive_path: str | Path,
        base_model_path: str | Path,
        *,
        name: str | None = None,
    ) -> ManagedModelVersion:
        archive_path = Path(archive_path).expanduser().resolve()
        if not archive_path.is_file():
            raise ModelVersionError("Adapter archive does not exist.")
        temporary = self.staging_dir / f"import-{uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                infos = [item for item in archive.infolist() if not item.is_dir()]
                if len(infos) > self.MAX_ARCHIVE_FILES:
                    raise ModelVersionError("Adapter archive contains too many files.")
                total_size = 0
                for item in infos:
                    pure = PurePosixPath(item.filename)
                    if "\\" in item.filename or ":" in item.filename or pure.is_absolute() or ".." in pure.parts or "" in pure.parts:
                        raise ModelVersionError("Adapter archive contains an unsafe path.")
                    if item.flag_bits & 0x1:
                        raise ModelVersionError("Encrypted adapter archives are not supported.")
                    if (item.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ModelVersionError("Symbolic links are not allowed in adapter archives.")
                    total_size += int(item.file_size)
                    if total_size > self.MAX_ADAPTER_BYTES:
                        raise ModelVersionError("Adapter archive exceeds the size limit.")
                    destination = (temporary / Path(*pure.parts)).resolve()
                    if temporary.resolve() not in destination.parents:
                        raise ModelVersionError("Adapter archive path escaped the import directory.")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item, "r") as source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
            metadata_path = temporary / "tanya_dora_model.json"
            exported = _read_json(metadata_path) if metadata_path.is_file() else {}
            adapter_root = temporary
            if not (adapter_root / "adapter_config.json").is_file():
                nested_configs = list(temporary.rglob("adapter_config.json"))
                if len(nested_configs) != 1:
                    raise ModelVersionError("Archive must contain exactly one adapter_config.json file.")
                adapter_root = nested_configs[0].parent
            self._validate_adapter_directory(adapter_root)
            supplied_base = Path(base_model_path).expanduser().resolve()
            supplied_fingerprint = _base_model_fingerprint(supplied_base)
            if isinstance(exported, Mapping) and exported.get("base_model_fingerprint"):
                if str(exported["base_model_fingerprint"]) != supplied_fingerprint:
                    raise ModelVersionError("The imported adapter was registered for a different base model.")
            adapter_digest = _directory_sha256(
                adapter_root,
                excluded_paths={"tanya_dora_model.json"} if adapter_root == temporary else set(),
            )
            if isinstance(exported, Mapping) and exported.get("adapter_sha256"):
                if str(exported["adapter_sha256"]) != adapter_digest:
                    raise ModelVersionError("Adapter archive integrity check failed.")
            return self.register_adapter(
                name or (str(exported.get("name") or "Imported adapter") if isinstance(exported, Mapping) else "Imported adapter"),
                adapter_root,
                supplied_base,
                dataset_version_id=(str(exported["dataset_version_id"]) if isinstance(exported, Mapping) and exported.get("dataset_version_id") else None),
                license_name=(str(exported.get("license_name") or "Not recorded") if isinstance(exported, Mapping) else "Not recorded"),
                training_config=(exported.get("training_config", {}) if isinstance(exported, Mapping) and isinstance(exported.get("training_config", {}), Mapping) else {}),
                evaluation_metrics=(exported.get("evaluation_metrics", {}) if isinstance(exported, Mapping) and isinstance(exported.get("evaluation_metrics", {}), Mapping) else {}),
            )
        except zipfile.BadZipFile as error:
            raise ModelVersionError("Adapter archive is not a valid ZIP file.") from error
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def delete_version(self, version_id: str) -> bool:
        version = self.get_version(version_id)
        if version.selected:
            raise ModelVersionError("Clear the selected version before deleting it.")
        with self._connection() as connection:
            connection.execute("DELETE FROM model_versions WHERE version_id = ?", (str(version_id),))
        shutil.rmtree(self._resolve_adapter_path(version.adapter_path), ignore_errors=True)
        return True

    def _validate_adapter_directory(self, path: Path) -> None:
        if not path.is_dir():
            raise ModelVersionError("Adapter path must be an existing local directory.")
        _reject_symlinks(path)
        if not (path / "adapter_config.json").is_file():
            raise ModelVersionError("Adapter directory is missing adapter_config.json.")
        if not tuple(path.glob("adapter_model.*")):
            raise ModelVersionError("Adapter directory is missing adapter model weights.")
        total_size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        if total_size > self.MAX_ADAPTER_BYTES:
            raise ModelVersionError("Adapter exceeds the configured local storage limit.")

    def _resolve_adapter_path(self, relative_path: str) -> Path:
        candidate = (self.storage_dir / relative_path).resolve()
        root = self.adapter_dir.resolve()
        if root not in candidate.parents:
            raise ModelVersionError("Stored adapter path is invalid.")
        return candidate


def _base_model_fingerprint(base_path: Path) -> str:
    if not base_path.is_dir():
        raise ModelVersionError("Base model must be available in a local directory.")
    config_path = base_path / "config.json"
    if not config_path.is_file():
        raise ModelVersionError("Base model directory is missing config.json.")
    digest = hashlib.sha256()
    digest.update(config_path.read_bytes())
    weight_files = sorted(
        item for item in base_path.rglob("*")
        if item.is_file() and item.suffix.casefold() in {".safetensors", ".bin"}
    )
    if not weight_files:
        raise ModelVersionError("Base model directory contains no local model weights.")
    for item in weight_files:
        digest.update(item.relative_to(base_path).as_posix().encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("ascii"))
    return digest.hexdigest()


def _directory_sha256(path: Path, *, excluded_paths: set[str] | None = None) -> str:
    digest = hashlib.sha256()
    excluded = excluded_paths or set()
    for item in sorted(child for child in path.rglob("*") if child.is_file()):
        relative_path = item.relative_to(path).as_posix()
        if relative_path in excluded:
            continue
        digest.update(relative_path.encode("utf-8"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _reject_symlinks(path: Path) -> None:
    if path.is_symlink() or any(item.is_symlink() for item in path.rglob("*")):
        raise ModelVersionError("Symbolic links are not allowed in model directories.")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ModelVersionError(f"Model metadata could not be read: {error}") from error


def _safe_json(value: Any, *, indent: int | None = None) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=indent)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Model version metadata must be JSON-compatible: {error}") from error


def _version_from_row(row: sqlite3.Row, selected: bool) -> ManagedModelVersion:
    return ManagedModelVersion(
        version_id=str(row["version_id"]),
        name=str(row["name"]),
        created_at=str(row["created_at"]),
        base_model_path=str(row["base_model_path"]),
        base_model_fingerprint=str(row["base_model_fingerprint"]),
        adapter_path=str(row["adapter_path"]),
        adapter_sha256=str(row["adapter_sha256"]),
        dataset_version_id=str(row["dataset_version_id"]) if row["dataset_version_id"] else None,
        license_name=str(row["license_name"]),
        training_config=_json_object(row["training_config_json"]),
        evaluation_metrics=_json_object(row["evaluation_metrics_json"]),
        selected=selected,
    )


def _json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}
