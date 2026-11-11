"""Tanya Dora Instruction Dataset (TDID) validation, splitting and versioning."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


TDID_CATEGORIES = frozenset(
    {
        "data_science",
        "statistics",
        "machine_learning",
        "deep_learning",
        "computer_vision",
        "programming",
        "research_methodology",
        "research_gap",
        "dora_knowledge",
        "response_behavior",
    }
)
TDID_DIFFICULTIES = frozenset({"beginner", "intermediate", "advanced"})
TDID_INSTRUCTION_TYPES = frozenset(
    {
        "knowledge",
        "reasoning",
        "coding",
        "research",
        "dora_context",
        "behavioral",
        "uncertainty",
        "tool_calling",
    }
)
TDID_ROLES = frozenset({"system", "user", "assistant", "tool"})


class TDIDError(RuntimeError):
    """Raised when a TDID dataset cannot be safely prepared or stored."""


@dataclass(frozen=True, slots=True)
class TDIDIssue:
    record_index: int
    field: str
    message: str


@dataclass(frozen=True, slots=True)
class TDIDSplit:
    train: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    test: tuple[dict[str, Any], ...]
    excluded_unverified: tuple[str, ...]
    excluded_duplicates: tuple[str, ...]
    seed: int
    source_grouped: bool

    def manifest(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "source_grouped": self.source_grouped,
            "counts": {
                "train": len(self.train),
                "validation": len(self.validation),
                "test": len(self.test),
            },
            "record_ids": {
                "train": [_record_id(record) for record in self.train],
                "validation": [_record_id(record) for record in self.validation],
                "test": [_record_id(record) for record in self.test],
            },
            "excluded_unverified": list(self.excluded_unverified),
            "excluded_duplicates": list(self.excluded_duplicates),
        }


@dataclass(frozen=True, slots=True)
class TDIDDatasetVersion:
    version_id: str
    name: str
    created_at: str
    status: str
    record_count: int
    verified_count: int
    sha256: str
    snapshot_path: str
    split_manifest: dict[str, Any]


class TDIDDatasetBuilder:
    """Editable in-memory dataset with explicit validation and leakage-safe splits."""

    MAX_RECORDS = 100_000
    MAX_MESSAGE_CHARS = 40_000

    def __init__(self, records: Iterable[Mapping[str, Any]] = ()) -> None:
        self.records = []
        for record in records:
            normalized = dict(record)
            if not str(normalized.get("record_id") or "").strip():
                normalized["record_id"] = uuid4().hex
            self.records.append(normalized)
        if len(self.records) > self.MAX_RECORDS:
            raise ValueError(f"TDID cannot contain more than {self.MAX_RECORDS:,} examples.")
        self.import_issues: list[TDIDIssue] = []

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TDIDDatasetBuilder":
        source = Path(path).expanduser()
        if not source.is_file():
            raise TDIDError("JSONL source file does not exist.")
        builder = cls()
        try:
            with source.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    if len(builder.records) >= cls.MAX_RECORDS:
                        builder.import_issues.append(
                            TDIDIssue(line_number, "record", "The record limit was reached.")
                        )
                        break
                    try:
                        record = json.loads(line)
                    except (TypeError, ValueError) as error:
                        builder.import_issues.append(
                            TDIDIssue(line_number, "json", f"Invalid JSON: {error}")
                        )
                        continue
                    if not isinstance(record, Mapping):
                        builder.import_issues.append(
                            TDIDIssue(line_number, "record", "Each JSONL line must be an object.")
                        )
                        continue
                    normalized = dict(record)
                    if not str(normalized.get("record_id") or "").strip():
                        normalized["record_id"] = uuid4().hex
                    builder.records.append(normalized)
        except OSError as error:
            raise TDIDError(f"JSONL source could not be read: {error}") from error
        return builder

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        field_map: Mapping[str, str],
        encoding: str = "utf-8-sig",
    ) -> "TDIDDatasetBuilder":
        source = Path(path).expanduser()
        if not source.is_file():
            raise TDIDError("CSV source file does not exist.")
        mapping = {str(key): str(value) for key, value in field_map.items() if value}
        if "messages" not in mapping and not {"user", "assistant"}.issubset(mapping):
            raise ValueError("Map a JSON messages column or both user and assistant columns.")
        builder = cls()
        try:
            with source.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise TDIDError("CSV must have a header row.")
                missing_columns = set(mapping.values()) - set(reader.fieldnames)
                if missing_columns:
                    raise ValueError(
                        "CSV mapping references missing columns: "
                        + ", ".join(sorted(missing_columns))
                    )
                for line_number, row in enumerate(reader, start=2):
                    if len(builder.records) >= cls.MAX_RECORDS:
                        builder.import_issues.append(
                            TDIDIssue(line_number, "record", "The record limit was reached.")
                        )
                        break
                    try:
                        if "messages" in mapping:
                            messages = json.loads(row[mapping["messages"]] or "")
                        else:
                            messages = []
                            if mapping.get("system") and row.get(mapping["system"], "").strip():
                                messages.append({"role": "system", "content": row[mapping["system"]]})
                            messages.extend(
                                (
                                    {"role": "user", "content": row[mapping["user"]]},
                                    {"role": "assistant", "content": row[mapping["assistant"]]},
                                )
                            )
                        verified_value = _csv_value(row, mapping, "verified").casefold()
                        metadata = {
                            "category": _csv_value(row, mapping, "category"),
                            "difficulty": _csv_value(row, mapping, "difficulty") or "intermediate",
                            "instruction_type": _csv_value(row, mapping, "instruction_type") or "knowledge",
                            "verified": verified_value in {"true", "1", "yes", "verified"},
                        }
                        for key in ("source", "source_url", "source_group", "source_id"):
                            value = _csv_value(row, mapping, key)
                            if value:
                                metadata[key] = value
                        record_id = _csv_value(row, mapping, "record_id") or uuid4().hex
                        builder.records.append({
                            "record_id": record_id,
                            "messages": messages,
                            "metadata": metadata,
                        })
                    except (TypeError, ValueError, KeyError) as error:
                        builder.import_issues.append(
                            TDIDIssue(line_number, "csv_mapping", str(error))
                        )
        except (OSError, UnicodeError) as error:
            raise TDIDError(f"CSV source could not be read: {error}") from error
        return builder

    def validate(self) -> tuple[TDIDIssue, ...]:
        issues = list(self.import_issues)
        seen_ids: set[str] = set()
        for index, record in enumerate(self.records, start=1):
            record_id = _record_id(record)
            if not record_id:
                issues.append(TDIDIssue(index, "record_id", "Record ID cannot be empty."))
            elif record_id in seen_ids:
                issues.append(TDIDIssue(index, "record_id", "Record ID is duplicated."))
            seen_ids.add(record_id)
            issues.extend(validate_tdid_record(record, record_index=index))
        return tuple(issues)

    def add_record(self, record: Mapping[str, Any]) -> int:
        if len(self.records) >= self.MAX_RECORDS:
            raise ValueError(f"TDID cannot contain more than {self.MAX_RECORDS:,} examples.")
        normalized = dict(record)
        if not str(normalized.get("record_id") or "").strip():
            normalized["record_id"] = uuid4().hex
        if any(_record_id(item) == str(normalized["record_id"]) for item in self.records):
            raise ValueError("TDID record ID is already in use.")
        self.records.append(normalized)
        issues = validate_tdid_record(self.records[-1], record_index=len(self.records))
        if issues:
            self.records.pop()
            raise ValueError("; ".join(issue.message for issue in issues))
        return len(self.records) - 1

    def edit_record(self, index: int, record: Mapping[str, Any]) -> None:
        if index < 0 or index >= len(self.records):
            raise IndexError("TDID record index is out of range.")
        normalized = dict(record)
        if not str(normalized.get("record_id") or "").strip():
            normalized["record_id"] = _record_id(self.records[index]) or uuid4().hex
        if any(
            other_index != index and _record_id(item) == str(normalized["record_id"])
            for other_index, item in enumerate(self.records)
        ):
            raise ValueError("TDID record ID is already in use.")
        issues = validate_tdid_record(normalized, record_index=index + 1)
        if issues:
            raise ValueError("; ".join(issue.message for issue in issues))
        self.records[index] = normalized

    def delete_record(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.records):
            raise IndexError("TDID record index is out of range.")
        return self.records.pop(index)

    def duplicate_groups(self) -> tuple[tuple[int, ...], ...]:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            groups[_content_fingerprint(record)].append(index)
        return tuple(tuple(indices) for indices in groups.values() if len(indices) > 1)

    def near_duplicate_prompt_groups(self) -> tuple[tuple[int, ...], ...]:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            prompt = _normalized_prompt(record)
            if prompt:
                groups[prompt].append(index)
        return tuple(tuple(indices) for indices in groups.values() if len(indices) > 1)

    def category_counts(self) -> dict[str, int]:
        counts = Counter()
        for record in self.records:
            metadata = record.get("metadata")
            category = metadata.get("category") if isinstance(metadata, Mapping) else None
            counts[str(category or "invalid_or_missing")] += 1
        return dict(counts)

    def difficulty_counts(self) -> dict[str, int]:
        counts = Counter()
        for record in self.records:
            metadata = record.get("metadata")
            difficulty = metadata.get("difficulty") if isinstance(metadata, Mapping) else None
            counts[str(difficulty or "invalid_or_missing")] += 1
        return dict(counts)

    def verification_counts(self) -> dict[str, int]:
        counts = Counter()
        for record in self.records:
            metadata = record.get("metadata")
            verified = metadata.get("verified") if isinstance(metadata, Mapping) else None
            counts["verified" if verified is True else "unverified"] += 1
        return dict(counts)

    def split(
        self,
        *,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        group_by_source: bool = True,
        remove_duplicates: bool = True,
    ) -> TDIDSplit:
        ratios = (float(train_ratio), float(validation_ratio), float(test_ratio))
        if any(not 0 < value < 1 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("Train, validation and test ratios must be positive and sum to 1.")
        # Import diagnostics carry CSV/JSONL source line numbers. Only validate
        # the in-memory records here so a malformed earlier source line cannot
        # accidentally exclude a different, valid record from the split.
        invalid_records = {
            index
            for index, record in enumerate(self.records)
            if validate_tdid_record(record, record_index=index + 1)
        }
        seen_ids: set[str] = set()
        for index, record in enumerate(self.records):
            record_id = _record_id(record)
            if not record_id or record_id in seen_ids:
                invalid_records.add(index)
            seen_ids.add(record_id)
        duplicate_indexes = {
            index for group in self.duplicate_groups() for index in group[1:]
        } if remove_duplicates else set()

        eligible: list[tuple[int, dict[str, Any]]] = []
        excluded_unverified = []
        excluded_duplicates = []
        for index, record in enumerate(self.records):
            record_id = _record_id(record)
            if index in invalid_records:
                continue
            metadata = record.get("metadata", {})
            if not isinstance(metadata, Mapping) or metadata.get("verified") is not True:
                excluded_unverified.append(record_id)
                continue
            if index in duplicate_indexes:
                excluded_duplicates.append(record_id)
                continue
            eligible.append((index, record))

        if len(eligible) < 3:
            raise ValueError("At least three verified, valid, nonduplicate examples are required for a split.")

        union = _UnionFind(index for index, _record in eligible)
        prompt_roots: dict[str, int] = {}
        for index, record in eligible:
            prompt = _normalized_prompt(record)
            if prompt:
                if prompt in prompt_roots:
                    union.union(index, prompt_roots[prompt])
                else:
                    prompt_roots[prompt] = index
        if group_by_source:
            source_roots: dict[str, int] = {}
            for index, record in eligible:
                source_group = _source_group(record)
                if source_group:
                    if source_group in source_roots:
                        union.union(index, source_roots[source_group])
                    else:
                        source_roots[source_group] = index
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for index, record in eligible:
            grouped[union.find(index)].append(record)
        groups = list(grouped.values())
        if len(groups) < 3:
            raise ValueError(
                "At least three independent source groups are required to keep train/validation/test isolated."
            )

        rng = random.Random(int(seed))
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        targets = [len(eligible) * ratio for ratio in ratios]
        assigned: list[list[dict[str, Any]]] = [[], [], []]
        for group_index, group in enumerate(groups):
            if group_index < 3:
                assigned[group_index].extend(group)
                continue
            deficits = [targets[index] - len(assigned[index]) for index in range(3)]
            best = max(range(3), key=lambda index: deficits[index])
            assigned[best].extend(group)

        return TDIDSplit(
            train=tuple(assigned[0]),
            validation=tuple(assigned[1]),
            test=tuple(assigned[2]),
            excluded_unverified=tuple(excluded_unverified),
            excluded_duplicates=tuple(excluded_duplicates),
            seed=int(seed),
            source_grouped=bool(group_by_source),
        )

    def export_jsonl(
        self,
        path: str | Path,
        records: Sequence[Mapping[str, Any]] | None = None,
    ) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        selected = self.records if records is None else records
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for record in selected:
                handle.write(json.dumps(dict(record), ensure_ascii=False, allow_nan=False))
                handle.write("\n")
        return destination


def validate_tdid_record(
    record: Mapping[str, Any],
    *,
    record_index: int = 0,
) -> tuple[TDIDIssue, ...]:
    issues: list[TDIDIssue] = []
    if not isinstance(record, Mapping):
        return (TDIDIssue(record_index, "record", "Example must be an object."),)
    messages = record.get("messages")
    metadata = record.get("metadata")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or len(messages) < 2:
        issues.append(TDIDIssue(record_index, "messages", "Provide at least a user and assistant message."))
        messages = ()
    has_user = has_assistant = False
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            issues.append(TDIDIssue(record_index, f"messages[{message_index}]", "Message must be an object."))
            continue
        role = str(message.get("role") or "").strip()
        content = message.get("content")
        if role not in TDID_ROLES:
            issues.append(TDIDIssue(record_index, f"messages[{message_index}].role", "Unsupported message role."))
        has_user |= role == "user"
        has_assistant |= role == "assistant"
        if not isinstance(content, str) or not content.strip():
            issues.append(TDIDIssue(record_index, f"messages[{message_index}].content", "Message content cannot be empty."))
        elif len(content) > TDIDDatasetBuilder.MAX_MESSAGE_CHARS:
            issues.append(TDIDIssue(record_index, f"messages[{message_index}].content", "Message exceeds the 40,000-character limit."))
        if "tool_calls" in message and role != "assistant":
            issues.append(TDIDIssue(record_index, f"messages[{message_index}].tool_calls", "Tool calls must be attached to an assistant message."))
    if not has_user:
        issues.append(TDIDIssue(record_index, "messages", "At least one user message is required."))
    if not has_assistant:
        issues.append(TDIDIssue(record_index, "messages", "At least one assistant message is required."))
    if messages and isinstance(messages[-1], Mapping) and messages[-1].get("role") != "assistant":
        issues.append(TDIDIssue(record_index, "messages", "The last message must be from the assistant."))

    if not isinstance(metadata, Mapping):
        issues.append(TDIDIssue(record_index, "metadata", "Metadata must be an object."))
        return tuple(issues)
    category = str(metadata.get("category") or "").strip().casefold()
    if category not in TDID_CATEGORIES:
        issues.append(TDIDIssue(record_index, "metadata.category", "Unknown TDID category."))
    difficulty = str(metadata.get("difficulty") or "").strip().casefold()
    if difficulty not in TDID_DIFFICULTIES:
        issues.append(TDIDIssue(record_index, "metadata.difficulty", "Difficulty must be beginner, intermediate or advanced."))
    instruction_type = str(metadata.get("instruction_type") or "").strip().casefold()
    if instruction_type not in TDID_INSTRUCTION_TYPES:
        issues.append(TDIDIssue(record_index, "metadata.instruction_type", "Unknown instruction type."))
    if not isinstance(metadata.get("verified"), bool):
        issues.append(TDIDIssue(record_index, "metadata.verified", "Verification status must be true or false."))
    if metadata.get("verified") is True and not (
        str(metadata.get("source") or "").strip()
        or str(metadata.get("source_url") or "").strip()
    ):
        issues.append(TDIDIssue(record_index, "metadata.source", "Verified examples must identify a source."))
    return tuple(issues)


class TDIDVersionStore:
    """Immutable JSONL snapshots with a local SQLite manifest."""

    def __init__(self, storage_dir: str | Path | None = None, *, db_path: str | Path | None = None) -> None:
        if storage_dir is None:
            from app.core.config import get_data_dir

            storage_dir = get_data_dir() / "tanya_dora" / "tdid"
        self.storage_dir = Path(storage_dir).expanduser().resolve()
        self.snapshot_dir = self.storage_dir / "versions"
        self.db_path = Path(db_path).expanduser().resolve() if db_path else self.storage_dir / "tdid.db"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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
                """CREATE TABLE IF NOT EXISTS tdid_versions (
                    version_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    verified_count INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    snapshot_path TEXT NOT NULL,
                    split_manifest_json TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_tdid_versions_created ON tdid_versions(created_at DESC)"
            )

    def save_version(
        self,
        name: str,
        builder: TDIDDatasetBuilder,
        *,
        status: str = "draft",
        split: TDIDSplit | None = None,
    ) -> TDIDDatasetVersion:
        normalized_name = " ".join(str(name or "").split())
        if not normalized_name or len(normalized_name) > 100:
            raise ValueError("Dataset version name must contain 1–100 characters.")
        normalized_status = str(status or "draft").casefold()
        if normalized_status not in {"draft", "final"}:
            raise ValueError("Dataset version status must be draft or final.")
        issues = builder.validate()
        if issues:
            raise TDIDError(f"Cannot save a dataset with {len(issues)} validation issue(s).")
        duplicates = builder.duplicate_groups()
        if duplicates:
            raise TDIDError(f"Resolve {len(duplicates)} duplicate group(s) before saving a version.")
        counts = builder.verification_counts()
        verified_count = counts.get("verified", 0)
        if normalized_status == "final" and verified_count != len(builder.records):
            raise TDIDError("A final TDID version can contain only verified examples.")
        split_data = split.manifest() if split is not None else {}
        if split is not None:
            _validate_split_records(builder.records, split)
        version_id = uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        snapshot_path = self.snapshot_dir / f"{version_id}.jsonl"
        builder.export_jsonl(snapshot_path)
        digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        relative_path = snapshot_path.relative_to(self.storage_dir).as_posix()
        try:
            with self._connection() as connection:
                connection.execute(
                    """INSERT INTO tdid_versions
                       (version_id, name, created_at, status, record_count, verified_count,
                        sha256, snapshot_path, split_manifest_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        version_id,
                        normalized_name,
                        created_at,
                        normalized_status,
                        len(builder.records),
                        verified_count,
                        digest,
                        relative_path,
                        json.dumps(split_data, ensure_ascii=False, allow_nan=False),
                    ),
                )
        except Exception:
            snapshot_path.unlink(missing_ok=True)
            raise
        return self.get_version(version_id)

    def list_versions(self) -> tuple[TDIDDatasetVersion, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM tdid_versions ORDER BY created_at DESC, version_id"
            ).fetchall()
        return tuple(_version_from_row(row) for row in rows)

    def get_version(self, version_id: str) -> TDIDDatasetVersion:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM tdid_versions WHERE version_id = ?", (str(version_id),)
            ).fetchone()
        if row is None:
            raise TDIDError("TDID version was not found.")
        return _version_from_row(row)

    def load_version(self, version_id: str) -> TDIDDatasetBuilder:
        version = self.get_version(version_id)
        path = self._resolve_snapshot(version.snapshot_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != version.sha256:
            raise TDIDError("TDID snapshot hash does not match its manifest.")
        return TDIDDatasetBuilder.from_jsonl(path)

    def delete_version(self, version_id: str) -> bool:
        version = None
        try:
            version = self.get_version(version_id)
        except TDIDError:
            return False
        with self._connection() as connection:
            connection.execute("DELETE FROM tdid_versions WHERE version_id = ?", (str(version_id),))
        self._resolve_snapshot(version.snapshot_path).unlink(missing_ok=True)
        return True

    def _resolve_snapshot(self, relative_path: str) -> Path:
        root = self.snapshot_dir.resolve()
        candidate = (self.storage_dir / relative_path).resolve()
        if root not in candidate.parents or candidate.suffix.lower() != ".jsonl":
            raise TDIDError("TDID snapshot path is invalid.")
        return candidate


def _csv_value(row: Mapping[str, Any], mapping: Mapping[str, str], key: str) -> str:
    column = mapping.get(key)
    return str(row.get(column) or "").strip() if column else ""


def _record_id(record: Mapping[str, Any]) -> str:
    return str(record.get("record_id") or "")


def _content_fingerprint(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    normalized = []
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            normalized.append(
                {
                    "role": str(message.get("role") or "").strip().casefold(),
                    "content": _normalize_text(str(message.get("content") or "")),
                    "tool_calls": message.get("tool_calls"),
                }
            )
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_prompt(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return ""
    return next(
        (
            _normalize_text(str(message.get("content") or ""))
            for message in messages
            if isinstance(message, Mapping) and message.get("role") == "user"
        ),
        "",
    )


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _source_group(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return ""
    value = metadata.get("source_group") or metadata.get("source_id") or metadata.get("source") or metadata.get("source_url")
    return _normalize_text(str(value or ""))


def _validate_split_records(records: Sequence[Mapping[str, Any]], split: TDIDSplit) -> None:
    parts = (split.train, split.validation, split.test)
    if any(not part for part in parts):
        raise TDIDError("Train, validation and test splits must all contain examples.")
    by_id = {_record_id(record): record for record in records}
    selected_ids = [
        _record_id(record)
        for part in parts
        for record in part
    ]
    if any(not item for item in selected_ids) or len(set(selected_ids)) != len(selected_ids):
        raise TDIDError("Split contains missing or cross-split duplicate record IDs.")
    if not set(selected_ids).issubset(by_id):
        raise TDIDError("Split contains records that are not in this TDID builder.")
    selected_records = [by_id[item] for item in selected_ids]
    for record in selected_records:
        metadata = record.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("verified") is not True
            or validate_tdid_record(record)
        ):
            raise TDIDError("Split can contain only valid, verified records.")
    duplicate_ids = set()
    for group in TDIDDatasetBuilder(records).duplicate_groups():
        group_ids = {_record_id(records[index]) for index in group}
        selected = group_ids & set(selected_ids)
        if len(selected) > 1:
            duplicate_ids.update(selected)
    if duplicate_ids:
        raise TDIDError("Resolve duplicate conversations across the saved data split.")
    if split.source_grouped:
        grouped_sources: list[set[str]] = []
        for part in parts:
            grouped_sources.append(
                {
                    _source_group(record)
                    for record in part
                    if _source_group(record)
                }
            )
        for left in range(3):
            for right in range(left + 1, 3):
                if grouped_sources[left] & grouped_sources[right]:
                    raise TDIDError("A TDID source group appears in more than one split.")


class _UnionFind:
    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            next_value = self.parent[value]
            self.parent[value] = root
            value = next_value
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _version_from_row(row: sqlite3.Row) -> TDIDDatasetVersion:
    try:
        split = json.loads(row["split_manifest_json"] or "{}")
    except (TypeError, ValueError):
        split = {}
    if not isinstance(split, dict):
        split = {}
    return TDIDDatasetVersion(
        version_id=str(row["version_id"]),
        name=str(row["name"]),
        created_at=str(row["created_at"]),
        status=str(row["status"]),
        record_count=int(row["record_count"]),
        verified_count=int(row["verified_count"]),
        sha256=str(row["sha256"]),
        snapshot_path=str(row["snapshot_path"]),
        split_manifest=split,
    )
