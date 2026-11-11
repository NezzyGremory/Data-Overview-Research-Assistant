"""Persistent local conversation history for Tanya DORA.

Conversation data lives in its own SQLite file under the Tanya DORA data
directory. It is intentionally separate from DORA project/session databases
and is never added to the Knowledge Base automatically.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4


class ConversationStoreError(RuntimeError):
    """Raised for invalid or unavailable local conversation operations."""


@dataclass(frozen=True, slots=True)
class Conversation:
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    model: str | None


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    message_id: int
    conversation_id: str
    role: str
    content: str
    created_at: str
    model: str | None
    sources: tuple[dict[str, Any], ...]
    status: str


class ConversationStore:
    """CRUD for local chat history with predictable ordering and isolation."""

    TITLE_LIMIT = 96
    MESSAGE_LIMIT = 200_000
    VALID_ROLES = frozenset({"user", "assistant", "system"})
    VALID_STATUSES = frozenset({"complete", "interrupted", "error"})

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        *,
        db_path: str | Path | None = None,
    ) -> None:
        if storage_dir is None:
            from app.core.config import get_data_dir

            storage_dir = get_data_dir() / "tanya_dora" / "conversations"
        self.storage_dir = Path(storage_dir).expanduser().resolve()
        self.db_path = (
            Path(db_path).expanduser().resolve()
            if db_path
            else self.storage_dir / "conversations.db"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    model TEXT
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    model TEXT,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'complete'
                        CHECK(status IN ('complete', 'interrupted', 'error'))
                );
                CREATE INDEX IF NOT EXISTS idx_tanya_messages_conversation_order
                    ON messages(conversation_id, message_id);
                CREATE INDEX IF NOT EXISTS idx_tanya_conversations_updated
                    ON conversations(updated_at DESC);
                """
            )

    def create_conversation(self, title: str = "Percakapan baru") -> Conversation:
        normalized_title = self._validate_title(title)
        conversation_id = uuid4().hex
        now = self._now()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO conversations
                   (conversation_id, title, created_at, updated_at, model)
                   VALUES (?, ?, ?, ?, NULL)""",
                (conversation_id, normalized_title, now, now),
            )
        return Conversation(conversation_id, normalized_title, now, now, None)

    def list_conversations(self) -> tuple[Conversation, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT conversation_id, title, created_at, updated_at, model
                   FROM conversations ORDER BY updated_at DESC, conversation_id"""
            ).fetchall()
        return tuple(self._conversation_from_row(row) for row in rows)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT conversation_id, title, created_at, updated_at, model
                   FROM conversations WHERE conversation_id = ?""",
                (str(conversation_id),),
            ).fetchone()
        return self._conversation_from_row(row) if row else None

    def get_messages(self, conversation_id: str) -> tuple[ConversationMessage, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT message_id, conversation_id, role, content, created_at,
                          model, sources_json, status
                   FROM messages WHERE conversation_id = ? ORDER BY message_id""",
                (str(conversation_id),),
            ).fetchall()
        return tuple(self._message_from_row(row) for row in rows)

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        model: str | None = None,
        sources: Sequence[Mapping[str, Any]] = (),
        status: str = "complete",
    ) -> ConversationMessage:
        conversation_id = str(conversation_id or "").strip()
        normalized_role = str(role or "").strip().casefold()
        normalized_content = str(content or "").strip()
        normalized_status = str(status or "complete").strip().casefold()
        if not conversation_id:
            raise ValueError("A conversation ID is required.")
        if normalized_role not in self.VALID_ROLES:
            raise ValueError("Unsupported conversation message role.")
        if not normalized_content:
            raise ValueError("Conversation message must contain text.")
        if len(normalized_content) > self.MESSAGE_LIMIT:
            raise ValueError("Conversation message exceeds the local storage limit.")
        if normalized_status not in self.VALID_STATUSES:
            raise ValueError("Unsupported conversation message status.")
        source_entries = []
        for source in sources:
            if isinstance(source, Mapping):
                source_entries.append(dict(source))
        sources_json = json.dumps(source_entries, ensure_ascii=False, allow_nan=False)
        now = self._now()
        model_name = str(model or "").strip() or None
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT title FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if exists is None:
                raise ConversationStoreError("Conversation does not exist.")
            cursor = connection.execute(
                """INSERT INTO messages
                   (conversation_id, role, content, created_at, model, sources_json, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    conversation_id,
                    normalized_role,
                    normalized_content,
                    now,
                    model_name,
                    sources_json,
                    normalized_status,
                ),
            )
            title = str(exists["title"])
            if normalized_role == "user" and title == "Percakapan baru":
                title = self._title_from_message(normalized_content)
            connection.execute(
                """UPDATE conversations SET title = ?, updated_at = ?,
                   model = COALESCE(?, model) WHERE conversation_id = ?""",
                (title, now, model_name, conversation_id),
            )
            row = connection.execute(
                """SELECT message_id, conversation_id, role, content, created_at,
                          model, sources_json, status
                   FROM messages WHERE message_id = ?""",
                (cursor.lastrowid,),
            ).fetchone()
        return self._message_from_row(row)

    def rename_conversation(self, conversation_id: str, title: str) -> Conversation:
        normalized_title = self._validate_title(title)
        now = self._now()
        with self._connection() as connection:
            cursor = connection.execute(
                """UPDATE conversations SET title = ?, updated_at = ?
                   WHERE conversation_id = ?""",
                (normalized_title, now, str(conversation_id)),
            )
            if cursor.rowcount != 1:
                raise ConversationStoreError("Conversation does not exist.")
        return self.get_conversation(str(conversation_id))  # type: ignore[return-value]

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (str(conversation_id),),
            )
        return cursor.rowcount > 0

    def delete_message(self, message_id: int) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT conversation_id FROM messages WHERE message_id = ?",
                (int(message_id),),
            ).fetchone()
            if row is None:
                return False
            connection.execute("DELETE FROM messages WHERE message_id = ?", (int(message_id),))
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (self._now(), str(row["conversation_id"])),
            )
        return True

    def clear(self) -> int:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM conversations")
        return cursor.rowcount

    @classmethod
    def _validate_title(cls, title: str) -> str:
        normalized = " ".join(str(title or "").split())
        if not normalized:
            raise ValueError("Conversation title must contain text.")
        if len(normalized) > cls.TITLE_LIMIT:
            raise ValueError(f"Conversation title cannot exceed {cls.TITLE_LIMIT} characters.")
        return normalized

    @classmethod
    def _title_from_message(cls, content: str) -> str:
        first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
        title = first_line[: cls.TITLE_LIMIT]
        if len(first_line) > cls.TITLE_LIMIT:
            title = title.rsplit(" ", 1)[0].rstrip(" ,.;:-") or title
        return title or "Percakapan baru"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _conversation_from_row(row: sqlite3.Row) -> Conversation:
        return Conversation(
            conversation_id=str(row["conversation_id"]),
            title=str(row["title"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            model=str(row["model"]) if row["model"] else None,
        )

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> ConversationMessage:
        try:
            raw_sources = json.loads(row["sources_json"] or "[]")
        except (TypeError, ValueError):
            raw_sources = []
        sources = tuple(item for item in raw_sources if isinstance(item, dict)) if isinstance(raw_sources, list) else ()
        return ConversationMessage(
            message_id=int(row["message_id"]),
            conversation_id=str(row["conversation_id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            created_at=str(row["created_at"]),
            model=str(row["model"]) if row["model"] else None,
            sources=sources,
            status=str(row["status"]),
        )
