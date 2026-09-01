"""Local conversation management and persistence."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .ollama import ChatMessage, ToolCall


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def title_for_messages(messages: list[ChatMessage]) -> str:
    """Create a compact, useful title from the first user message."""

    for message in messages:
        if message.role == "user" and message.content.strip():
            title = " ".join(message.content.split())
            return title if len(title) <= 42 else f"{title[:41].rstrip()}…"
    return "New chat"


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    title: str = "New chat"
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @classmethod
    def create(cls) -> "Conversation":
        return cls()

    def update_messages(self, messages: list[ChatMessage]) -> None:
        self.messages = list(messages)
        self.title = title_for_messages(self.messages)
        self.updated_at = _now()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "messages": [message.as_dict() for message in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "Conversation":
        if not isinstance(raw, dict):
            raise ValueError("Conversation must be an object.")
        conversation_id = raw.get("id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("Conversation is missing an id.")

        raw_messages = raw.get("messages", [])
        if not isinstance(raw_messages, list):
            raise ValueError("Conversation messages must be a list.")
        messages: list[ChatMessage] = []
        for raw_message in raw_messages:
            if not isinstance(raw_message, dict):
                raise ValueError("Conversation message must be an object.")
            role = raw_message.get("role")
            content = raw_message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("Conversation message has invalid fields.")
            raw_tool_calls = raw_message.get("tool_calls", [])
            if not isinstance(raw_tool_calls, list):
                raise ValueError("Conversation tool calls must be a list.")
            tool_calls: list[ToolCall] = []
            for raw_tool_call in raw_tool_calls:
                if not isinstance(raw_tool_call, dict):
                    raise ValueError("Conversation tool call must be an object.")
                function = raw_tool_call.get("function")
                if not isinstance(function, dict):
                    raise ValueError("Conversation tool call has no function.")
                name = function.get("name")
                arguments = function.get("arguments", {})
                call_id = raw_tool_call.get("id", "")
                if not isinstance(name, str) or not isinstance(arguments, dict):
                    raise ValueError("Conversation tool call has invalid fields.")
                if not isinstance(call_id, str):
                    call_id = ""
                tool_calls.append(ToolCall(name, arguments, call_id))
            name = raw_message.get("name", "")
            if not isinstance(name, str):
                name = ""
            raw_images = raw_message.get("images", [])
            if not isinstance(raw_images, list) or not all(isinstance(image, str) for image in raw_images):
                raise ValueError("Conversation images must be file paths.")
            tool_call_id = raw_message.get("tool_call_id", "")
            if not isinstance(tool_call_id, str):
                tool_call_id = ""
            messages.append(
                ChatMessage(
                    role,
                    content,
                    tuple(tool_calls),
                    name,
                    tuple(raw_images),
                    tool_call_id,
                )
            )

        title = raw.get("title")
        created_at = raw.get("created_at")
        updated_at = raw.get("updated_at")
        return cls(
            id=conversation_id,
            title=title if isinstance(title, str) and title else title_for_messages(messages),
            messages=messages,
            created_at=created_at if isinstance(created_at, str) else _now(),
            updated_at=updated_at if isinstance(updated_at, str) else _now(),
        )


class ConversationStore:
    """Read and durably store the user's local conversation history in SQLite."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: Path | None = None,
        legacy_path: Path | None = None,
    ) -> None:
        self.path = path or self.default_path()
        self.legacy_path = legacy_path if legacy_path is not None else (
            self.legacy_default_path() if path is None else None
        )

    @staticmethod
    def default_path() -> Path:
        data_home = os.environ.get("XDG_DATA_HOME")
        base = Path(data_home) if data_home else Path.home() / ".local" / "share"
        return base / "local-ai-assistant" / "conversations.db"

    @staticmethod
    def legacy_default_path() -> Path:
        data_home = os.environ.get("XDG_DATA_HOME")
        base = Path(data_home) if data_home else Path.home() / ".local" / "share"
        return base / "local-ai-assistant" / "conversations.json"

    def load(self) -> list[Conversation]:
        if not self.path.exists():
            return self._migrate_legacy()

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT id, title, created_at, updated_at
                    FROM conversations
                    ORDER BY updated_at DESC, rowid DESC
                    """
                ).fetchall()
                conversations: list[Conversation] = []
                for row in rows:
                    try:
                        message_rows = connection.execute(
                            """
                            SELECT role, content, tool_calls, name, images
                            FROM messages
                            WHERE conversation_id = ?
                            ORDER BY position ASC
                            """,
                            (row["id"],),
                        ).fetchall()
                        raw_messages = [
                            {
                                "role": message["role"],
                                "content": message["content"],
                                "tool_calls": json.loads(message["tool_calls"]),
                                "name": message["name"],
                                "images": json.loads(message["images"]),
                            }
                            for message in message_rows
                        ]
                        conversations.append(
                            Conversation.from_dict(
                                {
                                    "id": row["id"],
                                    "title": row["title"],
                                    "created_at": row["created_at"],
                                    "updated_at": row["updated_at"],
                                    "messages": raw_messages,
                                }
                            )
                        )
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                return conversations
        except (OSError, sqlite3.DatabaseError, TypeError, ValueError):
            # Explicit callers may still point at the pre-Phase-5 JSON path.
            # Import it in place so subsequent saves use the SQLite schema.
            legacy_conversations = self._read_json(self.path)
            if legacy_conversations is not None:
                try:
                    self.save(legacy_conversations)
                except (OSError, sqlite3.DatabaseError, TypeError, ValueError):
                    pass
                return legacy_conversations
            # A damaged local database should not prevent the application from
            # opening. The next successful save will replace it transactionally.
            return []

    def save(self, conversations: list[Conversation]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("BEGIN")
            try:
                connection.execute("DELETE FROM messages")
                connection.execute("DELETE FROM conversations")
                for conversation in conversations:
                    connection.execute(
                        """
                        INSERT INTO conversations (id, title, created_at, updated_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            conversation.id,
                            conversation.title,
                            conversation.created_at,
                            conversation.updated_at,
                        ),
                    )
                    for position, message in enumerate(conversation.messages):
                        connection.execute(
                            """
                            INSERT INTO messages (
                                conversation_id, position, role, content,
                                tool_calls, name, images
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                conversation.id,
                                position,
                                message.role,
                                message.content,
                                json.dumps(
                                    [tool_call.as_dict() for tool_call in message.tool_calls],
                                    sort_keys=True,
                                ),
                                message.name,
                                json.dumps(list(message.images), sort_keys=True),
                            ),
                        )
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _migrate_legacy(self) -> list[Conversation]:
        if self.legacy_path is None or not self.legacy_path.is_file():
            return []
        conversations = self._read_json(self.legacy_path)
        if conversations is None:
            return []
        try:
            self.save(conversations)
            return conversations
        except (OSError, sqlite3.DatabaseError, TypeError, ValueError):
            return []

    @staticmethod
    def _read_json(path: Path) -> list[Conversation] | None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw_conversations = raw.get("conversations", []) if isinstance(raw, dict) else []
            if not isinstance(raw_conversations, list):
                return []
            conversations: list[Conversation] = []
            for item in raw_conversations:
                if not isinstance(item, dict):
                    continue
                try:
                    conversations.append(Conversation.from_dict(item))
                except (TypeError, ValueError):
                    continue
            return conversations
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        self._initialize_schema(connection)
        return connection

    @classmethod
    def _initialize_schema(cls, connection: sqlite3.Connection) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > cls.SCHEMA_VERSION:
            raise sqlite3.DatabaseError(
                f"Unsupported conversation database version: {version}"
            )
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                conversation_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_calls TEXT NOT NULL,
                name TEXT NOT NULL,
                images TEXT NOT NULL,
                PRIMARY KEY (conversation_id, position),
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations (id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS messages_conversation_idx
                ON messages (conversation_id, position);
            """
        )
        if version < cls.SCHEMA_VERSION:
            connection.execute(f"PRAGMA user_version = {cls.SCHEMA_VERSION}")