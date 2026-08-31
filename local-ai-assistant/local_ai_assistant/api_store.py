"""Authenticated API persistence for users, sessions, and conversations."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .conversations import Conversation, title_for_messages
from .ollama import ChatMessage, ToolCall


SESSION_TTL_SECONDS = 60 * 60 * 24 * 30
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DuplicateEmailError(ValueError):
    """Raised when an account already exists for an email address."""


class ApiStore:
    """SQLite store with ownership checks for all API-facing data."""

    def __init__(
        self,
        path: Path | None = None,
        session_secret: str | bytes | None = None,
        session_secret_path: Path | None = None,
    ) -> None:
        self.path = path or self.default_path()
        secret = (
            session_secret
            if session_secret is not None
            else os.environ.get("SESSION_SECRET")
        )
        if not secret:
            secret = self._load_or_create_local_secret(
                session_secret_path or self.default_secret_path()
            )
        self.session_secret = secret.encode("utf-8") if isinstance(secret, str) else secret
        if not self.session_secret or len(self.session_secret) < 16:
            raise ValueError("SESSION_SECRET must be configured with at least 16 characters.")
        self._lock = threading.RLock()

    @staticmethod
    def default_path() -> Path:
        data_home = os.environ.get("XDG_DATA_HOME")
        base = Path(data_home) if data_home else Path.home() / ".local" / "share"
        return base / "local-ai-assistant" / "api.db"

    @staticmethod
    def default_secret_path() -> Path:
        return Path.home() / ".config" / "local-ai-assistant" / "api-session.secret"

    def register(self, email: str, password: str) -> dict:
        normalized_email = self._validate_credentials(email, password)[0]
        salt = secrets.token_bytes(16).hex()
        password_hash = self._password_hash(password, salt)
        now = _now()
        with self._lock, self._connect() as connection:
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO users (email, password_salt, password_hash, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (normalized_email, salt, password_hash, now),
                )
            except sqlite3.IntegrityError as error:
                raise DuplicateEmailError(
                    "An account with that email already exists."
                ) from error
            return {"id": int(cursor.lastrowid), "email": normalized_email}

    def authenticate(self, email: str, password: str) -> dict | None:
        normalized_email = self._validate_credentials(email, password)[0]
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, email, password_salt, password_hash
                FROM users
                WHERE email = ? COLLATE NOCASE
                """,
                (normalized_email,),
            ).fetchone()
        if row is None:
            return None
        expected = self._password_hash(password, row["password_salt"])
        if not hmac.compare_digest(expected, row["password_hash"]):
            return None
        return {"id": int(row["id"]), "email": row["email"]}

    def create_session(self, user_id: int) -> tuple[str, int]:
        token = secrets.token_urlsafe(32)
        expires_at = int(time.time()) + SESSION_TTL_SECONDS
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (token_hash, user_id, expires_at)
                VALUES (?, ?, ?)
                """,
                (self._token_hash(token), user_id, expires_at),
            )
        return token, expires_at

    def user_for_session(self, token: str | None) -> dict | None:
        if not token:
            return None
        token_hash = self._token_hash(token)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT users.id, users.email, sessions.expires_at
                FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            if int(row["expires_at"]) <= int(time.time()):
                connection.execute(
                    "DELETE FROM sessions WHERE token_hash = ?", (token_hash,)
                )
                return None
            return {"id": int(row["id"]), "email": row["email"]}

    def delete_session(self, token: str | None) -> None:
        if not token:
            return
        with self._lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM sessions WHERE token_hash = ?",
                (self._token_hash(token),),
            )

    def list_conversations(self, user_id: int) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       COUNT(m.position) AS message_count
                FROM conversations AS c
                LEFT JOIN messages AS m ON m.conversation_id = c.id
                WHERE c.user_id = ?
                GROUP BY c.id
                ORDER BY c.updated_at DESC, c.rowid DESC
                """,
                (user_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": int(row["message_count"]),
            }
            for row in rows
        ]

    def create_conversation(self, user_id: int, title: str = "New chat") -> Conversation:
        title = self._conversation_title(title)
        conversation = Conversation(
            id=uuid.uuid4().hex,
            title=title,
            created_at=_now(),
            updated_at=_now(),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conversation.id,
                    user_id,
                    conversation.title,
                    conversation.created_at,
                    conversation.updated_at,
                ),
            )
        return conversation

    def get_conversation(self, user_id: int, conversation_id: str) -> Conversation | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                WHERE id = ? AND user_id = ?
                """,
                (conversation_id, user_id),
            ).fetchone()
            if row is None:
                return None
            message_rows = connection.execute(
                """
                SELECT role, content, tool_calls, name, images
                FROM messages
                WHERE conversation_id = ?
                ORDER BY position ASC
                """,
                (conversation_id,),
            ).fetchall()

        raw_messages = [
            {
                "role": row["role"],
                "content": row["content"],
                "tool_calls": json.loads(row["tool_calls"]),
                "name": row["name"],
                "images": json.loads(row["images"]),
            }
            for row in message_rows
        ]
        return Conversation.from_dict(
            {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "messages": raw_messages,
            }
        )

    def append_message(
        self,
        user_id: int,
        conversation_id: str,
        message: ChatMessage,
    ) -> Conversation:
        if message.role not in {"user", "assistant"}:
            raise ValueError("API messages must be user or assistant messages.")
        if len(message.content) > 200_000:
            raise ValueError("Message content is too large.")
        if message.tool_calls or message.name or message.images:
            raise ValueError("API messages cannot contain local tool or image data.")

        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                WHERE id = ? AND user_id = ?
                """,
                (conversation_id, user_id),
            ).fetchone()
            if row is None:
                raise KeyError("Conversation not found.")
            position = connection.execute(
                """
                SELECT COALESCE(MAX(position), -1) + 1
                FROM messages
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()[0]
            updated_at = _now()
            title = row["title"]
            if message.role == "user" and title == "New chat":
                title = title_for_messages([message])
            connection.execute(
                """
                INSERT INTO messages (
                    conversation_id, position, role, content,
                    tool_calls, name, images
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    position,
                    message.role,
                    message.content,
                    "[]",
                    "",
                    "[]",
                ),
            )
            connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (title, updated_at, conversation_id, user_id),
            )
        conversation = self.get_conversation(user_id, conversation_id)
        if conversation is None:
            raise KeyError("Conversation not found.")
        return conversation

    @staticmethod
    def _validate_credentials(email: str, password: str) -> tuple[str, str]:
        if not isinstance(email, str):
            raise ValueError("Email is required.")
        normalized_email = email.strip().casefold()
        if len(normalized_email) > 254 or not EMAIL_PATTERN.fullmatch(normalized_email):
            raise ValueError("Enter a valid email address.")
        if not isinstance(password, str) or not 8 <= len(password) <= 256:
            raise ValueError("Password must be between 8 and 256 characters.")
        return normalized_email, password

    @staticmethod
    def _password_hash(password: str, salt_hex: str) -> str:
        return hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=16_384,
            r=8,
            p=1,
        ).hex()

    def _token_hash(self, token: str) -> str:
        return hmac.new(
            self.session_secret,
            token.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _conversation_title(title: str) -> str:
        if not isinstance(title, str):
            return "New chat"
        normalized = " ".join(title.split()).strip()
        return normalized[:200] or "New chat"

    @staticmethod
    def _load_or_create_local_secret(path: Path) -> str:
        try:
            existing = path.read_text(encoding="utf-8").strip()
            if len(existing) >= 16:
                return existing
        except FileNotFoundError:
            pass
        except OSError as error:
            raise RuntimeError(
                f"Could not read the local API session secret: {error}"
            ) from error

        generated = secrets.token_urlsafe(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f"{path.name}.",
                delete=False,
            ) as temporary:
                os.chmod(temporary.name, 0o600)
                temporary.write(generated + "\n")
                temporary_path = temporary.name
            os.replace(temporary_path, path)
            os.chmod(path, 0o600)
            return generated
        except OSError as error:
            raise RuntimeError(
                f"Could not create the local API session secret: {error}"
            ) from error
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
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

            CREATE INDEX IF NOT EXISTS api_conversations_user_idx
                ON conversations (user_id, updated_at);
            CREATE INDEX IF NOT EXISTS api_messages_conversation_idx
                ON messages (conversation_id, position);
            """
        )
        return connection