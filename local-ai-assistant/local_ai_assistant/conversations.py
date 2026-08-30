"""Local conversation management and persistence."""

from __future__ import annotations

import json
import os
import tempfile
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
            messages.append(ChatMessage(role, content, tuple(tool_calls), name))

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
    """Read and atomically write the user's local conversation history."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or self.default_path()

    @staticmethod
    def default_path() -> Path:
        data_home = os.environ.get("XDG_DATA_HOME")
        base = Path(data_home) if data_home else Path.home() / ".local" / "share"
        return base / "local-ai-assistant" / "conversations.json"

    def load(self) -> list[Conversation]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
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
            return []

    def save(self, conversations: list[Conversation]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": 1, "conversations": [conversation.to_dict() for conversation in conversations]},
            indent=2,
            sort_keys=True,
        )
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f"{self.path.name}.",
                delete=False,
            ) as temporary:
                temporary.write(payload + "\n")
                temporary_path = temporary.name
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass