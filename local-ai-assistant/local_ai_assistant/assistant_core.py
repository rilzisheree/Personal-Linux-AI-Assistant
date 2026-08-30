"""Assistant service boundary.

Phase 1 deliberately exposes only conversational streaming. Future tool
calling should be added as an explicit interface here, never inferred by
parsing arbitrary assistant text.
"""

from __future__ import annotations

import threading
from typing import Iterator, Protocol

from .ollama import ChatMessage, OllamaClient, StreamEvent


class AssistantBackend(Protocol):
    def stream_reply(
        self,
        messages: list[ChatMessage],
        model: str,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        ...


class AssistantService:
    """Conversation-independent service used by the Qt workers."""

    def __init__(self, ollama: OllamaClient) -> None:
        self.ollama = ollama

    def stream_reply(
        self,
        messages: list[ChatMessage],
        model: str,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        return self.ollama.stream_chat(messages, model, cancel_event)
