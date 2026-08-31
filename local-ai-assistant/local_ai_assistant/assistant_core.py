"""Assistant service boundary.

The service owns the transport boundary while workers coordinate streaming
and explicit Ollama-native tool calls. Tool requests are never inferred by
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
        tools: list[dict] | None = None,
        context_size: int | None = None,
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
        tools: list[dict] | None = None,
        context_size: int | None = None,
    ) -> Iterator[StreamEvent]:
        return self.ollama.stream_chat(messages, model, cancel_event, tools, context_size)
