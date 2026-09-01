"""Assistant service boundary.

The service owns the transport boundary while workers coordinate streaming
and explicit Ollama-native tool calls. Tool requests are never inferred by
parsing arbitrary assistant text.
"""

from __future__ import annotations

import threading
from typing import Iterator, Protocol

from .ollama import ChatMessage, StreamEvent


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

    def list_models(self) -> list[str]:
        ...

    def cancel_active_request(self) -> None:
        ...


class AssistantService:
    """Conversation-independent service used by the Qt workers."""

    def __init__(self, backend: AssistantBackend) -> None:
        self.backend = backend

    @property
    def backend_name(self) -> str:
        return str(getattr(self.backend, "display_name", "AI backend"))

    def stream_reply(
        self,
        messages: list[ChatMessage],
        model: str,
        cancel_event: threading.Event | None = None,
        tools: list[dict] | None = None,
        context_size: int | None = None,
    ) -> Iterator[StreamEvent]:
        return self.backend.stream_chat(
            messages, model, cancel_event, tools, context_size
        )

    def list_models(self) -> list[str]:
        return self.backend.list_models()

    def cancel_active_request(self) -> None:
        self.backend.cancel_active_request()
