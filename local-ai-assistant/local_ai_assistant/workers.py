"""Background Qt workers for network operations."""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal, Slot

from .assistant_core import AssistantService
from .errors import OllamaCancelledError, format_ollama_error
from .ollama import ChatMessage, OllamaClient


class ChatWorker(QObject):
    chunk = Signal(str)
    finished = Signal(str)
    failed = Signal(str, str)

    def __init__(
        self,
        service: AssistantService,
        messages: list[ChatMessage],
        model: str,
    ) -> None:
        super().__init__()
        self.service = service
        self.messages = messages
        self.model = model
        self.cancel_event = threading.Event()

    @Slot()
    def run(self) -> None:
        assembled: list[str] = []
        try:
            for event in self.service.stream_reply(self.messages, self.model, self.cancel_event):
                if event.content:
                    assembled.append(event.content)
                    self.chunk.emit(event.content)
                if event.done:
                    break
            if self.cancel_event.is_set():
                self.failed.emit("Generation stopped.", "cancelled")
            else:
                self.finished.emit("".join(assembled))
        except OllamaCancelledError:
            self.failed.emit("Generation stopped.", "cancelled")
        except Exception as error:
            self.failed.emit(format_ollama_error(error), "error")

    def cancel(self) -> None:
        self.cancel_event.set()
        self.service.ollama.cancel_active_request()


class ConnectionWorker(QObject):
    succeeded = Signal(list)
    failed = Signal(str)

    def __init__(self, client: OllamaClient) -> None:
        super().__init__()
        self.client = client

    @Slot()
    def run(self) -> None:
        try:
            self.succeeded.emit(self.client.list_models())
        except Exception as error:
            self.failed.emit(format_ollama_error(error))
