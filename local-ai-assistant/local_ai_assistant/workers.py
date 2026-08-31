"""Background Qt workers for network operations."""

from __future__ import annotations

import threading
import subprocess
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from .assistant_core import AssistantService
from .errors import OllamaCancelledError, format_ollama_error
from .ollama import ChatMessage, OllamaClient, ToolCall
from .tools import PermissionLevel, ToolManager, ToolCallResult, ToolConfirmationRequired
from .voice import VoiceError, VoiceService


MAX_TOOL_ROUNDS = 8


class ChatWorker(QObject):
    chunk = Signal(str)
    finished = Signal(str)
    failed = Signal(str, str)
    conversation_ready = Signal(object)
    tool_started = Signal(str, str, object, str)
    tool_requested = Signal(str, str, object, str)
    tool_completed = Signal(str, str, str, bool)

    def __init__(
        self,
        service: AssistantService,
        messages: list[ChatMessage],
        model: str,
        tool_manager: ToolManager,
    ) -> None:
        super().__init__()
        self.service = service
        self.messages = messages
        self.model = model
        self.tool_manager = tool_manager
        self.cancel_event = threading.Event()
        self._approval_events: dict[str, threading.Event] = {}
        self._approval_results: dict[str, bool] = {}
        self._approval_lock = threading.Lock()

    @Slot()
    def run(self) -> None:
        messages = list(self.messages)
        complete_response: list[str] = []
        tool_rounds = 0
        try:
            while not self.cancel_event.is_set():
                cycle_response: list[str] = []
                tool_calls: list[ToolCall] = []
                for event in self.service.stream_reply(
                    messages,
                    self.model,
                    self.cancel_event,
                    self.tool_manager.definitions_for_ollama(),
                ):
                    if event.content:
                        cycle_response.append(event.content)
                        complete_response.append(event.content)
                        self.chunk.emit(event.content)
                    tool_calls.extend(event.tool_calls)
                    if event.done:
                        break
                if self.cancel_event.is_set():
                    self.conversation_ready.emit(messages)
                    self.failed.emit("Generation stopped.", "cancelled")
                    return
                if not tool_calls:
                    response = "".join(cycle_response)
                    messages.append(ChatMessage("assistant", response))
                    self.conversation_ready.emit(messages)
                    self.finished.emit("".join(complete_response))
                    return

                tool_rounds += 1
                if tool_rounds > MAX_TOOL_ROUNDS:
                    self.conversation_ready.emit(messages)
                    self.failed.emit(
                        f"Stopped after {MAX_TOOL_ROUNDS} tool rounds to prevent an endless loop.",
                        "error",
                    )
                    return
                messages.append(ChatMessage("assistant", "".join(cycle_response), tuple(tool_calls)))
                for tool_call in tool_calls:
                    if self.cancel_event.is_set():
                        self.conversation_ready.emit(messages)
                        self.failed.emit("Generation stopped.", "cancelled")
                        return
                    call_id = tool_call.id or f"{tool_call.name}-{uuid.uuid4().hex}"
                    permission = self.tool_manager.permission_for(tool_call.name, tool_call.arguments)
                    self.tool_started.emit(call_id, tool_call.name, tool_call.arguments, permission.value)
                    approved = permission == PermissionLevel.SAFE
                    if not approved:
                        approved = self._wait_for_approval(call_id, tool_call, permission)
                    try:
                        result = self.tool_manager.execute(tool_call.name, tool_call.arguments, approved)
                    except ToolConfirmationRequired:
                        result = ToolCallResult(False, "The user did not approve this action.")
                    self.tool_completed.emit(call_id, tool_call.name, result.content, result.success)
                    messages.append(
                        ChatMessage(
                            "tool",
                            result.content,
                            name=tool_call.name,
                            images=result.images,
                        )
                    )
            self.failed.emit("Generation stopped.", "cancelled")
        except OllamaCancelledError:
            self.conversation_ready.emit(messages)
            self.failed.emit("Generation stopped.", "cancelled")
        except Exception as error:
            self.conversation_ready.emit(messages)
            self.failed.emit(format_ollama_error(error), "error")

    def cancel(self) -> None:
        self.cancel_event.set()
        self.service.ollama.cancel_active_request()
        with self._approval_lock:
            for approval_event in self._approval_events.values():
                approval_event.set()

    def _wait_for_approval(self, call_id: str, tool_call: ToolCall, permission: PermissionLevel) -> bool:
        approval_event = threading.Event()
        with self._approval_lock:
            self._approval_events[call_id] = approval_event
        self.tool_requested.emit(call_id, tool_call.name, tool_call.arguments, permission.value)
        try:
            while not self.cancel_event.is_set() and not approval_event.wait(0.1):
                pass
            with self._approval_lock:
                return self._approval_results.pop(call_id, False) and not self.cancel_event.is_set()
        finally:
            with self._approval_lock:
                self._approval_events.pop(call_id, None)

    @Slot(str, bool)
    def resolve_tool_call(self, call_id: str, approved: bool) -> None:
        with self._approval_lock:
            self._approval_results[call_id] = approved
            approval_event = self._approval_events.get(call_id)
        if approval_event:
            approval_event.set()


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

    def cancel(self) -> None:
        self.client.cancel_active_request()


class VoiceRecordWorker(QObject):
    """Record microphone input without blocking the Qt event loop."""

    started = Signal()
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, service: VoiceService, destination: Path) -> None:
        super().__init__()
        self.service = service
        self.destination = destination
        self._stop_event = threading.Event()
        self._process: subprocess.Popen | None = None

    @Slot()
    def run(self) -> None:
        process = None
        try:
            process = self.service.start_recorder(self.destination)
            self._process = process
            self.started.emit()
            while process.poll() is None and not self._stop_event.wait(0.1):
                pass
            if process.poll() is None:
                self.service.stop_recorder(process)
            try:
                process.wait(timeout=2)
            except Exception:
                process.kill()
                process.wait()
            self.service.finish_recording(process, self.destination)
            self.finished.emit(str(self.destination))
        except (VoiceError, OSError, subprocess.SubprocessError) as error:
            self.failed.emit(str(error))
        finally:
            self._process = None

    def stop(self) -> None:
        self._stop_event.set()
        process = self._process
        if process is not None and process.poll() is None:
            self.service.stop_recorder(process)

    def cancel(self) -> None:
        self.stop()


class VoiceTranscriptionWorker(QObject):
    """Run a local Whisper backend outside the Qt event loop."""

    finished = Signal(str, str)
    failed = Signal(str, str)

    def __init__(self, service: VoiceService, audio_path: Path) -> None:
        super().__init__()
        self.service = service
        self.audio_path = audio_path

    @Slot()
    def run(self) -> None:
        try:
            text = self.service.transcribe(self.audio_path)
            self.finished.emit(text, str(self.audio_path))
        except Exception as error:
            self.failed.emit(str(error), str(self.audio_path))


class SpeechWorker(QObject):
    """Generate and play a local TTS response outside the Qt event loop."""

    finished = Signal()
    failed = Signal(str)

    def __init__(self, service: VoiceService, text: str) -> None:
        super().__init__()
        self.service = service
        self.text = text
        self.cancel_event = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            self.service.speak(self.text, self.cancel_event)
            self.finished.emit()
        except Exception as error:
            self.failed.emit(str(error))

    def cancel(self) -> None:
        self.cancel_event.set()
        self.service.cancel()
