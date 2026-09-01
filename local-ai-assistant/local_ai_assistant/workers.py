"""Background Qt workers for network operations."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from .assistant_core import AssistantService
from .errors import (
    AssistantCancelledError,
    OllamaCancelledError,
    format_backend_error,
)
from .ollama import ChatMessage, OllamaClient, ToolCall
from .telegram_bot import TelegramBotRunner, TelegramConfig
from .tools import PermissionLevel, ToolManager, ToolCallResult, ToolConfirmationRequired
from .voice import (
    VoiceActivityDetector,
    VoiceError,
    VoiceService,
    is_no_speech_error,
    wake_word_matches,
)


MAX_TOOL_ROUNDS = 8


class ChatWorker(QObject):
    chunk = Signal(str)
    ollama_started = Signal()
    finished = Signal(str)
    failed = Signal(str, str)
    conversation_ready = Signal(object)
    tool_started = Signal(str, str, object, str)
    tool_requested = Signal(str, str, object, str)
    tool_completed = Signal(str, str, str, bool, object)

    def __init__(
        self,
        service: AssistantService,
        messages: list[ChatMessage],
        model: str,
        tool_manager: ToolManager,
        context_size: int | None = None,
        system_prompt: str = "",
    ) -> None:
        super().__init__()
        self.service = service
        self.messages = messages
        self.model = model
        self.tool_manager = tool_manager
        self.context_size = context_size
        self.system_prompt = system_prompt.strip()
        self.cancel_event = threading.Event()
        self._approval_events: dict[str, threading.Event] = {}
        self._approval_results: dict[str, bool] = {}
        self._approval_lock = threading.Lock()

    @Slot()
    def run(self) -> None:
        visible_messages = list(self.messages)
        messages = (
            [ChatMessage("system", self.system_prompt), *visible_messages]
            if self.system_prompt
            else list(visible_messages)
        )
        complete_response: list[str] = []
        tool_rounds = 0
        try:
            while not self.cancel_event.is_set():
                cycle_response: list[str] = []
                tool_calls: list[ToolCall] = []
                self.ollama_started.emit()
                for event in self.service.stream_reply(
                    messages,
                    self.model,
                    self.cancel_event,
                    self.tool_manager.definitions_for_ollama(),
                    self.context_size,
                ):
                    if event.content:
                        cycle_response.append(event.content)
                        complete_response.append(event.content)
                        self.chunk.emit(event.content)
                    tool_calls.extend(event.tool_calls)
                    if event.done:
                        break
                if self.cancel_event.is_set():
                    self.conversation_ready.emit(visible_messages)
                    self.failed.emit("Generation stopped.", "cancelled")
                    return
                if not tool_calls:
                    response = "".join(cycle_response)
                    assistant_message = ChatMessage("assistant", response)
                    messages.append(assistant_message)
                    visible_messages.append(assistant_message)
                    self.conversation_ready.emit(visible_messages)
                    self.finished.emit("".join(complete_response))
                    return

                tool_rounds += 1
                if tool_rounds > MAX_TOOL_ROUNDS:
                    self.conversation_ready.emit(visible_messages)
                    self.failed.emit(
                        f"Stopped after {MAX_TOOL_ROUNDS} tool rounds to prevent an endless loop.",
                        "error",
                    )
                    return
                normalized_tool_calls = [
                    tool_call
                    if tool_call.id
                    else ToolCall(
                        tool_call.name,
                        tool_call.arguments,
                        f"call_{uuid.uuid4().hex}",
                        tool_call.thought_signature,
                    )
                    for tool_call in tool_calls
                ]
                assistant_message = ChatMessage(
                    "assistant",
                    "".join(cycle_response),
                    tuple(normalized_tool_calls),
                )
                messages.append(assistant_message)
                visible_messages.append(assistant_message)
                for tool_call in normalized_tool_calls:
                    if self.cancel_event.is_set():
                        self.conversation_ready.emit(visible_messages)
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
                    self.tool_completed.emit(
                        call_id,
                        tool_call.name,
                        result.content,
                        result.success,
                        result.images,
                    )
                    tool_message = ChatMessage(
                        "tool",
                        result.content,
                        name=tool_call.name,
                        images=result.images,
                        tool_call_id=call_id,
                    )
                    messages.append(tool_message)
                    visible_messages.append(tool_message)
            self.failed.emit("Generation stopped.", "cancelled")
        except (OllamaCancelledError, AssistantCancelledError):
            self.conversation_ready.emit(visible_messages)
            self.failed.emit("Generation stopped.", "cancelled")
        except Exception as error:
            self.conversation_ready.emit(visible_messages)
            self.failed.emit(
                format_backend_error(error, self.service.backend_name), "error"
            )

    def cancel(self) -> None:
        self.cancel_event.set()
        self.service.cancel_active_request()
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

    def __init__(self, client) -> None:
        super().__init__()
        self.client = client

    @Slot()
    def run(self) -> None:
        try:
            self.succeeded.emit(self.client.list_models())
        except Exception as error:
            backend_name = str(getattr(self.client, "display_name", "AI backend"))
            self.failed.emit(format_backend_error(error, backend_name))

    def cancel(self) -> None:
        self.client.cancel_active_request()


class TelegramBotWorker(QObject):
    """Run the Telegram long-polling bridge away from the Qt UI thread."""

    connected = Signal(str)
    status = Signal(str)
    failed = Signal(str)
    stopped = Signal()

    def __init__(self, config: TelegramConfig) -> None:
        super().__init__()
        self.config = config
        self.runner: TelegramBotRunner | None = None

    @Slot()
    def run(self) -> None:
        try:
            self.runner = TelegramBotRunner(self.config)
            self.runner.run(
                on_connected=self.connected.emit,
                on_status=self.status.emit,
            )
        except Exception as error:
            if self.runner is None or not self.runner._stop_event.is_set():
                self.failed.emit(format_ollama_error(error))
        finally:
            self.stopped.emit()

    def cancel(self) -> None:
        if self.runner is not None:
            self.runner.stop()


class VoiceRecordWorker(QObject):
    """Record microphone input without blocking the Qt event loop."""

    started = Signal()
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        service: VoiceService,
        destination: Path,
        duration_seconds: int | None = None,
    ) -> None:
        super().__init__()
        self.service = service
        self.destination = destination
        self.duration_seconds = duration_seconds
        self._stop_event = threading.Event()
        self._process: subprocess.Popen | None = None

    @Slot()
    def run(self) -> None:
        process = None
        try:
            process = self.service.start_recorder(self.destination)
            self._process = process
            self.started.emit()
            if self.duration_seconds is None:
                while process.poll() is None and not self._stop_event.wait(0.1):
                    pass
            else:
                self._record_until_silence(process)
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
        except Exception as error:
            # Keep an unexpected recorder/backend error from leaving the Orb
            # stuck in its thinking state without feedback.
            self.failed.emit(f"Microphone recording failed: {error}")
        finally:
            self._process = None
            if process is not None:
                self.service.release_recorder(process)

    def stop(self) -> None:
        self._stop_event.set()
        process = self._process
        if process is not None and process.poll() is None:
            self.service.stop_recorder(process)

    def cancel(self) -> None:
        self.stop()

    def _record_until_silence(self, process: subprocess.Popen) -> None:
        """Wait for speech, then stop after a short quiet period.

        The duration setting is deliberately a maximum listening window, not
        a command to record dead air for the full duration.
        """
        deadline = time.monotonic() + float(self.duration_seconds or 1)
        detector = VoiceActivityDetector(
            threshold=self.service.config.voice_vad_threshold,
            silence_duration=self.service.config.voice_silence_duration,
            min_speech_duration=self.service.config.voice_min_speech_duration,
        )
        offset = 44
        while process.poll() is None and not self._stop_event.is_set():
            now = time.monotonic()
            if now >= deadline:
                return
            try:
                with self.destination.open("rb") as audio:
                    audio.seek(offset)
                    samples = audio.read()
                if len(samples) % 2:
                    samples = samples[:-1]
                if samples:
                    offset += len(samples)
                    if detector.consume(samples, now):
                        return
            except OSError:
                pass
            if detector.should_stop(now):
                return
            self._stop_event.wait(0.1)


class WakeWordWorker(QObject):
    """Continuously capture one stream while scanning overlapping audio windows."""

    detected = Signal()
    failed = Signal(str)

    def __init__(
        self,
        service: VoiceService,
        wake_word: str,
        chunk_seconds: float = 4.0,
    ) -> None:
        super().__init__()
        self.service = service
        self.wake_word = wake_word.casefold().strip()
        self.chunk_seconds = chunk_seconds
        self._stop_event = threading.Event()
        self._process: subprocess.Popen | None = None

    @Slot()
    def run(self) -> None:
        stream_path: Path | None = None
        process: subprocess.Popen | None = None
        try:
            stream_path = self.service.new_recording_path()
            process = self.service.start_recorder(stream_path)
            self._process = process
            if self.chunk_seconds <= 0:
                # A zero-length window is useful for deterministic worker
                # tests and keeps the worker's public seam backwards
                # compatible; normal wake listening always uses a real window.
                transcript = self.service.transcribe(stream_path, device="cpu").casefold()
                if wake_word_matches(transcript, self.wake_word):
                    self.detected.emit()
                return
            window_bytes = max(1, int(self.chunk_seconds * 16_000 * 2))
            overlap_bytes = min(window_bytes // 2, int(2.0 * 16_000 * 2))
            pcm = b""
            offset = 44
            while not self._stop_event.is_set():
                if process.poll() is not None:
                    raise VoiceError("The microphone stream stopped unexpectedly.")
                try:
                    with stream_path.open("rb") as audio:
                        audio.seek(offset)
                        new_pcm = audio.read()
                except OSError:
                    new_pcm = b""
                if new_pcm:
                    usable = len(new_pcm) - (len(new_pcm) % 2)
                    pcm += new_pcm[:usable]
                    offset += usable
                if len(pcm) >= window_bytes:
                    snapshot = self.service.new_recording_path()
                    try:
                        with wave.open(str(snapshot), "wb") as audio:
                            audio.setnchannels(1)
                            audio.setsampwidth(2)
                            audio.setframerate(16_000)
                            audio.writeframes(pcm[-window_bytes:])
                        # Keep wake listening off the GPU used by Ollama and
                        # other desktop workloads. The recorder itself stays
                        # open, so Whisper processing never creates a
                        # microphone-monitoring gap.
                        transcript = self.service.transcribe(
                            snapshot, device="cpu"
                        ).casefold()
                        if wake_word_matches(transcript, self.wake_word):
                            self.detected.emit()
                            return
                    except VoiceError as error:
                        if not is_no_speech_error(str(error)):
                            self.failed.emit(str(error))
                            return
                    finally:
                        try:
                            snapshot.unlink()
                        except OSError:
                            pass
                    pcm = pcm[-overlap_bytes:]
                self._stop_event.wait(0.1)
        except Exception as error:
            if not self._stop_event.is_set():
                self.failed.emit(str(error))
        finally:
            if process is not None:
                if process.poll() is None:
                    self.service.stop_recorder(process)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                if stream_path is not None:
                    try:
                        self.service.finish_recording(process, stream_path)
                    except VoiceError as error:
                        if not self._stop_event.is_set():
                            self.failed.emit(str(error))
                self.service.release_recorder(process)
            self._process = None
            if stream_path is not None:
                try:
                    stream_path.unlink()
                except OSError:
                    pass

    def cancel(self) -> None:
        self._stop_event.set()
        process = self._process
        if process is not None and process.poll() is None:
            self.service.stop_recorder(process)


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


class SpeechStreamWorker(QObject):
    """Synthesize and play sentence-sized chunks while Ollama is still running."""

    started = Signal()
    synthesis_started = Signal()
    playback_started = Signal()
    finished = Signal()
    failed = Signal(str)

    def __init__(self, service: VoiceService) -> None:
        super().__init__()
        self.service = service
        self.cancel_event = threading.Event()
        self._chunks: queue.Queue[str | None] = queue.Queue()
        self._started = False

    @Slot()
    def run(self) -> None:
        try:
            while not self.cancel_event.is_set():
                try:
                    chunk = self._chunks.get(timeout=0.1)
                except queue.Empty:
                    continue
                if chunk is None:
                    self.finished.emit()
                    return
                if not chunk.strip():
                    continue
                if not self._started:
                    self._started = True
                    self.started.emit()
                self.service.speak(
                    chunk,
                    self.cancel_event,
                    self.synthesis_started.emit,
                    self.playback_started.emit,
                )
            self.finished.emit()
        except Exception as error:
            self.failed.emit(str(error))

    def append(self, text: str) -> None:
        if text.strip() and not self.cancel_event.is_set():
            self._chunks.put(text)

    def finish(self) -> None:
        if not self.cancel_event.is_set():
            self._chunks.put(None)

    def cancel(self) -> None:
        self.cancel_event.set()
        self.service.cancel()
