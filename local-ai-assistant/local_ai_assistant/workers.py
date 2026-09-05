"""Background Qt workers for network operations."""

from __future__ import annotations

import queue
import logging
import os
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
    OllamaProtocolError,
    format_backend_error,
)
from .ollama import ChatMessage, OllamaClient, ToolCall
from .telegram_bot import TelegramBotRunner, TelegramConfig
from .tools import PermissionLevel, ToolManager, ToolCallResult, ToolConfirmationRequired
from .voice import (
    VoiceActivityDetector,
    VoiceError,
    VoiceService,
    find_wake_word,
    is_no_speech_error,
    remove_wake_word,
    wake_word_match_score,
)


MAX_TOOL_ROUNDS = 8
LOGGER = logging.getLogger("lura.workers")
DEFAULT_WAKE_WORD_WINDOW_SECONDS = 2.5
WAKE_WORD_SAMPLE_RATE = 16_000
WAKE_WORD_CHANNELS = 1
WAKE_WORD_SAMPLE_WIDTH = 2
DIRECT_REMINDER_TOOLS = frozenset(
    {
        "list_reminders",
        "complete_reminder",
        "cancel_reminder",
        "reschedule_reminder",
    }
)


def _wake_debug_enabled() -> bool:
    return os.environ.get("LURA_VOICE_DEBUG", "").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _stream_debug_enabled() -> bool:
    return os.environ.get("LURA_STREAM_DEBUG", "").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
        ollama_tools = self.tool_manager.definitions_for_ollama()
        if _stream_debug_enabled():
            tool_names = [
                schema.get("function", {}).get("name", "")
                for schema in ollama_tools
                if isinstance(schema, dict)
                and isinstance(schema.get("function"), dict)
                and isinstance(schema.get("function", {}).get("name"), str)
            ]
            reminder_schema_present = "create_reminder" in tool_names
            LOGGER.info(
                "[TOOL FLOW] reminder_tool_registered=%s "
                "reminder_schema_present=%s ollama_tools_count=%d "
                "ollama_tool_names=%s",
                str(self.tool_manager.has_tool("create_reminder")).lower(),
                str(reminder_schema_present).lower(),
                len(ollama_tools),
                ",".join(tool_names) or "none",
            )
        complete_response: list[str] = []
        cycle_response: list[str] = []
        tool_rounds = 0
        stream_chunks = 0
        stream_characters = 0
        stream_completion_received = False
        stream_done_reason = ""
        stream_generated_tokens: int | float | None = None
        try:
            self.ollama_started.emit()
            current_request = next(
                (
                    message.content
                    for message in reversed(visible_messages)
                    if message.role == "user" and message.content.strip()
                ),
                "",
            )
            self.tool_manager.set_active_model(self.model)
            direct_call = self.tool_manager.direct_tool_call_for_request(
                current_request
            )
            route_tools = None
            if direct_call is not None:
                tool_name, arguments = direct_call
                direct_tool_call = ToolCall(
                    tool_name,
                    arguments,
                    f"direct_{uuid.uuid4().hex}",
                )
                messages.append(ChatMessage("assistant", "", (direct_tool_call,)))
                visible_messages.append(messages[-1])
                result = self._execute_tool_call(direct_tool_call)
                tool_message = ChatMessage(
                    "tool",
                    result.content,
                    name=tool_name,
                    images=result.images,
                    tool_call_id=direct_tool_call.id,
                )
                messages.append(tool_message)
                visible_messages.append(tool_message)
                # The application already selected and executed the only
                # authoritative tool needed for this request. Qwen only
                # formats the result and cannot replace it with a refusal or
                # another guessed tool call.
                route_tools = None
                if tool_name in DIRECT_REMINDER_TOOLS:
                    # Reminder state and timing are application facts. Avoid
                    # a second no-tools model request, which can cause a
                    # model to request the same reminder tool again.
                    final_response = result.content
                    visible_messages.append(ChatMessage("assistant", final_response))
                    self.conversation_ready.emit(visible_messages)
                    self.finished.emit(final_response)
                    return
            else:
                route = self.service.route_request(
                    messages,
                    ollama_tools,
                    self.cancel_event,
                )
                if self.cancel_event.is_set():
                    self.conversation_ready.emit(visible_messages)
                    self.failed.emit("Generation stopped.", "cancelled")
                    return
                route_tools = ollama_tools if route.route != "simple" else None
                if (
                    route_tools is None
                    and self.tool_manager.request_requires_tools(current_request)
                ):
                    route_tools = ollama_tools
                if _stream_debug_enabled():
                    LOGGER.info(
                        "[TOOL FLOW] classification=%s selected_tools=%s",
                        route.route,
                        ",".join(
                            schema.get("function", {}).get("name", "")
                            for schema in (route_tools or [])
                            if isinstance(schema, dict)
                            and isinstance(schema.get("function"), dict)
                            and isinstance(schema.get("function", {}).get("name"), str)
                        )
                        or "none",
                    )
            while not self.cancel_event.is_set():
                cycle_response = []
                tool_calls: list[ToolCall] = []
                stream_complete = False
                cycle_chunks = 0
                cycle_characters = 0
                self.ollama_started.emit()
                for event in self.service.stream_reply(
                    messages,
                    self.model,
                    self.cancel_event,
                    route_tools,
                    self.context_size,
                ):
                    cycle_chunks += 1
                    cycle_characters += len(event.content)
                    stream_chunks += 1
                    stream_characters += len(event.content)
                    if event.content:
                        cycle_response.append(event.content)
                        complete_response.append(event.content)
                        self.chunk.emit(event.content)
                    tool_calls.extend(event.tool_calls)
                    if event.done:
                        stream_complete = True
                        stream_completion_received = True
                        stream_done_reason = event.done_reason
                        raw_eval_count = (event.metrics or {}).get("eval_count")
                        if isinstance(raw_eval_count, (int, float)):
                            stream_generated_tokens = raw_eval_count
                        break
                LOGGER.info(
                    "[BACKEND] worker_cycle_chunks=%d worker_cycle_chars=%d "
                    "worker_stream_chunks=%d worker_stream_chars=%d "
                    "stream_finished=%s done_reason=%s generated_tokens=%s",
                    cycle_chunks,
                    cycle_characters,
                    stream_chunks,
                    stream_characters,
                    str(stream_complete).lower(),
                    stream_done_reason or "unspecified",
                    stream_generated_tokens if stream_generated_tokens is not None else "unknown",
                )
                if not stream_complete:
                    raise OllamaProtocolError(
                        "The AI stream ended before sending a completion event."
                    )
                if self.cancel_event.is_set():
                    self.conversation_ready.emit(visible_messages)
                    self.failed.emit("Generation stopped.", "cancelled")
                    return
                if tool_calls and route_tools is None:
                    LOGGER.warning(
                        "[BACKEND] model_returned_tools_without_tools "
                        "tool_calls=%d cycle_chars=%d",
                        len(tool_calls),
                        cycle_characters,
                    )
                    if not cycle_response:
                        raise OllamaProtocolError(
                            "The model requested a tool although no tools were enabled."
                        )
                    # Preserve visible text from a malformed mixed response, but
                    # never execute a tool that was not offered to the model.
                    tool_calls = []
                if not tool_calls:
                    response = "".join(cycle_response)
                    assistant_message = ChatMessage("assistant", response)
                    messages.append(assistant_message)
                    visible_messages.append(assistant_message)
                    self.conversation_ready.emit(visible_messages)
                    self.finished.emit("".join(complete_response))
                    LOGGER.info(
                        "[BACKEND] worker_complete chunks_received=%d "
                        "chars_received=%d stream_finished=%s final_message_chars=%d",
                        stream_chunks,
                        stream_characters,
                        str(stream_completion_received).lower(),
                        len("".join(complete_response)),
                    )
                    return

                if _stream_debug_enabled():
                    LOGGER.info(
                        "[TOOL FLOW] model_tool_call=%s",
                        ",".join(tool_call.name for tool_call in tool_calls) or "none",
                    )
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
                cycle_response = []
                tool_results = []
                for tool_call in normalized_tool_calls:
                    if self.cancel_event.is_set():
                        self.conversation_ready.emit(visible_messages)
                        self.failed.emit("Generation stopped.", "cancelled")
                        return
                    call_id = tool_call.id or f"{tool_call.name}-{uuid.uuid4().hex}"
                    result = self._execute_tool_call(tool_call, call_id)
                    tool_results.append(result)
                    tool_message = ChatMessage(
                        "tool",
                        result.content,
                        name=tool_call.name,
                        images=result.images,
                        tool_call_id=call_id,
                    )
                    messages.append(tool_message)
                    visible_messages.append(tool_message)
                if (
                    len(normalized_tool_calls) == 1
                    and normalized_tool_calls[0].name == "create_reminder"
                ):
                    # Reminder timing is an application fact. Do not ask the
                    # model to paraphrase it, because it can turn a relative
                    # delay such as five minutes into an incorrect day label.
                    final_response = tool_results[0].content
                    visible_messages.append(ChatMessage("assistant", final_response))
                    self.conversation_ready.emit(visible_messages)
                    self.finished.emit(final_response)
                    return
            self.failed.emit("Generation stopped.", "cancelled")
        except (OllamaCancelledError, AssistantCancelledError):
            LOGGER.info(
                "[BACKEND] worker_cancelled chunks_received=%d chars_received=%d "
                "stream_finished=%s",
                stream_chunks,
                stream_characters,
                str(stream_completion_received).lower(),
            )
            self._preserve_partial_response(visible_messages, cycle_response)
            self.conversation_ready.emit(visible_messages)
            self.failed.emit("Generation stopped.", "cancelled")
        except Exception as error:
            LOGGER.error(
                "[BACKEND] worker_error chunks_received=%d chars_received=%d "
                "stream_finished=%s exception=%s",
                stream_chunks,
                stream_characters,
                str(stream_completion_received).lower(),
                type(error).__name__,
            )
            self._preserve_partial_response(visible_messages, cycle_response)
            self.conversation_ready.emit(visible_messages)
            self.failed.emit(
                format_backend_error(error, self.service.backend_name), "error"
            )

    @staticmethod
    def _preserve_partial_response(
        visible_messages: list[ChatMessage],
        cycle_response: list[str],
    ) -> None:
        partial = "".join(cycle_response)
        if partial:
            visible_messages.append(ChatMessage("assistant", partial))

    def _execute_tool_call(
        self,
        tool_call: ToolCall,
        call_id: str | None = None,
    ) -> ToolCallResult:
        resolved_call_id = call_id or tool_call.id or f"{tool_call.name}-{uuid.uuid4().hex}"
        permission = self.tool_manager.permission_for(
            tool_call.name, tool_call.arguments
        )
        self.tool_started.emit(
            resolved_call_id,
            tool_call.name,
            tool_call.arguments,
            permission.value,
        )
        approved = permission in {PermissionLevel.SAFE, PermissionLevel.NORMAL}
        if permission not in {PermissionLevel.SAFE, PermissionLevel.NORMAL, PermissionLevel.BLOCKED}:
            approved = self._wait_for_approval(
                resolved_call_id, tool_call, permission
            )
        try:
            result = self.tool_manager.execute(
                tool_call.name, tool_call.arguments, approved
            )
        except ToolConfirmationRequired:
            result = ToolCallResult(False, "The user did not approve this action.")
        self.tool_completed.emit(
            resolved_call_id,
            tool_call.name,
            result.content,
            result.success,
            result.images,
        )
        if _stream_debug_enabled():
            LOGGER.info(
                "[TOOL FLOW] dispatched_tool=%s tool_result_success=%s",
                tool_call.name,
                str(result.success).lower(),
            )
        return result

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
    speech_started = Signal()
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        service: VoiceService,
        destination: Path,
        duration_seconds: int | None = None,
        detect_speech: bool = False,
        min_speech_duration: float | None = None,
    ) -> None:
        super().__init__()
        self.service = service
        self.destination = destination
        self.duration_seconds = duration_seconds
        self.detect_speech = detect_speech
        self.min_speech_duration = min_speech_duration
        self.speech_detected = False
        self._stop_event = threading.Event()
        self._process: subprocess.Popen | None = None

    @Slot()
    def run(self) -> None:
        process = None
        try:
            LOGGER.info("[Voice] Initializing recording: %s", self.destination)
            process = self.service.start_recorder(self.destination)
            self._process = process
            self.started.emit()
            LOGGER.info("[Voice] Listening for recorded input")
            if self.duration_seconds is None and not self.detect_speech:
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
            LOGGER.info("[Voice] Audio finalized: %s", self.destination)
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
        deadline = (
            time.monotonic() + float(self.duration_seconds)
            if self.duration_seconds is not None
            else None
        )
        detector = VoiceActivityDetector(
            threshold=self.service.config.voice_vad_threshold,
            silence_duration=self.service.config.voice_silence_duration,
            min_speech_duration=(
                self.min_speech_duration
                if self.min_speech_duration is not None
                else self.service.config.voice_min_speech_duration
            ),
        )
        offset = 44
        while process.poll() is None and not self._stop_event.is_set():
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                return
            try:
                with self.destination.open("rb") as audio:
                    audio.seek(offset)
                    samples = audio.read()
                if len(samples) % 2:
                    samples = samples[:-1]
                if samples:
                    offset += len(samples)
                    was_started = detector.speech_started
                    should_stop = detector.consume(samples, now)
                    if detector.speech_started and not was_started:
                        self.speech_detected = True
                        self.speech_started.emit()
                    if should_stop:
                        return
            except OSError:
                pass
            if detector.should_stop(now):
                return
            self._stop_event.wait(0.1)


class WakeWordWorker(QObject):
    """Continuously capture one stream while scanning overlapping audio windows."""

    detected = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        service: VoiceService,
        wake_word: str | tuple[str, ...],
        chunk_seconds: float = DEFAULT_WAKE_WORD_WINDOW_SECONDS,
    ) -> None:
        super().__init__()
        self.service = service
        candidates = (wake_word,) if isinstance(wake_word, str) else wake_word
        self.wake_words = tuple(
            value.casefold().strip()
            for value in candidates
            if isinstance(value, str) and value.strip()
        )
        if not self.wake_words:
            raise ValueError("At least one wake word is required.")
        # Preserve the old attribute for callers that display the primary alias.
        self.wake_word = self.wake_words[0]
        self.chunk_seconds = chunk_seconds
        configured_threshold = getattr(
            getattr(service, "config", None),
            "wake_word_match_threshold",
            0.68,
        )
        try:
            self.match_threshold = float(configured_threshold)
        except (TypeError, ValueError):
            self.match_threshold = 0.68
        self._stop_event = threading.Event()
        self._process: subprocess.Popen | None = None

    @Slot()
    def run(self) -> None:
        stream_path: Path | None = None
        process: subprocess.Popen | None = None
        try:
            LOGGER.info("[WakeWord] Initializing aliases: %s", ", ".join(self.wake_words))
            stream_path = self.service.new_recording_path()
            process = self.service.start_recorder(stream_path)
            self._process = process
            if _wake_debug_enabled():
                LOGGER.info(
                    "[WakeWord] capture device=%r sample_rate=%d channels=%d "
                    "sample_width=%d window_bytes=%d threshold=%.2f state=listening",
                    getattr(self.service.config, "microphone_device", ""),
                    WAKE_WORD_SAMPLE_RATE,
                    WAKE_WORD_CHANNELS,
                    WAKE_WORD_SAMPLE_WIDTH,
                    max(
                        1,
                        int(
                            self.chunk_seconds
                            * WAKE_WORD_SAMPLE_RATE
                            * WAKE_WORD_SAMPLE_WIDTH
                        ),
                    ),
                    self.match_threshold,
                )
            else:
                LOGGER.info("[WakeWord] Microphone connected")
            if self.chunk_seconds <= 0:
                # A zero-length window is useful for deterministic worker
                # tests and keeps the worker's public seam backwards
                # compatible; normal wake listening always uses a real window.
                transcript = self.service.transcribe(stream_path, device="cpu").casefold()
                matched_wake_word = find_wake_word(transcript, self.wake_words)
                if matched_wake_word is not None:
                    LOGGER.info("[WakeWord] Wake word detected")
                    self.detected.emit(
                        remove_wake_word(transcript, matched_wake_word) or ""
                    )
                return
            overlap_seconds = min(self.chunk_seconds / 2, 2.0)
            LOGGER.info(
                "[WakeWord] Listening: %.1fs windows with %.1fs overlap",
                self.chunk_seconds,
                overlap_seconds,
            )
            window_bytes = max(
                WAKE_WORD_SAMPLE_WIDTH,
                int(
                    self.chunk_seconds
                    * WAKE_WORD_SAMPLE_RATE
                    * WAKE_WORD_SAMPLE_WIDTH
                ),
            )
            window_bytes -= window_bytes % WAKE_WORD_SAMPLE_WIDTH
            overlap_bytes = min(
                window_bytes // 2,
                int(
                    overlap_seconds
                    * WAKE_WORD_SAMPLE_RATE
                    * WAKE_WORD_SAMPLE_WIDTH
                ),
            )
            overlap_bytes -= overlap_bytes % WAKE_WORD_SAMPLE_WIDTH
            pcm = b""
            offset = 44
            frames_received = 0
            frames_dropped = 0
            windows_processed = 0
            last_detection_at: float | None = None
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
                    frames_received += usable // WAKE_WORD_SAMPLE_WIDTH
                    if _wake_debug_enabled():
                        LOGGER.debug(
                            "[WakeWord] audio frames_received=%d bytes=%d",
                            frames_received,
                            usable,
                        )
                if len(pcm) >= window_bytes:
                    snapshot = self.service.new_recording_path()
                    try:
                        with wave.open(str(snapshot), "wb") as audio:
                            audio.setnchannels(1)
                            audio.setsampwidth(2)
                            audio.setframerate(WAKE_WORD_SAMPLE_RATE)
                            audio.writeframes(pcm[-window_bytes:])
                        # Keep wake listening off the GPU used by Ollama and
                        # other desktop workloads. The recorder itself stays
                        # open, so Whisper processing never creates a
                        # microphone-monitoring gap.
                        transcript = self.service.transcribe(
                            snapshot, device="cpu"
                        ).casefold()
                        windows_processed += 1
                        scores = {
                            wake_word: wake_word_match_score(transcript, wake_word)
                            for wake_word in self.wake_words
                        }
                        best_score = max(scores.values(), default=0.0)
                        if _wake_debug_enabled():
                            LOGGER.info(
                                "[WakeWord] device=%r sample_rate=%d channels=%d "
                                "format=s16le window_bytes=%d window=%d "
                                "frames_received=%d frames_dropped=%d "
                                "confidence=%.3f threshold=%.2f state=%s "
                                "last_detection=%s cooldown=inactive",
                                getattr(self.service.config, "microphone_device", ""),
                                WAKE_WORD_SAMPLE_RATE,
                                WAKE_WORD_CHANNELS,
                                window_bytes,
                                windows_processed,
                                frames_received,
                                frames_dropped,
                                best_score,
                                self.match_threshold,
                                "listening",
                                last_detection_at,
                            )
                        matched_wake_word = find_wake_word(
                            transcript,
                            self.wake_words,
                            threshold=self.match_threshold,
                        )
                        if matched_wake_word is not None:
                            last_detection_at = time.monotonic()
                            LOGGER.info(
                                "[WakeWord] Wake word detected "
                                "(confidence=%.3f threshold=%.2f)",
                                scores.get(matched_wake_word, 0.0),
                                self.match_threshold,
                            )
                            self.detected.emit(
                                remove_wake_word(transcript, matched_wake_word) or ""
                            )
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
                LOGGER.exception("[WakeWord] Listener failed")
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


class ReminderAlarmWorker(QObject):
    """Repeat a spoken reminder until the user dismisses it."""

    finished = Signal()
    failed = Signal(str)

    def __init__(
        self,
        service: VoiceService,
        message: str,
        repeat_interval: float = 6.0,
    ) -> None:
        super().__init__()
        self.service = service
        self.message = message
        self.repeat_interval = repeat_interval
        self.cancel_event = threading.Event()

    @Slot()
    def run(self) -> None:
        alert = f"Reminder. It's time to {self.message}."
        try:
            while not self.cancel_event.is_set():
                self.service.speak(alert, self.cancel_event)
                if self.cancel_event.wait(self.repeat_interval):
                    break
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
