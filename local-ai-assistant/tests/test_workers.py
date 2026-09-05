from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from local_ai_assistant.assistant_core import RouteDecision
from local_ai_assistant.errors import OllamaProtocolError
from local_ai_assistant.ollama import ChatMessage, StreamEvent, ToolCall
from local_ai_assistant.applications import ApplicationRecord
from local_ai_assistant.reminders import ReminderService, ReminderStore
from local_ai_assistant.tools import ToolManager
from local_ai_assistant.voice import VoiceError
from local_ai_assistant.workers import (
    ChatWorker,
    DEFAULT_WAKE_WORD_WINDOW_SECONDS,
    ReminderAlarmWorker,
    VoiceRecordWorker,
    VoiceTranscriptionWorker,
)


class VoiceWorkerTests(unittest.TestCase):
    def test_reminder_alarm_repeats_until_cancelled(self) -> None:
        service = Mock()
        worker = ReminderAlarmWorker(service, "drink water", repeat_interval=0)
        finished: list[bool] = []
        worker.finished.connect(lambda: finished.append(True))

        def speak(_text: str, _cancel_event) -> None:
            worker.cancel()

        service.speak.side_effect = speak
        worker.run()

        service.speak.assert_called_once()
        self.assertEqual(
            service.speak.call_args.args[0],
            "Reminder. It's time to drink water.",
        )
        self.assertEqual(finished, [True])

    def test_wake_word_worker_uses_a_short_low_latency_window_by_default(self) -> None:
        from local_ai_assistant.workers import WakeWordWorker

        worker = WakeWordWorker(Mock(), "Lura")

        self.assertEqual(worker.chunk_seconds, DEFAULT_WAKE_WORD_WINDOW_SECONDS)
        self.assertLessEqual(worker.chunk_seconds, 1.5)

    def test_wake_word_worker_transcribes_on_cpu(self) -> None:
        from local_ai_assistant.workers import WakeWordWorker

        service = Mock()
        process = Mock()
        service.new_recording_path.return_value = Path("/tmp/lura-wake-test.wav")
        service.start_recorder.return_value = process
        service.transcribe.return_value = "Luda, what time is it?"
        process.poll.return_value = None

        worker = WakeWordWorker(service, ("Luna", "Luda"), chunk_seconds=0)
        commands: list[str] = []
        worker.detected.connect(commands.append)
        worker.run()

        service.transcribe.assert_called_once_with(
            Path("/tmp/lura-wake-test.wav"), device="cpu"
        )
        self.assertEqual(commands, ["what time is it?"])

    def test_wake_word_worker_uses_configured_match_threshold(self) -> None:
        from local_ai_assistant.workers import WakeWordWorker

        service = Mock()
        service.config.wake_word_match_threshold = 0.81
        worker = WakeWordWorker(service, "Lura")

        self.assertEqual(worker.match_threshold, 0.81)

    def test_record_worker_stop_sets_cancel_and_terminates_process(self) -> None:
        service = Mock()
        process = Mock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            worker = VoiceRecordWorker(service, Path(directory) / "recording.wav")
            worker._process = process
            worker.stop()
        self.assertTrue(worker._stop_event.is_set())
        service.stop_recorder.assert_called_once_with(process)
        process.send_signal.assert_not_called()

    def test_record_worker_reports_backend_failure(self) -> None:
        service = Mock()
        service.start_recorder.side_effect = VoiceError("recorder missing")
        failed: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            worker = VoiceRecordWorker(service, Path(directory) / "recording.wav")
            worker.failed.connect(failed.append)
            worker.run()
        self.assertEqual(failed, ["recorder missing"])

    def test_record_worker_processes_audio_after_release_even_if_start_is_pending(self) -> None:
        service = Mock()
        process = Mock()
        process.poll.return_value = None
        service.start_recorder.return_value = process
        service.finish_recording.return_value = None
        finished: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "recording.wav"
            audio_path.write_bytes(b"RIFF" + b"\0" * 100)
            worker = VoiceRecordWorker(service, audio_path)
            worker.finished.connect(finished.append)
            worker.stop()
            worker.run()

        service.transcribe.assert_not_called()
        service.finish_recording.assert_called_once_with(process, audio_path)
        self.assertEqual(finished, [str(audio_path)])

    def test_record_worker_can_report_speech_start_for_barge_in(self) -> None:
        service = Mock()
        service.config.voice_vad_threshold = 350
        service.config.voice_silence_duration = 0.05
        service.config.voice_min_speech_duration = 0.2
        process = Mock()
        process.poll.side_effect = [None, None, None, 0, 0]
        service.start_recorder.return_value = process
        service.finish_recording.return_value = None
        speech_started: list[bool] = []
        finished: list[str] = []

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "recording.wav"
            # The worker reads after the header offset. Each poll appends a
            # calibration frame followed by sustained voice and then silence.
            audio_path.write_bytes(
                b"RIFF"
                + b"\0" * 40
                + b"\0\0" * 3200
                + (900).to_bytes(2, "little", signed=True) * 6400
            )
            worker = VoiceRecordWorker(
                service,
                audio_path,
                detect_speech=True,
            )
            worker.speech_started.connect(lambda: speech_started.append(True))
            worker.finished.connect(finished.append)
            worker.run()

        self.assertEqual(speech_started, [True])
        self.assertTrue(worker.speech_detected)
        self.assertEqual(finished, [str(audio_path)])

    def test_transcription_worker_reports_failure_and_path(self) -> None:
        service = Mock()
        service.transcribe.side_effect = VoiceError("Whisper missing")
        failed: list[tuple[str, str]] = []
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "recording.wav"
            worker = VoiceTranscriptionWorker(service, audio_path)
            worker.failed.connect(lambda message, path: failed.append((message, path)))
            worker.run()
        self.assertEqual(failed, [("Whisper missing", str(audio_path))])


class DirectToolDispatchTests(unittest.TestCase):
    def test_chat_worker_preserves_partial_response_when_stream_fails(self) -> None:
        class FailingService:
            display_name = "Fake Ollama"
            backend_name = "Fake Ollama"

            def route_request(self, *args, **kwargs):
                return RouteDecision("simple")

            def stream_reply(
                self,
                messages,
                model,
                cancel_event=None,
                tools=None,
                context_size=None,
            ):
                del messages, model, cancel_event, tools, context_size
                yield StreamEvent("A partial sentence")
                raise OllamaProtocolError("connection ended unexpectedly")

            def cancel_active_request(self) -> None:
                return None

        service = FailingService()
        worker = ChatWorker(
            service,
            [ChatMessage("user", "Explain Linux.")],
            "qwen3.5:2b",
            ToolManager(),
        )
        ready_messages: list[list[ChatMessage]] = []
        failures: list[tuple[str, str]] = []
        worker.conversation_ready.connect(ready_messages.append)
        worker.failed.connect(lambda message, kind: failures.append((message, kind)))

        worker.run()

        self.assertEqual(
            ready_messages[-1][-1],
            ChatMessage("assistant", "A partial sentence"),
        )
        self.assertEqual(failures[0][1], "error")
        self.assertIn("connection ended unexpectedly", failures[0][0])

    def test_chat_worker_rejects_a_stream_without_completion_event(self) -> None:
        class IncompleteService:
            backend_name = "Fake Ollama"

            def route_request(self, *args, **kwargs):
                return RouteDecision("simple")

            def stream_reply(self, *args, **kwargs):
                yield StreamEvent("A partial answer")

            def cancel_active_request(self) -> None:
                return None

        worker = ChatWorker(
            IncompleteService(),
            [ChatMessage("user", "Explain Linux.")],
            "qwen3.5:2b",
            ToolManager(),
        )
        failures: list[tuple[str, str]] = []
        worker.failed.connect(lambda message, kind: failures.append((message, kind)))

        worker.run()

        self.assertEqual(failures[0][1], "error")
        self.assertIn("completion event", failures[0][0])

    def test_chat_worker_rejects_tool_calls_when_no_tools_were_offered(self) -> None:
        class UnexpectedToolService:
            backend_name = "Fake Ollama"

            def route_request(self, *args, **kwargs):
                return RouteDecision("simple")

            def stream_reply(
                self,
                messages,
                model,
                cancel_event=None,
                tools=None,
                context_size=None,
            ):
                del messages, model, cancel_event, tools, context_size
                yield StreamEvent(
                    "",
                    True,
                    (ToolCall("web_search", {"query": "unexpected"}, "call-1"),),
                )

            def cancel_active_request(self) -> None:
                return None

        worker = ChatWorker(
            UnexpectedToolService(),
            [ChatMessage("user", "Say hello.")],
            "qwen3.5:2b",
            ToolManager(),
        )
        failures: list[tuple[str, str]] = []
        worker.failed.connect(lambda message, kind: failures.append((message, kind)))

        worker.run()

        self.assertEqual(failures[0][1], "error")
        self.assertIn("no tools were enabled", failures[0][0])

    def test_chat_worker_reports_the_selected_model_without_routing(self) -> None:
        class FakeService:
            display_name = "Fake Ollama"

            def __init__(self) -> None:
                self.seen_messages: list[ChatMessage] = []

            def route_request(self, *args, **kwargs):
                raise AssertionError("Model identification should be direct-dispatched.")

            def stream_reply(
                self,
                messages,
                model,
                cancel_event=None,
                tools=None,
                context_size=None,
            ):
                self.seen_messages = list(messages)
                yield StreamEvent("The selected model is qwen3.5:2b.", True)

            def cancel_active_request(self) -> None:
                return None

        service = FakeService()
        worker = ChatWorker(
            service,
            [ChatMessage("user", "What model are you currently using?")],
            "qwen3.5:2b",
            ToolManager(),
        )
        finished: list[str] = []
        worker.finished.connect(finished.append)

        worker.run()

        self.assertEqual(finished, ["The selected model is qwen3.5:2b."])
        tool_messages = [message for message in service.seen_messages if message.role == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(
            json.loads(tool_messages[0].content)["data"]["active_model"],
            "qwen3.5:2b",
        )

    def test_chat_worker_executes_explicit_app_request_before_model_reply(self) -> None:
        class FakeService:
            display_name = "Fake Ollama"

            def __init__(self) -> None:
                self.seen_messages: list[ChatMessage] = []

            def route_request(self, messages, tools=None, cancel_event=None):
                return RouteDecision("reasoning")

            def stream_reply(
                self,
                messages,
                model,
                cancel_event=None,
                tools=None,
                context_size=None,
            ):
                self.seen_messages = list(messages)
                yield StreamEvent("Firefox is open, Sir.", True)

            def cancel_active_request(self) -> None:
                return None

        manager = ToolManager()
        manager.application_registry.resolve = Mock(
            return_value=ApplicationRecord(
                app_id="org.mozilla.firefox",
                name="Firefox",
                kind="flatpak",
                launch_command=("flatpak", "run", "org.mozilla.firefox"),
            )
        )
        service = FakeService()
        worker = ChatWorker(
            service,
            [ChatMessage("user", "Open Firefox")],
            "qwen3.5:2b",
            manager,
        )
        finished: list[str] = []
        worker.finished.connect(finished.append)

        with unittest.mock.patch("local_ai_assistant.tools.subprocess.Popen"):
            worker.run()

        self.assertEqual(finished, ["Firefox is open, Sir."])
        tool_messages = [message for message in service.seen_messages if message.role == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("org.mozilla.firefox", tool_messages[0].content)

    def test_chat_worker_sends_reminder_schema_and_persists_model_tool_call(self) -> None:
        class FakeService:
            display_name = "Fake Ollama"

            def __init__(self) -> None:
                self.calls: list[dict] = []

            def route_request(self, messages, tools=None, cancel_event=None):
                del messages, tools, cancel_event
                return RouteDecision("function")

            def stream_reply(
                self,
                messages,
                model,
                cancel_event=None,
                tools=None,
                context_size=None,
            ):
                del model, cancel_event, context_size
                self.calls.append({"messages": list(messages), "tools": list(tools or [])})
                if len(self.calls) == 1:
                    yield StreamEvent(
                        "",
                        True,
                        (
                            ToolCall(
                                "create_reminder",
                                {"message": "drink water", "delay_seconds": 5},
                                "call-reminder-1",
                            ),
                        ),
                    )
                else:
                    yield StreamEvent("Reminder created successfully.", True)

            def cancel_active_request(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as directory:
            manager = ToolManager()
            manager.reminder_service = ReminderService(
                ReminderStore(Path(directory) / "reminders.json"),
                notify_send="",
                start_scheduler=False,
            )
            service = FakeService()
            worker = ChatWorker(
                service,
                [ChatMessage("user", "Remind me in 5 seconds to drink water.")],
                "qwen3.5:2b",
                manager,
            )
            finished: list[str] = []
            failures: list[tuple[str, str]] = []
            worker.finished.connect(finished.append)
            worker.failed.connect(lambda message, kind: failures.append((message, kind)))

            worker.run()

            for call in service.calls:
                names = {
                    tool["function"]["name"]
                    for tool in call["tools"]
                }
                self.assertIn("create_reminder", names)
            self.assertEqual(len(service.calls), 2)
            self.assertEqual(finished, ["Reminder created successfully."])
            self.assertEqual(failures, [])
            reminders = manager.reminder_service.store.list()
            self.assertEqual(len(reminders), 1)
            self.assertEqual(reminders[0].message, "drink water")


if __name__ == "__main__":
    unittest.main()