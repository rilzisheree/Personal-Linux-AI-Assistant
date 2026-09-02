from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from local_ai_assistant.assistant_core import RouteDecision
from local_ai_assistant.ollama import ChatMessage, StreamEvent
from local_ai_assistant.applications import ApplicationRecord
from local_ai_assistant.tools import ToolManager
from local_ai_assistant.voice import VoiceError
from local_ai_assistant.workers import (
    ChatWorker,
    DEFAULT_WAKE_WORD_WINDOW_SECONDS,
    VoiceRecordWorker,
    VoiceTranscriptionWorker,
)


class VoiceWorkerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()