from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from local_ai_assistant.voice import VoiceError
from local_ai_assistant.workers import VoiceRecordWorker, VoiceTranscriptionWorker


class VoiceWorkerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()