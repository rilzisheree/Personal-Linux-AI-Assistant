from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from local_ai_assistant.config import AppConfig
from local_ai_assistant.voice import VoiceError, VoiceService


class VoiceServiceTests(unittest.TestCase):
    def test_recorder_prefers_pipewire_and_passes_device(self) -> None:
        config = AppConfig(microphone_device="my-mic")
        service = VoiceService(config)
        with patch(
            "local_ai_assistant.voice.shutil.which",
            side_effect=lambda name: "/usr/bin/pw-record" if name == "pw-record" else None,
        ):
            command = service.recorder_command(Path("/tmp/recording.wav"))
        self.assertEqual(
            command,
            [
                "pw-record",
                "--rate",
                "16000",
                "--channels",
                "1",
                "--format",
                "s16",
                "--target",
                "my-mic",
                "/tmp/recording.wav",
            ],
        )

    def test_alsa_device_uses_arecord_when_pipewire_is_available(self) -> None:
        service = VoiceService(AppConfig(microphone_device="plughw:2,0"))
        with patch(
            "local_ai_assistant.voice.shutil.which",
            side_effect=lambda name: f"/usr/bin/{name}"
            if name in {"pw-record", "arecord"}
            else None,
        ):
            command = service.recorder_command(Path("/tmp/recording.wav"))
        self.assertEqual(command[:2], ["arecord", "-q"])
        self.assertIn("-D", command)
        self.assertIn("plughw:2,0", command)

    def test_transcribe_reports_missing_local_backend(self) -> None:
        service = VoiceService(AppConfig())
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "recording.wav"
            audio.write_bytes(b"RIFF" + b"\0" * 42)
            with patch("local_ai_assistant.voice.shutil.which", return_value=None):
                with self.assertRaisesRegex(VoiceError, "No local Whisper backend"):
                    service.transcribe(audio)

    def test_piper_receives_response_text_on_stdin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "voice.onnx"
            model.write_bytes(b"model")
            service = VoiceService(
                AppConfig(tts_engine="piper", tts_voice=str(model))
            )
            with patch(
                "local_ai_assistant.voice.shutil.which",
                side_effect=lambda name: "/usr/bin/piper" if name == "piper" else None,
            ), patch(
                "local_ai_assistant.voice.subprocess.run",
                return_value=Mock(returncode=0, stdout="", stderr=""),
            ) as run_mock:
                service._synthesize("Hello locally", Path(directory) / "out.wav")
        self.assertEqual(run_mock.call_args.kwargs["input"], "Hello locally")
        self.assertEqual(run_mock.call_args.args[0][0], "/usr/bin/piper")


if __name__ == "__main__":
    unittest.main()