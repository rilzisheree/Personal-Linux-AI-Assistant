from __future__ import annotations

import json
import tempfile
import unittest
import signal
import sys
from pathlib import Path
from unittest.mock import Mock, patch

from local_ai_assistant.config import AppConfig
from local_ai_assistant.voice import VoiceError, VoiceService


class VoiceServiceTests(unittest.TestCase):
    def test_list_microphones_filters_monitors_and_keeps_source_names(self) -> None:
        pactl_output = json.dumps(
            [
                {
                    "name": "alsa_input.usb-hyperx.analog-stereo",
                    "description": "HyperX SoloCast",
                    "properties": {},
                    "monitor_of_sink": None,
                },
                {
                    "name": "alsa_output.pci.monitor",
                    "description": "Built-in Audio Monitor",
                    "properties": {},
                    "monitor_of_sink": 12,
                },
            ]
        )
        with patch(
            "local_ai_assistant.voice.shutil.which",
            side_effect=lambda name: "/usr/bin/pactl" if name == "pactl" else None,
        ), patch(
            "local_ai_assistant.voice.subprocess.run",
            return_value=Mock(returncode=0, stdout=pactl_output),
        ):
            microphones = VoiceService.list_microphones()
        self.assertEqual(
            microphones,
            [("alsa_input.usb-hyperx.analog-stereo", "HyperX SoloCast")],
        )

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
            with patch("local_ai_assistant.voice.shutil.which", return_value=None), patch.dict(
                sys.modules, {"whisper": None}
            ):
                with self.assertRaisesRegex(VoiceError, "No local Whisper backend"):
                    service.transcribe(audio)

    def test_stop_recorder_uses_sigint_to_finalize_audio(self) -> None:
        service = VoiceService(AppConfig())
        process = Mock()
        process.poll.return_value = None
        service.stop_recorder(process)
        process.send_signal.assert_called_once_with(signal.SIGINT)
        process.terminate.assert_not_called()

    def test_finish_accepts_pw_record_exit_one_with_valid_wav(self) -> None:
        service = VoiceService(AppConfig())
        process = Mock(returncode=1)
        process.stderr.read.return_value = "recording.wav"
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "recording.wav"
            audio.write_bytes(b"RIFF" + b"\0" * 100)
            service.finish_recording(process, audio)

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
                side_effect=lambda command, **kwargs: (
                    Path(command[command.index("--output_file") + 1]).write_bytes(b"RIFF"),
                    Mock(returncode=0, stdout="", stderr=""),
                )[1],
            ) as run_mock:
                service._synthesize("Hello locally", Path(directory) / "out.wav")
        self.assertEqual(run_mock.call_args.kwargs["input"], "Hello locally")
        self.assertEqual(run_mock.call_args.args[0][0], "/usr/bin/piper")

    def test_piper_requires_an_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "voice.onnx"
            model.write_bytes(b"model")
            service = VoiceService(AppConfig(tts_engine="piper", tts_voice=str(model)))
            with patch(
                "local_ai_assistant.voice.shutil.which",
                side_effect=lambda name: "/usr/bin/piper" if name == "piper" else None,
            ), patch(
                "local_ai_assistant.voice.subprocess.run",
                return_value=Mock(returncode=0, stdout="", stderr=""),
            ):
                with self.assertRaisesRegex(VoiceError, "without producing"):
                    service._synthesize("Hello locally", Path(directory) / "out.wav")

    def test_piper_does_not_fall_back_to_espeak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.wav"
            service = VoiceService(
                AppConfig(tts_engine="piper", tts_voice="~/missing.onnx")
            )
            with patch(
                "local_ai_assistant.voice.shutil.which",
                side_effect=lambda name: (
                    "/usr/bin/espeak-ng" if name == "espeak-ng" else None
                ),
            ) as run_mock:
                with self.assertRaisesRegex(VoiceError, "model was not found"):
                    service._synthesize("Hello locally", output)


if __name__ == "__main__":
    unittest.main()