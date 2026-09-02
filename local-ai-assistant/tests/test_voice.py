from __future__ import annotations

import json
import tempfile
import unittest
import signal
import sys
from pathlib import Path
from unittest.mock import Mock, patch

from local_ai_assistant.config import AppConfig
from local_ai_assistant.voice import (
    SpeechChunker,
    VoiceActivityDetector,
    VoiceError,
    VoiceService,
    conversation_end_requested,
    is_no_speech_error,
    remove_wake_word,
    find_wake_word,
    speech_text,
    wake_word_matches,
)


class VoiceServiceTests(unittest.TestCase):
    def test_silent_whisper_window_is_not_a_fatal_wake_error(self) -> None:
        self.assertTrue(is_no_speech_error("Whisper failed: no speech was detected"))
        self.assertTrue(is_no_speech_error("silence detected"))
        self.assertFalse(is_no_speech_error("model file is missing"))

    def test_speech_text_removes_non_speech_markup(self) -> None:
        self.assertEqual(
            speech_text("**GPU** is at 42% 🔥\n`nvidia-smi`"),
            "GPU is at 42% nvidia-smi",
        )

    def test_speech_chunker_emits_complete_sentences_and_flushes_remainder(self) -> None:
        chunker = SpeechChunker()
        self.assertEqual(chunker.feed("Certainly, Sir. The weather today is"), ["Certainly, Sir."])
        self.assertEqual(chunker.feed(" clear."), ["The weather today is clear."])
        self.assertEqual(chunker.flush(), [])

    def test_wake_word_matching_tolerates_common_whisper_spelling(self) -> None:
        self.assertTrue(wake_word_matches("hey Laura", "Lura"))
        self.assertTrue(wake_word_matches("Lara", "Lura"))
        self.assertFalse(wake_word_matches("coloration", "Lura"))
        self.assertEqual(remove_wake_word("Laura, what's up?", "Lura"), "what's up?")
        self.assertEqual(find_wake_word("Luda, what's up?", ("Luna", "Luda")), "Luda")
        self.assertEqual(
            remove_wake_word(
                "Luda, what's up?",
                find_wake_word("Luda, what's up?", ("Luna", "Luda")) or "",
            ),
            "what's up?",
        )

    def test_conversation_end_phrases_are_conservative(self) -> None:
        for phrase in (
            "bye",
            "Goodbye.",
            "see you later",
            "see ya",
            "I'm done",
            "that's all",
            "end the conversation",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(conversation_end_requested(phrase))
        self.assertTrue(conversation_end_requested("Okay, goodbye"))
        self.assertTrue(conversation_end_requested("go to sleep now"))
        self.assertTrue(conversation_end_requested("please stop listening"))
        self.assertTrue(conversation_end_requested("I am done now"))
        self.assertTrue(conversation_end_requested("that's all"))
        self.assertTrue(conversation_end_requested("please stop listenin"))
        self.assertTrue(conversation_end_requested("goodby"))
        self.assertFalse(conversation_end_requested("don't stop listening, keep going"))
        self.assertFalse(conversation_end_requested("I am not done yet"))
        self.assertFalse(conversation_end_requested("please don't say goodbye"))
        self.assertFalse(conversation_end_requested("stand by for instructions"))
        self.assertFalse(conversation_end_requested("stop the timer"))
        self.assertFalse(conversation_end_requested("tell me about goodbyes"))

    def test_vad_waits_for_sustained_silence_after_speech(self) -> None:
        detector = VoiceActivityDetector(
            threshold=350,
            silence_duration=0.9,
            min_speech_duration=0.2,
        )
        silence = b"\0\0" * 3200  # 200 ms at 16 kHz
        voice = (900).to_bytes(2, "little", signed=True) * 3200

        self.assertFalse(detector.consume(silence, 0.2))
        self.assertFalse(detector.consume(voice, 0.4))
        self.assertTrue(detector.speech_started)
        self.assertFalse(detector.consume(silence, 0.7))
        self.assertFalse(detector.should_stop(1.0))
        self.assertTrue(detector.should_stop(1.35))

    def test_vad_ignores_short_impulses_and_requires_continuous_voice(self) -> None:
        detector = VoiceActivityDetector(
            threshold=350,
            silence_duration=0.9,
            min_speech_duration=0.3,
        )
        silence = b"\0\0" * 3200  # 200 ms calibration
        noise = (500).to_bytes(2, "little", signed=True) * 800  # 50 ms click
        voice = (900).to_bytes(2, "little", signed=True) * 3200

        self.assertFalse(detector.consume(silence, 0.2))
        self.assertFalse(detector.consume(noise, 0.25))
        self.assertFalse(detector.speech_started)
        self.assertFalse(detector.consume(voice, 0.45))
        self.assertFalse(detector.speech_started)
        self.assertFalse(detector.consume(voice, 0.65))
        self.assertTrue(detector.speech_started)

    def test_vad_does_not_treat_calibration_noise_as_speech(self) -> None:
        detector = VoiceActivityDetector(
            threshold=350,
            silence_duration=0.9,
            min_speech_duration=0.2,
        )
        background = (300).to_bytes(2, "little", signed=True) * 3200
        voice = (900).to_bytes(2, "little", signed=True) * 3200

        self.assertFalse(detector.consume(background, 0.2))
        self.assertFalse(detector.speech_started)
        self.assertFalse(detector.consume(voice, 0.4))
        self.assertTrue(detector.speech_started)

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

    def test_openai_whisper_can_be_forced_to_cpu(self) -> None:
        service = VoiceService(AppConfig())
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "recording.wav"
            audio.write_bytes(b"RIFF" + b"\0" * 42)

            def run_whisper(command: list[str], *args, **kwargs) -> Mock:
                output_directory = Path(command[command.index("--output_dir") + 1])
                (output_directory / "recording.txt").write_text(
                    "Lura", encoding="utf-8"
                )
                return Mock(returncode=0, stdout="", stderr="")

            with patch(
                "local_ai_assistant.voice.shutil.which",
                side_effect=lambda name: "/usr/bin/whisper"
                if name == "whisper"
                else None,
            ), patch.object(
                service, "_run_checked", side_effect=run_whisper
            ) as run:
                self.assertEqual(service.transcribe(audio, device="cpu"), "Lura")

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--device") + 1], "cpu")

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