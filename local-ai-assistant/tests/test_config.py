from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_ai_assistant.config import (
    AppConfig,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_TTS_ENGINE,
    DEFAULT_WHISPER_MODEL,
)


class ConfigTests(unittest.TestCase):
    def test_defaults_are_stable(self) -> None:
        config = AppConfig.defaults()
        self.assertEqual(config.ollama_url, DEFAULT_OLLAMA_URL)
        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertEqual(config.ollama_context_size, DEFAULT_CONTEXT_SIZE)
        self.assertTrue(config.voice_input_enabled)
        self.assertFalse(config.voice_responses_enabled)
        self.assertEqual(config.whisper_model, DEFAULT_WHISPER_MODEL)
        self.assertEqual(config.tts_engine, DEFAULT_TTS_ENGINE)

    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            original = AppConfig(
                "http://127.0.0.1:11435/",
                "llama3.2:3b",
                16384,
                voice_input_enabled=False,
                voice_responses_enabled=True,
                microphone_device="alsa_input.pci",
                whisper_model="small",
                whisper_language="en",
                tts_engine="piper",
                tts_voice="/models/en_US.onnx",
                background_mode_enabled=True,
                autostart_enabled=True,
            )
            original.save(path)
            loaded = AppConfig.load(path)
        self.assertEqual(
            loaded,
            AppConfig(
                "http://127.0.0.1:11435",
                "llama3.2:3b",
                16384,
                voice_input_enabled=False,
                voice_responses_enabled=True,
                microphone_device="alsa_input.pci",
                whisper_model="small",
                whisper_language="en",
                tts_engine="piper",
                tts_voice="/models/en_US.onnx",
                background_mode_enabled=True,
                autostart_enabled=True,
            ),
        )

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AppConfig("localhost:11434", DEFAULT_MODEL)
        with self.assertRaises(ValueError):
            AppConfig(DEFAULT_OLLAMA_URL, " ")
        with self.assertRaises(ValueError):
            AppConfig(ollama_context_size=5000)
        with self.assertRaises(ValueError):
            AppConfig(tts_engine="unknown")
        with self.assertRaises(ValueError):
            AppConfig(whisper_model=" ")

    def test_damaged_config_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(AppConfig.load(path), AppConfig.defaults())
