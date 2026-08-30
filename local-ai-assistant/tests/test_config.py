from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_ai_assistant.config import AppConfig, DEFAULT_MODEL, DEFAULT_OLLAMA_URL


class ConfigTests(unittest.TestCase):
    def test_defaults_are_stable(self) -> None:
        config = AppConfig.defaults()
        self.assertEqual(config.ollama_url, DEFAULT_OLLAMA_URL)
        self.assertEqual(config.model, DEFAULT_MODEL)

    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            original = AppConfig("http://127.0.0.1:11435/", "llama3.2:3b")
            original.save(path)
            loaded = AppConfig.load(path)
        self.assertEqual(loaded, AppConfig("http://127.0.0.1:11435", "llama3.2:3b"))

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AppConfig("localhost:11434", DEFAULT_MODEL)
        with self.assertRaises(ValueError):
            AppConfig(DEFAULT_OLLAMA_URL, " ")

    def test_damaged_config_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(AppConfig.load(path), AppConfig.defaults())
