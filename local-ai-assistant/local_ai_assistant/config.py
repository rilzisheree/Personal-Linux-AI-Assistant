"""Persistent, validated application settings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:4b"


@dataclass
class AppConfig:
    """Settings that are safe to persist locally."""

    ollama_url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
        self.ollama_url = self.ollama_url.strip().rstrip("/")
        self.model = self.model.strip()
        self.validate()

    def validate(self) -> None:
        parsed = urlparse(self.ollama_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Ollama URL must be a complete http:// or https:// URL.")
        if not self.model:
            raise ValueError("Ollama model cannot be empty.")

    @classmethod
    def defaults(cls) -> "AppConfig":
        return cls()

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        config_path = path or cls.default_path()
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            return cls(
                ollama_url=str(raw.get("ollama_url", DEFAULT_OLLAMA_URL)),
                model=str(raw.get("model", DEFAULT_MODEL)),
            )
        except FileNotFoundError:
            return cls.defaults()
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            # A damaged local config should not prevent the application from
            # opening. The user can replace it through Settings.
            return cls.defaults()

    def save(self, path: Path | None = None) -> None:
        config_path = path or self.default_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def default_path() -> Path:
        return Path.home() / ".config" / "local-ai-assistant" / "config.json"
