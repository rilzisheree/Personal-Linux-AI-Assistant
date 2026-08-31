"""Persistent, validated application settings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_TTS_ENGINE = "espeak-ng"
DEFAULT_TTS_VOICE = "en-gb+m3"
TTS_ENGINES = ("disabled", "espeak-ng", "piper")
ESPEAK_VOICE_PRESETS = (
    ("British · Jarvis-style", "en-gb+m3"),
    ("Female-sounding", "en-gb+f2"),
)
PIPER_VOICE_PRESETS = (
    ("British · Alan (Piper)", "~/Models/piper/en_GB-alan-medium.onnx"),
    ("Female · Amy (Piper)", "~/Models/piper/en_US-amy-medium.onnx"),
)
TTS_VOICE_PRESETS = ESPEAK_VOICE_PRESETS + PIPER_VOICE_PRESETS


@dataclass
class AppConfig:
    """Settings that are safe to persist locally."""

    ollama_url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_MODEL
    ollama_context_size: int = DEFAULT_CONTEXT_SIZE
    voice_input_enabled: bool = True
    voice_responses_enabled: bool = False
    microphone_device: str = ""
    whisper_model: str = DEFAULT_WHISPER_MODEL
    whisper_language: str = "auto"
    tts_engine: str = DEFAULT_TTS_ENGINE
    tts_voice: str = DEFAULT_TTS_VOICE
    background_mode_enabled: bool = False
    autostart_enabled: bool = False
    telegram_enabled: bool = False
    telegram_allowed_user_id: str = ""

    def __post_init__(self) -> None:
        self.ollama_url = self.ollama_url.strip().rstrip("/")
        self.model = self.model.strip()
        if isinstance(self.ollama_context_size, str):
            self.ollama_context_size = int(self.ollama_context_size.strip())
        self.microphone_device = self.microphone_device.strip()
        self.whisper_model = self.whisper_model.strip()
        self.whisper_language = self.whisper_language.strip() or "auto"
        self.tts_engine = self.tts_engine.strip().lower()
        self.tts_voice = self.tts_voice.strip()
        self.telegram_allowed_user_id = str(self.telegram_allowed_user_id).strip()
        self.validate()

    def validate(self) -> None:
        parsed = urlparse(self.ollama_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Ollama URL must be a complete http:// or https:// URL.")
        if not self.model:
            raise ValueError("Ollama model cannot be empty.")
        if (
            isinstance(self.ollama_context_size, bool)
            or not isinstance(self.ollama_context_size, int)
            or not 2048 <= self.ollama_context_size <= 131072
            or self.ollama_context_size % 1024
        ):
            raise ValueError("Ollama context size must be a multiple of 1024 between 2048 and 131072.")
        if not isinstance(self.voice_input_enabled, bool):
            raise ValueError("Voice input setting must be true or false.")
        if not isinstance(self.voice_responses_enabled, bool):
            raise ValueError("Voice response setting must be true or false.")
        if not isinstance(self.background_mode_enabled, bool):
            raise ValueError("Background mode setting must be true or false.")
        if not isinstance(self.autostart_enabled, bool):
            raise ValueError("Autostart setting must be true or false.")
        if not isinstance(self.telegram_enabled, bool):
            raise ValueError("Telegram setting must be true or false.")
        if self.telegram_allowed_user_id:
            try:
                if int(self.telegram_allowed_user_id) <= 0:
                    raise ValueError
            except ValueError as error:
                raise ValueError(
                    "Telegram user ID must be a positive numeric Telegram ID."
                ) from error
        if self.telegram_enabled and not self.telegram_allowed_user_id:
            raise ValueError(
                "Enter your Telegram numeric user ID before enabling Telegram."
            )
        if not self.whisper_model:
            raise ValueError("Whisper model cannot be empty.")
        if self.tts_engine not in TTS_ENGINES:
            raise ValueError("TTS engine must be disabled, espeak-ng, or piper.")
        if not self.tts_voice:
            raise ValueError("TTS voice cannot be empty.")

    @classmethod
    def defaults(cls) -> "AppConfig":
        return cls()

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        config_path = path or cls.default_path()
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("Configuration must be an object.")
            return cls(
                ollama_url=str(raw.get("ollama_url", DEFAULT_OLLAMA_URL)),
                model=str(raw.get("model", DEFAULT_MODEL)),
                ollama_context_size=_int_setting(
                    raw.get("ollama_context_size", DEFAULT_CONTEXT_SIZE),
                    DEFAULT_CONTEXT_SIZE,
                ),
                voice_input_enabled=_bool_setting(raw.get("voice_input_enabled", True), True),
                voice_responses_enabled=_bool_setting(
                    raw.get("voice_responses_enabled", False), False
                ),
                microphone_device=str(raw.get("microphone_device", "")),
                whisper_model=str(raw.get("whisper_model", DEFAULT_WHISPER_MODEL)),
                whisper_language=str(raw.get("whisper_language", "auto")),
                tts_engine=str(raw.get("tts_engine", DEFAULT_TTS_ENGINE)),
                tts_voice=str(raw.get("tts_voice", DEFAULT_TTS_VOICE)),
                background_mode_enabled=_bool_setting(
                    raw.get("background_mode_enabled", False), False
                ),
                autostart_enabled=_bool_setting(
                    raw.get("autostart_enabled", False), False
                ),
                telegram_enabled=_bool_setting(
                    raw.get("telegram_enabled", False), False
                ),
                telegram_allowed_user_id=str(
                    raw.get("telegram_allowed_user_id", "")
                ),
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


def _bool_setting(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


def _int_setting(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
