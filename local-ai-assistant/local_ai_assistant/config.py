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
DEFAULT_TTS_ENGINE = "piper"
DEFAULT_AI_PROVIDER = "ollama"
DEFAULT_HOSTED_PROVIDER = "openai"
DEFAULT_HOSTED_API_URL = "https://api.openai.com/v1"
DEFAULT_HOSTED_MODEL = "gpt-4o-mini"
HOSTED_PROVIDER_PRESETS = (
    ("OpenAI", "openai", "https://api.openai.com/v1", "gpt-4o-mini"),
    ("OpenRouter", "openrouter", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
    ("Custom OpenAI-compatible", "custom", "", ""),
)
PIPER_VOICE_DIRECTORY = "~/.local/share/lura/piper"
PIPER_VOICE_PRESETS = (
    ("Jarvis", f"{PIPER_VOICE_DIRECTORY}/en_GB-alan-medium.onnx"),
    ("Laura", f"{PIPER_VOICE_DIRECTORY}/en_US-amy-medium.onnx"),
)
DEFAULT_TTS_VOICE = PIPER_VOICE_PRESETS[0][1]
TTS_ENGINES = ("disabled", "piper")
TTS_VOICE_PRESETS = PIPER_VOICE_PRESETS


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
    assistant_name: str = "Lura"
    wake_word_enabled: bool = False
    wake_word: str = "Lura"
    active_listening_duration: int = 15
    theme: str = "obsidian"
    orb_intensity: int = 65
    animation_intensity: int = 70
    ai_provider: str = DEFAULT_AI_PROVIDER
    hosted_provider: str = DEFAULT_HOSTED_PROVIDER
    hosted_api_url: str = DEFAULT_HOSTED_API_URL
    hosted_model: str = DEFAULT_HOSTED_MODEL

    def __post_init__(self) -> None:
        self.ai_provider = self.ai_provider.strip().lower()
        self.ollama_url = self.ollama_url.strip().rstrip("/")
        self.model = self.model.strip()
        self.hosted_provider = self.hosted_provider.strip().lower()
        self.hosted_api_url = self.hosted_api_url.strip().rstrip("/")
        self.hosted_model = self.hosted_model.strip()
        if isinstance(self.ollama_context_size, str):
            self.ollama_context_size = int(self.ollama_context_size.strip())
        self.microphone_device = self.microphone_device.strip()
        self.whisper_model = self.whisper_model.strip()
        self.whisper_language = self.whisper_language.strip() or "auto"
        self.tts_engine = self.tts_engine.strip().lower()
        self.tts_voice = self.tts_voice.strip()
        # Migrate settings saved by older versions to the two supported Piper
        # voices instead of leaving an obsolete eSpeak/custom voice selected.
        if self.tts_engine == "espeak-ng":
            self.tts_engine = DEFAULT_TTS_ENGINE
        if self.tts_voice in {"en-gb+m3", "en-gb+f2", "~/Models/piper/en_GB-alan-medium.onnx"}:
            self.tts_voice = DEFAULT_TTS_VOICE
        elif self.tts_voice == "~/Models/piper/en_US-amy-medium.onnx":
            self.tts_voice = PIPER_VOICE_PRESETS[1][1]
        self.telegram_allowed_user_id = str(self.telegram_allowed_user_id).strip()
        self.assistant_name = self.assistant_name.strip() or "Lura"
        self.wake_word = self.wake_word.strip() or self.assistant_name
        self.theme = self.theme.strip().lower()
        self.validate()

    def validate(self) -> None:
        if self.ai_provider not in {"ollama", "hosted"}:
            raise ValueError("AI provider must be Ollama or hosted API.")
        parsed = urlparse(self.ollama_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Ollama URL must be a complete http:// or https:// URL.")
        if not self.model:
            raise ValueError("Ollama model cannot be empty.")
        if self.hosted_provider not in {"openai", "openrouter", "custom"}:
            raise ValueError("Hosted provider must be OpenAI, OpenRouter, or custom.")
        hosted_parsed = urlparse(self.hosted_api_url)
        if (
            hosted_parsed.scheme not in {"http", "https"}
            or not hosted_parsed.netloc
        ):
            raise ValueError(
                "Hosted API URL must be a complete http:// or https:// URL."
            )
        if not self.hosted_model:
            raise ValueError("Hosted API model cannot be empty.")
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
        if not isinstance(self.wake_word_enabled, bool):
            raise ValueError("Wake-word setting must be true or false.")
        if (
            isinstance(self.active_listening_duration, bool)
            or not isinstance(self.active_listening_duration, int)
            or not 1 <= self.active_listening_duration <= 60
        ):
            raise ValueError("Active listening duration must be between 1 and 60 seconds.")
        if self.theme not in {"obsidian"}:
            raise ValueError("Theme must be obsidian.")
        for value, label in (
            (self.orb_intensity, "Orb intensity"),
            (self.animation_intensity, "Motion intensity"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 100
            ):
                raise ValueError(f"{label} must be between 1 and 100.")
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
            raise ValueError("TTS engine must be disabled or piper.")
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
                ai_provider=str(raw.get("ai_provider", DEFAULT_AI_PROVIDER)),
                ollama_url=str(raw.get("ollama_url", DEFAULT_OLLAMA_URL)),
                model=str(raw.get("model", DEFAULT_MODEL)),
                hosted_provider=str(
                    raw.get("hosted_provider", DEFAULT_HOSTED_PROVIDER)
                ),
                hosted_api_url=str(
                    raw.get("hosted_api_url", DEFAULT_HOSTED_API_URL)
                ),
                hosted_model=str(raw.get("hosted_model", DEFAULT_HOSTED_MODEL)),
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
                assistant_name=str(raw.get("assistant_name", "Lura")),
                wake_word_enabled=_bool_setting(
                    raw.get("wake_word_enabled", False), False
                ),
                wake_word=str(raw.get("wake_word", "Lura")),
                active_listening_duration=_int_setting(
                    raw.get("active_listening_duration", 15), 15
                ),
                theme=str(raw.get("theme", "obsidian")),
                orb_intensity=_int_setting(raw.get("orb_intensity", 65), 65),
                animation_intensity=_int_setting(
                    raw.get("animation_intensity", 70), 70
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
