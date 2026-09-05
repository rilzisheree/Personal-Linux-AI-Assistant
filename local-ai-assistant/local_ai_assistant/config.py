"""Persistent, validated application settings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_TTS_ENGINE = "piper"
PIPER_VOICE_DIRECTORY = "~/.local/share/lura/piper"
PIPER_VOICE_PRESETS = (
    ("Jarvis", f"{PIPER_VOICE_DIRECTORY}/en_GB-alan-medium.onnx"),
    ("Laura", f"{PIPER_VOICE_DIRECTORY}/en_US-amy-medium.onnx"),
    ("Arabic (Kareem)", f"{PIPER_VOICE_DIRECTORY}/ar_JO-kareem-medium.onnx"),
)
DEFAULT_TTS_VOICE = PIPER_VOICE_PRESETS[0][1]
DEFAULT_ARABIC_TTS_VOICE = PIPER_VOICE_PRESETS[2][1]
TTS_ENGINES = ("disabled", "piper")
TTS_VOICE_PRESETS = PIPER_VOICE_PRESETS
DEFAULT_VOICE_SILENCE_DURATION = 0.9
DEFAULT_VOICE_MIN_SPEECH_DURATION = 0.3
DEFAULT_VOICE_VAD_THRESHOLD = 350
DEFAULT_WAKE_WORD_MATCH_THRESHOLD = 0.62
DEFAULT_CONVERSATION_TIMEOUT = 8
DEFAULT_CONVERSATION_TRANSITION_DELAY = 0.35
DEFAULT_WAKE_WORD = "Lura"
PERMISSION_POLICIES = ("default", "always_allow", "ask", "blocked")
PERMISSION_POLICY_LABELS = {
    "default": "Default safety",
    "always_allow": "Always allow",
    "ask": "Ask every time",
    "blocked": "Block",
}


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
    wake_word: str = DEFAULT_WAKE_WORD
    wake_words: tuple[str, ...] = ()
    active_listening_duration: int = 15
    continuous_conversation_enabled: bool = False
    conversation_timeout: int = DEFAULT_CONVERSATION_TIMEOUT
    conversation_transition_delay: float = DEFAULT_CONVERSATION_TRANSITION_DELAY
    voice_silence_duration: float = DEFAULT_VOICE_SILENCE_DURATION
    voice_min_speech_duration: float = DEFAULT_VOICE_MIN_SPEECH_DURATION
    voice_vad_threshold: int = DEFAULT_VOICE_VAD_THRESHOLD
    wake_word_match_threshold: float = DEFAULT_WAKE_WORD_MATCH_THRESHOLD
    theme: str = "obsidian"
    orb_intensity: int = 65
    animation_intensity: int = 70
    tool_permissions: dict[str, str] = field(default_factory=dict)
    custom_app_commands: dict[str, str] = field(default_factory=dict)

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
        configured_wake_words = self.wake_words
        if isinstance(configured_wake_words, str):
            configured_wake_words = (configured_wake_words,)
        if not configured_wake_words:
            configured_wake_words = (self.wake_word,)
        normalized_wake_words: list[str] = []
        seen_wake_words: set[str] = set()
        for value in configured_wake_words:
            if not isinstance(value, str):
                continue
            wake_word = value.strip()
            folded = wake_word.casefold()
            if wake_word and folded not in seen_wake_words:
                normalized_wake_words.append(wake_word)
                seen_wake_words.add(folded)
        if not normalized_wake_words:
            normalized_wake_words = [self.assistant_name]
        self.wake_words = tuple(normalized_wake_words)
        # Keep the original field as the primary alias for older callers and
        # settings files that only know about one wake word.
        self.wake_word = self.wake_words[0]
        if isinstance(self.voice_silence_duration, str):
            self.voice_silence_duration = float(self.voice_silence_duration.strip())
        if isinstance(self.voice_min_speech_duration, str):
            self.voice_min_speech_duration = float(
                self.voice_min_speech_duration.strip()
            )
        if isinstance(self.voice_vad_threshold, str):
            self.voice_vad_threshold = int(self.voice_vad_threshold.strip())
        if isinstance(self.wake_word_match_threshold, str):
            self.wake_word_match_threshold = float(
                self.wake_word_match_threshold.strip()
            )
        if isinstance(self.conversation_timeout, str):
            self.conversation_timeout = int(self.conversation_timeout.strip())
        if isinstance(self.conversation_transition_delay, str):
            self.conversation_transition_delay = float(
                self.conversation_transition_delay.strip()
            )
        self.theme = self.theme.strip().lower()
        self.tool_permissions = _normalized_settings_map(
            self.tool_permissions,
            allowed_values=PERMISSION_POLICIES,
        )
        self.custom_app_commands = _normalized_settings_map(
            self.custom_app_commands,
            max_value_length=500,
        )
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
        if not isinstance(self.wake_word_enabled, bool):
            raise ValueError("Wake-word setting must be true or false.")
        if not self.wake_words:
            raise ValueError("At least one wake word is required.")
        if not isinstance(self.continuous_conversation_enabled, bool):
            raise ValueError("Continuous conversation setting must be true or false.")
        if (
            isinstance(self.active_listening_duration, bool)
            or not isinstance(self.active_listening_duration, int)
            or not 1 <= self.active_listening_duration <= 60
        ):
            raise ValueError("Active listening duration must be between 1 and 60 seconds.")
        if (
            isinstance(self.conversation_timeout, bool)
            or not isinstance(self.conversation_timeout, int)
            or not 3 <= self.conversation_timeout <= 120
        ):
            raise ValueError("Conversation timeout must be between 3 and 120 seconds.")
        if (
            isinstance(self.conversation_transition_delay, bool)
            or not isinstance(self.conversation_transition_delay, (int, float))
            or not 0.1 <= self.conversation_transition_delay <= 2.0
        ):
            raise ValueError(
                "Conversation transition delay must be between 0.1 and 2 seconds."
            )
        if (
            isinstance(self.voice_silence_duration, bool)
            or not isinstance(self.voice_silence_duration, (int, float))
            or not 0.5 <= self.voice_silence_duration <= 3.0
        ):
            raise ValueError("Voice silence duration must be between 0.5 and 3 seconds.")
        if (
            isinstance(self.voice_min_speech_duration, bool)
            or not isinstance(self.voice_min_speech_duration, (int, float))
            or not 0.1 <= self.voice_min_speech_duration <= 1.0
        ):
            raise ValueError(
                "Minimum speech duration must be between 0.1 and 1 second."
            )
        if (
            isinstance(self.voice_vad_threshold, bool)
            or not isinstance(self.voice_vad_threshold, int)
            or not 100 <= self.voice_vad_threshold <= 4000
        ):
            raise ValueError("Voice activity threshold must be between 100 and 4000.")
        if (
            isinstance(self.wake_word_match_threshold, bool)
            or not isinstance(self.wake_word_match_threshold, (int, float))
            or not 0.5 <= self.wake_word_match_threshold <= 0.95
        ):
            raise ValueError(
                "Wake-word match threshold must be between 0.5 and 0.95."
            )
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
        if not isinstance(self.tool_permissions, dict):
            raise ValueError("Tool permissions must be an object.")
        if not isinstance(self.custom_app_commands, dict):
            raise ValueError("Custom app commands must be an object.")

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
                assistant_name=str(raw.get("assistant_name", "Lura")),
                wake_word_enabled=_bool_setting(
                    raw.get("wake_word_enabled", False), False
                ),
                wake_word=str(raw.get("wake_word", "Lura")),
                wake_words=(
                    raw.get("wake_words")
                    if raw.get("wake_words") is not None
                    else (raw.get("wake_word", DEFAULT_WAKE_WORD),)
                ),
                active_listening_duration=_int_setting(
                    raw.get("active_listening_duration", 15), 15
                ),
                continuous_conversation_enabled=_bool_setting(
                    raw.get("continuous_conversation_enabled", False), False
                ),
                conversation_timeout=_int_setting(
                    raw.get("conversation_timeout", DEFAULT_CONVERSATION_TIMEOUT),
                    DEFAULT_CONVERSATION_TIMEOUT,
                ),
                conversation_transition_delay=_float_setting(
                    raw.get(
                        "conversation_transition_delay",
                        DEFAULT_CONVERSATION_TRANSITION_DELAY,
                    ),
                    DEFAULT_CONVERSATION_TRANSITION_DELAY,
                ),
                voice_silence_duration=_float_setting(
                    raw.get("voice_silence_duration", DEFAULT_VOICE_SILENCE_DURATION),
                    DEFAULT_VOICE_SILENCE_DURATION,
                ),
                voice_min_speech_duration=_float_setting(
                    raw.get(
                        "voice_min_speech_duration",
                        DEFAULT_VOICE_MIN_SPEECH_DURATION,
                    ),
                    DEFAULT_VOICE_MIN_SPEECH_DURATION,
                ),
                voice_vad_threshold=_int_setting(
                    raw.get("voice_vad_threshold", DEFAULT_VOICE_VAD_THRESHOLD),
                    DEFAULT_VOICE_VAD_THRESHOLD,
                ),
                wake_word_match_threshold=_float_setting(
                    raw.get(
                        "wake_word_match_threshold",
                        DEFAULT_WAKE_WORD_MATCH_THRESHOLD,
                    ),
                    DEFAULT_WAKE_WORD_MATCH_THRESHOLD,
                ),
                theme=str(raw.get("theme", "obsidian")),
                orb_intensity=_int_setting(raw.get("orb_intensity", 65), 65),
                animation_intensity=_int_setting(
                    raw.get("animation_intensity", 70), 70
                ),
                tool_permissions=raw.get("tool_permissions", {}),
                custom_app_commands=raw.get("custom_app_commands", {}),
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


def _normalized_settings_map(
    value: object,
    *,
    allowed_values: tuple[str, ...] | None = None,
    max_value_length: int = 120,
) -> dict[str, str]:
    """Normalize user-editable string maps without trusting config contents."""

    if not isinstance(value, dict):
        return {}
    normalized: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            continue
        key = raw_key.strip()
        item = raw_value.strip()
        if not key or not item or len(key) > 120 or len(item) > max_value_length:
            continue
        if allowed_values is not None and item not in allowed_values:
            continue
        normalized[key] = item
    return normalized


def _int_setting(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_setting(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
