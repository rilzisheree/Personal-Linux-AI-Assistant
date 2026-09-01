"""Optional local voice backends for recording, transcription, and playback."""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

from .config import AppConfig, DEFAULT_TTS_VOICE

LOGGER = logging.getLogger("lura.voice")


class VoiceError(RuntimeError):
    """A user-facing error from an unavailable or failed local voice backend."""


def is_no_speech_error(message: str) -> bool:
    """Treat normal silent Whisper windows as empty input, not backend failure."""
    normalized = message.casefold()
    return any(
        phrase in normalized
        for phrase in (
            "no speech",
            "no voice",
            "silence",
            "speech not detected",
        )
    )


class SpeechChunker:
    """Split streamed model output at natural boundaries for local TTS."""

    def __init__(self, max_chars: int = 260) -> None:
        self.max_chars = max_chars
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        return self._extract_ready_chunks()

    def flush(self) -> list[str]:
        remainder = self._buffer.strip()
        self._buffer = ""
        return [remainder] if remainder else []

    def _extract_ready_chunks(self) -> list[str]:
        chunks: list[str] = []
        while self._buffer:
            boundary = self._find_sentence_boundary()
            if boundary is not None:
                chunk = self._buffer[:boundary].strip()
                self._buffer = self._buffer[boundary:].lstrip()
                if chunk:
                    chunks.append(chunk)
                continue
            if len(self._buffer) <= self.max_chars:
                break
            split_at = self._buffer.rfind(" ", 80, self.max_chars + 1)
            if split_at <= 0:
                break
            chunk = self._buffer[:split_at].strip()
            self._buffer = self._buffer[split_at:].lstrip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _find_sentence_boundary(self) -> int | None:
        for index, character in enumerate(self._buffer):
            if character == "\n":
                return index + 1
            if character not in ".!?":
                continue
            next_character = self._buffer[index + 1] if index + 1 < len(self._buffer) else ""
            if next_character and not next_character.isspace():
                continue
            # Avoid turning common short abbreviations into spoken fragments.
            word_start = index
            while word_start > 0 and not self._buffer[word_start - 1].isspace():
                word_start -= 1
            if character == "." and index - word_start <= 2:
                continue
            end = index + 1
            while end < len(self._buffer) and self._buffer[end] in "\"'”’)]":
                end += 1
            return end
        return None


def _normalise_spoken_text(text: str) -> list[str]:
    """Return lowercase word-like tokens for tolerant wake-word matching."""
    cleaned = "".join(character.casefold() if character.isalnum() else " " for character in text)
    return [token for token in cleaned.split() if token]


def wake_word_matches(text: str, wake_word: str, threshold: float = 0.72) -> bool:
    """Match Whisper's small pronunciation/spelling variations safely.

    Whisper commonly renders short names such as "Lura" as "Laura" or "Lara".
    Comparing individual words (rather than arbitrary substrings) avoids
    treating a word that merely contains the wake word as a trigger.
    """
    from difflib import SequenceMatcher

    expected = _normalise_spoken_text(wake_word)
    actual = _normalise_spoken_text(text)
    if not expected or not actual:
        return False
    width = len(expected)
    for start in range(len(actual) - width + 1):
        candidate = actual[start : start + width]
        score = SequenceMatcher(None, " ".join(expected), " ".join(candidate)).ratio()
        if score >= threshold:
            return True
    return False


def remove_wake_word(text: str, wake_word: str, threshold: float = 0.72) -> str | None:
    """Return the command after a tolerant wake word, or None when absent."""
    from difflib import SequenceMatcher

    expected = _normalise_spoken_text(wake_word)
    matches = list(re.finditer(r"[^\W_]+", text, flags=re.UNICODE))
    if not expected or not matches:
        return None
    width = len(expected)
    for start in range(len(matches) - width + 1):
        candidate = [match.group(0).casefold() for match in matches[start : start + width]]
        score = SequenceMatcher(None, " ".join(expected), " ".join(candidate)).ratio()
        if score >= threshold:
            end = matches[start + width - 1].end()
            return text[end:].strip(" ,.!?:;-")
    return None


class VoiceActivityDetector:
    """Adaptive activity detector for 16-bit, 16 kHz mono WAV recordings."""

    def __init__(
        self,
        *,
        threshold: int,
        silence_duration: float,
        min_speech_duration: float,
        sample_rate: int = 16_000,
    ) -> None:
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.sample_rate = sample_rate
        self.speech_started = False
        self._voiced_duration = 0.0
        self._last_voice_at: float | None = None
        self._noise_floor = float(threshold) * 0.25

    @staticmethod
    def _levels(samples: bytes) -> tuple[float, int]:
        usable = len(samples) - (len(samples) % 2)
        if usable <= 0:
            return 0.0, 0
        values = [
            int.from_bytes(samples[index : index + 2], "little", signed=True)
            for index in range(0, usable, 2)
        ]
        rms = math.sqrt(sum(value * value for value in values) / len(values))
        return rms, max(abs(value) for value in values)

    def consume(self, samples: bytes, now: float) -> bool:
        """Consume new PCM bytes and return True after confirmed end-of-speech."""
        usable = len(samples) - (len(samples) % 2)
        if usable <= 0:
            return False
        rms, peak = self._levels(samples[:usable])
        duration = usable / 2 / self.sample_rate

        if not self.speech_started:
            # Let the room establish a baseline, but retain an absolute floor
            # so a quiet normal voice is not rejected just because the first
            # few frames happened to be quiet.
            self._noise_floor = min(
                self._noise_floor * 0.8 + rms * 0.2,
                float(self.threshold) * 0.8,
            )
        dynamic_floor = max(float(self.threshold) * 0.65, self._noise_floor * 2.2)
        voiced = rms >= dynamic_floor or (
            peak >= self.threshold * 1.5 and rms >= self.threshold * 0.25
        )
        if voiced:
            self._voiced_duration += duration
            self._last_voice_at = now
        else:
            self._voiced_duration = max(0.0, self._voiced_duration - duration * 0.35)

        if not self.speech_started:
            if self._voiced_duration >= self.min_speech_duration:
                self.speech_started = True
            return False
        return self.should_stop(now)

    def should_stop(self, now: float) -> bool:
        """Check for end-of-speech even when a recorder poll has no new bytes."""
        return (
            self._last_voice_at is not None
            and now - self._last_voice_at >= self.silence_duration
        )


def _is_monitor_source(
    name: str,
    source: dict[str, Any] | None = None,
    properties: dict[str, Any] | None = None,
) -> bool:
    """Exclude sink monitor streams from the microphone picker."""
    source = source or {}
    properties = properties or {}
    media_class = str(properties.get("media.class", "")).casefold()
    return (
        name.casefold().endswith(".monitor")
        or "monitor" in media_class
        or source.get("monitor_of_sink") not in (None, -1)
    )


def _dedupe_microphones(
    microphones: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for name, label in microphones:
        if name in seen:
            continue
        seen.add(name)
        result.append((name, label))
    return result


class VoiceService:
    """Keep optional audio tools outside the Qt widgets and chat transport."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._process_lock = threading.Lock()
        self._recorder_process: subprocess.Popen | None = None
        self._active_process: subprocess.Popen | None = None
        self._last_recorder_command: list[str] = []
        self._whisper_model: Any | None = None
        self._whisper_model_name: str | None = None
        self._whisper_device: str | None = None
        self._piper_voice: Any | None = None
        self._piper_model_path: Path | None = None

    @staticmethod
    def recordings_directory() -> Path:
        return Path.home() / ".cache" / "local-ai-assistant" / "recordings"

    def new_recording_path(self) -> Path:
        self.recordings_directory().mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return self.recordings_directory() / f"recording-{stamp}.wav"

    @staticmethod
    def list_microphones() -> list[tuple[str, str]]:
        """Return selectable physical input sources as (source_name, label)."""
        pactl = shutil.which("pactl")
        if pactl:
            try:
                result = subprocess.run(
                    [pactl, "-f", "json", "list", "sources"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    sources = json.loads(result.stdout)
                    microphones = []
                    for source in sources if isinstance(sources, list) else []:
                        if not isinstance(source, dict):
                            continue
                        properties = source.get("properties", {})
                        name = source.get("name") or properties.get("node.name")
                        if not isinstance(name, str) or _is_monitor_source(
                            name, source, properties
                        ):
                            continue
                        description = (
                            source.get("description")
                            or properties.get("device.description")
                            or properties.get("node.description")
                            or name
                        )
                        microphones.append((name, str(description)))
                    if microphones:
                        return _dedupe_microphones(microphones)
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
                pass

            try:
                result = subprocess.run(
                    [pactl, "list", "short", "sources"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                if result.returncode == 0:
                    microphones = []
                    for line in result.stdout.splitlines():
                        fields = line.split()
                        if len(fields) < 2 or _is_monitor_source(fields[1]):
                            continue
                        microphones.append((fields[1], fields[1]))
                    if microphones:
                        return _dedupe_microphones(microphones)
            except (OSError, subprocess.SubprocessError):
                pass

        pw_dump = shutil.which("pw-dump")
        if pw_dump:
            try:
                result = subprocess.run(
                    [pw_dump],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                objects = json.loads(result.stdout)
                microphones = []
                for item in objects if isinstance(objects, list) else []:
                    if not isinstance(item, dict):
                        continue
                    info = item.get("info", {})
                    properties = info.get("props", {})
                    media_class = str(properties.get("media.class", ""))
                    name = properties.get("node.name")
                    if (
                        not isinstance(name, str)
                        or "Audio/Source" not in media_class
                        or _is_monitor_source(name, item, properties)
                    ):
                        continue
                    label = (
                        properties.get("node.description")
                        or properties.get("device.description")
                        or name
                    )
                    microphones.append((name, str(label)))
                return _dedupe_microphones(microphones)
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
                pass
        return []

    def recorder_command(self, destination: Path) -> list[str]:
        device = self.config.microphone_device
        alsa_device = device.casefold().startswith(("hw:", "plughw:", "default"))
        if alsa_device and shutil.which("arecord"):
            return self._arecord_command(destination, device)
        if shutil.which("pw-record"):
            command = [
                "pw-record",
                "--rate",
                "16000",
                "--channels",
                "1",
                "--format",
                "s16",
            ]
            if device:
                command.extend(["--target", device])
            return [*command, str(destination)]
        if shutil.which("arecord"):
            return self._arecord_command(destination, device)
        raise VoiceError(
            "No local recorder found. Install PipeWire's pw-record or ALSA arecord."
        )

    def _arecord_command(self, destination: Path, device: str) -> list[str]:
        command = ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1"]
        if device:
            command.extend(["-D", device])
        return [*command, str(destination)]

    def start_recorder(self, destination: Path) -> subprocess.Popen:
        with self._process_lock:
            if (
                self._recorder_process is not None
                and self._recorder_process.poll() is None
            ):
                raise VoiceError(
                    "The microphone is already in use by another voice operation."
                )
            command = self.recorder_command(destination)
            self._last_recorder_command = command
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
            except OSError as error:
                raise VoiceError(f"Could not start microphone recording: {error}") from error
            self._recorder_process = process
            LOGGER.info("[Voice] Microphone connected: %s", shlex.join(command))
        return process

    def release_recorder(self, process: subprocess.Popen) -> None:
        """Release microphone ownership after the recorder has fully exited."""
        with self._process_lock:
            if self._recorder_process is process:
                self._recorder_process = None

    def finish_recording(self, process: subprocess.Popen, destination: Path) -> None:
        detail = process.stderr.read().strip() if process.stderr is not None else ""
        command_text = (
            shlex.join(self._last_recorder_command)
            if self._last_recorder_command
            else "unknown recorder command"
        )
        audio_is_valid = destination.is_file() and destination.stat().st_size >= 44
        stopped_with_signal = process.returncode in (
            0,
            1,
            -signal.SIGINT,
            -signal.SIGTERM,
        )
        if not stopped_with_signal or (process.returncode == 1 and not audio_is_valid):
            raise VoiceError(
                f"Microphone recording failed with exit code {process.returncode}. "
                f"Command: {command_text}."
                + (f" Recorder output: {detail[:400]}" if detail else "")
            )
        if not audio_is_valid:
            raise VoiceError(
                "The microphone produced no usable audio. Check the selected "
                f"input device. Command: {command_text}."
                + (f" Recorder output: {detail[:400]}" if detail else "")
            )

    @staticmethod
    def stop_recorder(process: subprocess.Popen) -> None:
        """Ask recorder CLIs to finalize their WAV headers before exiting."""
        if process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
            except OSError:
                process.terminate()

    def transcribe(
        self,
        audio_path: Path,
        *,
        device: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        if not audio_path.is_file():
            raise VoiceError(f"Recording not found: {audio_path}")
        whisper = shutil.which("whisper")
        if whisper:
            return self._transcribe_openai_whisper(
                whisper,
                audio_path,
                device=device,
                cancel_event=cancel_event,
            )
        whisper_cpp = shutil.which("whisper-cli") or shutil.which("whisper.cpp")
        if whisper_cpp:
            return self._transcribe_whisper_cpp(
                whisper_cpp,
                audio_path,
                cancel_event=cancel_event,
            )
        try:
            import whisper as whisper_module
        except ImportError:
            whisper_module = None
        if whisper_module is not None:
            return self._transcribe_openai_whisper_module(
                whisper_module, audio_path, device=device
            )
        raise VoiceError(
            "No local Whisper backend found. Install openai-whisper in Lura's "
            "Python environment (`python -m pip install openai-whisper`) or "
            "install whisper.cpp."
        )

    def _transcribe_openai_whisper(
        self,
        executable: str,
        audio_path: Path,
        *,
        device: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        with tempfile.TemporaryDirectory(prefix="lura-whisper-") as output_directory:
            command = [
                executable,
                str(audio_path),
                "--model",
                self.config.whisper_model,
                "--output_format",
                "txt",
                "--output_dir",
                output_directory,
                "--fp16",
                "False",
            ]
            if device:
                command.extend(["--device", device])
            if self.config.whisper_language.casefold() not in {"", "auto"}:
                command.extend(["--language", self.config.whisper_language])
            result = self._run_checked(
                command,
                600,
                "Whisper",
                cancel_event=cancel_event,
            )
            text_path = Path(output_directory) / f"{audio_path.stem}.txt"
            if not text_path.is_file():
                raise VoiceError(
                    "Whisper completed without producing a transcript."
                    + (f" {result.stderr.strip()[:240]}" if result.stderr else "")
                )
            return self._read_transcript(text_path)

    def _transcribe_openai_whisper_module(
        self,
        whisper_module: Any,
        audio_path: Path,
        *,
        device: str | None = None,
    ) -> str:
        try:
            if (
                self._whisper_model is None
                or self._whisper_model_name != self.config.whisper_model
                or self._whisper_device != device
            ):
                load_options = {"device": device} if device else {}
                self._whisper_model = whisper_module.load_model(
                    self.config.whisper_model,
                    **load_options,
                )
                self._whisper_model_name = self.config.whisper_model
                self._whisper_device = device
            model_device = str(getattr(self._whisper_model, "device", device or "cpu"))
            options: dict[str, Any] = {
                "fp16": model_device.casefold().startswith("cuda")
            }
            if self.config.whisper_language.casefold() not in {"", "auto"}:
                options["language"] = self.config.whisper_language
            result = self._whisper_model.transcribe(str(audio_path), **options)
        except Exception as error:
            raise VoiceError(f"Whisper could not transcribe the recording: {error}") from error
        if not isinstance(result, dict):
            raise VoiceError("Whisper returned an invalid transcription result.")
        return self._read_transcript_from_text(str(result.get("text", "")))

    def _transcribe_whisper_cpp(
        self,
        executable: str,
        audio_path: Path,
        *,
        cancel_event: threading.Event | None = None,
    ) -> str:
        model = Path(self.config.whisper_model).expanduser()
        if not model.is_file():
            raise VoiceError(
                "whisper.cpp needs a local model file. Set its path in Voice settings."
            )
        command = [executable, "-m", str(model), "-f", str(audio_path), "-otxt", "-nt"]
        self._run_checked(command, 600, "whisper.cpp", cancel_event=cancel_event)
        candidates = (
            audio_path.with_suffix(audio_path.suffix + ".txt"),
            audio_path.with_suffix(".txt"),
        )
        transcript = next((path for path in candidates if path.is_file()), None)
        if transcript is None:
            raise VoiceError("whisper.cpp completed without producing a transcript.")
        try:
            return self._read_transcript(transcript)
        finally:
            try:
                transcript.unlink()
            except OSError:
                pass

    @staticmethod
    def _read_transcript(path: Path) -> str:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as error:
            raise VoiceError(f"Could not read the transcript: {error}") from error
        return VoiceService._read_transcript_from_text(text)

    @staticmethod
    def _read_transcript_from_text(text: str) -> str:
        text = text.strip()
        if not text:
            raise VoiceError("No speech was detected.")
        return text

    @staticmethod
    def _run_checked(
        command: list[str],
        timeout: int,
        label: str,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                input=input_text,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise VoiceError(f"{label} could not start: {error}") from error
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise VoiceError(
                f"{label} failed with exit code {result.returncode}."
                + (f" {detail[:300]}" if detail else "")
            )
        return result

    def speak(
        self,
        text: str,
        cancel_event: threading.Event | None = None,
        on_synthesis_started: Callable[[], None] | None = None,
        on_playback_started: Callable[[], None] | None = None,
    ) -> None:
        if self.config.tts_engine == "disabled":
            return
        text = text.strip()
        if not text:
            return
        if len(text) > 10_000:
            text = text[:9_997] + "..."
        with tempfile.NamedTemporaryFile(suffix=".wav", prefix="lura-tts-", delete=False) as file:
            audio_path = Path(file.name)
        try:
            if on_synthesis_started is not None:
                on_synthesis_started()
            self._synthesize(text, audio_path)
            self._play(audio_path, cancel_event, on_playback_started)
        finally:
            try:
                audio_path.unlink()
            except OSError:
                pass

    def _synthesize(self, text: str, audio_path: Path) -> None:
        if self.config.tts_engine == "piper":
            model = Path(self.config.tts_voice).expanduser()
            if not model.is_file():
                model = self._ensure_piper_model(model)
            if not model.is_file():
                raise VoiceError(
                    f"The {self._voice_label(model)} Piper model is missing."
                )
            if model.with_suffix(".onnx.json").is_file():
                self._synthesize_piper(text, model, audio_path)
            else:
                executable = shutil.which("piper")
                if not executable:
                    raise VoiceError(
                        f"The {self._voice_label(model)} Piper model is missing "
                        "its .onnx.json config file."
                    )
                self._run_checked(
                    [
                        executable,
                        "--model",
                        str(model),
                        "--output_file",
                        str(audio_path),
                    ],
                    120,
                    "Piper",
                    input_text=text,
                )
            if not audio_path.is_file() or audio_path.stat().st_size == 0:
                raise VoiceError("Piper completed without producing an audio file.")
            return

        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if not executable:
            raise VoiceError(
                "No local TTS engine found. Install espeak-ng or configure Piper."
            )
        self._synthesize_espeak(executable, text, audio_path, self.config.tts_voice)

    @staticmethod
    def _voice_label(model: Path) -> str:
        return {
            "en_GB-alan-medium": "Jarvis",
            "en_US-amy-medium": "Laura",
        }.get(model.stem, "selected")

    def _synthesize_piper(self, text: str, model: Path, audio_path: Path) -> None:
        try:
            from piper import PiperVoice
        except ImportError as error:
            raise VoiceError(
                "Piper is not installed. Install piper-tts in the same "
                "Python environment Lura uses."
            ) from error
        try:
            if self._piper_voice is None or self._piper_model_path != model:
                self._piper_voice = PiperVoice.load(model)
                self._piper_model_path = model
            with wave.open(str(audio_path), "wb") as wav_file:
                self._piper_voice.synthesize_wav(text, wav_file)
        except Exception as error:
            raise VoiceError(f"Piper could not synthesize the response: {error}") from error

    def _ensure_piper_model(self, model: Path) -> Path:
        if model.is_file():
            return model
        voice_name = model.stem
        if model.suffix != ".onnx" or voice_name.count("-") < 2:
            raise VoiceError(
                "The selected Piper voice model was not found. Choose a valid "
                "Piper .onnx model path in Voice settings."
            )
        try:
            import piper.download_voices  # noqa: F401
        except ImportError as error:
            raise VoiceError(
                "The selected Piper voice is not downloaded. Install piper-tts "
                "or download the voice model and its .onnx.json file."
            ) from error
        try:
            model.parent.mkdir(parents=True, exist_ok=True)
            self._run_checked(
                [
                    sys.executable,
                    "-m",
                    "piper.download_voices",
                    voice_name,
                    "--download-dir",
                    str(model.parent),
                ],
                600,
                "Piper voice download",
            )
        except VoiceError as error:
            raise VoiceError(
                f"Piper voice {voice_name} is unavailable. Check your internet "
                f"connection or download it manually: {error}"
            ) from error
        if not model.is_file() or not model.with_suffix(".onnx.json").is_file():
            raise VoiceError(
                f"Piper voice download did not produce {model.name} and its config."
            )
        return model

    def _synthesize_espeak(
        self, executable: str, text: str, audio_path: Path, voice: str
    ) -> None:
        self._run_checked(
            [executable, "-w", str(audio_path), "-v", voice, "--", text],
            120,
            "eSpeak",
        )

    def _play(
        self,
        audio_path: Path,
        cancel_event: threading.Event | None,
        on_playback_started: Callable[[], None] | None = None,
    ) -> None:
        executable = next(
            (shutil.which(name) for name in ("pw-play", "aplay", "paplay") if shutil.which(name)),
            None,
        )
        if not executable:
            raise VoiceError("No local audio player found. Install pw-play or ALSA aplay.")
        try:
            process = subprocess.Popen(
                [executable, str(audio_path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as error:
            raise VoiceError(f"Could not play the response: {error}") from error
        with self._process_lock:
            self._active_process = process
        if on_playback_started is not None:
            on_playback_started()
        try:
            while process.poll() is None:
                if cancel_event is not None and cancel_event.wait(0.1):
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return
                if cancel_event is None:
                    time.sleep(0.1)
            if process.returncode:
                detail = process.stderr.read().strip() if process.stderr else ""
                raise VoiceError(
                    f"Audio playback failed with exit code {process.returncode}."
                    + (f" {detail[:240]}" if detail else "")
                )
        finally:
            with self._process_lock:
                if self._active_process is process:
                    self._active_process = None

    def cancel(self) -> None:
        with self._process_lock:
            process = self._active_process
        if process is not None and process.poll() is None:
            process.terminate()