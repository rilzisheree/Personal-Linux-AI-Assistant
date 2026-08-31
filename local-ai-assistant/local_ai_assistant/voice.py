"""Optional local voice backends for recording, transcription, and playback."""

from __future__ import annotations

import shutil
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig, DEFAULT_TTS_VOICE


class VoiceError(RuntimeError):
    """A user-facing error from an unavailable or failed local voice backend."""


class VoiceService:
    """Keep optional audio tools outside the Qt widgets and chat transport."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._process_lock = threading.Lock()
        self._active_process: subprocess.Popen | None = None
        self._last_recorder_command: list[str] = []
        self._whisper_model: Any | None = None
        self._whisper_model_name: str | None = None

    @staticmethod
    def recordings_directory() -> Path:
        return Path.home() / ".cache" / "local-ai-assistant" / "recordings"

    def new_recording_path(self) -> Path:
        self.recordings_directory().mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return self.recordings_directory() / f"recording-{stamp}.wav"

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
        return process

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

    def transcribe(self, audio_path: Path) -> str:
        if not audio_path.is_file():
            raise VoiceError(f"Recording not found: {audio_path}")
        whisper = shutil.which("whisper")
        if whisper:
            return self._transcribe_openai_whisper(whisper, audio_path)
        whisper_cpp = shutil.which("whisper-cli") or shutil.which("whisper.cpp")
        if whisper_cpp:
            return self._transcribe_whisper_cpp(whisper_cpp, audio_path)
        try:
            import whisper as whisper_module
        except ImportError:
            whisper_module = None
        if whisper_module is not None:
            return self._transcribe_openai_whisper_module(
                whisper_module, audio_path
            )
        raise VoiceError(
            "No local Whisper backend found. Install openai-whisper in Lura's "
            "Python environment (`python -m pip install openai-whisper`) or "
            "install whisper.cpp."
        )

    def _transcribe_openai_whisper(self, executable: str, audio_path: Path) -> str:
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
            if self.config.whisper_language.casefold() not in {"", "auto"}:
                command.extend(["--language", self.config.whisper_language])
            result = self._run_checked(command, 600, "Whisper")
            text_path = Path(output_directory) / f"{audio_path.stem}.txt"
            if not text_path.is_file():
                raise VoiceError(
                    "Whisper completed without producing a transcript."
                    + (f" {result.stderr.strip()[:240]}" if result.stderr else "")
                )
            return self._read_transcript(text_path)

    def _transcribe_openai_whisper_module(
        self, whisper_module: Any, audio_path: Path
    ) -> str:
        try:
            if self._whisper_model is None or self._whisper_model_name != self.config.whisper_model:
                self._whisper_model = whisper_module.load_model(self.config.whisper_model)
                self._whisper_model_name = self.config.whisper_model
            options: dict[str, Any] = {"fp16": False}
            if self.config.whisper_language.casefold() not in {"", "auto"}:
                options["language"] = self.config.whisper_language
            result = self._whisper_model.transcribe(str(audio_path), **options)
        except Exception as error:
            raise VoiceError(f"Whisper could not transcribe the recording: {error}") from error
        if not isinstance(result, dict):
            raise VoiceError("Whisper returned an invalid transcription result.")
        return self._read_transcript_from_text(str(result.get("text", "")))

    def _transcribe_whisper_cpp(self, executable: str, audio_path: Path) -> str:
        model = Path(self.config.whisper_model).expanduser()
        if not model.is_file():
            raise VoiceError(
                "whisper.cpp needs a local model file. Set its path in Voice settings."
            )
        command = [executable, "-m", str(model), "-f", str(audio_path), "-otxt", "-nt"]
        self._run_checked(command, 600, "whisper.cpp")
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

    def speak(self, text: str, cancel_event: threading.Event | None = None) -> None:
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
            self._synthesize(text, audio_path)
            self._play(audio_path, cancel_event)
        finally:
            try:
                audio_path.unlink()
            except OSError:
                pass

    def _synthesize(self, text: str, audio_path: Path) -> None:
        if self.config.tts_engine == "piper":
            executable = shutil.which("piper")
            model = Path(self.config.tts_voice).expanduser()
            if executable and model.is_file():
                self._run_checked(
                    [executable, "--model", str(model), "--output_file", str(audio_path)],
                    120,
                    "Piper",
                    input_text=text,
                )
                if not audio_path.is_file() or audio_path.stat().st_size == 0:
                    raise VoiceError("Piper completed without producing an audio file.")
                return

            fallback = shutil.which("espeak-ng") or shutil.which("espeak")
            if fallback:
                self._synthesize_espeak(
                    fallback, text, audio_path, DEFAULT_TTS_VOICE
                )
                return
            if not executable:
                raise VoiceError(
                    "Piper is not installed and no eSpeak fallback is available. "
                    "Install Piper or eSpeak-NG."
                )
            raise VoiceError(
                "The selected Piper voice model was not found and no eSpeak "
                "fallback is available."
            )

        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if not executable:
            raise VoiceError(
                "No local TTS engine found. Install espeak-ng or configure Piper."
            )
        self._synthesize_espeak(executable, text, audio_path, self.config.tts_voice)

    def _synthesize_espeak(
        self, executable: str, text: str, audio_path: Path, voice: str
    ) -> None:
        self._run_checked(
            [executable, "-w", str(audio_path), "-v", voice, "--", text],
            120,
            "eSpeak",
        )

    def _play(self, audio_path: Path, cancel_event: threading.Event | None) -> None:
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