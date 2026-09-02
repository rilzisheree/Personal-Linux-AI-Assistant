"""Persistent local reminders and their desktop notification scheduler."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


LOGGER = logging.getLogger("lura.reminders")
MAX_DELAY_SECONDS = 365 * 24 * 60 * 60


@dataclass(frozen=True)
class Reminder:
    reminder_id: str
    message: str
    due_at: float

    def as_dict(self) -> dict[str, str | float]:
        return {
            "id": self.reminder_id,
            "message": self.message,
            "due_at": self.due_at,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "Reminder | None":
        if not isinstance(payload, dict):
            return None
        reminder_id = payload.get("id")
        message = payload.get("message")
        due_at = payload.get("due_at")
        if (
            not isinstance(reminder_id, str)
            or not reminder_id.strip()
            or not isinstance(message, str)
            or not message.strip()
            or isinstance(due_at, bool)
            or not isinstance(due_at, (int, float))
            or due_at <= 0
        ):
            return None
        return cls(reminder_id.strip(), message.strip(), float(due_at))


class ReminderStore:
    """Read and atomically persist reminders in the user's config directory."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (
            Path.home() / ".config" / "local-ai-assistant" / "reminders.json"
        )
        self._lock = threading.Lock()

    def list(self) -> list[Reminder]:
        with self._lock:
            return self._read()

    def add(self, reminder: Reminder) -> None:
        with self._lock:
            reminders = self._read()
            reminders.append(reminder)
            self._save(reminders)

    def remove(self, reminder_id: str) -> bool:
        with self._lock:
            reminders = self._read()
            remaining = [
                reminder for reminder in reminders
                if reminder.reminder_id != reminder_id
            ]
            if len(remaining) == len(reminders):
                return False
            self._save(remaining)
            return True

    def _read(self) -> list[Reminder]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
            return []
        if not isinstance(raw, list):
            return []
        reminders = [Reminder.from_dict(item) for item in raw]
        return [reminder for reminder in reminders if reminder is not None]

    def _save(self, reminders: list[Reminder]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f"{self.path.name}.",
            dir=self.path.parent,
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    [reminder.as_dict() for reminder in reminders],
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
                handle.write("\n")
            temporary_path.chmod(0o600)
            temporary_path.replace(self.path)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


class ReminderService:
    """Schedule reminders and deliver them through the Linux notification daemon."""

    def __init__(
        self,
        store: ReminderStore | None = None,
        notify_send: str | None = None,
        *,
        clock=time.time,
        start_scheduler: bool = True,
        on_due: Callable[[Reminder], None] | None = None,
    ) -> None:
        self.store = store or ReminderStore()
        self._notify_send = (
            shutil.which("notify-send") if notify_send is None else notify_send
        )
        self._clock = clock
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._listener_lock = threading.Lock()
        self._due_listeners: list[Callable[[Reminder], None]] = []
        if on_due is not None:
            self._due_listeners.append(on_due)
        self._thread: threading.Thread | None = None
        if start_scheduler:
            self._thread = threading.Thread(
                target=self._run,
                name="lura-reminders",
                daemon=True,
            )
            self._thread.start()

    def schedule(self, message: str, delay_seconds: float) -> Reminder:
        cleaned_message = message.strip()
        if not cleaned_message:
            raise ValueError("Reminder message cannot be empty.")
        if (
            isinstance(delay_seconds, bool)
            or not isinstance(delay_seconds, (int, float))
            or delay_seconds <= 0
            or delay_seconds > MAX_DELAY_SECONDS
        ):
            raise ValueError(
                f"Reminder delay must be greater than 0 and no more than "
                f"{MAX_DELAY_SECONDS} seconds."
            )
        reminder = Reminder(
            uuid.uuid4().hex,
            cleaned_message,
            self._clock() + float(delay_seconds),
        )
        self.store.add(reminder)
        with self._condition:
            self._condition.notify_all()
        return reminder

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            reminders = self.store.list()
            now = self._clock()
            due = [reminder for reminder in reminders if reminder.due_at <= now]
            if due:
                for reminder in due:
                    self.store.remove(reminder.reminder_id)
                    self._notify(reminder)
                    self._emit_due(reminder)
                continue

            wait_seconds = 60.0
            if reminders:
                wait_seconds = max(0.1, min(reminder.due_at for reminder in reminders) - now)
            with self._condition:
                self._condition.wait(timeout=wait_seconds)

    def add_due_listener(self, listener: Callable[[Reminder], None]) -> None:
        """Register a callback invoked when a reminder becomes due."""
        with self._listener_lock:
            if listener not in self._due_listeners:
                self._due_listeners.append(listener)

    def remove_due_listener(self, listener: Callable[[Reminder], None]) -> None:
        """Remove a previously registered due callback."""
        with self._listener_lock:
            self._due_listeners = [
                registered
                for registered in self._due_listeners
                if registered != listener
            ]

    def _emit_due(self, reminder: Reminder) -> None:
        with self._listener_lock:
            listeners = tuple(self._due_listeners)
        for listener in listeners:
            try:
                listener(reminder)
            except Exception:
                LOGGER.exception("Reminder due listener failed")

    def _notify(self, reminder: Reminder) -> None:
        if not self._notify_send:
            LOGGER.warning(
                "Reminder is due but notify-send is unavailable: %s",
                reminder.message,
            )
            return
        try:
            result = subprocess.run(
                [
                    self._notify_send,
                    "--app-name=Lura",
                    "--urgency=normal",
                    "--expire-time=10000",
                    "Lura reminder",
                    reminder.message,
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            LOGGER.warning("Could not deliver reminder notification: %s", error)
            return
        if result.returncode != 0:
            LOGGER.warning(
                "notify-send failed for reminder: %s",
                result.stderr.strip() or f"exit code {result.returncode}",
            )


_DEFAULT_SERVICE: ReminderService | None = None
_DEFAULT_SERVICE_LOCK = threading.Lock()


def default_reminder_service() -> ReminderService:
    """Return the one scheduler shared by desktop, API, and Telegram tool managers."""
    global _DEFAULT_SERVICE
    with _DEFAULT_SERVICE_LOCK:
        if _DEFAULT_SERVICE is None:
            _DEFAULT_SERVICE = ReminderService()
        return _DEFAULT_SERVICE


def format_due_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )