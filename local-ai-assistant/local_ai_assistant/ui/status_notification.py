"""Desktop notification bridge for background voice activity."""

from __future__ import annotations

import logging
import shutil
import subprocess

from PySide6.QtCore import QObject


LOGGER = logging.getLogger("lura.ui")


class StatusNotification(QObject):
    """Send status updates through the desktop notification daemon.

    This intentionally does not create a Qt top-level window. Hyprland treats
    every Qt top-level surface as a client, even when it uses ToolTip or
    SplashScreen window flags.
    """

    STATE_LABELS = {
        "idle": "Idle",
        "listening": "Listening…",
        "processing": "Thinking…",
        "speaking": "Speaking…",
        "error": "Error",
    }
    STATE_COLORS = {
        "idle": "#75b9e6",
        "listening": "#6dd9b4",
        "processing": "#b189ff",
        "speaking": "#4aaeff",
        "error": "#e07f91",
    }

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state = "idle"
        self._notification_id: int | None = None
        self._notify_send = shutil.which("notify-send")
        self._warning_logged = False

    def set_state(self, state: str) -> None:
        normalized = "processing" if state == "thinking" else state
        self._state = normalized if normalized in self.STATE_LABELS else "idle"

    def show_for_desktop(self) -> None:
        if self._state == "idle":
            return
        if self._notify_send is None:
            if not self._warning_logged:
                LOGGER.warning(
                    "[Voice] notify-send is unavailable; status notifications are disabled"
                )
                self._warning_logged = True
            return

        command = [
            self._notify_send,
            "--app-name=Lura",
            "--print-id",
            "--urgency=normal",
            "--expire-time=0",
        ]
        if self._notification_id is not None:
            command.append(f"--replace-id={self._notification_id}")
        command.extend(["Lura", self.STATE_LABELS[self._state]])
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=0.5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            LOGGER.warning("[Voice] Could not show status notification: %s", error)
            return
        if result.returncode != 0:
            LOGGER.warning(
                "[Voice] notify-send failed: %s",
                result.stderr.strip() or f"exit code {result.returncode}",
            )
            return
        try:
            self._notification_id = int(result.stdout.strip())
        except ValueError:
            # Some notification daemons do not implement --print-id. The
            # notification is still visible; replacement is simply unavailable.
            self._notification_id = None

    def hide(self) -> None:
        if self._notify_send is None or self._notification_id is None:
            self._notification_id = None
            return
        try:
            subprocess.run(
                [
                    self._notify_send,
                    "--close",
                    f"--replace-id={self._notification_id}",
                ],
                capture_output=True,
                text=True,
                timeout=0.5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        finally:
            self._notification_id = None