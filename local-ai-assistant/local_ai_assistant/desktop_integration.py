"""Linux desktop integration helpers for autostart."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path


AUTOSTART_FILENAME = "lura.desktop"


def autostart_path(config_dir: Path | None = None) -> Path:
    """Return the user-level freedesktop autostart entry path."""

    base = config_dir or (Path.home() / ".config")
    return base / "autostart" / AUTOSTART_FILENAME


def autostart_entry(
    executable: str | None = None,
    *,
    application_name: str = "Lura",
) -> str:
    """Build a desktop entry that launches Lura hidden in background mode."""

    command = [
        executable or sys.executable,
        "-m",
        "local_ai_assistant.app",
        "--background",
    ]
    exec_command = shlex.join(command)
    return "\n".join(
        [
            "[Desktop Entry]",
            "Type=Application",
            f"Name={application_name}",
            "Comment=Local Ollama desktop assistant",
            f"Exec={exec_command}",
            f"TryExec={shlex.quote(executable or sys.executable)}",
            "Terminal=false",
            "StartupNotify=false",
            "X-GNOME-Autostart-enabled=true",
            "",
        ]
    )


def set_autostart_enabled(
    enabled: bool,
    *,
    config_dir: Path | None = None,
    executable: str | None = None,
) -> Path:
    """Create or remove Lura's user-level autostart entry."""

    path = autostart_path(config_dir)
    if enabled:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(autostart_entry(executable), encoding="utf-8")
        path.chmod(0o644)
    else:
        path.unlink(missing_ok=True)
    return path


def is_autostart_enabled(config_dir: Path | None = None) -> bool:
    """Return whether Lura's user-level autostart entry exists."""

    return autostart_path(config_dir).is_file()