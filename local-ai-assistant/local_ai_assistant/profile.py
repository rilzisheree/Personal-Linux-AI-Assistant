"""Editable local user profile and dynamically collected system profile."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_USER_PROFILE: dict[str, str] = {
    "name": "",
    "owner": "the current local user",
    "preferred_address": "Sir",
    "assistant_role": "Lura is my personal AI assistant.",
    "application_install_preference": "Flatpak",
}


class UserProfileStore:
    """Persist editable user facts separately from live machine state."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (
            Path.home() / ".config" / "local-ai-assistant" / "user_profile.json"
        )

    def profile(self) -> dict[str, str]:
        profile = dict(DEFAULT_USER_PROFILE)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
            return profile
        if isinstance(raw, dict):
            for key in profile:
                value = raw.get(key)
                if isinstance(value, str):
                    profile[key] = value.strip()
        return profile

    def update(self, updates: dict[str, Any]) -> dict[str, str]:
        profile = self.profile()
        for key in DEFAULT_USER_PROFILE:
            value = updates.get(key)
            if isinstance(value, str):
                profile[key] = value.strip()
        self._save(profile)
        return profile

    def context(self) -> str:
        profile = self.profile()
        lines = ["USER PROFILE (editable local facts):"]
        for key, value in profile.items():
            if value:
                lines.append(f"- {key}: {value}")
            else:
                lines.append(f"- {key}: not configured")
        return "\n".join(lines)

    def _save(self, profile: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f"{self.path.name}.", dir=self.path.parent, text=True
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(profile, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            temporary_path.chmod(0o600)
            temporary_path.replace(self.path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _run(command: list[str], timeout: int = 5) -> str:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def collect_system_profile() -> dict[str, Any]:
    """Collect relatively stable machine facts from the current host."""

    from .applications import ApplicationRegistry

    desktop = (
        os.environ.get("XDG_CURRENT_DESKTOP")
        or os.environ.get("XDG_SESSION_DESKTOP")
        or "unknown"
    )
    window_manager = os.environ.get("XDG_CURRENT_DESKTOP", "").strip() or "unknown"
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        window_manager = "Hyprland"
    cpu = platform.processor() or _run(["lscpu"]) or "unknown"
    if "\n" in cpu:
        cpu = next(
            (line.split(":", 1)[1].strip() for line in cpu.splitlines() if line.startswith("Model name:")),
            cpu.splitlines()[0],
        )
    gpu = _run(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ]
    )
    if not gpu and shutil.which("lspci"):
        lspci = _run(["lspci"])
        gpu = next(
            (
                line
                for line in lspci.splitlines()
                if any(token in line.casefold() for token in ("vga", "3d", "display"))
            ),
            "",
        )
    monitors = _run(["hyprctl", "monitors", "-j"]) if shutil.which("hyprctl") else ""
    try:
        monitor_setup: Any = json.loads(monitors) if monitors else None
    except json.JSONDecodeError:
        monitor_setup = monitors
    applications = [
        {"id": app.app_id, "name": app.name, "kind": app.kind}
        for app in ApplicationRegistry().list()[:200]
    ]
    return {
        "operating_system": platform.platform(),
        "kernel": platform.release(),
        "desktop_environment": desktop,
        "window_manager": window_manager,
        "cpu": cpu,
        "gpu": gpu or "unknown",
        "monitor_setup": monitor_setup or "Use get_display_info for live monitor data.",
        "installed_applications": applications,
        "flatpak_available": bool(shutil.which("flatpak")),
        "application_install_preference": "Flatpak",
    }