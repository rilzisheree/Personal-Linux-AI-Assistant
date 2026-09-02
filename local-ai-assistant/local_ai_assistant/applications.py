"""Discover installed Linux applications without guessing executable names."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApplicationRecord:
    """A launchable application discovered from the host system."""

    app_id: str
    name: str
    kind: str
    launch_command: tuple[str, ...]
    desktop_file: str = ""

    @property
    def search_text(self) -> str:
        return f"{self.name} {self.app_id}".casefold()


def _desktop_directories() -> list[Path]:
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    data_dirs = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(":")
    roots = [data_home, *(Path(item) for item in data_dirs if item)]
    directories: list[Path] = []
    for root in roots:
        directory = root / "applications"
        if directory not in directories:
            directories.append(directory)
    return directories


def _desktop_entry(path: Path) -> ApplicationRecord | None:
    section = ""
    values: dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue
            if section != "Desktop Entry" or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    except OSError:
        return None
    if values.get("Type") != "Application" or values.get("NoDisplay", "").lower() == "true":
        return None
    if values.get("Hidden", "").lower() == "true":
        return None
    name = values.get("Name", "").strip()
    exec_line = values.get("Exec", "").strip()
    if not name or not exec_line:
        return None
    try:
        command = tuple(
            part
            for part in shlex.split(exec_line)
            if not part.startswith("%") and "%" not in part
        )
    except ValueError:
        return None
    if not command:
        return None
    executable = command[0]
    try_exec = values.get("TryExec", "").strip()
    if try_exec and not shutil.which(try_exec) and not Path(try_exec).is_file():
        return None
    if not shutil.which(executable) and not Path(executable).is_file():
        return None
    return ApplicationRecord(
        app_id=path.stem,
        name=name,
        kind="native",
        launch_command=command,
        desktop_file=str(path),
    )


def discover_flatpak() -> list[ApplicationRecord]:
    """Return installed Flatpak applications from Flatpak's own registry."""

    executable = shutil.which("flatpak")
    if not executable:
        return []
    try:
        result = subprocess.run(
            [executable, "list", "--app", "--columns=application,name"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode:
        return []
    records: list[ApplicationRecord] = []
    for line in result.stdout.splitlines():
        app_id, separator, name = line.partition("\t")
        if not separator:
            parts = line.split(None, 1)
            app_id, name = (parts + [""])[:2]
        app_id, name = app_id.strip(), name.strip()
        if not app_id or not name:
            continue
        records.append(
            ApplicationRecord(
                app_id=app_id,
                name=name,
                kind="flatpak",
                launch_command=("flatpak", "run", app_id),
            )
        )
    return records


def discover_native() -> list[ApplicationRecord]:
    """Return launchable applications from freedesktop desktop entries."""

    records: list[ApplicationRecord] = []
    seen: set[str] = set()
    for directory in _desktop_directories():
        if not directory.is_dir():
            continue
        try:
            paths = sorted(directory.glob("*.desktop"))
        except OSError:
            continue
        for path in paths:
            record = _desktop_entry(path)
            if record is None or record.app_id in seen:
                continue
            seen.add(record.app_id)
            records.append(record)
    return records


class ApplicationRegistry:
    """Refreshable registry backed by current Flatpak and desktop-entry data."""

    def discover(self) -> list[ApplicationRecord]:
        flatpak = discover_flatpak()
        native = discover_native()
        # Flatpak is intentionally first: it is the user's stated primary
        # application source and avoids choosing a similarly named native shim.
        return [*flatpak, *native]

    def resolve(self, query: str) -> ApplicationRecord:
        normalized = query.strip().casefold()
        if not normalized:
            raise ValueError("Application name cannot be empty.")
        records = self.discover()
        exact = [
            record
            for record in records
            if record.name.casefold() == normalized
            or record.app_id.casefold() == normalized
        ]
        if len(exact) == 1:
            return exact[0]
        matches = [
            record
            for record in records
            if normalized in record.name.casefold()
            or normalized in record.app_id.casefold()
        ]
        if not matches:
            raise ValueError(
                f"Application not found in installed application data: {query}"
            )
        if len(matches) > 1:
            choices = ", ".join(f"{item.name} ({item.kind})" for item in matches[:6])
            raise ValueError(
                f"Multiple applications matched '{query}': {choices}. "
                "Use the exact display name or application ID."
            )
        return matches[0]

    def list(self) -> list[ApplicationRecord]:
        return self.discover()