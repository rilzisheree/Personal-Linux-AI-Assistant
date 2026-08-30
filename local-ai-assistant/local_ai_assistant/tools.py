"""Local tools exposed to Ollama with explicit permission boundaries."""

from __future__ import annotations

import glob
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


class PermissionLevel(str, Enum):
    SAFE = "safe"
    CONFIRMATION_REQUIRED = "confirmation_required"
    DANGEROUS = "dangerous"


@dataclass(frozen=True)
class ToolCallResult:
    """A user-facing result returned to the model and the tool status UI."""

    success: bool
    content: str


class ToolConfirmationRequired(Exception):
    """Raised when a tool is called without the required user approval."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    permission: PermissionLevel
    parameters: dict
    handler: Callable[[dict], ToolCallResult]

    def ollama_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_DANGEROUS_COMMANDS = re.compile(
    r"(^|[;&|]\s*)(sudo|rm|mkfs(?:\.[^\s]+)?|dd|shutdown|reboot|poweroff)"
    r"(\s|$)|\b(rm\s+-rf|rm\s+-fr)\b",
    re.IGNORECASE,
)


def _string_argument(arguments: dict, name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Tool argument '{name}' must be a non-empty string.")
    return value.strip()


def _failed(error: Exception) -> ToolCallResult:
    return ToolCallResult(False, str(error) or error.__class__.__name__)


class ToolManager:
    """Registry and safety gate for tools callable by the local model."""

    def __init__(self) -> None:
        self._definitions = {
            "open_app": ToolDefinition(
                "open_app",
                "Open a desktop application by executable name.",
                PermissionLevel.SAFE,
                {
                    "type": "object",
                    "properties": {"app": {"type": "string", "description": "Executable name, such as firefox"}},
                    "required": ["app"],
                },
                self._open_app,
            ),
            "close_app": ToolDefinition(
                "close_app",
                "Close all processes for a desktop application.",
                PermissionLevel.CONFIRMATION_REQUIRED,
                {
                    "type": "object",
                    "properties": {"app": {"type": "string", "description": "Executable name"}},
                    "required": ["app"],
                },
                self._close_app,
            ),
            "restart_app": ToolDefinition(
                "restart_app",
                "Close and reopen a desktop application.",
                PermissionLevel.CONFIRMATION_REQUIRED,
                {
                    "type": "object",
                    "properties": {"app": {"type": "string", "description": "Executable name"}},
                    "required": ["app"],
                },
                self._restart_app,
            ),
            "exec": ToolDefinition(
                "exec",
                "Run a terminal command on the local Linux machine. Use only when necessary.",
                PermissionLevel.CONFIRMATION_REQUIRED,
                {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "Shell command to run"}},
                    "required": ["command"],
                },
                self._exec,
            ),
            "get_cpu_usage": self._system_definition(
                "get_cpu_usage", "Get current CPU utilization as a percentage.", self._get_cpu_usage
            ),
            "get_gpu_usage": self._system_definition(
                "get_gpu_usage", "Get current NVIDIA GPU utilization as a percentage.", self._get_gpu_usage
            ),
            "get_ram_usage": self._system_definition(
                "get_ram_usage", "Get current RAM usage.", self._get_ram_usage
            ),
            "get_disk_usage": self._system_definition(
                "get_disk_usage", "Get disk usage for the root filesystem.", self._get_disk_usage
            ),
            "get_temperature": self._system_definition(
                "get_temperature", "Get available system temperature readings.", self._get_temperature
            ),
            "get_battery": self._system_definition(
                "get_battery", "Get battery charge and status.", self._get_battery
            ),
            "get_volume": self._system_definition(
                "get_volume", "Get the current default audio output volume.", self._get_volume
            ),
        }

    @staticmethod
    def _system_definition(
        name: str, description: str, handler: Callable[[dict], ToolCallResult]
    ) -> ToolDefinition:
        return ToolDefinition(
            name,
            description,
            PermissionLevel.SAFE,
            {"type": "object", "properties": {}},
            handler,
        )

    def definitions_for_ollama(self) -> list[dict]:
        return [definition.ollama_schema() for definition in self._definitions.values()]

    def permission_for(self, name: str, arguments: dict) -> PermissionLevel:
        definition = self._definitions.get(name)
        if definition is None:
            return PermissionLevel.DANGEROUS
        if name == "exec":
            command = arguments.get("command", "")
            if isinstance(command, str) and _DANGEROUS_COMMANDS.search(command):
                return PermissionLevel.DANGEROUS
        return definition.permission

    def execute(self, name: str, arguments: dict, approved: bool = False) -> ToolCallResult:
        if not isinstance(arguments, dict):
            return ToolCallResult(False, "Tool arguments must be an object.")
        definition = self._definitions.get(name)
        if definition is None:
            return ToolCallResult(False, f"Unknown tool: {name}")
        permission = self.permission_for(name, arguments)
        if permission != PermissionLevel.SAFE and not approved:
            raise ToolConfirmationRequired(f"{name} requires user confirmation.")
        try:
            return definition.handler(arguments)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            return _failed(error)

    @staticmethod
    def _application_command(arguments: dict) -> list[str]:
        app = _string_argument(arguments, "app")
        command = shlex.split(app)
        if not command or any(part.startswith("-") for part in command[1:]):
            raise ValueError("Application calls may include an executable name only.")
        executable = command[0]
        if not shutil.which(executable) and not Path(executable).is_file():
            raise ValueError(f"Application not found: {executable}")
        return command

    def _open_app(self, arguments: dict) -> ToolCallResult:
        command = self._application_command(arguments)
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return ToolCallResult(True, f"{command[0]} opened.")

    def _close_app(self, arguments: dict) -> ToolCallResult:
        command = self._application_command(arguments)
        result = subprocess.run(
            ["pkill", "-x", Path(command[0]).name],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 1:
            return ToolCallResult(False, f"{command[0]} is not running.")
        if result.returncode != 0:
            return ToolCallResult(False, result.stderr.strip() or f"Could not close {command[0]}.")
        return ToolCallResult(True, f"{command[0]} closed.")

    def _restart_app(self, arguments: dict) -> ToolCallResult:
        closed = self._close_app(arguments)
        if not closed.success:
            return closed
        time.sleep(0.4)
        return self._open_app(arguments)

    @staticmethod
    def _exec(arguments: dict) -> ToolCallResult:
        command = _string_argument(arguments, "command")
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        output = (completed.stdout + completed.stderr).strip()
        if len(output) > 4000:
            output = output[:3997] + "…"
        detail = output or "(no output)"
        if completed.returncode:
            return ToolCallResult(False, f"Command exited with code {completed.returncode}.\n{detail}")
        return ToolCallResult(True, detail)

    @staticmethod
    def _get_cpu_usage(arguments: dict) -> ToolCallResult:
        del arguments
        first = _read_cpu_times()
        time.sleep(0.1)
        second = _read_cpu_times()
        total_delta = second[0] - first[0]
        idle_delta = second[3] - first[3]
        if total_delta <= 0:
            return ToolCallResult(False, "CPU usage is unavailable.")
        usage = max(0.0, min(100.0, 100 * (1 - idle_delta / total_delta)))
        return ToolCallResult(True, f"{usage:.1f}% CPU usage")

    @staticmethod
    def _get_gpu_usage(arguments: dict) -> ToolCallResult:
        del arguments
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return ToolCallResult(False, f"NVIDIA GPU usage unavailable: {error}")
        values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode or not values:
            return ToolCallResult(False, "NVIDIA GPU usage is unavailable.")
        return ToolCallResult(True, f"{', '.join(values)}% GPU usage")

    @staticmethod
    def _get_ram_usage(arguments: dict) -> ToolCallResult:
        del arguments
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition(":")
                if separator:
                    values[key] = int(value.strip().split()[0])
        except (OSError, ValueError):
            return ToolCallResult(False, "RAM usage is unavailable.")
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        if not total or available is None:
            return ToolCallResult(False, "RAM usage is unavailable.")
        used = total - available
        return ToolCallResult(True, f"{used / 1024:.0f} MiB used of {total / 1024:.0f} MiB ({used / total * 100:.1f}%)")

    @staticmethod
    def _get_disk_usage(arguments: dict) -> ToolCallResult:
        del arguments
        usage = shutil.disk_usage("/")
        used = usage.total - usage.free
        return ToolCallResult(
            True,
            f"{used / 2**30:.1f} GiB used of {usage.total / 2**30:.1f} GiB ({used / usage.total * 100:.1f}%)",
        )

    @staticmethod
    def _get_temperature(arguments: dict) -> ToolCallResult:
        del arguments
        readings: list[str] = []
        for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            try:
                value = int(Path(path).read_text(encoding="utf-8").strip()) / 1000
            except (OSError, ValueError):
                continue
            if value > 0:
                zone = Path(path).parent.name
                readings.append(f"{zone}: {value:.1f}°C")
        return ToolCallResult(bool(readings), ", ".join(readings) if readings else "Temperature is unavailable.")

    @staticmethod
    def _get_battery(arguments: dict) -> ToolCallResult:
        del arguments
        batteries = glob.glob("/sys/class/power_supply/BAT*/")
        if not batteries:
            return ToolCallResult(False, "No battery detected.")
        readings: list[str] = []
        for battery in batteries:
            base = Path(battery)
            try:
                capacity = (base / "capacity").read_text(encoding="utf-8").strip()
                status = (base / "status").read_text(encoding="utf-8").strip()
            except OSError:
                continue
            readings.append(f"{base.name}: {capacity}% ({status})")
        return ToolCallResult(bool(readings), ", ".join(readings) if readings else "Battery status is unavailable.")

    @staticmethod
    def _get_volume(arguments: dict) -> ToolCallResult:
        del arguments
        for command in (["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"], ["pactl", "get-sink-volume", "@DEFAULT_SINK@"]):
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
            except (OSError, subprocess.SubprocessError):
                continue
            if result.returncode == 0 and result.stdout.strip():
                return ToolCallResult(True, result.stdout.strip())
        return ToolCallResult(False, "Audio volume is unavailable.")


def _read_cpu_times() -> tuple[int, ...]:
    line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    values = line.split()[1:]
    if len(values) < 4:
        raise ValueError("CPU statistics are unavailable.")
    return tuple(int(value) for value in values)