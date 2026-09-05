"""Local tools exposed to Ollama with explicit permission boundaries."""

from __future__ import annotations

import csv
import glob
import json
import logging
import math
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

from .applications import ApplicationRecord, ApplicationRegistry
from .information_tools import (
    InformationResult,
    currency as information_currency,
    directions as information_directions,
    find_places as information_find_places,
    game_search as information_game_search,
    knowledge_search as information_knowledge_search,
    news as information_news,
    travel_search as information_travel_search,
    weather as information_weather,
)
from .memory import MemoryStore
from .profile import UserProfileStore, collect_system_profile
from .reminders import (
    default_reminder_service,
    format_due_time,
    format_reminder_timing,
)


LOGGER = logging.getLogger("lura.tools")


class PermissionLevel(str, Enum):
    SAFE = "safe"
    NORMAL = "normal"
    CONFIRMATION_REQUIRED = "confirmation_required"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ToolCallResult:
    """A user-facing result returned to the model and the tool status UI."""

    success: bool
    content: str
    images: tuple[str, ...] = ()


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
_READ_ONLY_EXEC_COMMANDS = {
    "df",
    "free",
    "lscpu",
    "ls",
    "nvidia-smi",
    "ps",
    "uname",
    "uptime",
    "cat",
}
_UNSAFE_LAUNCH_EXECUTABLES = {
    "bash",
    "dash",
    "fish",
    "sh",
    "sudo",
    "doas",
    "zsh",
}
_LAUNCH_SHELL_OPERATORS = re.compile(r"[;&|<>]")
_WINDOW_ADDRESS = re.compile(r"^0x[0-9a-f]+$", re.IGNORECASE)
_WORKSPACE_NAME = re.compile(r"^[A-Za-z0-9_:-]+$")
_MAX_SEARCH_RESULTS = 50
_MAX_SEARCH_DIRECTORIES = 10_000
_MAX_SCREENSHOTS = 20
_WEB_SEARCH_TIMEOUT = 15
_WEB_SEARCH_MAX_RESPONSE_BYTES = 1_000_000
_GPU_TOOL_NAMES = frozenset({"get_gpu_info", "get_gpu_status", "get_gpu_usage"})
_GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]


class _DuckDuckGoParser(HTMLParser):
    """Extract the small result fields needed by the assistant."""

    def __init__(self) -> None:
        super().__init__()
        self.items: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._field: str | None = None
        self._field_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self.finish_current()
            self._current = {
                "title": "",
                "url": attributes.get("href") or "",
                "snippet": "",
            }
            self._field = "title"
            self._field_tag = None
        elif self._current is not None and "result__snippet" in classes:
            self._field = "snippet"
            self._field_tag = tag

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._field:
            self._current[self._field] += " ".join(data.split())

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if tag == "a" and self._field == "title":
            self._field = None
        elif self._field == "snippet" and tag == self._field_tag:
            self.items.append(self._current)
            self._current = None
            self._field = None
            self._field_tag = None

    def finish_current(self) -> None:
        """Commit a result even when a provider omits its snippet element."""
        if self._current is not None:
            self.items.append(self._current)
            self._current = None
            self._field = None
            self._field_tag = None


def _string_argument(arguments: dict, name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Tool argument '{name}' must be a non-empty string.")
    return value.strip()


def _optional_string_argument(arguments: dict, name: str) -> str:
    value = arguments.get(name, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Tool argument '{name}' must be a string.")
    return value.strip()


def _first_inline_data(payload: object) -> str | None:
    if isinstance(payload, dict):
        for key in ("inlineData", "inline_data"):
            value = payload.get(key)
            if isinstance(value, dict) and isinstance(value.get("data"), str):
                return value["data"]
        for value in payload.values():
            found = _first_inline_data(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _first_inline_data(value)
            if found:
                return found
    return None


def _failed(error: Exception) -> ToolCallResult:
    return ToolCallResult(False, str(error) or error.__class__.__name__)


def _information_result(result: InformationResult) -> ToolCallResult:
    return ToolCallResult(result.success, result.content)


def _tool_debug_enabled() -> bool:
    return os.environ.get("LURA_STREAM_DEBUG", "").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _gpu_failure(error: str, details: list[str] | None = None) -> ToolCallResult:
    payload: dict[str, object] = {
        "available": False,
        "vendor": "NVIDIA",
        "model": None,
        "vram_total_mb": None,
        "vram_used_mb": None,
        "utilization_percent": None,
        "temperature_c": None,
        "gpus": [],
        "error": error,
    }
    if details:
        payload["error_details"] = details
    return ToolCallResult(
        False,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )


def _parse_gpu_rows(stdout: str) -> tuple[list[dict[str, object]], list[str]]:
    gpus: list[dict[str, object]] = []
    errors: list[str] = []
    try:
        rows = csv.reader(stdout.splitlines(), strict=True)
        for row_number, fields in enumerate(rows, start=1):
            if not fields or not any(field.strip() for field in fields):
                continue
            if len(fields) != 5:
                errors.append(
                    f"row {row_number}: expected 5 fields, received {len(fields)}"
                )
                continue
            name, utilization, temperature, memory_used, memory_total = (
                field.strip() for field in fields
            )
            if not name:
                errors.append(f"row {row_number}: GPU name is empty")
                continue
            try:
                numeric_values = {
                    "utilization_percent": float(utilization),
                    "temperature_c": float(temperature),
                    "vram_used_mb": float(memory_used),
                    "vram_total_mb": float(memory_total),
                }
            except ValueError:
                errors.append(f"row {row_number}: GPU metrics are not numeric")
                continue
            if not all(math.isfinite(value) for value in numeric_values.values()):
                errors.append(f"row {row_number}: GPU metrics are not finite")
                continue
            gpus.append({"name": name, **numeric_values})
    except csv.Error as error:
        errors.append(f"CSV parsing failed: {error}")
    return gpus, errors


class ToolManager:
    """Registry and safety gate for tools callable by the local model."""

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        profile_store: UserProfileStore | None = None,
        assistant_name: str = "Lura",
        tool_permissions: dict[str, str] | None = None,
        custom_app_commands: dict[str, str] | None = None,
        active_model: str | None = None,
    ) -> None:
        self.memory_store = memory_store or MemoryStore()
        self.profile_store = profile_store or UserProfileStore()
        self.assistant_name = assistant_name.strip() or "Lura"
        self.application_registry = ApplicationRegistry()
        self.tool_permissions = {
            key.strip(): value.strip()
            for key, value in (tool_permissions or {}).items()
            if isinstance(key, str) and isinstance(value, str) and key.strip() and value.strip()
        }
        self.custom_app_commands = {
            key.strip(): value.strip()
            for key, value in (custom_app_commands or {}).items()
            if isinstance(key, str) and isinstance(value, str) and key.strip() and value.strip()
        }
        self._active_model = active_model.strip() if isinstance(active_model, str) and active_model.strip() else None
        self.reminder_service = default_reminder_service()
        custom_launcher_names = ", ".join(
            f'"{alias}"' for alias in self.custom_app_commands
        )
        open_app_description = (
            "Open an installed desktop application by its display name or "
            "application ID. Resolve it from installed Flatpak or native "
            "application data; never guess an executable."
        )
        if custom_launcher_names:
            open_app_description += (
                " The user also configured these exact custom app names: "
                f"{custom_launcher_names}. Use those names with open_app."
            )
        self._definitions = {
            "open_app": ToolDefinition(
                "open_app",
                open_app_description,
                PermissionLevel.NORMAL,
                {
                    "type": "object",
                    "properties": {"app": {"type": "string", "description": "Executable name, such as firefox"}},
                    "required": ["app"],
                },
                self._open_app,
            ),
            "close_app": ToolDefinition(
                "close_app",
                "Close an installed desktop application by display name or application ID.",
                PermissionLevel.NORMAL,
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
            "list_windows": self._system_definition(
                "list_windows", "List windows in the current Hyprland session.", self._list_windows
            ),
            "focus_window": self._window_definition(
                "focus_window", "Focus a Hyprland window by address, title, or application class.", self._focus_window
            ),
            "move_window": ToolDefinition(
                "move_window",
                "Move a Hyprland window to another workspace.",
                PermissionLevel.CONFIRMATION_REQUIRED,
                {
                    "type": "object",
                    "properties": {
                        "window": {"type": "string", "description": "Window address, title, or application class"},
                        "workspace": {"type": "string", "description": "Workspace name or number"},
                    },
                    "required": ["window", "workspace"],
                },
                self._move_window,
            ),
            "resize_window": ToolDefinition(
                "resize_window",
                "Resize a Hyprland window to an exact width and height.",
                PermissionLevel.CONFIRMATION_REQUIRED,
                {
                    "type": "object",
                    "properties": {
                        "window": {"type": "string", "description": "Window address, title, or application class"},
                        "width": {"type": "integer", "minimum": 1, "maximum": 10000},
                        "height": {"type": "integer", "minimum": 1, "maximum": 10000},
                    },
                    "required": ["window", "width", "height"],
                },
                self._resize_window,
            ),
            "close_window": ToolDefinition(
                "close_window",
                "Close a Hyprland window.",
                PermissionLevel.DANGEROUS,
                {
                    "type": "object",
                    "properties": {"window": {"type": "string", "description": "Window address, title, or application class"}},
                    "required": ["window"],
                },
                self._close_window,
            ),
            "take_screenshot": ToolDefinition(
                "take_screenshot",
                "Capture the current Linux desktop screen for local vision analysis.",
                PermissionLevel.NORMAL,
                {"type": "object", "properties": {}},
                self._take_screenshot,
            ),
            "search_files": ToolDefinition(
                "search_files",
                "Find files below a local directory by name.",
                PermissionLevel.SAFE,
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Case-insensitive part of a filename"},
                        "directory": {"type": "string", "description": "Directory to search, defaulting to the home directory"},
                    },
                    "required": ["query"],
                },
                self._search_files,
            ),
            "read_file": ToolDefinition(
                "read_file",
                "Read a UTF-8 text file from the local machine.",
                PermissionLevel.SAFE,
                {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "File path"}},
                    "required": ["path"],
                },
                self._read_file,
            ),
            "write_file": self._file_definition(
                "write_file", "Overwrite a UTF-8 text file.", self._write_file
            ),
            "create_file": self._file_definition(
                "create_file", "Create a new UTF-8 text file.", self._create_file
            ),
            "delete_file": ToolDefinition(
                "delete_file",
                "Delete one local file. Directories and protected home paths are rejected.",
                PermissionLevel.DANGEROUS,
                {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "File path"}},
                    "required": ["path"],
                },
                self._delete_file,
            ),
            "move_file": ToolDefinition(
                "move_file",
                "Move one local file to another path.",
                PermissionLevel.CONFIRMATION_REQUIRED,
                {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source file path"},
                        "destination": {"type": "string", "description": "Destination file path"},
                    },
                    "required": ["source", "destination"],
                },
                self._move_file,
            ),
            "copy_file": ToolDefinition(
                "copy_file",
                "Copy one local file to another path.",
                PermissionLevel.CONFIRMATION_REQUIRED,
                {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source file path"},
                        "destination": {"type": "string", "description": "Destination file path"},
                    },
                    "required": ["source", "destination"],
                },
                self._copy_file,
            ),
            "mouse_move": ToolDefinition(
                "mouse_move",
                "Move the pointer to screen coordinates.",
                PermissionLevel.CONFIRMATION_REQUIRED,
                {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "minimum": -20000, "maximum": 20000},
                        "y": {"type": "integer", "minimum": -20000, "maximum": 20000},
                    },
                    "required": ["x", "y"],
                },
                self._mouse_move,
            ),
            "mouse_click": ToolDefinition(
                "mouse_click",
                "Click a mouse button at screen coordinates.",
                PermissionLevel.CONFIRMATION_REQUIRED,
                {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "minimum": -20000, "maximum": 20000},
                        "y": {"type": "integer", "minimum": -20000, "maximum": 20000},
                        "button": {"type": "string", "enum": ["left", "right", "middle"]},
                    },
                    "required": ["x", "y", "button"],
                },
                self._mouse_click,
            ),
            "keyboard_type": ToolDefinition(
                "keyboard_type",
                "Type text into the currently focused application.",
                PermissionLevel.CONFIRMATION_REQUIRED,
                {
                    "type": "object",
                    "properties": {"text": {"type": "string", "maxLength": 2000}},
                    "required": ["text"],
                },
                self._keyboard_type,
            ),
            "keyboard_press": ToolDefinition(
                "keyboard_press",
                "Press one key in the currently focused application.",
                PermissionLevel.CONFIRMATION_REQUIRED,
                {
                    "type": "object",
                    "properties": {"key": {"type": "string", "description": "Key name, such as Return or Escape"}},
                    "required": ["key"],
                },
                self._keyboard_press,
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
            "open_website": ToolDefinition(
                "open_website",
                "Open an http or https website in the user's default browser.",
                PermissionLevel.NORMAL,
                {
                    "type": "object",
                    "properties": {"url": {"type": "string", "description": "Website URL"}},
                    "required": ["url"],
                },
                self._open_website,
            ),
            "web_search": ToolDefinition(
                "web_search",
                "Perform a live search of the public web for current information. Use the "
                "returned sources to answer or summarize; do not dump links without an explanation.",
                PermissionLevel.SAFE,
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "A focused web search query"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
                    },
                    "required": ["query"],
                },
                self._web_search,
            ),
            "get_weather": ToolDefinition(
                "get_weather",
                "Get current weather and a forecast for a city or location. Use this for "
                "weather, temperature, rain, and forecast questions. The location may be "
                "omitted only when LURA_DEFAULT_LOCATION is configured.",
                PermissionLevel.SAFE,
                {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City, region, address, or other location",
                        },
                        "date": {
                            "type": "string",
                            "description": "today, tomorrow, or a date in YYYY-MM-DD format",
                        },
                        "units": {
                            "type": "string",
                            "enum": ["metric", "imperial"],
                        },
                    },
                },
                self._get_weather,
            ),
            "search_news": ToolDefinition(
                "search_news",
                "Search recent news and return dated headlines and source summaries. "
                "Use this for today's news, breaking news, or news about a country, topic, "
                "game, company, or person.",
                PermissionLevel.SAFE,
                {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Topic or location; omit for the latest general news",
                        },
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
                    },
                },
                self._search_news,
            ),
            "knowledge_search": ToolDefinition(
                "knowledge_search",
                "Search Wikipedia for reliable general-knowledge information and a concise "
                "article summary. Use this for people, history, places, science, and concepts.",
                PermissionLevel.SAFE,
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Subject to look up"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["query"],
                },
                self._knowledge_search,
            ),
            "convert_currency": ToolDefinition(
                "convert_currency",
                "Convert money using a current exchange rate. Use ISO 4217 three-letter "
                "currency codes such as SAR, USD, EUR, or GBP; never use a hardcoded rate.",
                PermissionLevel.SAFE,
                {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number", "minimum": 0},
                        "from_currency": {"type": "string", "description": "Three-letter source currency code"},
                        "to_currency": {"type": "string", "description": "Three-letter target currency code"},
                    },
                    "required": ["amount", "from_currency", "to_currency"],
                },
                self._convert_currency,
            ),
            "find_places": ToolDefinition(
                "find_places",
                "Find places, businesses, or services and return map coordinates and links. "
                "Use this for restaurants, pharmacies, shops, and nearby places.",
                PermissionLevel.SAFE,
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Place, business, or service to find"},
                        "near": {"type": "string", "description": "City, neighborhood, or area to search near"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
                    },
                    "required": ["query"],
                },
                self._find_places,
            ),
            "get_directions": ToolDefinition(
                "get_directions",
                "Find a route and travel time between two natural-language locations. "
                "Use this for directions and distance questions.",
                PermissionLevel.SAFE,
                {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "Starting city, address, or landmark"},
                        "destination": {"type": "string", "description": "Destination city, address, or landmark"},
                        "mode": {
                            "type": "string",
                            "enum": ["driving", "walking", "cycling"],
                        },
                    },
                    "required": ["origin", "destination"],
                },
                self._get_directions,
            ),
            "travel_search": ToolDefinition(
                "travel_search",
                "Research travel information with current web sources. Use this for "
                "destinations, travel requirements, itineraries, attractions, prices, "
                "schedules, or flight questions; current details must be verified.",
                PermissionLevel.SAFE,
                {
                    "type": "object",
                    "properties": {
                        "destination": {"type": "string", "description": "Destination or route"},
                        "topic": {"type": "string", "description": "Travel question or information needed"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
                    },
                    "required": ["destination"],
                },
                self._travel_search,
            ),
            "game_search": ToolDefinition(
                "game_search",
                "Search current gaming information and guides. Use this for game strategies, "
                "builds, bosses, release dates, seasons, patches, and updates.",
                PermissionLevel.SAFE,
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Game, character, boss, or gaming question"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
                    },
                    "required": ["query"],
                },
                self._game_search,
            ),
            "create_reminder": ToolDefinition(
                "create_reminder",
                "Create a local desktop reminder that will notify the user at the requested "
                "time. Use this for requests such as 'remind me in 5 seconds to drink water'. "
                "The user's request itself is sufficient authorization; do not claim reminders "
                "are unavailable.",
                PermissionLevel.NORMAL,
                {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "What the user should be reminded about",
                        },
                        "delay_seconds": {
                            "type": "number",
                            "minimum": 0.1,
                            "maximum": 31536000,
                            "description": (
                                "Seconds from now until the reminder. Convert relative "
                                "requests such as 'in 5 minutes' to 300."
                            ),
                        },
                    },
                    "required": ["message", "delay_seconds"],
                },
                self._create_reminder,
            ),
            "list_reminders": ToolDefinition(
                "list_reminders",
                "List the user's upcoming reminders and reminder history.",
                PermissionLevel.SAFE,
                {"type": "object", "properties": {}},
                self._list_reminders,
            ),
            "complete_reminder": ToolDefinition(
                "complete_reminder",
                "Mark a matching reminder as completed. Use after the user says they finished it.",
                PermissionLevel.NORMAL,
                {
                    "type": "object",
                    "properties": {"reminder": {"type": "string", "description": "Task name or reminder ID"}},
                    "required": ["reminder"],
                },
                self._complete_reminder,
            ),
            "cancel_reminder": ToolDefinition(
                "cancel_reminder",
                "Cancel a matching upcoming reminder.",
                PermissionLevel.NORMAL,
                {
                    "type": "object",
                    "properties": {"reminder": {"type": "string", "description": "Task name or reminder ID"}},
                    "required": ["reminder"],
                },
                self._cancel_reminder,
            ),
            "reschedule_reminder": ToolDefinition(
                "reschedule_reminder",
                "Move a reminder to a new future time expressed as seconds from now.",
                PermissionLevel.NORMAL,
                {
                    "type": "object",
                    "properties": {
                        "reminder": {"type": "string", "description": "Task name or reminder ID"},
                        "delay_seconds": {"type": "number", "minimum": 1, "maximum": 31536000},
                    },
                    "required": ["reminder", "delay_seconds"],
                },
                self._reschedule_reminder,
            ),
            "remember": ToolDefinition(
                "remember",
                "Save one user-approved fact or preference for future conversations.",
                PermissionLevel.CONFIRMATION_REQUIRED,
                {
                    "type": "object",
                    "properties": {"fact": {"type": "string", "description": "Fact to remember"}},
                    "required": ["fact"],
                },
                self._remember,
            ),
            "forget_memory": ToolDefinition(
                "forget_memory",
                "Remove an exact fact from Lura's local memory.",
                PermissionLevel.CONFIRMATION_REQUIRED,
                {
                    "type": "object",
                    "properties": {"fact": {"type": "string", "description": "Exact fact to forget"}},
                    "required": ["fact"],
                },
                self._forget_memory,
            ),
            "list_memory": self._system_definition(
                "list_memory", "List facts the user explicitly asked Lura to remember.", self._list_memory
            ),
            "get_user_profile": self._system_definition(
                "get_user_profile", "Read the editable local user profile.", self._get_user_profile
            ),
            "get_identity": self._system_definition(
                "get_identity",
                "Read the authoritative assistant name and editable user identity.",
                self._get_identity,
            ),
            "get_active_model": self._system_definition(
                "get_active_model",
                "Read the exact Ollama model selected for the current conversation request. "
                "Do not infer this from GPU, CPU, memory, system information, or conversation history.",
                self._get_active_model,
            ),
            "update_user_profile": ToolDefinition(
                "update_user_profile",
                "Update editable local user profile fields. Only save fields explicitly supplied by the user.",
                PermissionLevel.CONFIRMATION_REQUIRED,
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "owner": {"type": "string"},
                        "preferred_address": {"type": "string"},
                        "assistant_role": {"type": "string"},
                        "application_install_preference": {"type": "string"},
                    },
                },
                self._update_user_profile,
            ),
            "get_system_profile": self._system_definition(
                "get_system_profile",
                "Collect current stable Linux, desktop, CPU, GPU, monitor, and Flatpak facts.",
                self._get_system_profile,
            ),
            "get_system_info": self._system_definition(
                "get_system_info",
                "Collect a complete current system profile and live status snapshot.",
                self._get_system_info,
            ),
            "get_gpu_info": self._system_definition(
                "get_gpu_info",
                "Read authoritative current GPU identity, utilization, temperature, and VRAM.",
                self._get_gpu_status,
            ),
            "get_cpu_info": self._system_definition(
                "get_cpu_info",
                "Read authoritative current CPU identity, core counts, and utilization.",
                self._get_cpu_status,
            ),
            "get_ram_info": self._system_definition(
                "get_ram_info",
                "Read authoritative current RAM totals, availability, and utilization.",
                self._get_memory_status,
            ),
            "get_disk_info": self._system_definition(
                "get_disk_info",
                "Read authoritative current root filesystem storage totals and utilization.",
                self._get_storage_status,
            ),
            "get_system_status": self._system_definition(
                "get_system_status",
                "Collect live CPU, memory, disk, GPU, process, window, network, and power status.",
                self._get_system_status,
            ),
            "get_gpu_status": self._system_definition(
                "get_gpu_status", "Get current NVIDIA GPU identity, utilization, temperature, and VRAM.", self._get_gpu_status
            ),
            "get_cpu_status": self._system_definition(
                "get_cpu_status", "Get current CPU identity and utilization.", self._get_cpu_status
            ),
            "get_memory_status": self._system_definition(
                "get_memory_status", "Get current RAM usage.", self._get_memory_status
            ),
            "get_storage_status": self._system_definition(
                "get_storage_status", "Get current root filesystem storage.", self._get_storage_status
            ),
            "get_processes": self._system_definition(
                "get_processes", "List processes using CPU or memory.", self._get_processes
            ),
            "get_gpu_processes": self._system_definition(
                "get_gpu_processes", "List processes currently using NVIDIA GPU resources.", self._get_gpu_processes
            ),
            "get_display_info": self._system_definition(
                "get_display_info", "Get current monitor resolution and refresh rate.", self._get_display_info
            ),
            "list_applications": self._system_definition(
                "list_applications", "List applications discovered from installed Flatpak and native desktop data.", self._list_applications
            ),
            "set_volume": ToolDefinition(
                "set_volume",
                "Set the default audio output volume to a percentage.",
                PermissionLevel.NORMAL,
                {
                    "type": "object",
                    "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 150}},
                    "required": ["level"],
                },
                self._set_volume,
            ),
            "shutdown": ToolDefinition(
                "shutdown",
                "Shut down the computer.",
                PermissionLevel.DANGEROUS,
                {"type": "object", "properties": {}},
                self._shutdown,
            ),
            "reboot": ToolDefinition(
                "reboot",
                "Restart the computer.",
                PermissionLevel.DANGEROUS,
                {"type": "object", "properties": {}},
                self._reboot,
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

    @staticmethod
    def _window_definition(
        name: str, description: str, handler: Callable[[dict], ToolCallResult]
    ) -> ToolDefinition:
        return ToolDefinition(
            name,
            description,
            PermissionLevel.SAFE,
            {
                "type": "object",
                "properties": {"window": {"type": "string", "description": "Window address, title, or application class"}},
                "required": ["window"],
            },
            handler,
        )

    @staticmethod
    def _file_definition(
        name: str, description: str, handler: Callable[[dict], ToolCallResult]
    ) -> ToolDefinition:
        return ToolDefinition(
            name,
            description,
            PermissionLevel.CONFIRMATION_REQUIRED,
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "UTF-8 text content", "maxLength": 200000},
                },
                "required": ["path", "content"],
            },
            handler,
        )

    def definitions_for_ollama(self) -> list[dict]:
        return [definition.ollama_schema() for definition in self._definitions.values()]

    def has_tool(self, name: str) -> bool:
        return name in self._definitions

    @staticmethod
    def request_requires_tools(request: str) -> bool:
        """Keep obvious reminder requests on the tool-enabled model path."""
        return bool(
            re.search(
                r"\b(?:remind(?:er)?|timer|scheduled\s+notification)s?\b",
                request.casefold(),
            )
        )

    def set_active_model(self, model: str | None) -> None:
        """Track the model that the next/current conversation request sends to Ollama."""
        self._active_model = (
            model.strip()
            if isinstance(model, str) and model.strip()
            else None
        )

    def direct_tool_call_for_request(self, request: str) -> tuple[str, dict] | None:
        """Map unambiguous local actions/facts to tools before the LLM responds.

        The router still runs for every request, but a model must not be the
        authority for local state or whether an explicit computer action is
        executed. Ambiguous requests continue through normal tool calling.
        """
        text = re.sub(r"\s+", " ", request.strip()).strip(" .!?")
        folded = text.casefold()
        if not text:
            return None

        def decision(
            tool_name: str | None,
            arguments: dict | None,
            reason: str,
        ) -> tuple[str, dict] | None:
            LOGGER.debug(
                "[ROUTER] user_request=%r classification=%s selected_tool=%s "
                "why_tool=%s",
                text[:600],
                "direct_tool" if tool_name else "no_tool",
                tool_name or "none",
                reason,
            )
            return (tool_name, arguments or {}) if tool_name else None

        model_question = (
            bool(
                re.search(
                    r"\b(?:what(?:'s| is)?|which|tell me)\b"
                    r".*\b(?:model|llm|language model)\b"
                    r".*\b(?:currently|current|active|running|using|use|selected|loaded)\b",
                    folded,
                )
            )
            or bool(
                re.search(
                    r"\b(?:currently|current|active|running|selected|loaded)\b"
                    r".*\b(?:model|llm|language model)\b",
                    folded,
                )
            )
            or bool(
                re.search(
                    r"\b(?:am i|are you)\s+using\b"
                    r".*\b(?:model|llm|language model)\b",
                    folded,
                )
            )
        )
        if model_question:
            return decision(
                "get_active_model",
                {},
                "explicit question about the active model",
            )

        identity_question = (
            "who am i" in folded
            or "who are you" in folded
            or bool(
                re.search(
                    r"\b(?:what(?:'s| is)|who is|do you know|remind me|tell me)\b"
                    r".*\b(?:my name|your name|who i am)\b",
                    folded,
                )
            )
            or bool(re.search(r"\b(?:what(?:'s| is))\s+(?:my|your)\s+name\b", folded))
        )
        if identity_question:
            return decision(
                "get_identity",
                {},
                "explicit question about the user or assistant identity",
            )

        if self.custom_app_commands and re.search(
            r"\b(?:open|launch|start|run|use|show|bring\s+up)\b",
            folded,
        ):
            for alias in self.custom_app_commands:
                if re.search(re.escape(alias.casefold()), folded):
                    return decision(
                        "open_app",
                        {"app": alias},
                        "explicit configured application launch request",
                    )

        url = re.search(r"https?://[^\s]+", text, re.IGNORECASE)
        if url and re.match(r"^(?:please\s+)?(?:open|launch|visit|go to)\b", folded):
            return decision(
                "open_website",
                {"url": url.group(0).rstrip(".,!?")},
                "explicit request to open a URL",
            )

        live_search_match = re.match(
            r"^(?:(?:please|can you|could you|would you|will you)\s+)?"
            r"(?:perform|do|run|make|conduct|carry out)?\s*"
            r"(?:a\s+)?"
            r"(?:live|web|online|internet)(?:\s+(?:web|internet))?\s+search\s+"
            r"(?:(?:the|on)\s+(?:web|internet)\s+)?"
            r"(?:for|about|on)?\s*(?P<query>.+)$",
            text,
            re.IGNORECASE,
        )
        if live_search_match:
            query = live_search_match.group("query").strip(" .!?")
            if query:
                search_tool = (
                    "search_news"
                    if re.search(
                        r"\b(?:news|headlines?|breaking)\b",
                        query,
                        re.IGNORECASE,
                    )
                    else "web_search"
                )
                return decision(
                    search_tool,
                    {"query": query},
                    "explicit live/web search request",
                )

        if re.search(r"\b(?:what are|show|list)\s+(?:my\s+)?reminders\b", folded):
            return decision(
                "list_reminders",
                {},
                "explicit request to list reminders",
            )
        completion_match = re.search(
            r"\b(?:mark|set)\s+(?P<reminder>.+?)\s+(?:as\s+)?(?:done|complete|completed)\b",
            text,
            re.IGNORECASE,
        )
        if completion_match:
            return decision(
                "complete_reminder",
                {"reminder": completion_match.group("reminder").strip()},
                "explicit request to complete a reminder",
            )
        cancellation_match = re.search(
            r"\b(?:cancel|delete|remove)\s+(?:my\s+)?(?P<reminder>.+?)\s+reminder\b",
            text,
            re.IGNORECASE,
        )
        if cancellation_match:
            return decision(
                "cancel_reminder",
                {"reminder": cancellation_match.group("reminder").strip()},
                "explicit request to cancel a reminder",
            )
        move_match = re.search(
            r"\b(?:move|reschedule| postpone)\s+(?:my\s+)?(?P<reminder>.+?)\s+reminder\b"
            r".*?\b(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>seconds?|minutes?|hours?|days?)\b",
            text,
            re.IGNORECASE,
        )
        if move_match:
            unit_seconds = {
                "second": 1,
                "seconds": 1,
                "minute": 60,
                "minutes": 60,
                "hour": 3600,
                "hours": 3600,
                "day": 86400,
                "days": 86400,
            }
            unit = move_match.group("unit").casefold()
            return decision(
                "reschedule_reminder",
                {
                    "reminder": move_match.group("reminder").strip(),
                    "delay_seconds": float(move_match.group("amount")) * unit_seconds[unit],
                },
                "explicit request to reschedule a reminder",
            )

        search_match = re.match(
            r"^(?:(?:please|can you|could you|would you|will you)\s+)?"
            r"(?:search|look\s+up|lookup|google|browse)\s+"
            r"(?:(?:the|on)\s+(?:web|internet)\s+)?(?:for\s+)?(?P<query>.+)$",
            text,
            re.IGNORECASE,
        )
        if search_match:
            query = search_match.group("query").strip(" .!?")
            if query:
                search_tool = (
                    "search_news"
                    if re.search(
                        r"\b(?:news|headlines?|breaking)\b",
                        query,
                        re.IGNORECASE,
                    )
                    else "web_search"
                )
                return decision(
                    search_tool,
                    {"query": query},
                    "explicit search or lookup request",
                )

        # Current external questions must reach a live-information tool even
        # when the user does not say "search". Specialized tools below still
        # own weather/currency/maps/etc. A time word alone is not enough:
        # conversational requests commonly contain "today" or "now". Require
        # an external-information cue as well so uncertainty falls through to
        # the normal model route instead of causing an unnecessary search.
        current_marker = re.search(
            r"\b(?:latest|today|now|recently|this week|breaking|current)\b",
            folded,
        )
        specialized_current = re.search(
            r"\b(?:weather|forecast|temperature|currency|exchange rate|"
            r"directions?|route|nearby|restaurant|travel|flight|game|gaming)\b",
            folded,
        )
        external_information_cue = re.search(
            r"\b(?:news|headlines?|breaking|release|releases|price|prices|"
            r"stock|stocks|market|markets|score|scores|result|results|"
            r"updates?|events?|race|races|won|happened|happening|"
            r"what(?:'s| is) new)\b",
            folded,
        )
        if current_marker and external_information_cue and not specialized_current:
            query = re.sub(
                r"^(?:(?:what(?:'s| is)|what are|tell me|show me|give me)\s+)",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip(" .!?")
            if query:
                search_tool = (
                    "search_news"
                    if re.search(r"\b(?:news|headlines?|breaking)\b", query, re.IGNORECASE)
                    else "web_search"
                )
                return decision(
                    search_tool,
                    {"query": query},
                    "current marker paired with an external-information cue",
                )

        app_match = re.match(
            r"^(?:(?:please|can you|could you|would you|will you)\s+)?"
            r"(open|launch|start|run|close|quit|exit|stop)\s+"
            r"(?:the\s+)?(.+)$",
            text,
            re.IGNORECASE,
        )
        if app_match:
            action = app_match.group(1).casefold()
            app = re.sub(r"\s+(?:for me|please)$", "", app_match.group(2).strip(), flags=re.IGNORECASE)
            app = re.sub(
                r"\s+on\s+(?:my|the)\s+computer$",
                "",
                app,
                flags=re.IGNORECASE,
            ).strip()
            is_generic_instruction = bool(
                re.match(
                    r"^(?:a|an|the)\s+(?:python\s+)?"
                    r"(?:program|command|script|application|app|file)\b",
                    app,
                    re.IGNORECASE,
                )
            )
            if (
                not is_generic_instruction
                and app.casefold() not in {"listening", "the app", "an app"}
                and app
            ):
                tool_name = "close_app" if action in {"close", "quit", "exit", "stop"} else "open_app"
                return decision(
                    tool_name,
                    {"app": app},
                    "explicit application action",
                )

        local_state_marker = re.search(
            r"\b(?:my|your|i have|do i have|am i using|current|usage|"
            r"do you have|what do you have|utilization|temperature|temp|"
            r"vram|cores?|threads?|how much|how many)\b",
            folded,
        )
        if local_state_marker and re.search(
            r"\b(?:gpu|graphics card|video card|vram|graphics memory)\b", folded
        ):
            return decision(
                "get_gpu_info",
                {},
                "explicit request for local GPU state",
            )
        if local_state_marker and re.search(
            r"\b(?:cpu|processor|processing unit|cpu cores?|threads?)\b", folded
        ):
            return decision(
                "get_cpu_info",
                {},
                "explicit request for local CPU state",
            )
        if local_state_marker and re.search(r"\b(?:ram|memory)\b", folded):
            if re.search(r"(?:using|uses|top|most|process)", folded):
                return decision(
                    "get_processes",
                    {},
                    "explicit request for processes using local memory",
                )
            return decision(
                "get_ram_info",
                {},
                "explicit request for local memory state",
            )
        if local_state_marker and re.search(
            r"\b(?:disk|storage|drive space|free space)\b", folded
        ):
            return decision(
                "get_disk_info",
                {},
                "explicit request for local storage state",
            )
        if re.search(r"\b(?:system info|system information|computer status|machine status|linux version)\b", folded):
            return decision(
                "get_system_info",
                {},
                "explicit request for local system information",
            )
        if re.search(r"\b(?:screenshot|screen capture|capture my screen)\b", folded):
            return decision(
                "take_screenshot",
                {},
                "explicit request for a desktop screenshot",
            )
        if re.search(r"\b(?:open windows?|windows? do i have)\b", folded):
            return decision(
                "list_windows",
                {},
                "explicit request to list desktop windows",
            )
        return decision(None, None, "no unambiguous direct tool intent")

    def permission_for(self, name: str, arguments: dict) -> PermissionLevel:
        definition = self._definitions.get(name)
        if definition is None:
            return PermissionLevel.DANGEROUS
        policy = self.tool_permissions.get(name, "default")
        if policy == "blocked":
            return PermissionLevel.BLOCKED
        if name == "exec":
            command = arguments.get("command", "")
            if isinstance(command, str) and _DANGEROUS_COMMANDS.search(command):
                base_permission = PermissionLevel.DANGEROUS
            elif isinstance(command, str) and _is_read_only_exec(command):
                base_permission = PermissionLevel.SAFE
            else:
                base_permission = definition.permission
        else:
            base_permission = definition.permission
        if policy == "ask" and base_permission != PermissionLevel.DANGEROUS:
            return PermissionLevel.CONFIRMATION_REQUIRED
        # Explicit permission can remove the confirmation step from safe,
        # normal, and confirmation-gated actions. Dangerous actions retain
        # their confirmation gate even when the user chooses Always allow.
        if policy == "always_allow" and base_permission == PermissionLevel.CONFIRMATION_REQUIRED:
            return PermissionLevel.NORMAL
        return base_permission

    def execute(self, name: str, arguments: dict, approved: bool = False) -> ToolCallResult:
        if not isinstance(arguments, dict):
            return ToolCallResult(False, "Tool arguments must be an object.")
        definition = self._definitions.get(name)
        if definition is None:
            return ToolCallResult(False, f"Unknown tool: {name}")
        if _tool_debug_enabled() and name in _GPU_TOOL_NAMES:
            LOGGER.info("[GPU TOOL] selected=%s arguments=%s", name, arguments)
        permission = self.permission_for(name, arguments)
        if permission == PermissionLevel.BLOCKED:
            return ToolCallResult(False, f"Permission blocked for {name}.")
        if permission not in {PermissionLevel.SAFE, PermissionLevel.NORMAL} and not approved:
            raise ToolConfirmationRequired(f"{name} requires user confirmation.")
        try:
            return definition.handler(arguments)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            return _failed(error)

    def _application(self, arguments: dict) -> ApplicationRecord:
        app = _string_argument(arguments, "app")
        return self.application_registry.resolve(app)

    def _custom_application(self, app: str) -> tuple[str, tuple[str, ...]] | None:
        candidates = [app]
        for prefix in ("my ", "the ", "a ", "an "):
            if app.casefold().startswith(prefix):
                candidates.append(app[len(prefix):].strip())
        configured = next(
            (
                (alias, command)
                for alias, command in self.custom_app_commands.items()
                if any(alias.casefold() == candidate.casefold() for candidate in candidates)
            ),
            None,
        )
        if configured is None:
            return None
        alias, command = configured
        if _LAUNCH_SHELL_OPERATORS.search(command) or "\x00" in command:
            raise ValueError(
                f"Custom launcher for {alias} contains shell operators. "
                "Use a direct executable and arguments only."
            )
        try:
            parts = tuple(shlex.split(command))
        except ValueError as error:
            raise ValueError(f"Custom launcher for {alias} could not be parsed: {error}") from error
        if not parts:
            raise ValueError(f"Custom launcher for {alias} is empty.")
        executable = Path(parts[0]).name.casefold()
        if executable in _UNSAFE_LAUNCH_EXECUTABLES:
            raise ValueError(
                f"Custom launcher for {alias} must start with the application executable, "
                f"not {executable}."
            )
        if not shutil.which(parts[0]) and not (Path(parts[0]).is_file() and os.access(parts[0], os.X_OK)):
            raise ValueError(
                f"Custom launcher for {alias} cannot start because its executable "
                f"is not available on PATH: {parts[0]}. Arguments are allowed, "
                "but the first item must be an installed executable."
            )
        return alias, parts

    def _open_app(self, arguments: dict) -> ToolCallResult:
        app = _string_argument(arguments, "app")
        custom_application = self._custom_application(app)
        if custom_application is not None:
            application_name, launch_command = custom_application
            application_id = application_name
            kind = "custom"
        else:
            application = self.application_registry.resolve(app)
            application_name = application.name
            application_id = application.app_id
            launch_command = application.launch_command
            kind = application.kind
        launch_environment = None
        if kind == "custom":
            # Browser command-line clients still use the existing desktop
            # session through DISPLAY and D-Bus. Starting a new process session
            # prevents Lura's terminal/process group from owning the app.
            launch_environment = os.environ.copy()
            launch_environment.pop("MOZ_NO_REMOTE", None)
        subprocess.Popen(
            launch_command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=launch_environment,
        )
        return ToolCallResult(
            True,
            json.dumps(
                {
                    "action": "open_app",
                    "application": application_name,
                    "application_id": application_id,
                    "kind": kind,
                    "command": list(launch_command),
                    "status": "opened",
                }
            ),
        )

    @staticmethod
    def _open_website(arguments: dict) -> ToolCallResult:
        url = _string_argument(arguments, "url")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Website URL must start with http:// or https://.")
        executable = shutil.which("xdg-open")
        if not executable:
            raise ValueError("xdg-open is not available.")
        subprocess.Popen(
            [executable, url],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return ToolCallResult(True, f"Opened {url}.")

    @staticmethod
    def _web_search(arguments: dict) -> ToolCallResult:
        query = _string_argument(arguments, "query")
        max_results = _integer_argument(arguments, "max_results", 1, 8) if "max_results" in arguments else 5
        started_at = time.monotonic()
        LOGGER.info(
            "[WEB_SEARCH] start query=%r max_results=%d",
            query,
            max_results,
        )

        def failure(error_code: str, message: str) -> ToolCallResult:
            elapsed_ms = round((time.monotonic() - started_at) * 1000)
            LOGGER.info(
                "[WEB_SEARCH] complete query=%r success=false error_code=%s "
                "result_count=0 elapsed_ms=%d",
                query,
                error_code,
                elapsed_ms,
            )
            return ToolCallResult(
                False,
                json.dumps(
                    {
                        "success": False,
                        "query": query,
                        "results": [],
                        "error_code": error_code,
                        "message": message,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )

        request = Request(
            "https://html.duckduckgo.com/html/?" + urlencode({"q": query}),
            headers={
                "Accept": "text/html",
                "User-Agent": "Lura/1.0 (personal desktop assistant)",
            },
        )
        try:
            with urlopen(request, timeout=_WEB_SEARCH_TIMEOUT) as response:
                status = response.getcode()
                raw_html = response.read(_WEB_SEARCH_MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            error_code = "PERMISSION_DENIED" if error.code in {401, 403} else "NETWORK_ERROR"
            return failure(error_code, f"DuckDuckGo returned HTTP {error.code}.")
        except (TimeoutError, socket.timeout) as error:
            return failure("TIMEOUT", f"DuckDuckGo search timed out: {error}")
        except (URLError, ConnectionError, OSError) as error:
            return failure("NETWORK_ERROR", f"DuckDuckGo is unavailable: {error}")
        except Exception as error:
            return failure("WEB_SEARCH_UNAVAILABLE", f"DuckDuckGo search failed: {error}")

        if not isinstance(status, int) or status < 200 or status >= 300:
            return failure("WEB_SEARCH_UNAVAILABLE", f"DuckDuckGo returned HTTP {status}.")
        if len(raw_html) > _WEB_SEARCH_MAX_RESPONSE_BYTES:
            return failure("PARSER_ERROR", "DuckDuckGo returned an oversized response.")
        try:
            html = raw_html.decode("utf-8")
        except UnicodeDecodeError as error:
            return failure("PARSER_ERROR", f"DuckDuckGo returned invalid text: {error}")

        # DuckDuckGo currently responds with HTTP 202 and an anomaly challenge
        # when automated requests are rate-limited. Treat it as unavailable;
        # never reinterpret the challenge page as an empty successful search.
        lowered_html = html.casefold()
        if "challenge-form" in lowered_html or "anomaly-modal" in lowered_html:
            return failure(
                "WEB_SEARCH_UNAVAILABLE",
                "DuckDuckGo requires a human verification challenge.",
            )
        if not html.strip() or "<" not in html:
            return failure("PARSER_ERROR", "DuckDuckGo returned malformed HTML.")

        results = _DuckDuckGoParser()
        try:
            results.feed(html)
            results.finish_current()
        except Exception as error:
            return failure("PARSER_ERROR", f"DuckDuckGo returned invalid HTML: {error}")

        usable_results: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for item in results.items:
            title = " ".join(item.get("title", "").split())
            raw_url = item.get("url", "").strip()
            if raw_url.startswith("//"):
                raw_url = "https:" + raw_url
            parsed_url = urlparse(raw_url)
            if parsed_url.hostname == "duckduckgo.com" and parsed_url.path.startswith("/l/"):
                target = parse_qs(parsed_url.query).get("uddg", [""])[0]
                raw_url = unquote(target).strip() or raw_url
                parsed_url = urlparse(raw_url)
            if (
                not title
                or parsed_url.scheme not in {"http", "https"}
                or not parsed_url.netloc
            ):
                continue
            normalized_url = raw_url.split("#", 1)[0].rstrip("/")
            dedupe_key = normalized_url.casefold()
            if dedupe_key in seen_urls:
                continue
            seen_urls.add(dedupe_key)
            usable_results.append(
                {
                    "title": title,
                    "url": normalized_url,
                    "snippet": " ".join(item.get("snippet", "").split()),
                    "source": parsed_url.hostname or parsed_url.netloc,
                }
            )
            if len(usable_results) >= max_results:
                break

        if not usable_results:
            error_code = "INVALID_RESULT" if results.items else "NO_RESULTS"
            message = (
                "DuckDuckGo returned no valid result records."
                if error_code == "INVALID_RESULT"
                else f"No web results found for '{query}'."
            )
            return failure(error_code, message)

        elapsed_ms = round((time.monotonic() - started_at) * 1000)
        LOGGER.info(
            "[WEB_SEARCH] complete query=%r success=true result_count=%d elapsed_ms=%d",
            query,
            len(usable_results),
            elapsed_ms,
        )
        return ToolCallResult(
            True,
            json.dumps(
                {
                    "success": True,
                    "query": query,
                    "results": usable_results,
                    "source": "DuckDuckGo HTML",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    @staticmethod
    def _get_weather(arguments: dict) -> ToolCallResult:
        units = _optional_string_argument(arguments, "units") or "metric"
        return _information_result(
            information_weather(
                _optional_string_argument(arguments, "location"),
                _optional_string_argument(arguments, "date"),
                units,
            )
        )

    @staticmethod
    def _search_news(arguments: dict) -> ToolCallResult:
        max_results = (
            _integer_argument(arguments, "max_results", 1, 8)
            if "max_results" in arguments
            else 5
        )
        return _information_result(
            information_news(_optional_string_argument(arguments, "query"), max_results)
        )

    @staticmethod
    def _knowledge_search(arguments: dict) -> ToolCallResult:
        query = _string_argument(arguments, "query")
        max_results = (
            _integer_argument(arguments, "max_results", 1, 5)
            if "max_results" in arguments
            else 3
        )
        return _information_result(information_knowledge_search(query, max_results))

    @staticmethod
    def _convert_currency(arguments: dict) -> ToolCallResult:
        amount = arguments.get("amount")
        return _information_result(
            information_currency(
                amount,
                _string_argument(arguments, "from_currency"),
                _string_argument(arguments, "to_currency"),
            )
        )

    @staticmethod
    def _find_places(arguments: dict) -> ToolCallResult:
        max_results = (
            _integer_argument(arguments, "max_results", 1, 8)
            if "max_results" in arguments
            else 5
        )
        return _information_result(
            information_find_places(
                _string_argument(arguments, "query"),
                _optional_string_argument(arguments, "near"),
                max_results,
            )
        )

    @staticmethod
    def _get_directions(arguments: dict) -> ToolCallResult:
        return _information_result(
            information_directions(
                _string_argument(arguments, "origin"),
                _string_argument(arguments, "destination"),
                _optional_string_argument(arguments, "mode") or "driving",
            )
        )

    @staticmethod
    def _travel_search(arguments: dict) -> ToolCallResult:
        max_results = (
            _integer_argument(arguments, "max_results", 1, 8)
            if "max_results" in arguments
            else 5
        )
        return _information_result(
            information_travel_search(
                _string_argument(arguments, "destination"),
                _optional_string_argument(arguments, "topic"),
                max_results,
            )
        )

    @staticmethod
    def _game_search(arguments: dict) -> ToolCallResult:
        max_results = (
            _integer_argument(arguments, "max_results", 1, 8)
            if "max_results" in arguments
            else 5
        )
        return _information_result(
            information_game_search(_string_argument(arguments, "query"), max_results)
        )

    def _create_reminder(self, arguments: dict) -> ToolCallResult:
        message = _string_argument(arguments, "message")
        delay_seconds = arguments.get("delay_seconds")
        reminder = self.reminder_service.schedule(message, delay_seconds)
        timing = format_reminder_timing(
            reminder.due_at,
            float(delay_seconds),
            now=self.reminder_service._clock(),
        )
        return ToolCallResult(
            True,
            f"Reminder created successfully. It will fire {timing}: {reminder.message}",
        )

    def _list_reminders(self, arguments: dict) -> ToolCallResult:
        del arguments
        reminders = self.reminder_service.store.list()
        if not reminders:
            return ToolCallResult(True, "No reminders are scheduled.")
        active = [reminder for reminder in reminders if reminder.status in {"upcoming", "triggered"}]
        history = [reminder for reminder in reminders if reminder.status in {"completed", "missed", "cancelled"}]
        lines = [
            f"- {reminder.message} — {format_due_time(reminder.due_at)} "
            f"({reminder.priority}, {reminder.status})"
            for reminder in sorted(active, key=lambda item: item.due_at)
        ]
        if history:
            lines.append("History:")
            lines.extend(
                f"- {reminder.message} — {reminder.status} "
                f"({format_due_time(reminder.completed_at or reminder.due_at)})"
                for reminder in sorted(
                    history,
                    key=lambda item: item.completed_at or item.due_at,
                    reverse=True,
                )[:10]
            )
        return ToolCallResult(True, "\n".join(lines) or "No upcoming reminders.")

    def _matching_reminder(self, value: str):
        reminders = self.reminder_service.store.list()
        needle = value.strip().casefold()
        return next(
            (
                reminder
                for reminder in reminders
                if reminder.reminder_id == value.strip()
                or needle in reminder.message.casefold()
            ),
            None,
        )

    def _complete_reminder(self, arguments: dict) -> ToolCallResult:
        value = _string_argument(arguments, "reminder")
        reminder = self._matching_reminder(value)
        if reminder is None:
            return ToolCallResult(False, f"Reminder not found: {value}")
        updated = self.reminder_service.complete(reminder.reminder_id)
        return ToolCallResult(True, f"Completed reminder: {updated.message}")

    def _cancel_reminder(self, arguments: dict) -> ToolCallResult:
        value = _string_argument(arguments, "reminder")
        reminder = self._matching_reminder(value)
        if reminder is None:
            return ToolCallResult(False, f"Reminder not found: {value}")
        updated = self.reminder_service.cancel(reminder.reminder_id)
        return ToolCallResult(True, f"Cancelled reminder: {updated.message}")

    def _reschedule_reminder(self, arguments: dict) -> ToolCallResult:
        value = _string_argument(arguments, "reminder")
        delay_seconds = arguments.get("delay_seconds")
        reminder = self._matching_reminder(value)
        if reminder is None:
            return ToolCallResult(False, f"Reminder not found: {value}")
        updated = self.reminder_service.reschedule(reminder.reminder_id, delay_seconds)
        return ToolCallResult(True, f"Rescheduled {updated.message} for {format_due_time(updated.due_at)}.")

    def _remember(self, arguments: dict) -> ToolCallResult:
        fact = _string_argument(arguments, "fact")
        self.memory_store.add(fact)
        return ToolCallResult(True, f"Remembered: {fact}")

    def _forget_memory(self, arguments: dict) -> ToolCallResult:
        fact = _string_argument(arguments, "fact")
        if not self.memory_store.forget(fact):
            return ToolCallResult(False, f"That memory was not found: {fact}")
        return ToolCallResult(True, f"Forgotten: {fact}")

    def _list_memory(self, arguments: dict) -> ToolCallResult:
        del arguments
        context = self.memory_store.context()
        return ToolCallResult(True, context or "No saved memories.")

    def _get_user_profile(self, arguments: dict) -> ToolCallResult:
        del arguments
        return ToolCallResult(True, json.dumps(self.profile_store.profile(), ensure_ascii=False))

    def _get_identity(self, arguments: dict) -> ToolCallResult:
        del arguments
        return ToolCallResult(
            True,
            json.dumps(
                self.profile_store.identity(self.assistant_name),
                ensure_ascii=False,
            ),
        )

    def _get_active_model(self, arguments: dict) -> ToolCallResult:
        del arguments
        if not self._active_model:
            return ToolCallResult(
                False,
                json.dumps(
                    {
                        "success": False,
                        "data": None,
                        "error": "The active Ollama model is unavailable.",
                    }
                ),
            )
        return ToolCallResult(
            True,
            json.dumps(
                {
                    "success": True,
                    "data": {
                        "active_model": self._active_model,
                        "model_type": "main",
                    },
                    "error": None,
                }
            ),
        )

    def _update_user_profile(self, arguments: dict) -> ToolCallResult:
        profile = self.profile_store.update(arguments)
        return ToolCallResult(True, json.dumps(profile, ensure_ascii=False))

    def _get_system_profile(self, arguments: dict) -> ToolCallResult:
        del arguments
        return ToolCallResult(True, json.dumps(collect_system_profile(), ensure_ascii=False))

    def _get_system_info(self, arguments: dict) -> ToolCallResult:
        del arguments
        profile = collect_system_profile()
        status = self._status_payload()
        return ToolCallResult(
            True,
            json.dumps({"system_profile": profile, "live_system_state": status}, ensure_ascii=False),
        )

    def _get_system_status(self, arguments: dict) -> ToolCallResult:
        del arguments
        return ToolCallResult(True, json.dumps(self._status_payload(), ensure_ascii=False))

    def _status_payload(self) -> dict:
        def payload(result: ToolCallResult) -> object:
            try:
                return json.loads(result.content)
            except (TypeError, json.JSONDecodeError):
                return {"available": result.success, "summary": result.content}

        return {
            "gpu": payload(self._get_gpu_status({})),
            "cpu": payload(self._get_cpu_status({})),
            "memory": payload(self._get_memory_status({})),
            "storage": payload(self._get_storage_status({})),
            "processes": payload(self._get_processes({})),
            "gpu_processes": payload(self._get_gpu_processes({})),
            "windows": payload(self._list_windows({})),
            "display": payload(self._get_display_info({})),
            "battery": payload(self._get_battery({})),
            "volume": payload(self._get_volume({})),
            "uptime": _read_uptime(),
            "network": {"online_route": Path("/proc/net/route").is_file()},
        }

    @staticmethod
    def _get_gpu_status(arguments: dict) -> ToolCallResult:
        del arguments
        if _tool_debug_enabled():
            LOGGER.info("[GPU TOOL] command=%s", shlex.join(_GPU_QUERY))
        try:
            result = subprocess.run(
                _GPU_QUERY,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            if _tool_debug_enabled():
                LOGGER.info(
                    "[GPU TOOL] execution_failed error_type=%s raw_stdout_chars=0 "
                    "raw_stderr_chars=0",
                    type(error).__name__,
                )
            return _gpu_failure(f"NVIDIA GPU query failed: {error}")
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if _tool_debug_enabled():
            LOGGER.info(
                "[GPU TOOL] return_code=%d raw_stdout_chars=%d raw_stdout_lines=%d "
                "raw_stderr_chars=%d",
                result.returncode,
                len(stdout),
                len(stdout.splitlines()),
                len(stderr),
            )
        if result.returncode:
            return _gpu_failure(
                "NVIDIA GPU is unavailable.",
                [f"nvidia-smi exited with status {result.returncode}"],
            )
        gpus, parse_errors = _parse_gpu_rows(stdout)
        if parse_errors:
            if _tool_debug_enabled():
                LOGGER.info(
                    "[GPU TOOL] parsed_type=dict gpu_rows=%d parse_errors=%d "
                    "serialized_result_chars=0",
                    len(gpus),
                    len(parse_errors),
                )
            return _gpu_failure("NVIDIA GPU data was malformed.", parse_errors)
        if not gpus:
            if _tool_debug_enabled():
                LOGGER.info(
                    "[GPU TOOL] parsed_type=dict gpu_rows=0 parse_errors=0 "
                    "serialized_result_chars=0"
                )
            return _gpu_failure("NVIDIA GPU data is unavailable.")
        first_gpu = gpus[0]
        payload = {
            "available": True,
            "vendor": "NVIDIA",
            "model": first_gpu["name"],
            "vram_total_mb": first_gpu["vram_total_mb"],
            "vram_used_mb": first_gpu["vram_used_mb"],
            "utilization_percent": first_gpu["utilization_percent"],
            "temperature_c": first_gpu["temperature_c"],
            "gpus": gpus,
        }
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if _tool_debug_enabled():
            LOGGER.info(
                "[GPU TOOL] parsed_type=dict gpu_rows=%d parsed_fields=%s "
                "first_model_chars=%d serialized_result_chars=%d",
                len(gpus),
                ",".join(payload),
                len(str(first_gpu["name"])),
                len(serialized),
            )
        return ToolCallResult(
            True,
            serialized,
        )

    @staticmethod
    def _get_cpu_status(arguments: dict) -> ToolCallResult:
        del arguments
        try:
            first = _read_cpu_times()
            time.sleep(0.1)
            second = _read_cpu_times()
            total_delta = second[0] - first[0]
            idle_delta = second[3] - first[3]
            usage = (
                max(0.0, min(100.0, 100 * (1 - idle_delta / total_delta)))
                if total_delta > 0
                else None
            )
        except (OSError, ValueError, IndexError):
            usage = None
        identity = _read_cpu_identity()
        model = identity["model"]
        return ToolCallResult(
            usage is not None,
            json.dumps(
                {
                    "vendor": identity["vendor"],
                    "model": model,
                    "cores": identity["cores"],
                    "threads": identity["threads"],
                    "utilization_percent": round(usage, 1) if usage is not None else None,
                    "available": usage is not None,
                }
            ),
        )

    @staticmethod
    def _get_memory_status(arguments: dict) -> ToolCallResult:
        del arguments
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition(":")
                if separator:
                    values[key] = int(value.strip().split()[0])
        except (OSError, ValueError):
            return ToolCallResult(False, json.dumps({"available": False}))
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        if not total or available is None:
            return ToolCallResult(False, json.dumps({"available": False}))
        used = total - available
        return ToolCallResult(
            True,
            json.dumps(
                {
                    "available": True,
                    "used_mb": round(used / 1024, 1),
                    "available_mb": round(available / 1024, 1),
                    "total_mb": round(total / 1024, 1),
                    "utilization_percent": round(used / total * 100, 1),
                }
            ),
        )

    @staticmethod
    def _get_storage_status(arguments: dict) -> ToolCallResult:
        del arguments
        usage = shutil.disk_usage("/")
        used = usage.total - usage.free
        return ToolCallResult(
            True,
            json.dumps(
                {
                    "path": "/",
                    "used_gb": round(used / 2**30, 1),
                    "free_gb": round(usage.free / 2**30, 1),
                    "total_gb": round(usage.total / 2**30, 1),
                    "utilization_percent": round(used / usage.total * 100, 1),
                }
            ),
        )

    @staticmethod
    def _get_processes(arguments: dict) -> ToolCallResult:
        del arguments
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,comm=,%cpu=,%mem=", "--sort=-%cpu"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return ToolCallResult(False, json.dumps({"available": False, "error": str(error)}))
        processes: list[dict[str, object]] = []
        for line in result.stdout.splitlines()[:30]:
            fields = line.split()
            if len(fields) < 4:
                continue
            try:
                processes.append(
                    {
                        "pid": int(fields[0]),
                        "command": fields[1],
                        "cpu_percent": float(fields[2]),
                        "memory_percent": float(fields[3]),
                    }
                )
            except ValueError:
                continue
        return ToolCallResult(
            result.returncode == 0,
            json.dumps({"available": result.returncode == 0, "processes": processes}),
        )

    @staticmethod
    def _get_gpu_processes(arguments: dict) -> ToolCallResult:
        del arguments
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ToolCallResult(False, json.dumps({"available": False, "processes": []}))
        processes: list[dict[str, object]] = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 3:
                continue
            try:
                processes.append(
                    {"pid": int(fields[0]), "process": fields[1], "vram_used_mb": float(fields[2])}
                )
            except ValueError:
                continue
        return ToolCallResult(
            result.returncode == 0,
            json.dumps({"available": result.returncode == 0, "processes": processes}),
        )

    @staticmethod
    def _get_display_info(arguments: dict) -> ToolCallResult:
        del arguments
        if shutil.which("hyprctl"):
            try:
                result = subprocess.run(
                    ["hyprctl", "monitors", "-j"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                monitors = json.loads(result.stdout) if result.returncode == 0 else []
                if isinstance(monitors, list):
                    return ToolCallResult(
                        True,
                        json.dumps(
                            {
                                "available": True,
                                "monitors": [
                                    {
                                        "name": item.get("name"),
                                        "description": item.get("description"),
                                        "width": item.get("width"),
                                        "height": item.get("height"),
                                        "refresh_hz": item.get("refreshRate"),
                                        "x": item.get("x"),
                                        "y": item.get("y"),
                                    }
                                    for item in monitors
                                    if isinstance(item, dict)
                                ],
                            }
                        ),
                    )
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
                pass
        if shutil.which("xrandr"):
            try:
                result = subprocess.run(
                    ["xrandr", "--current"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                outputs = [
                    line.strip()
                    for line in result.stdout.splitlines()
                    if " connected" in line
                ]
                return ToolCallResult(
                    bool(outputs),
                    json.dumps({"available": bool(outputs), "monitors": outputs}),
                )
            except (OSError, subprocess.SubprocessError):
                pass
        return ToolCallResult(False, json.dumps({"available": False, "monitors": []}))

    def _list_applications(self, arguments: dict) -> ToolCallResult:
        del arguments
        records = self.application_registry.list()
        return ToolCallResult(
            True,
            json.dumps(
                {
                    "count": len(records),
                    "applications": [
                        {
                            "id": record.app_id,
                            "name": record.name,
                            "kind": record.kind,
                            "launch_command": list(record.launch_command),
                        }
                        for record in records[:200]
                    ],
                },
                ensure_ascii=False,
            ),
        )

    @staticmethod
    def _set_volume(arguments: dict) -> ToolCallResult:
        level = _integer_argument(arguments, "level", 0, 150)
        if shutil.which("wpctl"):
            result = subprocess.run(
                ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{level}%"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        elif shutil.which("pactl"):
            result = subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        else:
            return ToolCallResult(False, "No supported volume control utility is available.")
        return ToolCallResult(
            result.returncode == 0,
            json.dumps(
                {
                    "level_percent": level,
                    "status": "updated" if result.returncode == 0 else "failed",
                    "error": result.stderr.strip() if result.returncode else "",
                }
            ),
        )

    @staticmethod
    def _shutdown(arguments: dict) -> ToolCallResult:
        del arguments
        return _power_action("poweroff")

    @staticmethod
    def _reboot(arguments: dict) -> ToolCallResult:
        del arguments
        return _power_action("reboot")

    def _close_app(self, arguments: dict) -> ToolCallResult:
        application = self._application(arguments)
        if application.kind == "flatpak":
            if not shutil.which("flatpak"):
                return ToolCallResult(False, "Flatpak is unavailable.")
            result = subprocess.run(
                ["flatpak", "kill", application.app_id],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode:
                return ToolCallResult(False, result.stderr.strip() or f"{application.name} is not running.")
            return ToolCallResult(
                True,
                json.dumps({"action": "close_app", "application": application.name, "kind": "flatpak", "status": "closed"}),
            )
        executable = Path(application.launch_command[0]).name
        result = subprocess.run(
            ["pkill", "-x", executable],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 1:
            return ToolCallResult(False, f"{application.name} is not running.")
        if result.returncode != 0:
            return ToolCallResult(False, result.stderr.strip() or f"Could not close {application.name}.")
        return ToolCallResult(
            True,
            json.dumps({"action": "close_app", "application": application.name, "kind": "native", "status": "closed"}),
        )

    def _restart_app(self, arguments: dict) -> ToolCallResult:
        closed = self._close_app(arguments)
        if not closed.success:
            return closed
        time.sleep(0.4)
        return self._open_app(arguments)

    @staticmethod
    def _exec(arguments: dict) -> ToolCallResult:
        command = _string_argument(arguments, "command")
        try:
            parts = shlex.split(command)
        except ValueError as error:
            return ToolCallResult(False, f"Command could not be parsed: {error}")
        if not parts or parts[0] not in {
            "df", "free", "lscpu", "ls", "nvidia-smi", "ps", "uname", "uptime",
            "flatpak", "cat", "printf",
        }:
            return ToolCallResult(
                False,
                "Arbitrary shell commands are disabled. Use Lura's controlled tools instead.",
            )
        if any(part in {";", "&&", "||", "|", ">", ">>", "<"} for part in parts):
            return ToolCallResult(False, "Shell operators are not allowed.")
        completed = subprocess.run(
            parts,
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
    def _hyprland_clients() -> tuple[list[dict] | None, str | None]:
        if not shutil.which("hyprctl"):
            return None, "Hyprland is unavailable: hyprctl was not found."
        try:
            result = subprocess.run(
                ["hyprctl", "clients", "-j"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return None, f"Could not query Hyprland: {error}"
        if result.returncode:
            return None, result.stderr.strip() or "Hyprland did not return its window list."
        try:
            clients = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None, "Hyprland returned invalid window data."
        if not isinstance(clients, list) or not all(isinstance(client, dict) for client in clients):
            return None, "Hyprland returned an invalid window list."
        return clients, None

    def _find_window(self, arguments: dict) -> tuple[dict | None, str | None]:
        query = _string_argument(arguments, "window")
        clients, error = self._hyprland_clients()
        if clients is None:
            return None, error
        normalized = query.casefold()
        matches = [
            client
            for client in clients
            if normalized
            in " ".join(
                str(client.get(field, ""))
                for field in ("address", "title", "class", "initialTitle", "initialClass")
            ).casefold()
        ]
        if not matches:
            return None, f"No Hyprland window matched '{query}'."
        if len(matches) > 1:
            choices = ", ".join(
                str(client.get("title") or client.get("class") or client.get("address"))
                for client in matches[:5]
            )
            return None, f"Multiple windows matched '{query}': {choices}. Use a unique address or title."
        return matches[0], None

    @staticmethod
    def _hyprland_dispatch(*arguments: str) -> ToolCallResult:
        try:
            result = subprocess.run(
                ["hyprctl", "dispatch", *arguments],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return ToolCallResult(False, f"Hyprland command failed: {error}")
        if result.returncode:
            return ToolCallResult(False, result.stderr.strip() or result.stdout.strip() or "Hyprland rejected the command.")
        return ToolCallResult(True, result.stdout.strip() or "Hyprland command completed.")

    @staticmethod
    def _list_windows(arguments: dict) -> ToolCallResult:
        del arguments
        clients, error = ToolManager._hyprland_clients()
        if clients is None:
            return ToolCallResult(False, error or "Hyprland is unavailable.")
        if not clients:
            return ToolCallResult(True, "No open windows.")
        lines = []
        for client in clients:
            address = str(client.get("address", "unknown"))
            app = str(client.get("class") or client.get("initialClass") or "unknown")
            title = str(client.get("title") or "(untitled)")
            workspace = client.get("workspace", {})
            workspace_name = workspace.get("name", "?") if isinstance(workspace, dict) else "?"
            lines.append(f"{address} | {app} | {title} | workspace {workspace_name}")
        return ToolCallResult(True, "\n".join(lines))

    def _focus_window(self, arguments: dict) -> ToolCallResult:
        window, error = self._find_window(arguments)
        if window is None:
            return ToolCallResult(False, error or "Window not found.")
        address = str(window.get("address", ""))
        result = self._hyprland_dispatch("focuswindow", f"address:{address}")
        return ToolCallResult(result.success, f"{window.get('title') or window.get('class')}: {result.content}")

    def _move_window(self, arguments: dict) -> ToolCallResult:
        window, error = self._find_window(arguments)
        if window is None:
            return ToolCallResult(False, error or "Window not found.")
        workspace = _string_argument(arguments, "workspace")
        if not _WORKSPACE_NAME.fullmatch(workspace):
            return ToolCallResult(False, "Workspace must be a name or number without shell characters.")
        address = self._window_address(window)
        if address is None:
            return ToolCallResult(False, "Hyprland returned a window without a valid address.")
        # Hyprland expects the workspace and window selector in one dispatcher
        # parameter: movetoworkspace WORKSPACE,address:ADDRESS.
        result = self._hyprland_dispatch("movetoworkspace", f"{workspace},address:{address}")
        return ToolCallResult(result.success, f"{window.get('title') or window.get('class')}: {result.content}")

    def _resize_window(self, arguments: dict) -> ToolCallResult:
        window, error = self._find_window(arguments)
        if window is None:
            return ToolCallResult(False, error or "Window not found.")
        width = _integer_argument(arguments, "width", 1, 10000)
        height = _integer_argument(arguments, "height", 1, 10000)
        address = self._window_address(window)
        if address is None:
            return ToolCallResult(False, "Hyprland returned a window without a valid address.")
        focus = self._hyprland_dispatch("focuswindow", f"address:{address}")
        if not focus.success:
            return focus
        result = self._hyprland_dispatch("resizeactive", "exact", str(width), str(height))
        return ToolCallResult(result.success, f"{window.get('title') or window.get('class')}: {result.content}")

    def _close_window(self, arguments: dict) -> ToolCallResult:
        window, error = self._find_window(arguments)
        if window is None:
            return ToolCallResult(False, error or "Window not found.")
        address = self._window_address(window)
        if address is None:
            return ToolCallResult(False, "Hyprland returned a window without a valid address.")
        result = self._hyprland_dispatch("closewindow", f"address:{address}")
        return ToolCallResult(result.success, f"{window.get('title') or window.get('class')}: {result.content}")

    @staticmethod
    def _take_screenshot(arguments: dict) -> ToolCallResult:
        del arguments
        screenshot_dir = Path.home() / ".cache" / "local-ai-assistant" / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = screenshot_dir / f"screenshot-{timestamp}.png"
        commands = (
            (["grim", str(destination)], "grim"),
            (["hyprshot", "-m", "output", "-f", str(destination)], "hyprshot"),
            (["gnome-screenshot", "-f", str(destination)], "gnome-screenshot"),
            (["spectacle", "-b", "-n", "-o", str(destination)], "spectacle"),
            (["scrot", str(destination)], "scrot"),
            (["import", "-window", "root", str(destination)], "ImageMagick"),
        )
        attempted: list[str] = []
        for command, label in commands:
            if not shutil.which(command[0]):
                continue
            attempted.append(label)
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
            except (OSError, subprocess.SubprocessError) as error:
                attempted[-1] = f"{label} ({error})"
                continue
            if result.returncode == 0 and destination.is_file() and destination.stat().st_size:
                ToolManager._prune_screenshots(screenshot_dir, destination)
                return ToolCallResult(
                    True,
                    f"Screenshot captured and attached to this chat: {destination}.",
                    (str(destination),),
                )
            detail = result.stderr.strip() or result.stdout.strip()
            if detail:
                attempted[-1] = f"{label} ({detail[:160]})"
        if attempted:
            return ToolCallResult(
                False,
                "Screenshot capture failed. Tried: " + ", ".join(attempted) + ".",
            )
        return ToolCallResult(
            False,
            "No supported screenshot utility is available. Install grim or hyprshot for Wayland.",
        )

    @staticmethod
    def _prune_screenshots(directory: Path, newest: Path) -> None:
        try:
            screenshots = sorted(
                (
                    path
                    for path in directory.glob("screenshot-*.png")
                    if path.is_file() and path != newest
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for old_path in screenshots[_MAX_SCREENSHOTS - 1 :]:
                try:
                    old_path.unlink()
                except OSError:
                    continue
        except OSError:
            # Screenshot retention is housekeeping, not a reason to fail a
            # screenshot that was already captured successfully.
            return

    @staticmethod
    def _search_files(arguments: dict) -> ToolCallResult:
        query = _string_argument(arguments, "query").casefold()
        directory_value = arguments.get("directory", str(Path.home()))
        if not isinstance(directory_value, str) or not directory_value.strip():
            return ToolCallResult(False, "Tool argument 'directory' must be a path.")
        directory = Path(directory_value).expanduser()
        if not directory.is_dir():
            return ToolCallResult(False, f"Directory not found: {directory}")
        results: list[str] = []
        visited = 0
        try:
            for root, directories, files in os.walk(directory, followlinks=False):
                visited += 1
                directories[:] = [name for name in directories if name not in {".git", ".cache", "node_modules"}]
                for name in files:
                    if query in name.casefold():
                        results.append(str(Path(root) / name))
                        if len(results) >= _MAX_SEARCH_RESULTS:
                            return ToolCallResult(
                                True,
                                "\n".join(results) + f"\n(Showing first {_MAX_SEARCH_RESULTS} matches.)",
                            )
                if visited >= _MAX_SEARCH_DIRECTORIES:
                    return ToolCallResult(
                        bool(results),
                        "\n".join(results)
                        + f"\n(Search stopped after {_MAX_SEARCH_DIRECTORIES} directories.)",
                    )
        except OSError as error:
            return ToolCallResult(False, f"Could not search {directory}: {error}")
        return ToolCallResult(bool(results), "\n".join(results) if results else f"No files matched '{query}'.")

    @staticmethod
    def _read_file(arguments: dict) -> ToolCallResult:
        path = _file_path(arguments, "path")
        if not path.is_file():
            return ToolCallResult(False, f"File not found: {path}")
        try:
            if path.stat().st_size > 100_000:
                return ToolCallResult(False, "File is larger than the 100 KiB tool limit.")
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            return ToolCallResult(False, f"Could not read {path}: {error}")
        return ToolCallResult(True, content or "(empty file)")

    @staticmethod
    def _write_file(arguments: dict) -> ToolCallResult:
        path, content = _file_and_content(arguments)
        if path.is_dir():
            return ToolCallResult(False, f"Path is a directory: {path}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(path, content)
        except OSError as error:
            return ToolCallResult(False, f"Could not write {path}: {error}")
        return ToolCallResult(True, f"Wrote {len(content)} characters to {path}.")

    @staticmethod
    def _create_file(arguments: dict) -> ToolCallResult:
        path, content = _file_and_content(arguments)
        if path.exists():
            return ToolCallResult(False, f"File already exists: {path}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(path, content, exclusive=True)
        except OSError as error:
            return ToolCallResult(False, f"Could not create {path}: {error}")
        return ToolCallResult(True, f"Created {path}.")

    @staticmethod
    def _delete_file(arguments: dict) -> ToolCallResult:
        path = _file_path(arguments, "path")
        if path in {Path("/"), Path.home()}:
            return ToolCallResult(False, "Refusing to delete a protected path.")
        if not path.exists():
            return ToolCallResult(False, f"File not found: {path}")
        if not path.is_file():
            return ToolCallResult(False, "Only individual files can be deleted.")
        try:
            path.unlink()
        except OSError as error:
            return ToolCallResult(False, f"Could not delete {path}: {error}")
        return ToolCallResult(True, f"Deleted {path}.")

    @staticmethod
    def _move_file(arguments: dict) -> ToolCallResult:
        source, destination = _source_destination(arguments)
        if not source.is_file():
            return ToolCallResult(False, f"Source file not found: {source}")
        if destination.exists():
            return ToolCallResult(False, f"Destination already exists: {destination}")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        except OSError as error:
            return ToolCallResult(False, f"Could not move file: {error}")
        return ToolCallResult(True, f"Moved {source} to {destination}.")

    @staticmethod
    def _copy_file(arguments: dict) -> ToolCallResult:
        source, destination = _source_destination(arguments)
        if not source.is_file():
            return ToolCallResult(False, f"Source file not found: {source}")
        if destination.exists():
            return ToolCallResult(False, f"Destination already exists: {destination}")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        except OSError as error:
            return ToolCallResult(False, f"Could not copy file: {error}")
        return ToolCallResult(True, f"Copied {source} to {destination}.")

    @staticmethod
    def _mouse_move(arguments: dict) -> ToolCallResult:
        x = _integer_argument(arguments, "x", -20000, 20000)
        y = _integer_argument(arguments, "y", -20000, 20000)
        if shutil.which("hyprctl"):
            return ToolManager._hyprland_dispatch("movecursor", str(x), str(y))
        if shutil.which("ydotool"):
            return _run_input_command(["ydotool", "mousemove", "--absolute", str(x), str(y)])
        if shutil.which("xdotool"):
            return _run_input_command(["xdotool", "mousemove", str(x), str(y)])
        return ToolCallResult(False, "No supported pointer-control utility is available.")

    @staticmethod
    def _mouse_click(arguments: dict) -> ToolCallResult:
        button = _string_argument(arguments, "button").lower()
        button_code = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}.get(button)
        if not button_code:
            return ToolCallResult(False, "Button must be left, right, or middle.")
        move = ToolManager._mouse_move(arguments)
        if not move.success:
            return move
        if shutil.which("ydotool"):
            return _run_input_command(["ydotool", "click", button_code])
        if shutil.which("xdotool"):
            xdotool_button = {"left": "1", "right": "3", "middle": "2"}[button]
            return _run_input_command(["xdotool", "click", xdotool_button])
        return ToolCallResult(False, "ydotool is required for Wayland mouse clicks.")

    @staticmethod
    def _keyboard_type(arguments: dict) -> ToolCallResult:
        text = _string_argument(arguments, "text")
        if len(text) > 2000:
            return ToolCallResult(False, "Text exceeds the 2000 character limit.")
        if shutil.which("wtype"):
            return _run_input_command(["wtype", "--", text])
        if shutil.which("ydotool"):
            return _run_input_command(["ydotool", "type", "--", text])
        if shutil.which("xdotool"):
            return _run_input_command(["xdotool", "type", "--", text])
        return ToolCallResult(False, "wtype or ydotool is required for Wayland keyboard input.")

    @staticmethod
    def _keyboard_press(arguments: dict) -> ToolCallResult:
        key = _string_argument(arguments, "key")
        if shutil.which("wtype"):
            return _run_input_command(["wtype", "-k", key])
        if shutil.which("xdotool"):
            return _run_input_command(["xdotool", "key", "--", key])
        return ToolCallResult(False, "wtype is required for Wayland key presses.")

    @staticmethod
    def _get_cpu_usage(arguments: dict) -> ToolCallResult:
        return ToolManager._get_cpu_status(arguments)

    @staticmethod
    def _get_gpu_usage(arguments: dict) -> ToolCallResult:
        return ToolManager._get_gpu_status(arguments)

    @staticmethod
    def _get_ram_usage(arguments: dict) -> ToolCallResult:
        return ToolManager._get_memory_status(arguments)

    @staticmethod
    def _get_disk_usage(arguments: dict) -> ToolCallResult:
        return ToolManager._get_storage_status(arguments)

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

    @staticmethod
    def _window_address(window: dict) -> str | None:
        address = str(window.get("address", "")).strip()
        return address if _WINDOW_ADDRESS.fullmatch(address) else None


def _integer_argument(arguments: dict, name: str, minimum: int, maximum: int) -> int:
    value = arguments.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"Tool argument '{name}' must be an integer from {minimum} to {maximum}.")
    return value


def _is_read_only_exec(command: str) -> bool:
    """Recognize only the small legacy read-only command allowlist."""

    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    if not parts or any(part in {";", "&&", "||", "|", ">", ">>", "<"} for part in parts):
        return False
    executable = Path(parts[0]).name
    if executable in _READ_ONLY_EXEC_COMMANDS:
        return True
    if executable != "flatpak":
        return False
    # Options may appear before the subcommand, but only `flatpak list` is
    # read-only. Install, uninstall, run, and other lifecycle commands remain
    # behind the normal confirmation gate.
    subcommands = [part for part in parts[1:] if not part.startswith("-")]
    return bool(subcommands) and subcommands[0] == "list"


def _file_path(arguments: dict, name: str) -> Path:
    value = _string_argument(arguments, name)
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _file_and_content(arguments: dict) -> tuple[Path, str]:
    path = _file_path(arguments, "path")
    content = arguments.get("content")
    if not isinstance(content, str):
        raise ValueError("Tool argument 'content' must be text.")
    if len(content.encode("utf-8")) > 200_000:
        raise ValueError("File content exceeds the 200 KiB tool limit.")
    return path, content


def _atomic_write_text(path: Path, content: str, exclusive: bool = False) -> None:
    """Write UTF-8 text beside the destination, then replace it atomically."""

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary_path = temporary.name
        if exclusive and path.exists():
            raise FileExistsError(path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _source_destination(arguments: dict) -> tuple[Path, Path]:
    return _file_path(arguments, "source"), _file_path(arguments, "destination")


def _run_input_command(command: list[str]) -> ToolCallResult:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return ToolCallResult(False, f"Input command failed: {error}")
    if result.returncode:
        return ToolCallResult(False, result.stderr.strip() or result.stdout.strip() or "Input command was rejected.")
    return ToolCallResult(True, result.stdout.strip() or "Input command completed.")


def _read_cpu_times() -> tuple[int, ...]:
    line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    values = line.split()[1:]
    if len(values) < 4:
        raise ValueError("CPU statistics are unavailable.")
    return tuple(int(value) for value in values)


def _read_cpu_identity() -> dict[str, str | int]:
    """Read CPU identity and topology from Linux, not from model context."""
    fields: dict[str, str] = {}
    try:
        result = subprocess.run(
            ["lscpu"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                key, separator, value = line.partition(":")
                if separator:
                    fields[key.strip().casefold()] = value.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    def integer(*names: str) -> int | None:
        for name in names:
            value = fields.get(name.casefold(), "")
            try:
                return int(value.split()[0])
            except (IndexError, ValueError):
                continue
        return None

    model = fields.get("model name") or fields.get("model") or platform.processor() or "unknown"
    vendor = fields.get("vendor id") or fields.get("vendor") or "unknown"
    cores = integer("core(s) per socket", "cores per socket")
    sockets = integer("socket(s)", "sockets")
    threads = integer("cpu(s)", "cpus", "logical cpu(s)")
    if cores is not None and sockets is not None:
        cores *= sockets
    if cores is None:
        cores = integer("cpu(s)") or 0
    if threads is None:
        threads = cores
    return {
        "vendor": vendor,
        "model": model,
        "cores": cores,
        "threads": threads,
    }


def _read_uptime() -> dict[str, float | bool]:
    try:
        seconds = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return {"available": False}
    return {"available": True, "seconds": seconds, "hours": round(seconds / 3600, 2)}


def _power_action(action: str) -> ToolCallResult:
    command = ["systemctl", action] if shutil.which("systemctl") else [action]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return ToolCallResult(False, json.dumps({"action": action, "error": str(error)}))
    return ToolCallResult(
        result.returncode == 0,
        json.dumps(
            {
                "action": action,
                "status": "requested" if result.returncode == 0 else "failed",
                "error": result.stderr.strip() if result.returncode else "",
            }
        ),
    )