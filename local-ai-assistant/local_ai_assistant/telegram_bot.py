"""Telegram bridge for controlling the local Lura assistant.

This process is intended to run on the same Linux desktop as Ollama. It uses
Telegram long polling, so it does not require an inbound port or Cloudflare
Tunnel. Only the configured Telegram user can interact with it, and only the
safe ``open_app`` tool is exposed remotely.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import DEFAULT_CONTEXT_SIZE, DEFAULT_MODEL, DEFAULT_OLLAMA_URL
from .errors import format_ollama_error
from .ollama import ChatMessage, OllamaClient, OllamaProtocolError, ToolCall
from .tools import PermissionLevel, ToolCallResult, ToolManager


TELEGRAM_MAX_MESSAGE_LENGTH = 4096
TELEGRAM_POLL_TIMEOUT = 30
TELEGRAM_RETRY_DELAY = 5
TELEGRAM_STREAM_UPDATE_INTERVAL = 1.0
MAX_TOOL_ROUNDS = 4
MAX_HISTORY_MESSAGES = 40
DEFAULT_OLLAMA_TIMEOUT = 120.0
REMOTE_TOOL_NAMES = frozenset({"open_app"})
TELEGRAM_TOKEN_PATH = (
    Path.home() / ".config" / "local-ai-assistant" / "telegram-bot.token"
)


def load_telegram_token() -> str:
    try:
        return TELEGRAM_TOKEN_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except OSError as error:
        raise RuntimeError(f"Could not read the Telegram token file: {error}") from error


def save_telegram_token(token: str) -> None:
    token = token.strip()
    if not token:
        raise ValueError("Telegram bot token cannot be empty.")
    TELEGRAM_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = TELEGRAM_TOKEN_PATH.with_suffix(".tmp")
    temporary_path.write_text(token + "\n", encoding="utf-8")
    temporary_path.chmod(0o600)
    temporary_path.replace(TELEGRAM_TOKEN_PATH)


class TelegramApiError(RuntimeError):
    """Raised when Telegram rejects or cannot complete a Bot API request."""


class TelegramClient:
    """Small dependency-free Telegram Bot API client."""

    def __init__(self, token: str, timeout: float = 40.0) -> None:
        token = token.strip()
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required.")
        self._base_url = f"https://api.telegram.org/bot{token}"
        self.timeout = timeout
        self._response: object | None = None
        self._response_lock = threading.Lock()

    def call(self, method: str, payload: dict | None = None) -> object:
        body = json.dumps(payload or {}).encode("utf-8")
        request = Request(
            f"{self._base_url}/{method}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                with self._response_lock:
                    self._response = response
                try:
                    raw = response.read()
                finally:
                    with self._response_lock:
                        self._response = None
        except HTTPError as error:
            detail = self._error_detail(error)
            raise TelegramApiError(f"Telegram API HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise TelegramApiError("Could not reach the Telegram API.") from error

        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TelegramApiError("Telegram returned an invalid response.") from error
        if not isinstance(result, dict) or result.get("ok") is not True:
            description = result.get("description", "Unknown Telegram API error") if isinstance(result, dict) else "Invalid response"
            raise TelegramApiError(str(description))
        return result.get("result")

    def cancel_active_request(self) -> None:
        with self._response_lock:
            response = self._response
        if response is not None:
            close = getattr(response, "close", None)
            if close is not None:
                close()

    @staticmethod
    def _error_detail(error: HTTPError) -> str:
        try:
            raw = error.read()
            result = json.loads(raw.decode("utf-8"))
            if isinstance(result, dict) and isinstance(result.get("description"), str):
                return result["description"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        return "Request was rejected."

    def get_me(self) -> dict:
        result = self.call("getMe")
        if not isinstance(result, dict):
            raise TelegramApiError("Telegram returned invalid bot information.")
        return result

    def delete_webhook(self) -> None:
        self.call("deleteWebhook", {"drop_pending_updates": False})

    def get_updates(self, offset: int) -> list[dict]:
        result = self.call(
            "getUpdates",
            {
                "offset": offset,
                "timeout": TELEGRAM_POLL_TIMEOUT,
                "allowed_updates": ["message"],
            },
        )
        if not isinstance(result, list):
            raise TelegramApiError("Telegram returned invalid updates.")
        return [update for update in result if isinstance(update, dict)]

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        self.call("sendChatAction", {"chat_id": chat_id, "action": action})

    def send_message(self, chat_id: int, text: str) -> dict | None:
        first_message: dict | None = None
        for chunk in _message_chunks(text):
            result = self.call("sendMessage", {"chat_id": chat_id, "text": chunk})
            if first_message is None and isinstance(result, dict):
                first_message = result
        return first_message

    def edit_message_text(self, chat_id: int, message_id: int, text: str) -> None:
        self.call(
            "editMessageText",
            {"chat_id": chat_id, "message_id": message_id, "text": text},
        )


@dataclass(frozen=True)
class TelegramConfig:
    token: str
    allowed_user_id: int
    ollama_url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_MODEL
    context_size: int = DEFAULT_CONTEXT_SIZE
    ollama_timeout: float = DEFAULT_OLLAMA_TIMEOUT

    @classmethod
    def from_environment(cls) -> "TelegramConfig":
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise ValueError("Set TELEGRAM_BOT_TOKEN before starting the Telegram bot.")
        try:
            allowed_user_id = int(os.environ.get("TELEGRAM_ALLOWED_USER_ID", "").strip())
        except ValueError as error:
            raise ValueError("TELEGRAM_ALLOWED_USER_ID must be a Telegram numeric user ID.") from error
        if allowed_user_id <= 0:
            raise ValueError("TELEGRAM_ALLOWED_USER_ID must be a positive Telegram numeric user ID.")

        ollama_url = os.environ.get("LURA_OLLAMA_URL", DEFAULT_OLLAMA_URL).strip().rstrip("/")
        model = os.environ.get("LURA_MODEL", DEFAULT_MODEL).strip()
        try:
            context_size = int(
                os.environ.get("LURA_CONTEXT_SIZE", str(DEFAULT_CONTEXT_SIZE)).strip()
            )
        except ValueError as error:
            raise ValueError("LURA_CONTEXT_SIZE must be an integer.") from error
        if not ollama_url or not model:
            raise ValueError("LURA_OLLAMA_URL and LURA_MODEL cannot be empty.")
        if context_size < 2048 or context_size > 131072 or context_size % 1024:
            raise ValueError("LURA_CONTEXT_SIZE must be a multiple of 1024 between 2048 and 131072.")
        try:
            ollama_timeout = float(
                os.environ.get("LURA_OLLAMA_TIMEOUT", str(DEFAULT_OLLAMA_TIMEOUT)).strip()
            )
        except ValueError as error:
            raise ValueError("LURA_OLLAMA_TIMEOUT must be a number.") from error
        if ollama_timeout < 8 or ollama_timeout > 600:
            raise ValueError("LURA_OLLAMA_TIMEOUT must be between 8 and 600 seconds.")
        return cls(token, allowed_user_id, ollama_url, model, context_size, ollama_timeout)


class LocalAssistant:
    """Runs Ollama chat plus the explicitly allowlisted local tool."""

    def __init__(self, config: TelegramConfig) -> None:
        self.config = config
        self.ollama = OllamaClient(config.ollama_url, timeout=config.ollama_timeout)
        self.tool_manager = ToolManager()
        self.remote_tools = [
            schema
            for schema in self.tool_manager.definitions_for_ollama()
            if schema.get("function", {}).get("name") in REMOTE_TOOL_NAMES
        ]
        self.histories: dict[int, list[ChatMessage]] = {}

    def reply(
        self,
        chat_id: int,
        content: str,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        messages = self.histories.setdefault(chat_id, [])
        messages.append(ChatMessage("user", content.strip()))
        response_parts: list[str] = []
        tool_rounds = 0

        while True:
            cycle_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            for event in self.ollama.stream_chat(
                messages,
                self.config.model,
                tools=self.remote_tools,
                context_size=self.config.context_size,
            ):
                if event.content:
                    cycle_parts.append(event.content)
                    response_parts.append(event.content)
                    if on_chunk is not None:
                        on_chunk(event.content)
                tool_calls.extend(event.tool_calls)
                if event.done:
                    break

            if not tool_calls:
                response = "".join(response_parts).strip()
                messages.append(ChatMessage("assistant", response))
                self._trim_history(messages)
                return response or "I did not receive a response from Ollama."

            tool_rounds += 1
            if tool_rounds > MAX_TOOL_ROUNDS:
                raise OllamaProtocolError(
                    f"Stopped after {MAX_TOOL_ROUNDS} tool rounds."
                )
            messages.append(
                ChatMessage("assistant", "".join(cycle_parts), tuple(tool_calls))
            )
            for tool_call in tool_calls:
                if tool_call.name not in REMOTE_TOOL_NAMES:
                    result = ToolCallResult(
                        False,
                        f"Remote tool '{tool_call.name}' is not enabled.",
                    )
                elif (
                    self.tool_manager.permission_for(
                        tool_call.name, tool_call.arguments
                    )
                    != PermissionLevel.SAFE
                ):
                    result = ToolCallResult(
                        False,
                        f"Remote tool '{tool_call.name}' requires local approval.",
                    )
                else:
                    result = self.tool_manager.execute(
                        tool_call.name, tool_call.arguments, approved=True
                    )
                messages.append(
                    ChatMessage("tool", result.content, name=tool_call.name)
                )

    @staticmethod
    def _trim_history(messages: list[ChatMessage]) -> None:
        if len(messages) > MAX_HISTORY_MESSAGES:
            del messages[:-MAX_HISTORY_MESSAGES]


class TelegramResponseStreamer:
    """Show an Ollama response in Telegram while it is still generating.

    Telegram does not support a streaming response body. A placeholder message
    is sent immediately and then edited at a safe cadence while Ollama emits
    tokens. The final edit also handles Telegram's 4096-character limit.
    """

    def __init__(self, telegram: TelegramClient, chat_id: int) -> None:
        self.telegram = telegram
        self.chat_id = chat_id
        self.message_id: int | None = None
        self.text = ""
        self._last_update_at = 0.0
        self._last_published_text = ""
        self._updates_disabled = False

    def start(self) -> None:
        result = self.telegram.send_message(self.chat_id, "Lura is thinking…")
        if isinstance(result, dict) and isinstance(result.get("message_id"), int):
            self.message_id = result["message_id"]
            self._last_update_at = time.monotonic()

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        self.text += chunk
        self._publish(force=False)

    def finish(self, text: str) -> None:
        self.text = text
        if self.message_id is None or self._updates_disabled:
            self.telegram.send_message(self.chat_id, text)
            return

        chunks = _message_chunks(text)
        try:
            if chunks[0] != self._last_published_text:
                self.telegram.edit_message_text(
                    self.chat_id,
                    self.message_id,
                    chunks[0],
                )
                self._last_published_text = chunks[0]
            for chunk in chunks[1:]:
                self.telegram.send_message(self.chat_id, chunk)
        except TelegramApiError:
            # The model response is more important than a failed live edit.
            # Try delivering it as a normal message instead.
            self.message_id = None
            self._updates_disabled = True
            self.telegram.send_message(self.chat_id, text)

    def _publish(self, force: bool) -> None:
        if self.message_id is None or self._updates_disabled:
            return
        now = time.monotonic()
        if (
            not force
            and now - self._last_update_at < TELEGRAM_STREAM_UPDATE_INTERVAL
        ):
            return
        visible_text = self.text[:TELEGRAM_MAX_MESSAGE_LENGTH] or "Lura is thinking…"
        try:
            self.telegram.edit_message_text(
                self.chat_id,
                self.message_id,
                visible_text,
            )
            self._last_published_text = visible_text
            self._last_update_at = now
        except TelegramApiError:
            self._updates_disabled = True


def _message_chunks(text: str) -> list[str]:
    text = text or "I did not receive a response."
    return [
        text[start : start + TELEGRAM_MAX_MESSAGE_LENGTH]
        for start in range(0, len(text), TELEGRAM_MAX_MESSAGE_LENGTH)
    ]


def _command_name(text: str) -> str:
    first = text.strip().split(maxsplit=1)[0]
    return first.split("@", maxsplit=1)[0].casefold()


def _handle_message(
    telegram: TelegramClient,
    assistant: LocalAssistant,
    allowed_user_id: int,
    message: dict,
    on_status=None,
) -> None:
    sender = message.get("from")
    chat = message.get("chat")
    text = message.get("text")
    if not isinstance(sender, dict) or not isinstance(chat, dict) or not isinstance(text, str):
        return
    if chat.get("type") != "private":
        return
    if sender.get("id") != allowed_user_id:
        if on_status is not None:
            on_status(
                "Ignored a private Telegram message from a different user ID."
            )
        return
    chat_id = chat.get("id")
    if not isinstance(chat_id, int):
        return

    command = _command_name(text)
    if command in {"/start", "/help"}:
        telegram.send_message(
            chat_id,
            "Lura is connected to this Linux PC.\n"
            "Send a normal message to chat with the local model.\n"
            "Try: Open Firefox on my PC\n\n"
            "/reset clears this Telegram chat's local context.",
        )
        return
    if command == "/reset":
        assistant.histories.pop(chat_id, None)
        telegram.send_message(chat_id, "Local Telegram conversation context cleared.")
        return

    telegram.send_chat_action(chat_id)
    streamer = TelegramResponseStreamer(telegram, chat_id)
    try:
        streamer.start()
        response = assistant.reply(chat_id, text, on_chunk=streamer.append)
    except Exception as error:
        response = f"Lura could not complete that request: {format_ollama_error(error)}"
    streamer.finish(response)


class TelegramBotRunner:
    """Long-polling loop shared by the CLI and the desktop worker."""

    def __init__(self, config: TelegramConfig) -> None:
        self.config = config
        self.telegram = TelegramClient(
            config.token, timeout=TELEGRAM_POLL_TIMEOUT + 10
        )
        self.assistant = LocalAssistant(config)
        self._stop_event = threading.Event()

    def run(
        self,
        on_connected=None,
        on_status=None,
    ) -> None:
        bot = self.telegram.get_me()
        username = bot.get("username", "unknown") if isinstance(bot, dict) else "unknown"
        if on_connected is not None:
            on_connected(str(username))
        self.telegram.delete_webhook()
        if on_status is not None:
            on_status("Waiting for private Telegram messages.")

        offset = 0
        seen_update_ids: set[int] = set()
        while not self._stop_event.is_set():
            try:
                for update in self.telegram.get_updates(offset):
                    update_id = update.get("update_id")
                    if not isinstance(update_id, int):
                        continue
                    if update_id in seen_update_ids:
                        continue
                    seen_update_ids.add(update_id)
                    if len(seen_update_ids) > 1000:
                        seen_update_ids.clear()
                    offset = max(offset, update_id + 1)
                    message = update.get("message")
                    if isinstance(message, dict):
                        _handle_message(
                            self.telegram,
                            self.assistant,
                            self.config.allowed_user_id,
                            message,
                            on_status=on_status,
                        )
            except TelegramApiError:
                if self._stop_event.is_set():
                    return
                if on_status is not None:
                    on_status(
                        "Telegram connection lost; retrying in "
                        f"{TELEGRAM_RETRY_DELAY} seconds."
                    )
                time.sleep(TELEGRAM_RETRY_DELAY)

    def stop(self) -> None:
        self._stop_event.set()
        self.telegram.cancel_active_request()


def run() -> int:
    config = TelegramConfig.from_environment()
    runner = TelegramBotRunner(config)
    try:
        runner.run(
            on_connected=lambda username: print(
                f"Lura Telegram bot connected as @{username}."
            ),
            on_status=print,
        )
    except KeyboardInterrupt:
        print("\nLura Telegram bot stopped.")
        return 0
    except TelegramApiError as error:
        print(f"Telegram connection error: {error}")
        return 1
    except Exception as error:
        print(f"Telegram bot error: {format_ollama_error(error)}")
        return 1
    print("Lura Telegram bot stopped.")
    return 0


def main() -> int:
    try:
        return run()
    except (ValueError, TelegramApiError) as error:
        print(f"Lura Telegram bot could not start: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())