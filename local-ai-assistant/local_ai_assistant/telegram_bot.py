"""Telegram bridge for controlling the local Lura assistant.

This process is intended to run on the same Linux desktop as Ollama. It uses
Telegram long polling, so it does not require an inbound port or Cloudflare
Tunnel. Only the configured Telegram user can interact with it, and only the
safe ``open_app`` tool is exposed remotely.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import DEFAULT_CONTEXT_SIZE, DEFAULT_MODEL, DEFAULT_OLLAMA_URL
from .errors import format_ollama_error
from .ollama import ChatMessage, OllamaClient, OllamaProtocolError, ToolCall
from .tools import PermissionLevel, ToolCallResult, ToolManager


TELEGRAM_MAX_MESSAGE_LENGTH = 4096
TELEGRAM_POLL_TIMEOUT = 30
TELEGRAM_RETRY_DELAY = 5
MAX_TOOL_ROUNDS = 4
MAX_HISTORY_MESSAGES = 40
REMOTE_TOOL_NAMES = frozenset({"open_app"})


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
                raw = response.read()
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

    def send_message(self, chat_id: int, text: str) -> None:
        for chunk in _message_chunks(text):
            self.call("sendMessage", {"chat_id": chat_id, "text": chunk})


@dataclass(frozen=True)
class TelegramConfig:
    token: str
    allowed_user_id: int
    ollama_url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_MODEL
    context_size: int = DEFAULT_CONTEXT_SIZE

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
        return cls(token, allowed_user_id, ollama_url, model, context_size)


class LocalAssistant:
    """Runs Ollama chat plus the explicitly allowlisted local tool."""

    def __init__(self, config: TelegramConfig) -> None:
        self.config = config
        self.ollama = OllamaClient(config.ollama_url)
        self.tool_manager = ToolManager()
        self.remote_tools = [
            schema
            for schema in self.tool_manager.definitions_for_ollama()
            if schema.get("function", {}).get("name") in REMOTE_TOOL_NAMES
        ]
        self.histories: dict[int, list[ChatMessage]] = {}

    def reply(self, chat_id: int, content: str) -> str:
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
) -> None:
    sender = message.get("from")
    chat = message.get("chat")
    text = message.get("text")
    if not isinstance(sender, dict) or not isinstance(chat, dict) or not isinstance(text, str):
        return
    if sender.get("id") != allowed_user_id or chat.get("type") != "private":
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
    try:
        response = assistant.reply(chat_id, text)
    except Exception as error:
        response = f"Lura could not complete that request: {format_ollama_error(error)}"
    telegram.send_message(chat_id, response)


def run() -> int:
    config = TelegramConfig.from_environment()
    telegram = TelegramClient(config.token)
    assistant = LocalAssistant(config)
    bot = telegram.get_me()
    username = bot.get("username", "unknown") if isinstance(bot, dict) else "unknown"
    print(f"Lura Telegram bot connected as @{username}.")
    telegram.delete_webhook()
    print("Waiting for private messages from the configured Telegram user.")

    offset = 0
    while True:
        try:
            for update in telegram.get_updates(offset):
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    offset = max(offset, update_id + 1)
                message = update.get("message")
                if isinstance(message, dict):
                    _handle_message(
                        telegram, assistant, config.allowed_user_id, message
                    )
        except KeyboardInterrupt:
            print("\nLura Telegram bot stopped.")
            return 0
        except TelegramApiError as error:
            print(f"Telegram connection error: {error}")
            time.sleep(TELEGRAM_RETRY_DELAY)
        except Exception as error:
            print(f"Telegram bot error: {format_ollama_error(error)}")
            time.sleep(TELEGRAM_RETRY_DELAY)


def main() -> int:
    try:
        return run()
    except (ValueError, TelegramApiError) as error:
        print(f"Lura Telegram bot could not start: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())