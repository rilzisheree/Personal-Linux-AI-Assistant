"""Single-user password-protected HTTP API for Lura clients."""

from __future__ import annotations

import json
import logging
import os
import threading
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from .config import (
    AppConfig,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
)
from .errors import format_backend_error
from .gemini_api import GeminiApiClient
from .ollama import ChatMessage, OllamaClient, OllamaProtocolError, ToolCall
from .api_store import ApiStore, SESSION_TTL_SECONDS
from .tools import PermissionLevel, ToolCallResult, ToolManager


MAX_REQUEST_BYTES = 1_000_000
MAX_REMOTE_TOOL_ROUNDS = 4
# Remote clients may use read-only current-information tools and the existing
# allowlisted app launcher. Desktop mutation, input, shell, and file tools stay
# local-only because they require the trusted Qt permission flow.
REMOTE_TOOL_NAMES = frozenset(
    {
        "open_app",
        "get_active_model",
        "web_search",
        "search_news",
        "knowledge_search",
        "get_weather",
        "convert_currency",
        "find_places",
        "get_directions",
        "travel_search",
        "game_search",
    }
)
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
LOGGER = logging.getLogger("lura.api")


class ApiHttpError(Exception):
    """An expected HTTP response from the API."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class ApiServer(ThreadingHTTPServer):
    """Threaded API server with isolated Ollama clients per request."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        store: ApiStore,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        default_model: str = DEFAULT_MODEL,
        context_size: int = DEFAULT_CONTEXT_SIZE,
        *,
        provider: str = "ollama",
        gemini_api_key: str = "",
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        tool_permissions: dict[str, str] | None = None,
        custom_app_commands: dict[str, str] | None = None,
    ) -> None:
        super().__init__(address, ApiRequestHandler)
        self.store = store
        self.ollama_url = ollama_url.rstrip("/")
        self.default_model = (
            gemini_model.strip()
            if provider.strip().lower() == "gemini"
            else default_model
        )
        self.context_size = context_size
        self.provider = provider.strip().lower()
        self.gemini_api_key = gemini_api_key.strip()
        self.gemini_model = gemini_model.strip() or DEFAULT_GEMINI_MODEL
        self.tool_permissions = dict(tool_permissions or {})
        self.custom_app_commands = dict(custom_app_commands or {})


def start_background_server(
    ollama_url: str = DEFAULT_OLLAMA_URL,
    default_model: str = DEFAULT_MODEL,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    *,
    provider: str = "ollama",
    gemini_api_key: str = "",
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    tool_permissions: dict[str, str] | None = None,
    custom_app_commands: dict[str, str] | None = None,
) -> tuple[ApiServer | None, threading.Thread | None]:
    """Start a localhost API for the desktop app without blocking Qt."""

    if os.environ.get("LURA_API_AUTOSTART", "1").casefold() in {"0", "false", "no"}:
        return None, None
    try:
        port = int(os.environ.get("LURA_API_PORT", "8000"))
        server = ApiServer(
            (os.environ.get("LURA_API_HOST", "127.0.0.1"), port),
            ApiStore(),
            ollama_url,
            default_model,
            context_size,
            provider=provider,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            tool_permissions=tool_permissions,
            custom_app_commands=custom_app_commands,
        )
    except (OSError, ValueError, RuntimeError) as error:
        # An independently running API is valid; do not prevent the desktop
        # client from opening when its port is already occupied.
        print(f"Lura API autostart skipped: {error}")
        return None, None
    thread = threading.Thread(
        target=server.serve_forever,
        name="lura-api",
        daemon=True,
    )
    thread.start()
    return server, thread


class ApiRequestHandler(BaseHTTPRequestHandler):
    """Small JSON/SSE API surface kept dependency-free for the desktop project."""

    protocol_version = "HTTP/1.1"
    server_version = "LuraAPI/1.0"

    def do_OPTIONS(self) -> None:
        self._send_empty(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        try:
            parts = self._path_parts()
            if parts == ["api", "health"]:
                self._send_json(
                    {
                        "ok": True,
                        "service": "lura-api",
                        "provider": self._api_server().provider,
                        "ollama_url": self._api_server().ollama_url,
                    }
                )
                return
            user = self._require_user()
            if parts == ["api", "me"]:
                self._send_json(
                    {"user": {"id": "local", "authenticated": True}}
                )
            elif parts == ["api", "models"]:
                models = self._ai_client().list_models()
                self._send_json({"models": models})
            elif parts == ["api", "conversations"]:
                self._send_json(
                    {"conversations": self._api_server().store.list_conversations(user["id"])}
                )
            elif len(parts) == 3 and parts[:2] == ["api", "conversations"]:
                conversation = self._api_server().store.get_conversation(
                    user["id"], parts[2]
                )
                if conversation is None:
                    raise ApiHttpError(HTTPStatus.NOT_FOUND, "Conversation not found.")
                self._send_json({"conversation": conversation.to_dict()})
            else:
                raise ApiHttpError(HTTPStatus.NOT_FOUND, "Endpoint not found.")
        except Exception as error:
            self._handle_error(error)

    def do_POST(self) -> None:
        try:
            parts = self._path_parts()
            payload = self._json_body()
            if parts == ["api", "auth", "register"]:
                raise ApiHttpError(HTTPStatus.NOT_FOUND, "Endpoint not found.")
            if parts == ["api", "auth", "login"]:
                self._login(payload)
            elif parts == ["api", "auth", "logout"]:
                self._logout()
            else:
                user = self._require_user()
                if parts == ["api", "conversations"]:
                    self._create_conversation(user["id"], payload)
                elif (
                    len(parts) == 4
                    and parts[:2] == ["api", "conversations"]
                    and parts[3] == "messages"
                ):
                    self._stream_message(user["id"], parts[2], payload)
                else:
                    raise ApiHttpError(HTTPStatus.NOT_FOUND, "Endpoint not found.")
        except Exception as error:
            self._handle_error(error)

    def _login(self, payload: dict[str, Any]) -> None:
        password = payload.get("password")
        if not self._api_server().store.has_password():
            raise ApiHttpError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "The local API password has not been configured.",
            )
        if not self._api_server().store.verify_password(password):
            raise ApiHttpError(
                HTTPStatus.UNAUTHORIZED, "Password is incorrect."
            )
        token, expires_at = self._api_server().store.create_session()
        self._send_json(
            {"authenticated": True, "session_token": token},
            headers={"Set-Cookie": self._session_cookie(token, expires_at)},
        )

    def _logout(self) -> None:
        token = self._session_token()
        self._api_server().store.delete_session(token)
        self._send_json(
            {"ok": True},
            headers={"Set-Cookie": self._session_cookie("", 0)},
        )

    def _create_conversation(self, user_id: int, payload: dict[str, Any]) -> None:
        title = payload.get("title", "New chat")
        conversation = self._api_server().store.create_conversation(user_id, title)
        self._send_json(
            {"conversation": conversation.to_dict()},
            HTTPStatus.CREATED,
        )

    def _stream_message(
        self,
        user_id: int,
        conversation_id: str,
        payload: dict[str, Any],
    ) -> None:
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ApiHttpError(HTTPStatus.BAD_REQUEST, "Message content is required.")
        if len(content) > 200_000:
            raise ApiHttpError(HTTPStatus.BAD_REQUEST, "Message content is too large.")
        model = payload.get("model", self._api_server().default_model)
        if not isinstance(model, str) or not model.strip() or len(model) > 200:
            raise ApiHttpError(HTTPStatus.BAD_REQUEST, "Model name is invalid.")

        store = self._api_server().store
        conversation = store.get_conversation(user_id, conversation_id)
        if conversation is None:
            raise ApiHttpError(HTTPStatus.NOT_FOUND, "Conversation not found.")
        conversation = store.append_message(
            user_id,
            conversation_id,
            ChatMessage("user", content.strip()),
        )

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self._add_cors_headers()
        self.end_headers()

        cancel_event = threading.Event()
        response_parts: list[str] = []
        sse_events_sent = 0
        sse_chars_sent = 0
        completion_event_sent = False
        backend_done_reason = ""
        backend_generated_tokens: int | float | None = None
        backend_total_chunks = 0
        backend_total_chars = 0
        client = self._ai_client()
        server = self._api_server()
        tool_manager = ToolManager(
            tool_permissions=server.tool_permissions,
            custom_app_commands=server.custom_app_commands,
            active_model=model.strip(),
        )
        remote_tools = []
        if server.provider in {"ollama", "gemini"}:
            remote_tools = [
                schema
                for schema in tool_manager.definitions_for_ollama()
                if schema.get("function", {}).get("name") in REMOTE_TOOL_NAMES
            ]
        messages = list(conversation.messages)

        def write_sse(event: str, event_payload: dict) -> None:
            nonlocal sse_events_sent, sse_chars_sent, completion_event_sent
            sse_events_sent += 1
            if event == "token":
                token_content = event_payload.get("content", "")
                if isinstance(token_content, str):
                    sse_chars_sent += len(token_content)
            if event == "done":
                completion_event_sent = True
            self._write_sse(event, event_payload)

        LOGGER.info(
            "[SSE] stream_start conversation_id=%s input_messages=%d input_chars=%d",
            conversation_id,
            len(messages),
            sum(len(message.content) for message in messages),
        )
        write_sse("started", {"conversation_id": conversation_id})
        direct_call = tool_manager.direct_tool_call_for_request(content.strip())
        if direct_call is not None and direct_call[0] in REMOTE_TOOL_NAMES:
            tool_name, arguments = direct_call
            tool_call = ToolCall(tool_name, arguments, f"direct_{threading.get_ident()}")
            try:
                result = tool_manager.execute(tool_name, arguments)
            except Exception as error:
                result = ToolCallResult(False, str(error))
            write_sse(
                "tool",
                {"name": tool_name, "success": result.success, "message": result.content},
            )
            messages.append(ChatMessage("assistant", "", (tool_call,)))
            messages.append(
                ChatMessage(
                    "tool",
                    result.content,
                    name=tool_name,
                    tool_call_id=tool_call.id,
                )
            )
        tool_rounds = 0
        try:
            while True:
                cycle_response: list[str] = []
                tool_calls = []
                stream_complete = False
                backend_chunks = 0
                backend_chars = 0
                for event in client.stream_chat(
                    messages,
                    model.strip(),
                    cancel_event,
                    tools=remote_tools,
                    context_size=self._api_server().context_size,
                ):
                    backend_chunks += 1
                    backend_chars += len(event.content)
                    backend_total_chunks += 1
                    backend_total_chars += len(event.content)
                    if event.done:
                        backend_done_reason = event.done_reason
                        raw_eval_count = (event.metrics or {}).get("eval_count")
                        if isinstance(raw_eval_count, (int, float)):
                            backend_generated_tokens = raw_eval_count
                    if event.content:
                        cycle_response.append(event.content)
                        response_parts.append(event.content)
                        write_sse("token", {"content": event.content})
                    tool_calls.extend(event.tool_calls)
                    if event.done:
                        stream_complete = True
                        break

                if not stream_complete:
                    raise OllamaProtocolError(
                        "The AI stream ended before sending a completion event."
                    )
                LOGGER.info(
                    "[BACKEND] conversation_id=%s chunks_received=%d "
                    "chars_received=%d stream_finished=true done_reason=%s "
                    "generated_tokens=%s",
                    conversation_id,
                    backend_total_chunks,
                    backend_total_chars,
                    backend_done_reason or "unspecified",
                    backend_generated_tokens if backend_generated_tokens is not None else "unknown",
                )
                if cancel_event.is_set():
                    return
                if not tool_calls:
                    response = "".join(response_parts)
                    if response:
                        store.append_message(
                            user_id,
                            conversation_id,
                            ChatMessage("assistant", response),
                        )
                    write_sse(
                        "done",
                        {
                            "conversation_id": conversation_id,
                            "message": response,
                        },
                    )
                    return

                tool_rounds += 1
                if tool_rounds > MAX_REMOTE_TOOL_ROUNDS:
                    raise OllamaProtocolError(
                        f"Stopped after {MAX_REMOTE_TOOL_ROUNDS} remote tool rounds."
                    )

                messages.append(
                    ChatMessage("assistant", "".join(cycle_response), tuple(tool_calls))
                )
                for tool_call in tool_calls:
                    if tool_call.name not in REMOTE_TOOL_NAMES:
                        result = ToolCallResult(
                            False,
                            f"Remote tool '{tool_call.name}' is not enabled.",
                        )
                    elif (
                        tool_manager.permission_for(
                            tool_call.name, tool_call.arguments
                        )
                        not in {PermissionLevel.SAFE, PermissionLevel.NORMAL}
                    ):
                        result = ToolCallResult(
                            False,
                            f"Remote tool '{tool_call.name}' requires local approval.",
                        )
                    else:
                        result = tool_manager.execute(
                            tool_call.name, tool_call.arguments, approved=True
                        )
                    write_sse(
                        "tool",
                        {
                            "name": tool_call.name,
                            "success": result.success,
                            "message": result.content,
                        },
                    )
                    messages.append(
                        ChatMessage(
                            "tool",
                            result.content,
                            name=tool_call.name,
                        )
                    )
        except (BrokenPipeError, ConnectionResetError):
            cancel_event.set()
        except Exception as error:
            LOGGER.error(
                "[BACKEND] conversation_id=%s chunks_received=%d "
                "chars_received=%d stream_finished=false exception=%s",
                conversation_id,
                backend_total_chunks,
                backend_total_chars,
                type(error).__name__,
            )
            try:
                error_message = format_backend_error(
                    error,
                    getattr(client, "display_name", "AI backend"),
                )
                write_sse(
                    "error",
                    {
                        "status": "error",
                        "partial": bool(response_parts),
                        "error": {
                            "code": "STREAM_ERROR",
                            "message": error_message,
                        },
                        "message": error_message,
                    },
                )
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        finally:
            LOGGER.info(
                "[SSE] events_sent=%d chars_sent=%d completion_event_sent=%s "
                "final_message_chars=%d done_reason=%s generated_tokens=%s",
                sse_events_sent,
                sse_chars_sent,
                str(completion_event_sent).lower(),
                len("".join(response_parts)),
                backend_done_reason or "unspecified",
                backend_generated_tokens if backend_generated_tokens is not None else "unknown",
            )

    def _require_user(self) -> dict:
        user = self._api_server().store.user_for_session(self._session_token())
        if user is None:
            raise ApiHttpError(HTTPStatus.UNAUTHORIZED, "Authentication is required.")
        return user

    def _session_token(self) -> str | None:
        authorization = self.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            return authorization[7:].strip() or None
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except ValueError:
            return None
        morsel = cookie.get("lura_session")
        return morsel.value if morsel else None

    def _session_cookie(self, token: str, expires_at: int) -> str:
        allowed_origin = os.environ.get("LURA_ALLOWED_ORIGIN", "").strip()
        request_origin = self.headers.get("Origin", "").strip()
        cross_origin_https = (
            bool(allowed_origin)
            and request_origin == allowed_origin
            and request_origin.casefold().startswith("https://")
        )
        same_site = "None" if cross_origin_https else "Lax"
        cookie = f"lura_session={token}; HttpOnly; Path=/; SameSite={same_site}"
        if expires_at:
            cookie += f"; Max-Age={SESSION_TTL_SECONDS}"
        else:
            cookie += "; Max-Age=0"
        if cross_origin_https or os.environ.get("LURA_COOKIE_SECURE", "").casefold() in {
            "1",
            "true",
            "yes",
        }:
            cookie += "; Secure"
        return cookie

    def _path_parts(self) -> list[str]:
        return [
            unquote(part)
            for part in urlsplit(self.path).path.split("/")
            if part
        ]

    def _json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ApiHttpError(HTTPStatus.BAD_REQUEST, "Invalid request body.") from error
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ApiHttpError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request body is too large.")
        if length == 0:
            return {}
        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ApiHttpError(HTTPStatus.BAD_REQUEST, "Request body must be valid JSON.") from error
        if not isinstance(payload, dict):
            raise ApiHttpError(HTTPStatus.BAD_REQUEST, "Request body must be an object.")
        return cast(dict[str, Any], payload)

    def _send_json(
        self,
        payload: dict,
        status: int = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._add_cors_headers()
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self._add_cors_headers()
        self.end_headers()

    def _write_sse(self, event: str, payload: dict) -> None:
        body = (
            f"event: {event}\n"
            f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
        ).encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()

    def _add_cors_headers(self) -> None:
        allowed_origin = os.environ.get("LURA_ALLOWED_ORIGIN", "").strip()
        request_origin = self.headers.get("Origin", "").strip()
        if allowed_origin and request_origin == allowed_origin:
            self.send_header("Access-Control-Allow-Origin", request_origin)
            self.send_header("Access-Control-Allow-Credentials", "true")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Vary", "Origin")

    def _handle_error(self, error: Exception) -> None:
        if isinstance(error, ApiHttpError):
            status, message = error.status, error.message
        elif isinstance(error, KeyError):
            status, message = HTTPStatus.NOT_FOUND, str(error).strip("'")
        elif isinstance(error, ValueError):
            status, message = HTTPStatus.BAD_REQUEST, str(error)
        else:
            status, message = (
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "The API could not complete the request.",
            )
        if not self.wfile.closed:
            try:
                self._send_json({"error": message}, status)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    def _api_server(self) -> ApiServer:
        return cast(ApiServer, self.server)

    def _ai_client(self):
        server = self._api_server()
        if server.provider == "gemini":
            return GeminiApiClient(server.gemini_api_key)
        if server.provider == "ollama":
            return OllamaClient(server.ollama_url)
        raise ApiHttpError(
            HTTPStatus.BAD_REQUEST,
            f"Unsupported API provider: {server.provider}.",
        )

    def log_message(self, format: str, *args: object) -> None:
        # Keep credentials, request bodies, and bearer tokens out of logs.
        super().log_message("%s", format % args)


def main() -> int:
    """Run the API service using environment-based deployment configuration."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        port = int(os.environ.get("PORT", "8000"))
    except ValueError as error:
        raise SystemExit("PORT must be an integer.") from error
    if not 1 <= port <= 65535:
        raise SystemExit("PORT must be between 1 and 65535.")

    host = os.environ.get("LURA_API_HOST", "0.0.0.0")
    ollama_url = os.environ.get("LURA_OLLAMA_URL", DEFAULT_OLLAMA_URL).strip().rstrip("/")
    configured_gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    provider = os.environ.get(
        "LURA_AI_PROVIDER",
        "gemini" if configured_gemini_key else "ollama",
    ).strip().lower()
    default_model = DEFAULT_GEMINI_MODEL if provider == "gemini" else DEFAULT_MODEL
    model = os.environ.get("LURA_MODEL", default_model).strip()
    gemini_model = os.environ.get("LURA_GEMINI_MODEL", model).strip()
    local_config = AppConfig.load()
    database_path = Path(
        os.environ.get("LURA_API_DATABASE", str(ApiStore.default_path()))
    ).expanduser()
    store = ApiStore(database_path)
    configured_password = os.environ.get("LURA_API_PASSWORD")
    if not store.has_password() and configured_password:
        store.set_password(configured_password)
    server = ApiServer(
        (host, port),
        store,
        ollama_url,
        model,
        provider=provider,
        gemini_api_key=configured_gemini_key,
        gemini_model=gemini_model,
        tool_permissions=local_config.tool_permissions,
        custom_app_commands=local_config.custom_app_commands,
    )
    print(f"Lura API listening on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())