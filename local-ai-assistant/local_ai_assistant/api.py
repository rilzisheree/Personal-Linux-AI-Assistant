"""Authenticated HTTP API for remote Lura clients."""

from __future__ import annotations

import json
import os
import threading
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from .config import DEFAULT_CONTEXT_SIZE, DEFAULT_MODEL, DEFAULT_OLLAMA_URL
from .errors import format_ollama_error
from .ollama import ChatMessage, OllamaClient, OllamaProtocolError
from .api_store import ApiStore, DuplicateEmailError, SESSION_TTL_SECONDS


MAX_REQUEST_BYTES = 1_000_000


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
    ) -> None:
        super().__init__(address, ApiRequestHandler)
        self.store = store
        self.ollama_url = ollama_url.rstrip("/")
        self.default_model = default_model
        self.context_size = context_size


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
                        "ollama_url": self._api_server().ollama_url,
                    }
                )
                return
            user = self._require_user()
            if parts == ["api", "me"]:
                self._send_json({"user": user})
            elif parts == ["api", "models"]:
                models = OllamaClient(self._api_server().ollama_url).list_models()
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
                self._register(payload)
            elif parts == ["api", "auth", "login"]:
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

    def _register(self, payload: dict[str, Any]) -> None:
        email = payload.get("email")
        password = payload.get("password")
        user = self._api_server().store.register(email, password)
        token, expires_at = self._api_server().store.create_session(user["id"])
        self._send_json(
            {"user": user},
            HTTPStatus.CREATED,
            {"Set-Cookie": self._session_cookie(token, expires_at)},
        )

    def _login(self, payload: dict[str, Any]) -> None:
        email = payload.get("email")
        password = payload.get("password")
        user = self._api_server().store.authenticate(email, password)
        if user is None:
            raise ApiHttpError(
                HTTPStatus.UNAUTHORIZED, "Email or password is incorrect."
            )
        token, expires_at = self._api_server().store.create_session(user["id"])
        self._send_json(
            {"user": user},
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
        client = OllamaClient(self._api_server().ollama_url)
        try:
            self._write_sse("started", {"conversation_id": conversation_id})
            for event in client.stream_chat(
                conversation.messages,
                model.strip(),
                cancel_event,
                tools=[],
                context_size=self._api_server().context_size,
            ):
                if event.tool_calls:
                    raise OllamaProtocolError(
                        "Remote tool execution is disabled; use the trusted desktop app "
                        "for local tools."
                    )
                if event.content:
                    response_parts.append(event.content)
                    self._write_sse("token", {"content": event.content})
                if event.done:
                    break
            response = "".join(response_parts)
            if response:
                store.append_message(
                    user_id,
                    conversation_id,
                    ChatMessage("assistant", response),
                )
            self._write_sse(
                "done",
                {
                    "conversation_id": conversation_id,
                    "message": response,
                },
            )
        except (BrokenPipeError, ConnectionResetError):
            cancel_event.set()
        except Exception as error:
            try:
                self._write_sse(
                    "error",
                    {"message": format_ollama_error(error)},
                )
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

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

    @staticmethod
    def _session_cookie(token: str, expires_at: int) -> str:
        cookie = f"lura_session={token}; HttpOnly; Path=/; SameSite=Lax"
        if expires_at:
            cookie += f"; Max-Age={SESSION_TTL_SECONDS}"
        else:
            cookie += "; Max-Age=0"
        if os.environ.get("LURA_COOKIE_SECURE", "").casefold() in {"1", "true", "yes"}:
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
        if allowed_origin:
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Access-Control-Allow-Credentials", "true")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Vary", "Origin")

    def _handle_error(self, error: Exception) -> None:
        if isinstance(error, ApiHttpError):
            status, message = error.status, error.message
        elif isinstance(error, DuplicateEmailError):
            status, message = HTTPStatus.CONFLICT, str(error)
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

    def log_message(self, format: str, *args: object) -> None:
        # Keep credentials, request bodies, and bearer tokens out of logs.
        super().log_message("%s", format % args)


def main() -> int:
    """Run the API service using environment-based deployment configuration."""

    try:
        port = int(os.environ.get("PORT", "8000"))
    except ValueError as error:
        raise SystemExit("PORT must be an integer.") from error
    if not 1 <= port <= 65535:
        raise SystemExit("PORT must be between 1 and 65535.")

    host = os.environ.get("LURA_API_HOST", "0.0.0.0")
    ollama_url = os.environ.get("LURA_OLLAMA_URL", DEFAULT_OLLAMA_URL).strip().rstrip("/")
    model = os.environ.get("LURA_MODEL", DEFAULT_MODEL).strip()
    database_path = Path(
        os.environ.get("LURA_API_DATABASE", str(ApiStore.default_path()))
    ).expanduser()
    store = ApiStore(database_path)
    server = ApiServer((host, port), store, ollama_url, model)
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