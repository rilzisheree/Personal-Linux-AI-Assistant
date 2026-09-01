"""Direct Ollama HTTP adapter with newline-delimited JSON streaming."""

from __future__ import annotations

import base64
import json
import logging
import socket
import threading
import time
from dataclasses import dataclass
from http.client import HTTPResponse
from pathlib import Path
from typing import Iterator, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import (
    OllamaCancelledError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaProtocolError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)

LOGGER = logging.getLogger("lura.ollama")


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict
    id: str = ""
    thought_signature: str = ""

    def as_dict(self) -> dict:
        function = {"name": self.name, "arguments": self.arguments}
        payload = {"function": function}
        if self.id:
            payload["id"] = self.id
        if self.thought_signature:
            payload["thought_signature"] = self.thought_signature
        return payload


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    name: str = ""
    images: tuple[str, ...] = ()
    tool_call_id: str = ""

    def as_dict(self) -> dict:
        payload: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [tool_call.as_dict() for tool_call in self.tool_calls]
        if self.name:
            payload["name"] = self.name
        if self.images:
            payload["images"] = list(self.images)
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(frozen=True)
class StreamEvent:
    """One parsed event from Ollama's NDJSON chat response."""

    content: str = ""
    done: bool = False
    tool_calls: tuple[ToolCall, ...] = ()


def parse_stream_line(line: str | bytes) -> StreamEvent | None:
    """Parse one Ollama NDJSON line.

    Blank keep-alive lines are ignored. Invalid JSON and explicit API errors
    are raised instead of being silently shown as assistant text.
    """

    if isinstance(line, bytes):
        line = line.decode("utf-8")
    line = line.strip()
    if not line:
        return None

    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise OllamaProtocolError("Ollama sent invalid JSON.") from error

    if not isinstance(payload, dict):
        raise OllamaProtocolError("Ollama sent an invalid event.")
    if payload.get("error"):
        raise OllamaProtocolError(str(payload["error"]))

    message = payload.get("message") or {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(content, str):
        raise OllamaProtocolError("Ollama sent a non-text message chunk.")
    tool_calls: list[ToolCall] = []
    raw_tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
    if raw_tool_calls is None:
        raw_tool_calls = []
    if not isinstance(raw_tool_calls, list):
        raise OllamaProtocolError("Ollama sent invalid tool calls.")
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            raise OllamaProtocolError("Ollama sent an invalid tool call.")
        function = raw_tool_call.get("function")
        if (
            not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not function["name"].strip()
        ):
            raise OllamaProtocolError("Ollama sent an invalid tool function.")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise OllamaProtocolError("Ollama sent invalid tool arguments.") from error
        if not isinstance(arguments, dict):
            raise OllamaProtocolError("Ollama sent non-object tool arguments.")
        call_id = raw_tool_call.get("id", "")
        if not isinstance(call_id, str):
            call_id = ""
        tool_calls.append(ToolCall(function["name"], arguments, call_id))
    return StreamEvent(
        content=content,
        done=bool(payload.get("done", False)),
        tool_calls=tuple(tool_calls),
    )


def parse_stream(lines: Iterable[str | bytes]) -> Iterator[StreamEvent]:
    """Parse a sequence of Ollama NDJSON lines for tests and adapters."""

    for line in lines:
        event = parse_stream_line(line)
        if event is not None:
            yield event


class OllamaClient:
    """Small, synchronous HTTP client intended to run off the Qt UI thread."""

    display_name = "Ollama"
    default_keep_alive = "10m"
    default_timeout = 120.0
    default_connection_timeout = 10.0
    default_stream_timeout = 600.0

    def __init__(
        self,
        base_url: str,
        timeout: float = default_timeout,
        keep_alive: str | int = default_keep_alive,
        connection_timeout: float = default_connection_timeout,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.connection_timeout = connection_timeout
        self._response: HTTPResponse | None = None
        self._response_lock = threading.Lock()

    def list_models(self) -> list[str]:
        payload = self._request_json("GET", "/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise OllamaProtocolError("Ollama returned an invalid model list.")
        names: list[str] = []
        for model in models:
            if isinstance(model, dict) and isinstance(model.get("name"), str):
                names.append(model["name"])
        return names

    def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str,
        cancel_event: threading.Event | None = None,
        tools: list[dict] | None = None,
        context_size: int | None = None,
    ) -> Iterator[StreamEvent]:
        payload = {
            "model": model,
            "messages": [self._message_payload(message) for message in messages],
            "stream": True,
            "keep_alive": self.keep_alive,
        }
        if context_size:
            payload["options"] = {"num_ctx": context_size}
        if tools:
            payload["tools"] = tools
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self._url("/api/chat"),
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
            method="POST",
        )
        request_started = time.monotonic()
        try:
            response = urlopen(request, timeout=self.connection_timeout)
        except (OllamaCancelledError, OllamaProtocolError, OllamaUnavailableError):
            raise
        except HTTPError as error:
            message = self._read_http_error(error)
            if error.code == 404 or "not found" in message.lower():
                raise OllamaModelNotFoundError(message) from error
            raise OllamaProtocolError(message) from error
        except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as error:
            raise OllamaConnectionError(self._network_message(error)) from error

        with response:
            self._set_response_timeout(response, max(self.timeout, self.default_stream_timeout))
            with self._response_lock:
                self._response = response
            LOGGER.info(
                "[Ollama] Connection established in %.2fs; waiting for first token",
                time.monotonic() - request_started,
            )
            first_token_at: float | None = None
            stream_finished = False
            try:
                while True:
                    if cancel_event and cancel_event.is_set():
                        raise OllamaCancelledError()
                    try:
                        line = response.readline()
                    except (socket.timeout, TimeoutError) as error:
                        stage = (
                            "the first token"
                            if first_token_at is None
                            else "the next token"
                        )
                        raise OllamaTimeoutError(
                            f"Ollama connected but did not produce {stage} "
                            f"within {max(self.timeout, self.default_stream_timeout):g} seconds"
                        ) from error
                    except (OSError, ValueError) as error:
                        if cancel_event and cancel_event.is_set():
                            raise OllamaCancelledError() from error
                        raise OllamaUnavailableError(
                            "The Ollama connection closed unexpectedly."
                        ) from error
                    if not line:
                        break
                    event = parse_stream_line(line)
                    if event is not None:
                        if event.content and first_token_at is None:
                            first_token_at = time.monotonic()
                            LOGGER.info(
                                "[Ollama] First token received %.2fs after connection",
                                first_token_at - request_started,
                            )
                        yield event
                        if event.done:
                            stream_finished = True
                            LOGGER.info(
                                "[Ollama] Generation completed in %.2fs",
                                time.monotonic() - request_started,
                            )
                            break
                if not stream_finished:
                    raise OllamaUnavailableError(
                        "Ollama closed the response before generation completed."
                    )
            finally:
                with self._response_lock:
                    self._response = None

    @staticmethod
    def _message_payload(message: ChatMessage) -> dict:
        payload = message.as_dict()
        image_paths = payload.pop("images", [])
        if image_paths:
            encoded_images: list[str] = []
            for image_path in image_paths:
                try:
                    encoded_images.append(base64.b64encode(Path(image_path).read_bytes()).decode("ascii"))
                except (OSError, TypeError) as error:
                    raise OllamaProtocolError(f"Could not read image input: {error}") from error
            payload["images"] = encoded_images
        return payload

    def cancel_active_request(self) -> None:
        """Close the active response so a worker can unwind promptly."""

        with self._response_lock:
            response = self._response
        if response is not None:
            response.close()

    def _request_json(self, method: str, path: str) -> dict:
        request = Request(self._url(path), headers={"Accept": "application/json"}, method=method)
        response: HTTPResponse | None = None
        try:
            response = urlopen(request, timeout=self.connection_timeout)
            self._set_response_timeout(response, self.timeout)
            with self._response_lock:
                self._response = response
            raw = response.read()
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise OllamaProtocolError("Ollama returned an invalid JSON object.")
            return payload
        except HTTPError as error:
            message = self._read_http_error(error)
            raise OllamaProtocolError(message) from error
        except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as error:
            if response is None:
                raise OllamaConnectionError(self._network_message(error)) from error
            raise OllamaTimeoutError(
                f"Ollama connected but did not return {path} within {self.timeout:g} seconds"
            ) from error
        except json.JSONDecodeError as error:
            raise OllamaProtocolError("Ollama returned invalid JSON.") from error
        finally:
            with self._response_lock:
                if self._response is response:
                    self._response = None
            if response is not None:
                response.close()

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    @staticmethod
    def _set_response_timeout(response: HTTPResponse, timeout: float) -> None:
        """Use a short connect timeout without cutting off slow generation."""
        file_object = getattr(response, "fp", None)
        raw = getattr(file_object, "raw", None)
        sock = getattr(raw, "_sock", None) or getattr(file_object, "_sock", None)
        if sock is not None:
            try:
                sock.settimeout(timeout)
            except (AttributeError, OSError):
                LOGGER.debug("[Ollama] Could not update response read timeout", exc_info=True)

    @staticmethod
    def _read_http_error(error: HTTPError) -> str:
        try:
            body = error.read().decode("utf-8", errors="replace")
            payload = json.loads(body)
            if isinstance(payload, dict) and payload.get("error"):
                return str(payload["error"])
            return body or error.reason
        except (OSError, json.JSONDecodeError):
            return str(error.reason)

    @staticmethod
    def _network_message(error: Exception) -> str:
        return str(error) or "Could not connect to the configured Ollama endpoint."
