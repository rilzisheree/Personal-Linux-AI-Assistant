"""Dependency-free client for OpenAI-compatible hosted chat APIs."""

from __future__ import annotations

import base64
import json
import socket
import threading
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .ollama import ChatMessage, StreamEvent, ToolCall
from .errors import (
    AssistantAuthenticationError,
    AssistantCancelledError,
    AssistantProtocolError,
    AssistantUnavailableError,
)


class HostedApiClient:
    """Synchronous streaming client for OpenAI-compatible APIs."""

    display_name = "Hosted API"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.timeout = timeout
        self._response = None
        self._response_lock = threading.Lock()

    def list_models(self) -> list[str]:
        payload = self._request_json("GET", "/models")
        models = payload.get("data", [])
        if not isinstance(models, list):
            raise AssistantProtocolError("Hosted API returned an invalid model list.")
        names = [
            model["id"]
            for model in models
            if isinstance(model, dict) and isinstance(model.get("id"), str)
        ]
        return names

    def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str,
        cancel_event: threading.Event | None = None,
        tools: list[dict] | None = None,
        context_size: int | None = None,
    ) -> Iterator[StreamEvent]:
        del context_size
        if not self.api_key:
            raise AssistantAuthenticationError(
                "No hosted API key is configured. Add one in Settings."
            )
        payload = {
            "model": model,
            "messages": [self._message_payload(message) for message in messages],
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        request = Request(
            self._url("/chat/completions"),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers("text/event-stream"),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                with self._response_lock:
                    self._response = response
                try:
                    tool_call_parts: dict[int, dict[str, str]] = {}
                    while True:
                        if cancel_event and cancel_event.is_set():
                            raise AssistantCancelledError()
                        line = response.readline()
                        if not line:
                            break
                        event = self._parse_sse_line(line, tool_call_parts)
                        if event is not None:
                            yield event
                            if event.done:
                                break
                finally:
                    with self._response_lock:
                        self._response = None
        except (
            AssistantCancelledError,
            AssistantProtocolError,
            AssistantUnavailableError,
            AssistantAuthenticationError,
        ):
            raise
        except HTTPError as error:
            message = self._read_http_error(error)
            if error.code in {401, 403}:
                raise AssistantAuthenticationError(message) from error
            raise AssistantProtocolError(message) from error
        except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as error:
            raise AssistantUnavailableError(str(error) or "Could not reach the hosted API.") from error

    def cancel_active_request(self) -> None:
        with self._response_lock:
            response = self._response
        if response is not None:
            response.close()

    def _request_json(self, method: str, path: str) -> dict:
        if not self.api_key:
            raise AssistantAuthenticationError(
                "No hosted API key is configured. Add one in Settings."
            )
        request = Request(self._url(path), headers=self._headers("application/json"), method=method)
        response = None
        try:
            response = urlopen(request, timeout=self.timeout)
            with self._response_lock:
                self._response = response
            payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise AssistantProtocolError("Hosted API returned invalid JSON.")
            return payload
        except (
            AssistantAuthenticationError,
            AssistantProtocolError,
            AssistantUnavailableError,
        ):
            raise
        except HTTPError as error:
            message = self._read_http_error(error)
            if error.code in {401, 403}:
                raise AssistantAuthenticationError(message) from error
            raise AssistantProtocolError(message) from error
        except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as error:
            raise AssistantUnavailableError(str(error) or "Could not reach the hosted API.") from error
        except json.JSONDecodeError as error:
            raise AssistantProtocolError("Hosted API returned invalid JSON.") from error
        finally:
            with self._response_lock:
                if self._response is response:
                    self._response = None
            if response is not None:
                response.close()

    def _headers(self, accept: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": accept,
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    @staticmethod
    def _message_payload(message: ChatMessage) -> dict:
        payload: dict = {"role": message.role, "content": message.content}
        if message.name:
            payload["name"] = message.name
        if message.role == "tool" and message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for index, call in enumerate(message.tool_calls)
            ]
        if message.images:
            content: list[dict] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            for image_path in message.images:
                try:
                    encoded = base64.b64encode(Path(image_path).read_bytes()).decode(
                        "ascii"
                    )
                except (OSError, TypeError) as error:
                    raise AssistantProtocolError(
                        f"Could not read image input: {error}"
                    ) from error
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    }
                )
            payload["content"] = content
        return payload

    @staticmethod
    def _parse_sse_line(
        line: str | bytes,
        tool_call_parts: dict[int, dict[str, str]],
    ) -> StreamEvent | None:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line or not line.startswith("data:"):
            return None
        data = line[5:].strip()
        if data == "[DONE]":
            calls = HostedApiClient._tool_calls(tool_call_parts)
            return StreamEvent(done=True, tool_calls=calls)
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as error:
            raise AssistantProtocolError("Hosted API sent invalid SSE JSON.") from error
        if payload.get("error"):
            error = payload["error"]
            message = error.get("message") if isinstance(error, dict) else error
            raise AssistantProtocolError(str(message))
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        if not isinstance(choice, dict):
            raise AssistantProtocolError("Hosted API sent an invalid choice.")
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise AssistantProtocolError("Hosted API sent an invalid delta.")
        content = delta.get("content", "")
        if not isinstance(content, str):
            content = ""
        for raw_call in delta.get("tool_calls", []) or []:
            if not isinstance(raw_call, dict):
                continue
            index = raw_call.get("index", 0)
            if not isinstance(index, int):
                index = 0
            part = tool_call_parts.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if isinstance(raw_call.get("id"), str):
                part["id"] = raw_call["id"]
            function = raw_call.get("function") or {}
            if isinstance(function, dict):
                if isinstance(function.get("name"), str):
                    part["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    part["arguments"] += function["arguments"]
        # The final [DONE] sentinel is the single completion boundary. A
        # finish_reason can arrive before the last tool-call fragments.
        return StreamEvent(content=content)

    @staticmethod
    def _tool_calls(parts: dict[int, dict[str, str]]) -> tuple[ToolCall, ...]:
        calls: list[ToolCall] = []
        for index in sorted(parts):
            part = parts[index]
            try:
                arguments = json.loads(part["arguments"] or "{}")
            except json.JSONDecodeError as error:
                raise AssistantProtocolError("Hosted API sent invalid tool arguments.") from error
            if not part["name"]:
                continue
            if not isinstance(arguments, dict):
                raise AssistantProtocolError("Hosted API sent non-object tool arguments.")
            calls.append(ToolCall(part["name"], arguments, part["id"]))
        return tuple(calls)

    @staticmethod
    def _read_http_error(error: HTTPError) -> str:
        try:
            body = error.read().decode("utf-8", errors="replace")
            payload = json.loads(body)
            if isinstance(payload, dict):
                detail = payload.get("error")
                if isinstance(detail, dict) and detail.get("message"):
                    return str(detail["message"])
                if detail:
                    return str(detail)
            return body or str(error.reason)
        except (OSError, json.JSONDecodeError):
            return str(error.reason)