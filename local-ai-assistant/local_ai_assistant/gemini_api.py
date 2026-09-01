"""Dependency-free streaming client for Google's Gemini text API."""

from __future__ import annotations

import base64
import json
import socket
import threading
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .errors import (
    AssistantAuthenticationError,
    AssistantBackendError,
    AssistantCancelledError,
    AssistantProtocolError,
    AssistantUnavailableError,
)
from .ollama import ChatMessage, StreamEvent, ToolCall


GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_CONNECTION_TIMEOUT = 15.0
GEMINI_GENERATION_TIMEOUT = 180.0


class GeminiApiClient:
    """Synchronous Gemini REST client intended to run off the UI thread."""

    display_name = "Gemini"

    def __init__(
        self,
        api_key: str,
        timeout: float = GEMINI_GENERATION_TIMEOUT,
        base_url: str = GEMINI_API_BASE_URL,
    ) -> None:
        self.api_key = api_key.strip()
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self._response = None
        self._response_lock = threading.Lock()

    def list_models(self) -> list[str]:
        if not self.api_key:
            raise AssistantAuthenticationError(
                "No Gemini API key is configured. Add GEMINI_API_KEY or add one in Settings."
            )
        request = Request(
            self._url("/models"),
            headers=self._headers("application/json"),
            method="GET",
        )
        response = None
        try:
            response = urlopen(request, timeout=min(self.timeout, GEMINI_CONNECTION_TIMEOUT))
            with self._response_lock:
                self._response = response
            payload = json.loads(response.read().decode("utf-8"))
            models = payload.get("models", [])
            if not isinstance(models, list):
                raise AssistantProtocolError("Gemini returned an invalid model list.")
            names: list[str] = []
            for item in models:
                if not isinstance(item, dict):
                    continue
                methods = item.get("supportedGenerationMethods", [])
                name = item.get("name")
                if (
                    isinstance(name, str)
                    and isinstance(methods, list)
                    and "generateContent" in methods
                ):
                    names.append(name.removeprefix("models/"))
            return names
        except (
            AssistantAuthenticationError,
            AssistantProtocolError,
            AssistantUnavailableError,
        ):
            raise
        except HTTPError as error:
            raise self._http_error(error) from error
        except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as error:
            raise AssistantUnavailableError(
                self._timeout_message(error, "Gemini model discovery")
            ) from error
        except json.JSONDecodeError as error:
            raise AssistantProtocolError("Gemini returned invalid JSON.") from error
        finally:
            with self._response_lock:
                if self._response is response:
                    self._response = None
            if response is not None:
                response.close()

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
                "No Gemini API key is configured. Add GEMINI_API_KEY or add one in Settings."
            )
        system_instruction, contents = self._contents(messages)
        payload: dict = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}],
            }
        if tools:
            payload["tools"] = [{"functionDeclarations": self._function_declarations(tools)}]
        request = Request(
            self._url(
                f"/models/{quote(model.strip(), safe='')}:streamGenerateContent",
                {"alt": "sse"},
            ),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers("text/event-stream"),
            method="POST",
        )
        response = None
        try:
            response = urlopen(request, timeout=self.timeout)
            with self._response_lock:
                self._response = response
            tool_calls: list[ToolCall] = []
            while True:
                if cancel_event and cancel_event.is_set():
                    raise AssistantCancelledError()
                line = response.readline()
                if not line:
                    break
                event = self._parse_sse_line(line, tool_calls)
                if event is not None:
                    if event.content:
                        yield event
            yield StreamEvent(done=True, tool_calls=tuple(tool_calls))
        except (
            AssistantCancelledError,
            AssistantProtocolError,
            AssistantUnavailableError,
            AssistantAuthenticationError,
        ):
            raise
        except HTTPError as error:
            raise self._http_error(error) from error
        except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as error:
            raise AssistantUnavailableError(
                self._timeout_message(error, "Gemini generation")
            ) from error
        finally:
            with self._response_lock:
                self._response = None
            if response is not None:
                response.close()

    def cancel_active_request(self) -> None:
        with self._response_lock:
            response = self._response
        if response is not None:
            response.close()

    @staticmethod
    def _contents(messages: list[ChatMessage]) -> tuple[str, list[dict]]:
        system_parts: list[str] = []
        contents: list[dict] = []
        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue
            if message.role == "tool":
                parts = [
                    {
                        "functionResponse": {
                            "name": message.name,
                            "response": {"content": message.content},
                        }
                    }
                ]
                contents.append({"role": "user", "parts": parts})
                continue

            parts: list[dict] = []
            if message.content:
                parts.append({"text": message.content})
            for image_path in message.images:
                try:
                    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
                except (OSError, TypeError) as error:
                    raise AssistantProtocolError(
                        f"Could not read image input: {error}"
                    ) from error
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": encoded,
                        }
                    }
                )
            for tool_call in message.tool_calls:
                function_call = {
                    "name": tool_call.name,
                    "args": tool_call.arguments,
                }
                function_call_part = {"functionCall": function_call}
                if tool_call.thought_signature:
                    # Gemini requires the signature returned with a thought-enabled
                    # function call to be replayed on the matching model turn. It
                    # belongs to the Part, alongside functionCall, not inside it.
                    function_call_part["thoughtSignature"] = tool_call.thought_signature
                parts.append(function_call_part)
            if parts:
                role = "model" if message.role == "assistant" else "user"
                contents.append({"role": role, "parts": parts})
        return "\n\n".join(system_parts), contents

    @staticmethod
    def _function_declarations(tools: list[dict]) -> list[dict]:
        declarations: list[dict] = []
        for tool in tools:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            declaration = {
                "name": name,
                "description": str(function.get("description", "")),
                "parameters": function.get(
                    "parameters",
                    {"type": "object", "properties": {}},
                ),
            }
            declarations.append(declaration)
        return declarations

    @staticmethod
    def _parse_sse_line(
        line: str | bytes,
        tool_calls: list[ToolCall],
    ) -> StreamEvent | None:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line or not line.startswith("data:"):
            return None
        try:
            payload = json.loads(line[5:].strip())
        except json.JSONDecodeError as error:
            raise AssistantProtocolError("Gemini sent invalid SSE JSON.") from error
        if not isinstance(payload, dict):
            raise AssistantProtocolError("Gemini sent an invalid stream event.")
        if payload.get("error"):
            error = payload["error"]
            message = error.get("message") if isinstance(error, dict) else error
            raise AssistantProtocolError(str(message))
        content = payload.get("candidates", [])
        if not isinstance(content, list):
            raise AssistantProtocolError("Gemini sent invalid candidates.")
        text_parts: list[str] = []
        for candidate in content:
            if not isinstance(candidate, dict):
                continue
            response_content = candidate.get("content", {})
            parts = response_content.get("parts", []) if isinstance(response_content, dict) else []
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
                function_call = part.get("functionCall")
                if isinstance(function_call, dict):
                    name = function_call.get("name")
                    arguments = function_call.get("args", {})
                    if (
                        isinstance(name, str)
                        and name.strip()
                        and isinstance(arguments, dict)
                    ):
                        thought_signature = part.get("thoughtSignature", "")
                        if not isinstance(thought_signature, str):
                            thought_signature = ""
                        tool_calls.append(
                            ToolCall(
                                name,
                                arguments,
                                thought_signature=thought_signature,
                            )
                        )
        return StreamEvent(content="".join(text_parts))

    def _headers(self, accept: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": accept,
            "x-goog-api-key": self.api_key,
        }

    def _url(self, path: str, query: dict[str, str] | None = None) -> str:
        url = f"{self.base_url}/{path.lstrip('/')}"
        return f"{url}?{urlencode(query)}" if query else url

    @staticmethod
    def _http_error(error: HTTPError) -> AssistantBackendError:
        try:
            raw = error.read().decode("utf-8", errors="replace")
            payload = json.loads(raw)
            detail = payload.get("error", {}).get("message", raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            detail = str(error)
        message = str(detail).strip() or "Gemini rejected the request."
        if error.code in {401, 403}:
            return AssistantAuthenticationError(message)
        return AssistantProtocolError(message)

    def _timeout_message(self, error: Exception, operation: str) -> str:
        detail = str(error).strip()
        if "timed out" in detail.casefold() or isinstance(error, (TimeoutError, socket.timeout)):
            return f"Timed out waiting for {operation} after {self.timeout:g} seconds."
        return detail or "Could not reach Gemini."