"""Direct Ollama HTTP adapter with newline-delimited JSON streaming."""

from __future__ import annotations

import base64
import json
import logging
import re
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
    thinking: str = ""
    metrics: dict[str, int | float] | None = None
    done_reason: str = ""


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
    thinking = message.get("thinking", "") if isinstance(message, dict) else ""
    if not isinstance(thinking, str):
        raise OllamaProtocolError("Ollama sent a non-text thinking chunk.")
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
    metrics: dict[str, int | float] = {}
    for name in (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ):
        value = payload.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[name] = value
    done_reason = payload.get("done_reason", "")
    if done_reason is None:
        done_reason = ""
    if not isinstance(done_reason, str):
        raise OllamaProtocolError("Ollama sent an invalid completion reason.")
    return StreamEvent(
        content=content,
        done=bool(payload.get("done", False)),
        tool_calls=tuple(tool_calls),
        thinking=thinking,
        metrics=metrics or None,
        done_reason=done_reason,
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
    _thinking_boolean_models = (
        "qwen3",
        "qwen3.5",
        "qwq",
        "deepseek-r1",
        "deepseek-v3",
    )
    _thinking_level_models = ("gpt-oss",)
    _complexity_markers = re.compile(
        r"\b("
        r"analy[sz]e|compare|contrast|debug|derive|design|"
        r"evaluate|explain|implement|in detail|plan|prove|"
        r"research|step by step|troubleshoot|why|write|draft|code"
        r")\b",
        re.IGNORECASE,
    )

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
        options: dict[str, int | float | str | bool] | None = None,
        response_format: str | dict | None = None,
    ) -> Iterator[StreamEvent]:
        payload = {
            "model": model,
            "messages": [self._message_payload(message) for message in messages],
            "stream": True,
            "keep_alive": self.keep_alive,
        }
        LOGGER.info(
            "[STREAM] request started model=%s tools=%d",
            model,
            len(tools or []),
        )
        thinking_option = self._thinking_option(model, messages)
        if thinking_option is not None:
            payload["think"] = thinking_option
        request_options = dict(options or {})
        if context_size:
            request_options["num_ctx"] = context_size
        if request_options:
            payload["options"] = request_options
        if response_format is not None:
            payload["format"] = response_format
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
            chunk_count = 0
            thinking_characters = 0
            output_characters = 0
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
                        chunk_count += 1
                        if (event.content or event.thinking) and first_token_at is None:
                            first_token_at = time.monotonic()
                            LOGGER.info(
                                "[Ollama] First model token received %.2fs after connection",
                                first_token_at - request_started,
                            )
                        thinking_characters += len(event.thinking)
                        output_characters += len(event.content)
                        LOGGER.info(
                            "[STREAM] chunk received #%d content_chars=%d thinking_chars=%d tool_calls=%d",
                            chunk_count,
                            len(event.content),
                            len(event.thinking),
                            len(event.tool_calls),
                        )
                        if event.done:
                            stream_finished = True
                            LOGGER.info(
                                "[STREAM] Ollama finished chunks=%d characters=%d completion_reason=%s done_reason=%s",
                                chunk_count,
                                output_characters,
                                self._completion_reason(event.done_reason),
                                event.done_reason or "unspecified",
                            )
                            self._log_generation_metrics(
                                request_started,
                                first_token_at,
                                thinking_characters,
                                output_characters,
                                event.metrics or {},
                            )
                        yield event
                        if event.done:
                            break
                if not stream_finished:
                    raise OllamaUnavailableError(
                        "Ollama closed the response before generation completed."
                    )
            except OllamaCancelledError:
                LOGGER.info(
                    "[STREAM] CANCELLED_BY_USER chunks_received=%d characters_received=%d",
                    chunk_count,
                    output_characters,
                )
                raise
            except Exception as error:
                LOGGER.error(
                    "[STREAM] ERROR error_type=%s chunks_received=%d characters_received=%d",
                    type(error).__name__,
                    chunk_count,
                    output_characters,
                )
                raise
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

    @classmethod
    def _thinking_option(
        cls,
        model: str,
        messages: list[ChatMessage],
    ) -> bool | str | None:
        """Choose Ollama's real reasoning mode for the current request.

        Ollama enables thinking by default for supported models when omitted.
        Routine requests explicitly disable it; complex requests leave it on
        so the model can still reason when that is actually useful.
        """
        normalized_model = model.strip().casefold()
        if normalized_model.startswith(cls._thinking_boolean_models):
            if cls._is_simple_request(messages):
                return False
            return True
        if normalized_model.startswith(cls._thinking_level_models):
            if cls._is_simple_request(messages):
                return "low"
        return None

    @classmethod
    def _is_simple_request(cls, messages: list[ChatMessage]) -> bool:
        user_prompt = next(
            (
                message.content.strip()
                for message in reversed(messages)
                if message.role == "user" and message.content.strip()
            ),
            "",
        )
        if not user_prompt:
            return True
        normalized = re.sub(r"\s+", " ", user_prompt)
        if len(normalized) > 220 or normalized.count(".") + normalized.count("?") > 1:
            return False
        return cls._complexity_markers.search(normalized) is None

    @staticmethod
    def _completion_reason(done_reason: str) -> str:
        normalized = done_reason.strip().casefold()
        if normalized in {"length", "max_tokens", "token_limit"}:
            return "MODEL_REACHED_TOKEN_LIMIT"
        if normalized in {"", "stop", "eos", "end_turn"}:
            return "NORMAL_COMPLETION"
        return normalized.upper()

    @staticmethod
    def _log_generation_metrics(
        request_started: float,
        first_token_at: float | None,
        thinking_characters: int,
        output_characters: int,
        metrics: dict[str, int | float],
    ) -> None:
        elapsed = time.monotonic() - request_started
        eval_duration_ns = metrics.get("eval_duration")
        generation_seconds = (
            float(eval_duration_ns) / 1_000_000_000
            if isinstance(eval_duration_ns, (int, float)) and eval_duration_ns > 0
            else elapsed
        )
        generated_tokens = metrics.get("eval_count")
        tokens_per_second = (
            float(generated_tokens) / generation_seconds
            if isinstance(generated_tokens, (int, float)) and generation_seconds > 0
            else None
        )
        summary = {
            "time_to_first_token_seconds": (
                round(first_token_at - request_started, 3)
                if first_token_at is not None
                else None
            ),
            "reasoning_characters": thinking_characters,
            "reasoning_tokens_estimate": round(thinking_characters / 4)
            if thinking_characters
            else 0,
            "output_characters": output_characters,
            "output_tokens_estimate": round(output_characters / 4)
            if output_characters
            else 0,
            "generated_tokens": generated_tokens,
            "tokens_per_second": (
                round(tokens_per_second, 2)
                if tokens_per_second is not None
                else None
            ),
            "generation_seconds": round(generation_seconds, 3),
            "total_elapsed_seconds": round(elapsed, 3),
        }
        LOGGER.info("[Ollama] GENERATION_METRICS %s", json.dumps(summary, sort_keys=True))

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
