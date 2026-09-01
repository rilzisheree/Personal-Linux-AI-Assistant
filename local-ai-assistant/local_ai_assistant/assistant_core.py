"""Assistant service boundary.

The service owns the transport boundary while workers coordinate streaming
and explicit Ollama-native tool calls. Tool requests are never inferred by
parsing arbitrary assistant text.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Iterator, Protocol

from .ollama import ChatMessage, StreamEvent


class AssistantBackend(Protocol):
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
        ...

    def list_models(self) -> list[str]:
        ...

    def cancel_active_request(self) -> None:
        ...


LOGGER = logging.getLogger("lura.assistant")
DEFAULT_ROUTER_MODEL = "gemma3:270m"
ROUTER_CONTEXT_SIZE = 2048
ROUTER_OUTPUT_TOKENS = 64


@dataclass(frozen=True)
class RouteDecision:
    route: str
    response: str = ""
    function: str = ""
    arguments: dict = field(default_factory=dict)


class AssistantService:
    """Conversation-independent service used by the Qt workers."""

    def __init__(self, backend: AssistantBackend) -> None:
        self.backend = backend

    @property
    def backend_name(self) -> str:
        return str(getattr(self.backend, "display_name", "AI backend"))

    def route_request(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> RouteDecision:
        """Keep non-Ollama and legacy services on the existing model path."""
        return RouteDecision("reasoning")

    def stream_reply(
        self,
        messages: list[ChatMessage],
        model: str,
        cancel_event: threading.Event | None = None,
        tools: list[dict] | None = None,
        context_size: int | None = None,
    ) -> Iterator[StreamEvent]:
        return self.backend.stream_chat(
            messages, model, cancel_event, tools, context_size
        )

    def list_models(self) -> list[str]:
        return self.backend.list_models()

    def cancel_active_request(self) -> None:
        self.backend.cancel_active_request()


class RoutedAssistantService(AssistantService):
    """Use a small Ollama model to select the cheapest useful response path."""

    _router_prompt = """You are Lura's fast request router. Return ONLY one compact
JSON object and nothing else. Never use markdown and never explain your decision.

Allowed routes:
- simple: greetings, basic conversation, easy factual questions, and basic arithmetic.
  Include a short final answer in the response field, matching Lura's calm,
  professional personality. Address the user as Sir only when it fits naturally.
- reasoning: coding, planning, multi-step tasks, difficult questions, detailed
  explanations, or anything that needs careful reasoning.
- function: a direct local computer action. Use the exact tool name from the catalog
  and provide an arguments object.

Use this shape:
{"route":"simple","response":"..."}
{"route":"reasoning"}
{"route":"function","function":"exact_tool_name","arguments":{}}

Never invent a tool. Choose reasoning if the request is ambiguous or cannot be
handled safely by one direct route."""

    def __init__(
        self,
        backend: AssistantBackend,
        router_model: str = DEFAULT_ROUTER_MODEL,
    ) -> None:
        super().__init__(backend)
        self.router_model = router_model

    def route_request(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> RouteDecision:
        router_messages = self._router_messages(messages, tools)
        try:
            response_parts: list[str] = []
            for event in self.backend.stream_chat(
                router_messages,
                self.router_model,
                cancel_event,
                context_size=ROUTER_CONTEXT_SIZE,
                options={
                    "num_predict": ROUTER_OUTPUT_TOKENS,
                    "temperature": 0,
                },
                response_format="json",
            ):
                if event.content:
                    response_parts.append(event.content)
                if event.done:
                    break
            decision = self._parse_decision("".join(response_parts), tools)
            if decision is not None:
                LOGGER.info("[Router] Gemma selected route=%s", decision.route)
                return decision
            LOGGER.warning("[Router] Gemma returned an invalid route; using Qwen")
        except Exception as error:
            LOGGER.warning("[Router] Gemma unavailable; using Qwen: %s", error)
        return RouteDecision("reasoning")

    def _router_messages(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None,
    ) -> list[ChatMessage]:
        prompt = self._router_prompt
        catalog = self._tool_catalog(tools)
        if catalog:
            prompt += "\n\nAvailable local tools:\n" + catalog
        recent_messages = [
            ChatMessage(message.role, message.content[:600])
            for message in messages
            if message.role in {"user", "assistant"} and message.content.strip()
        ][-4:]
        return [ChatMessage("system", prompt), *recent_messages]

    @staticmethod
    def _tool_catalog(tools: list[dict] | None) -> str:
        entries: list[str] = []
        for schema in tools or []:
            function = schema.get("function", {}) if isinstance(schema, dict) else {}
            name = function.get("name")
            description = function.get("description")
            parameters = function.get("parameters", {})
            if not isinstance(name, str) or not name.strip():
                continue
            properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
            argument_names = ", ".join(properties) if isinstance(properties, dict) else ""
            suffix = f" (arguments: {argument_names})" if argument_names else ""
            entries.append(f"- {name}: {description or 'local action'}{suffix}")
        return "\n".join(entries)

    @staticmethod
    def _parse_decision(
        raw_response: str,
        tools: list[dict] | None,
    ) -> RouteDecision | None:
        text = raw_response.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        if not isinstance(payload, dict):
            return None
        route = payload.get("route")
        if route == "simple":
            response = payload.get("response")
            return (
                RouteDecision("simple", response=response.strip())
                if isinstance(response, str) and response.strip()
                else None
            )
        if route == "reasoning":
            return RouteDecision("reasoning")
        if route == "function":
            function = payload.get("function")
            arguments = payload.get("arguments", {})
            available = {
                schema.get("function", {}).get("name")
                for schema in tools or []
                if isinstance(schema, dict)
            }
            if (
                isinstance(function, str)
                and function in available
                and isinstance(arguments, dict)
            ):
                return RouteDecision("function", function=function, arguments=arguments)
        return None
