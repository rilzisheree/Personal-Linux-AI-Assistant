"""Assistant service boundary.

The service owns the transport boundary while workers coordinate streaming
and explicit Ollama-native tool calls. Tool requests are never inferred by
parsing arbitrary assistant text.
"""

from __future__ import annotations

import json
import logging
import os
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
DEFAULT_ROUTER_MODEL = os.environ.get("LURA_ROUTER_MODEL", "gemma3:270m").strip() or "gemma3:270m"
ROUTER_CONTEXT_SIZE = 2048
ROUTER_OUTPUT_TOKENS = 8
ROUTER_RESPONSE_SCHEMA = {
    "type": "string",
    "enum": ["SIMPLE", "FUNCTION", "REASONING"],
}
ROUTER_LABELS = {
    "SIMPLE": "simple",
    "FUNCTION": "function",
    "REASONING": "reasoning",
}
ROUTER_SYSTEM_PROMPT = """You are a classifier, not an assistant.
Classify REQUEST into exactly one label and output only that uppercase label:
SIMPLE, FUNCTION, or REASONING. Never answer, explain, use JSON, or add punctuation.

Choose FUNCTION first for any computer action or live computer information, or
current external information that requires a tool:
open, close, restart, screenshot, windows, CPU, memory, volume, active model, website,
weather, news, web search, currency, maps, travel, or game information.
reminders, timers, or scheduled notifications.
Otherwise choose REASONING for coding, planning, analysis, research, explanations,
debugging, generation, multiple steps, or uncertainty.
Choose SIMPLE only for an obvious greeting, thanks, goodbye, easy fact, arithmetic,
or short joke that needs no tools or meaningful reasoning.

Examples:
Hello -> SIMPLE
How are you? -> SIMPLE
Thanks -> SIMPLE
What is 2 + 2? -> SIMPLE
Tell me a joke -> SIMPLE
Open Discord -> FUNCTION
What is my CPU usage? -> FUNCTION
Take a screenshot -> FUNCTION
List my open windows -> FUNCTION
What model are you currently using? -> FUNCTION
Which model is active? -> FUNCTION
What GPU do I have? -> FUNCTION
What is the weather in Jeddah? -> FUNCTION
Convert 500 SAR to USD -> FUNCTION
What is the latest news about Minecraft? -> FUNCTION
Find directions from Jeddah to Taif -> FUNCTION
Remind me in 5 seconds to drink water -> FUNCTION
Remind me in 30 minutes -> FUNCTION
Write a Python program -> REASONING
Plan a trip -> REASONING
Explain black holes -> REASONING
Debug this error -> REASONING

REQUEST: output one label now."""


@dataclass(frozen=True)
class RouteDecision:
    route: str
    response: str = ""
    function: str = ""
    arguments: dict = field(default_factory=dict)
    used_fallback: bool = False


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
        return RouteDecision("reasoning", used_fallback=True)

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
    """Use a small Ollama model to classify the request before responding."""

    # This is a stable compatibility prompt for base and fine-tuned models. The
    # classification behavior is learned from the dataset, not from editing this
    # string for individual benchmark examples.
    _router_prompt = ROUTER_SYSTEM_PROMPT

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
                response_format=ROUTER_RESPONSE_SCHEMA,
            ):
                if event.content:
                    response_parts.append(event.content)
                if event.done:
                    break
            decision = self._parse_decision("".join(response_parts), tools)
            if decision is not None:
                LOGGER.info("[Router] Gemma selected route=%s", decision.route)
                return decision
        except Exception as error:
            LOGGER.warning("[Router] Gemma unavailable; using Qwen: %s", error)
            return RouteDecision("reasoning", used_fallback=True)
        raw_response = "".join(response_parts)
        LOGGER.warning(
            "[Router] Gemma returned an invalid route %r; using Qwen",
            raw_response[:200],
        )
        return RouteDecision("reasoning", used_fallback=True)

    def _router_messages(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None,
    ) -> list[ChatMessage]:
        del tools
        current_request = next(
            (
                message.content.strip()[:600]
                for message in reversed(messages)
                if message.role == "user" and message.content.strip()
            ),
            "",
        )
        return [
            ChatMessage("system", self._router_prompt),
            ChatMessage("user", current_request),
        ]

    @staticmethod
    def _parse_decision(
        raw_response: str,
        tools: list[dict] | None,
    ) -> RouteDecision | None:
        del tools
        text = raw_response.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = text
        if isinstance(payload, dict):
            if set(payload) != {"route"} or not isinstance(payload["route"], str):
                return None
            route = payload["route"].strip().upper()
        elif isinstance(payload, str):
            route = payload.strip().strip("`'\".,:;").upper()
        else:
            return None
        normalized_route = ROUTER_LABELS.get(route)
        return RouteDecision(normalized_route) if normalized_route else None
