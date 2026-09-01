from __future__ import annotations

import threading
import unittest

from local_ai_assistant.assistant_core import (
    DEFAULT_ROUTER_MODEL,
    ROUTER_CONTEXT_SIZE,
    ROUTER_OUTPUT_TOKENS,
    RouteDecision,
    RoutedAssistantService,
)
from local_ai_assistant.ollama import ChatMessage, StreamEvent


class FakeBackend:
    display_name = "Fake Ollama"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    def stream_chat(
        self,
        messages,
        model,
        cancel_event=None,
        tools=None,
        context_size=None,
        options=None,
        response_format=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "tools": tools,
                "context_size": context_size,
                "options": options,
                "response_format": response_format,
            }
        )
        yield StreamEvent(self.response, True)

    def list_models(self):
        return [DEFAULT_ROUTER_MODEL, "qwen3.5:2b"]

    def cancel_active_request(self):
        return None


class RoutedAssistantServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "open_app",
                    "description": "Open a desktop application.",
                    "parameters": {
                        "type": "object",
                        "properties": {"app": {"type": "string"}},
                        "required": ["app"],
                    },
                },
            }
        ]
        self.messages = [
            ChatMessage("system", "Lura system prompt"),
            ChatMessage("user", "Open Discord."),
        ]

    def test_simple_route_uses_router_only(self) -> None:
        backend = FakeBackend('{"route":"simple","response":"Hello, Sir."}')
        service = RoutedAssistantService(backend)

        decision = service.route_request(
            self.messages,
            self.tools,
            threading.Event(),
        )

        self.assertEqual(decision, RouteDecision("simple", response="Hello, Sir."))
        self.assertEqual(len(backend.calls), 1)
        call = backend.calls[0]
        self.assertEqual(call["model"], DEFAULT_ROUTER_MODEL)
        self.assertEqual(call["context_size"], ROUTER_CONTEXT_SIZE)
        self.assertEqual(
            call["options"],
            {"num_predict": ROUTER_OUTPUT_TOKENS, "temperature": 0},
        )
        self.assertEqual(call["response_format"], "json")
        self.assertIn("open_app", call["messages"][0].content)

    def test_function_route_validates_tool_name_and_arguments(self) -> None:
        backend = FakeBackend(
            '{"route":"function","function":"open_app",'
            '"arguments":{"app":"discord"}}'
        )
        service = RoutedAssistantService(backend)

        decision = service.route_request(self.messages, self.tools)

        self.assertEqual(decision.route, "function")
        self.assertEqual(decision.function, "open_app")
        self.assertEqual(decision.arguments, {"app": "discord"})

    def test_unknown_function_falls_back_to_reasoning(self) -> None:
        backend = FakeBackend(
            '{"route":"function","function":"delete_everything","arguments":{}}'
        )
        service = RoutedAssistantService(backend)

        decision = service.route_request(self.messages, self.tools)

        self.assertEqual(decision, RouteDecision("reasoning"))

    def test_invalid_router_output_falls_back_to_reasoning(self) -> None:
        backend = FakeBackend("not json")
        service = RoutedAssistantService(backend)

        decision = service.route_request(self.messages, self.tools)

        self.assertEqual(decision, RouteDecision("reasoning"))

    def test_json_embedded_in_router_filler_is_recovered(self) -> None:
        backend = FakeBackend('{"route":"reasoning"}')
        service = RoutedAssistantService(backend)

        decision = service.route_request(self.messages, self.tools)

        self.assertEqual(decision.route, "reasoning")


if __name__ == "__main__":
    unittest.main()