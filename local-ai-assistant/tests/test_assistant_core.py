from __future__ import annotations

import threading
import unittest

from local_ai_assistant.assistant_core import (
    DEFAULT_ROUTER_MODEL,
    ROUTER_CONTEXT_SIZE,
    ROUTER_OUTPUT_TOKENS,
    ROUTER_RESPONSE_SCHEMA,
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

    def test_simple_route_uses_short_route_only(self) -> None:
        backend = FakeBackend("SIMPLE")
        service = RoutedAssistantService(backend)

        decision = service.route_request(
            self.messages,
            self.tools,
            threading.Event(),
        )

        self.assertEqual(decision, RouteDecision("simple"))
        self.assertEqual(len(backend.calls), 1)
        call = backend.calls[0]
        self.assertEqual(call["model"], DEFAULT_ROUTER_MODEL)
        self.assertEqual(call["context_size"], ROUTER_CONTEXT_SIZE)
        self.assertEqual(
            call["options"],
            {"num_predict": ROUTER_OUTPUT_TOKENS, "temperature": 0},
        )
        self.assertEqual(call["response_format"], ROUTER_RESPONSE_SCHEMA)
        self.assertNotIn("open_app", call["messages"][0].content)
        self.assertEqual(
            [message.role for message in call["messages"]],
            ["system", "user"],
        )
        self.assertEqual(call["messages"][-1].content, "Open Discord.")

    def test_function_route_does_not_select_a_tool(self) -> None:
        backend = FakeBackend("FUNCTION")
        service = RoutedAssistantService(backend)

        decision = service.route_request(self.messages, self.tools)

        self.assertEqual(decision.route, "function")
        self.assertEqual(decision, RouteDecision("function"))

    def test_invalid_router_output_falls_back_to_reasoning(self) -> None:
        backend = FakeBackend("SIMPLE because this is easy")
        service = RoutedAssistantService(backend)

        decision = service.route_request(self.messages, self.tools)

        self.assertEqual(decision, RouteDecision("reasoning", used_fallback=True))

    def test_json_encoded_route_is_accepted_for_schema_compatibility(self) -> None:
        backend = FakeBackend('"REASONING"')
        service = RoutedAssistantService(backend)

        decision = service.route_request(self.messages, self.tools)

        self.assertEqual(decision.route, "reasoning")

    def test_route_only_json_object_is_accepted_for_schema_compatibility(self) -> None:
        self.assertEqual(
            RoutedAssistantService._parse_decision(
                '{"route":"FUNCTION"}',
                self.tools,
            ),
            RouteDecision("function"),
        )

    def test_router_prompt_keeps_function_boundary_explicit(self) -> None:
        self.assertIn("computer action or live computer information", service_prompt := RoutedAssistantService._router_prompt)
        self.assertIn("CPU", service_prompt)
        self.assertIn("memory", service_prompt)

    def test_legacy_json_objects_are_rejected(self) -> None:
        tools = self.tools
        self.assertIsNone(
            RoutedAssistantService._parse_decision(
                '{"route":"simple","response":"hi","reason":"extra"}',
                tools,
            )
        )
        self.assertEqual(
            RoutedAssistantService._parse_decision(" function \n", tools),
            RouteDecision("function"),
        )
        self.assertIsNone(
            RoutedAssistantService._parse_decision(
                '{"route":"reasoning","response":"I need to think"}',
                tools,
            )
        )


if __name__ == "__main__":
    unittest.main()