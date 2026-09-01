from __future__ import annotations

import unittest

from local_ai_assistant.gemini_api import GeminiApiClient
from local_ai_assistant.ollama import ChatMessage, ToolCall


class GeminiApiTests(unittest.TestCase):
    def test_parses_text_stream_event(self) -> None:
        event = GeminiApiClient._parse_sse_line(
            'data: {"candidates":[{"content":{"parts":[{"text":"Hello Gemini"}]}}]}',
            [],
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.content, "Hello Gemini")

    def test_parses_function_call_stream_event(self) -> None:
        tool_calls = []
        event = GeminiApiClient._parse_sse_line(
            'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"open_app","args":{"app":"firefox"}}}]}}]}',
            tool_calls,
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.content, "")
        self.assertEqual(tool_calls[0].name, "open_app")
        self.assertEqual(tool_calls[0].arguments, {"app": "firefox"})

    def test_maps_roles_system_tools_and_function_declarations(self) -> None:
        system, contents = GeminiApiClient._contents(
            [
                ChatMessage("system", "Be concise."),
                ChatMessage("user", "Open Firefox."),
                ChatMessage(
                    "assistant",
                    "",
                    tool_calls=(
                        ToolCall("open_app", {"app": "firefox"}),
                    ),
                ),
                ChatMessage("tool", "Opened.", name="open_app"),
            ]
        )
        self.assertEqual(system, "Be concise.")
        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(contents[1]["role"], "model")
        self.assertIn("functionCall", contents[1]["parts"][0])
        self.assertEqual(contents[2]["role"], "user")
        self.assertIn("functionResponse", contents[2]["parts"][0])

        declarations = GeminiApiClient._function_declarations(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "open_app",
                        "description": "Open an application.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        )
        self.assertEqual(declarations[0]["name"], "open_app")
        self.assertEqual(declarations[0]["parameters"]["type"], "object")

    def test_builds_gemini_stream_url_without_leaking_key(self) -> None:
        client = GeminiApiClient("test-key")
        url = client._url("/models/gemini-3.6-flash", {"alt": "sse"})
        self.assertIn("/models/gemini-3.6-flash?", url)
        self.assertNotIn("key=test-key", url)
        self.assertEqual(client._headers("text/event-stream")["x-goog-api-key"], "test-key")


if __name__ == "__main__":
    unittest.main()