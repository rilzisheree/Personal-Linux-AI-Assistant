from __future__ import annotations

import unittest
import base64
import tempfile
from pathlib import Path

from local_ai_assistant.errors import OllamaProtocolError
from local_ai_assistant.ollama import (
    ChatMessage,
    OllamaClient,
    StreamEvent,
    ToolCall,
    parse_stream,
    parse_stream_line,
)


class OllamaParserTests(unittest.TestCase):
    def test_parses_incremental_content_and_done_marker(self) -> None:
        events = list(
            parse_stream(
                [
                    b'{"model":"qwen3.5:4b","message":{"role":"assistant","content":"Hello"},"done":false}\n',
                    '{"model":"qwen3.5:4b","message":{"role":"assistant","content":" there"},"done":false}\n',
                    '{"model":"qwen3.5:4b","message":{"role":"assistant","content":""},"done":true}\n',
                ]
            )
        )
        self.assertEqual(
            events,
            [
                StreamEvent("Hello", False),
                StreamEvent(" there", False),
                StreamEvent("", True),
            ],
        )

    def test_ignores_blank_lines(self) -> None:
        self.assertIsNone(parse_stream_line("\n"))
        self.assertIsNone(parse_stream_line("   "))

    def test_surfaces_ollama_error_events(self) -> None:
        with self.assertRaisesRegex(OllamaProtocolError, "model missing"):
            parse_stream_line('{"error":"model missing"}')

    def test_rejects_malformed_events(self) -> None:
        with self.assertRaises(OllamaProtocolError):
            parse_stream_line('{"message":')

    def test_rejects_non_text_content(self) -> None:
        with self.assertRaises(OllamaProtocolError):
            parse_stream_line('{"message":{"content":42},"done":false}')

    def test_parses_native_tool_calls(self) -> None:
        event = parse_stream_line(
            '{"message":{"role":"assistant","content":"","tool_calls":['
            '{"id":"call-1","function":{"name":"open_app","arguments":{"app":"firefox"}}}'
            ']},"done":true}'
        )
        self.assertEqual(
            event,
            StreamEvent("", True, (ToolCall("open_app", {"app": "firefox"}, "call-1"),)),
        )

    def test_rejects_invalid_tool_arguments(self) -> None:
        with self.assertRaisesRegex(OllamaProtocolError, "tool arguments"):
            parse_stream_line(
                '{"message":{"tool_calls":[{"function":{"name":"open_app",'
                '"arguments":"not-json"}}]},"done":true}'
            )

    def test_rejects_tool_calls_without_a_name(self) -> None:
        with self.assertRaisesRegex(OllamaProtocolError, "tool function"):
            parse_stream_line(
                '{"message":{"tool_calls":[{"function":{"name":"",'
                '"arguments":{}}}]},"done":true}'
            )

    def test_encodes_image_paths_only_for_ollama_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "screen.png"
            image_path.write_bytes(b"png-bytes")
            payload = OllamaClient._message_payload(
                ChatMessage("tool", "Screenshot captured.", images=(str(image_path),))
            )
        self.assertEqual(payload["images"], [base64.b64encode(b"png-bytes").decode("ascii")])


if __name__ == "__main__":
    unittest.main()