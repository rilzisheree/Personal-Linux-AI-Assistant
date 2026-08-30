from __future__ import annotations

import unittest

from local_ai_assistant.errors import OllamaProtocolError
from local_ai_assistant.ollama import StreamEvent, parse_stream, parse_stream_line


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


if __name__ == "__main__":
    unittest.main()