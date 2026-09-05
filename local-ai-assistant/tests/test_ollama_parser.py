from __future__ import annotations

import unittest
import base64
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from local_ai_assistant.errors import (
    OllamaConnectionError,
    OllamaProtocolError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from local_ai_assistant.ollama import (
    ChatMessage,
    OllamaClient,
    StreamEvent,
    ToolCall,
    parse_stream,
    parse_stream_line,
)
from local_ai_assistant.tools import ToolManager


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

    def test_parses_thinking_chunks_and_generation_metrics(self) -> None:
        event = parse_stream_line(
            '{"message":{"role":"assistant","thinking":"short plan"},'
            '"done":true,"done_reason":"stop","eval_count":12,"eval_duration":2000000000,'
            '"total_duration":2500000000}'
        )
        self.assertEqual(event.thinking, "short plan")
        self.assertEqual(event.done_reason, "stop")
        self.assertEqual(
            event.metrics,
            {
                "eval_count": 12,
                "eval_duration": 2000000000,
                "total_duration": 2500000000,
            },
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

    def test_context_size_is_sent_as_ollama_option(self) -> None:
        client = OllamaClient("http://localhost:11434")
        self.assertEqual(client.timeout, 120.0)
        with patch("local_ai_assistant.ollama.urlopen") as urlopen:
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.readline.side_effect = [
                b'{"message":{"content":"ok"},"done":true}\n',
            ]
            urlopen.return_value = response
            list(client.stream_chat([], "qwen3.5:4b", context_size=8192))
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(
            payload["options"],
            {"num_predict": -1, "num_ctx": 8192},
        )
        self.assertFalse(payload["think"])
        self.assertEqual(payload["keep_alive"], "10m")

    def test_stream_chat_sends_the_reminder_tool_schema_to_ollama(self) -> None:
        client = OllamaClient("http://localhost:11434")
        reminder_schema = next(
            tool
            for tool in ToolManager().definitions_for_ollama()
            if tool["function"]["name"] == "create_reminder"
        )
        with patch("local_ai_assistant.ollama.urlopen") as urlopen:
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.readline.side_effect = [
                b'{"message":{"content":"ok"},"done":true}\n',
            ]
            urlopen.return_value = response
            list(
                client.stream_chat(
                    [ChatMessage("user", "Remind me in 5 minutes to stretch.")],
                    "qwen3.5:4b",
                    tools=[reminder_schema],
                )
            )

        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertIn("tools", payload)
        self.assertEqual(payload["tools"], [reminder_schema])

    def test_stream_chat_accumulates_chunks_and_logs_normal_completion(self) -> None:
        client = OllamaClient("http://localhost:11434")
        with patch("local_ai_assistant.ollama.urlopen") as urlopen:
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.readline.side_effect = [
                b'{"message":{"content":"first "},"done":false}\n',
                b'{"message":{"content":"second"},"done":false}\n',
                b'{"message":{"content":""},"done":true,"done_reason":"stop"}\n',
            ]
            urlopen.return_value = response
            with self.assertLogs("lura.ollama", level="INFO") as logs:
                events = list(client.stream_chat([], "qwen3.5:4b"))

        self.assertEqual("".join(event.content for event in events), "first second")
        self.assertTrue(events[-1].done)
        output = "\n".join(logs.output)
        self.assertIn("[STREAM] chunk received #1", output)
        self.assertIn("[STREAM] chunk received #3", output)
        self.assertIn("completion_reason=NORMAL_COMPLETION", output)

    def test_stream_chat_reports_model_token_limit_without_calling_it_a_stream_error(self) -> None:
        client = OllamaClient("http://localhost:11434")
        with patch("local_ai_assistant.ollama.urlopen") as urlopen:
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.readline.side_effect = [
                b'{"message":{"content":"partial"},"done":false}\n',
                b'{"message":{"content":""},"done":true,"done_reason":"length"}\n',
            ]
            urlopen.return_value = response
            with self.assertLogs("lura.ollama", level="INFO") as logs:
                events = list(client.stream_chat([], "qwen3.5:4b"))

        self.assertEqual("".join(event.content for event in events), "partial")
        output = "\n".join(logs.output)
        self.assertIn("completion_reason=MODEL_REACHED_TOKEN_LIMIT", output)
        self.assertNotIn("[STREAM] ERROR", output)

    def test_chat_once_uses_stream_false_and_preserves_completion_metadata(self) -> None:
        client = OllamaClient("http://localhost:11434")
        with patch("local_ai_assistant.ollama.urlopen") as urlopen:
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.read.return_value = (
                b'{"message":{"content":"complete answer"},'
                b'"done":true,"done_reason":"stop","eval_count":23}'
            )
            urlopen.return_value = response

            event = client.chat_once(
                [ChatMessage("user", "Explain Linux in detail.")],
                "qwen3.5:4b",
                context_size=8192,
                options={"temperature": 0.2},
            )

        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertFalse(payload["stream"])
        self.assertEqual(
            payload["options"],
            {"num_predict": -1, "num_ctx": 8192, "temperature": 0.2},
        )
        self.assertEqual(event.content, "complete answer")
        self.assertTrue(event.done)
        self.assertEqual(event.done_reason, "stop")
        self.assertEqual(event.metrics, {"eval_count": 23})

    def test_stream_chat_rejects_eof_before_done_and_logs_stream_error(self) -> None:
        client = OllamaClient("http://localhost:11434")
        with patch("local_ai_assistant.ollama.urlopen") as urlopen:
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.readline.side_effect = [
                b'{"message":{"content":"partial"},"done":false}\n',
                b"",
            ]
            urlopen.return_value = response
            with self.assertLogs("lura.ollama", level="ERROR") as logs:
                with self.assertRaisesRegex(
                    OllamaUnavailableError,
                    "closed the response",
                ):
                    list(client.stream_chat([], "qwen3.5:4b"))

        output = "\n".join(logs.output)
        self.assertIn("error_type=OllamaUnavailableError", output)
        self.assertIn("chunks_received=1", output)
        self.assertIn("characters_received=7", output)

    def test_complex_qwen_request_keeps_reasoning_available(self) -> None:
        client = OllamaClient("http://localhost:11434")
        with patch("local_ai_assistant.ollama.urlopen") as urlopen:
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.readline.side_effect = [
                b'{"message":{"content":"answer"},"done":true}\n',
            ]
            urlopen.return_value = response
            list(
                client.stream_chat(
                    [ChatMessage("user", "Explain why this architecture is resilient.")],
                    "qwen3.5:2b",
                )
            )
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertTrue(payload["think"])
        self.assertEqual(payload["options"]["num_predict"], -1)

    def test_router_request_can_set_deterministic_json_options(self) -> None:
        client = OllamaClient("http://localhost:11434")
        with patch("local_ai_assistant.ollama.urlopen") as urlopen:
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.readline.side_effect = [
                b'{"message":{"content":"{\\"route\\":\\"reasoning\\"}"},'
                b'"done":true}\n',
            ]
            urlopen.return_value = response
            list(
                client.stream_chat(
                    [ChatMessage("user", "Route this.")],
                    "gemma3:270m",
                    context_size=2048,
                    options={"num_predict": 64, "temperature": 0},
                    response_format="json",
                )
            )
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["model"], "gemma3:270m")
        self.assertEqual(payload["format"], "json")
        self.assertEqual(
            payload["options"],
            {"num_ctx": 2048, "num_predict": 64, "temperature": 0},
        )

    def test_connection_failure_is_distinguished_from_slow_generation(self) -> None:
        client = OllamaClient("http://localhost:11434")
        with patch(
            "local_ai_assistant.ollama.urlopen",
            side_effect=ConnectionRefusedError("connection refused"),
        ):
            with self.assertRaises(OllamaConnectionError):
                list(client.stream_chat([], "qwen3.5:4b"))

        with patch("local_ai_assistant.ollama.urlopen") as urlopen:
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.readline.side_effect = TimeoutError("read timed out")
            urlopen.return_value = response
            with self.assertRaises(OllamaTimeoutError):
                list(client.stream_chat([], "qwen3.5:4b"))


if __name__ == "__main__":
    unittest.main()