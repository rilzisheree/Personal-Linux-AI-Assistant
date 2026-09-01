from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from local_ai_assistant.credentials import load_hosted_api_key, save_hosted_api_key
from local_ai_assistant.hosted_api import (
    HOSTED_CONNECTION_TIMEOUT,
    HostedApiClient,
)
from local_ai_assistant.ollama import ChatMessage, StreamEvent, ToolCall


class HostedApiTests(unittest.TestCase):
    def test_parses_streamed_text_and_done_sentinel(self) -> None:
        parts: dict[int, dict[str, str]] = {}
        first = HostedApiClient._parse_sse_line(
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            parts,
        )
        second = HostedApiClient._parse_sse_line(
            'data: {"choices":[{"delta":{"content":" there"}}]}',
            parts,
        )
        done = HostedApiClient._parse_sse_line("data: [DONE]", parts)
        self.assertEqual(first, StreamEvent("Hello"))
        self.assertEqual(second, StreamEvent(" there"))
        self.assertEqual(done, StreamEvent(done=True))

    def test_assembles_incremental_tool_call_arguments(self) -> None:
        parts: dict[int, dict[str, str]] = {}
        HostedApiClient._parse_sse_line(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"id":"call-1","function":{"name":"open_app","arguments":"{\\"app\\":"}}]}}]}',
            parts,
        )
        HostedApiClient._parse_sse_line(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"\\"firefox\\"}"}}]}}]}',
            parts,
        )
        done = HostedApiClient._parse_sse_line("data: [DONE]", parts)
        self.assertEqual(
            done,
            StreamEvent(
                done=True,
                tool_calls=(ToolCall("open_app", {"app": "firefox"}, "call-1"),),
            ),
        )

    def test_preserves_gemini_thought_signature_for_tool_calls(self) -> None:
        parts: dict[int, dict[str, str]] = {}
        HostedApiClient._parse_sse_line(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"id":"call-1","extra_content":{"google":{"thought_signature":"sig-1"}},'
            '"function":{"name":"web_search","arguments":"{\\"query\\":\\"weather\\"}"}}]}}]}',
            parts,
        )
        done = HostedApiClient._parse_sse_line("data: [DONE]", parts)
        self.assertIsNotNone(done)
        assert done is not None
        self.assertEqual(done.tool_calls[0].thought_signature, "sig-1")
        payload = HostedApiClient._message_payload(
            ChatMessage("assistant", "", done.tool_calls)
        )
        self.assertEqual(
            payload["tool_calls"][0]["extra_content"],
            {"google": {"thought_signature": "sig-1"}},
        )
        legacy_payload = HostedApiClient._message_payload(
            ChatMessage("assistant", "", (ToolCall("web_search", {}, "call-2"),)),
            gemini=True,
        )
        self.assertEqual(
            legacy_payload["tool_calls"][0]["extra_content"]["google"][
                "thought_signature"
            ],
            "skip_thought_signature_validator",
        )

    def test_stream_chat_sends_openai_compatible_payload(self) -> None:
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.readline.side_effect = [
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
            b"data: [DONE]\n",
        ]
        client = HostedApiClient("https://api.example.test/v1", "secret-key")
        with patch("local_ai_assistant.hosted_api.urlopen", return_value=response) as urlopen:
            events = list(
                client.stream_chat(
                    [ChatMessage("user", "Hello")],
                    "example-model",
                )
            )
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "example-model")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Hello"}])
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-key")
        self.assertEqual(events, [StreamEvent("ok"), StreamEvent(done=True)])

    def test_gemini_omits_multi_tool_list_for_text_chat_compatibility(self) -> None:
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.readline.side_effect = [
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
            b"data: [DONE]\n",
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Tool {name}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in ("open_app", "list_windows")
        ]
        client = HostedApiClient(
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "secret-key",
        )
        with patch("local_ai_assistant.hosted_api.urlopen", return_value=response) as urlopen:
            events = list(
                client.stream_chat(
                    [ChatMessage("user", "Hello")],
                    "gemini-3.7-flash",
                    tools=tools,
                )
            )
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertNotIn("tools", payload)
        self.assertEqual(events, [StreamEvent("ok"), StreamEvent(done=True)])

    def test_gemini_keeps_single_tool_compatibility(self) -> None:
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.readline.side_effect = [
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
            b"data: [DONE]\n",
        ]
        tool = {
            "type": "function",
            "function": {
                "name": "open_app",
                "description": "Open an application",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        client = HostedApiClient(
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "secret-key",
        )
        with patch("local_ai_assistant.hosted_api.urlopen", return_value=response) as urlopen:
            list(
                client.stream_chat(
                    [ChatMessage("user", "Open Firefox")],
                    "gemini-3.7-flash",
                    tools=[tool],
                )
            )
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["tools"], [tool])

    def test_model_check_uses_short_connection_timeout(self) -> None:
        response = Mock()
        response.read.return_value = b'{"data":[{"id":"example-model"}]}'
        response.close = Mock()
        client = HostedApiClient("https://api.example.test/v1", "secret-key")
        with patch("local_ai_assistant.hosted_api.urlopen", return_value=response) as urlopen:
            self.assertEqual(client.list_models(), ["example-model"])
        self.assertEqual(
            urlopen.call_args.kwargs["timeout"],
            HOSTED_CONNECTION_TIMEOUT,
        )

    def test_model_list_strips_google_models_prefix(self) -> None:
        response = Mock()
        response.read.return_value = b'{"data":[{"id":"models/gemini-3.6-flash"}]}'
        response.close = Mock()
        client = HostedApiClient("https://generativelanguage.googleapis.com/v1beta/openai/", "secret-key")
        with patch("local_ai_assistant.hosted_api.urlopen", return_value=response):
            self.assertEqual(client.list_models(), ["gemini-3.6-flash"])


class HostedCredentialTests(unittest.TestCase):
    def test_key_is_saved_with_restricted_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hosted-api.key"
            with patch("local_ai_assistant.credentials.HOSTED_API_KEY_PATH", path):
                save_hosted_api_key("  secret-key  ")
                with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
                    self.assertEqual(load_hosted_api_key(), "secret-key")
            mode = stat.S_IMODE(path.stat().st_mode)
        self.assertEqual(mode, 0o600)


if __name__ == "__main__":
    unittest.main()