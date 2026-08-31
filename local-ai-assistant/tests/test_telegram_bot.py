from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest.mock import patch
from unittest.mock import Mock

from local_ai_assistant.telegram_bot import (
    LocalAssistant,
    REMOTE_TOOL_NAMES,
    TelegramClient,
    TelegramConfig,
    TelegramResponseStreamer,
    _handle_message,
    _message_chunks,
)
from local_ai_assistant.ollama import StreamEvent, ToolCall
from local_ai_assistant.tools import ToolCallResult


class FakeTelegram:
    def __init__(self) -> None:
        self.actions: list[tuple[str, object]] = []

    def send_message(self, chat_id: int, text: str) -> None:
        self.actions.append(("message", (chat_id, text)))
        return {"message_id": len(self.actions)}

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        self.actions.append(("action", (chat_id, action)))

    def send_photo(self, chat_id: int, photo_path: str, caption: str = "") -> None:
        self.actions.append(("photo", (chat_id, photo_path, caption)))

    def edit_message_text(self, chat_id: int, message_id: int, text: str) -> None:
        self.actions.append(("edit", (chat_id, message_id, text)))


class FakeAssistant:
    def __init__(self) -> None:
        self.histories: dict[int, list[object]] = {}
        self.reply_calls: list[tuple[int, str]] = []
        self.ollama = FakeOllama()

    def reply(
        self,
        chat_id: int,
        content: str,
        on_chunk=None,
        cancel_event=None,
        on_tool_image=None,
    ) -> str:
        self.reply_calls.append((chat_id, content))
        if on_chunk is not None:
            on_chunk("Local ")
            on_chunk("reply")
        return "Local reply"


class FakeOllama:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel_active_request(self) -> None:
        self.cancelled = True


class SlowAssistant(FakeAssistant):
    def reply(
        self,
        chat_id: int,
        content: str,
        on_chunk=None,
        cancel_event=None,
        on_tool_image=None,
    ) -> str:
        self.reply_calls.append((chat_id, content))
        if on_chunk is not None:
            on_chunk("Partial reply")
        while cancel_event is None or not cancel_event.is_set():
            time.sleep(0.01)
        return "Partial reply"


class ToolAwareOllama:
    def __init__(self) -> None:
        self.tool_names: list[str] = []
        self.call_count = 0

    def stream_chat(self, messages, model, cancel_event=None, tools=None, context_size=None):
        del messages, model, cancel_event, context_size
        self.call_count += 1
        self.tool_names = [
            tool["function"]["name"]
            for tool in (tools or [])
        ]
        if self.call_count == 1:
            yield StreamEvent(
                done=True,
                tool_calls=(ToolCall("close_app", {"app": "firefox"}),),
            )
        else:
            yield StreamEvent(content="Firefox closed.", done=True)


class ScreenshotOllama(ToolAwareOllama):
    def stream_chat(self, messages, model, cancel_event=None, tools=None, context_size=None):
        del messages, model, cancel_event, context_size
        self.call_count += 1
        self.tool_names = [tool["function"]["name"] for tool in (tools or [])]
        if self.call_count == 1:
            yield StreamEvent(
                done=True,
                tool_calls=(ToolCall("take_screenshot", {}),),
            )
        else:
            yield StreamEvent(content="Screenshot sent.", done=True)


class TelegramBotTests(unittest.TestCase):
    def test_remote_tools_include_controls_but_not_dangerous_operations(self) -> None:
        self.assertIn("close_app", REMOTE_TOOL_NAMES)
        self.assertIn("restart_app", REMOTE_TOOL_NAMES)
        self.assertIn("list_windows", REMOTE_TOOL_NAMES)
        self.assertIn("take_screenshot", REMOTE_TOOL_NAMES)
        self.assertIn("get_ram_usage", REMOTE_TOOL_NAMES)
        self.assertIn("read_file", REMOTE_TOOL_NAMES)
        self.assertNotIn("exec", REMOTE_TOOL_NAMES)
        self.assertNotIn("delete_file", REMOTE_TOOL_NAMES)
        self.assertNotIn("keyboard_type", REMOTE_TOOL_NAMES)

    def test_close_app_is_available_and_approved_for_telegram(self) -> None:
        config = TelegramConfig(token="not-printed", allowed_user_id=999)
        assistant = LocalAssistant(config)
        ollama = ToolAwareOllama()
        assistant.ollama = ollama

        with patch.object(
            assistant.tool_manager,
            "execute",
            return_value=unittest.mock.Mock(success=True, content="Firefox closed."),
        ) as execute:
            response = assistant.reply(456, "Close Firefox")

        self.assertEqual(response, "Firefox closed.")
        self.assertIn("close_app", ollama.tool_names)
        self.assertNotIn("exec", ollama.tool_names)
        execute.assert_called_once_with(
            "close_app",
            {"app": "firefox"},
            approved=True,
        )

    def test_screenshot_is_uploaded_and_added_to_next_model_turn(self) -> None:
        config = TelegramConfig(token="not-printed", allowed_user_id=999)
        assistant = LocalAssistant(config)
        ollama = ScreenshotOllama()
        assistant.ollama = ollama
        images: list[tuple[str, str]] = []

        with patch.object(
            assistant.tool_manager,
            "execute",
            return_value=ToolCallResult(
                True,
                "Screenshot captured.",
                ("/tmp/lura-test-screenshot.png",),
            ),
        ) as execute:
            response = assistant.reply(
                456,
                "Take a screenshot",
                on_tool_image=lambda path, caption: images.append((path, caption)),
            )

        self.assertEqual(response, "Screenshot sent.")
        self.assertIn("take_screenshot", ollama.tool_names)
        self.assertEqual(
            images,
            [("/tmp/lura-test-screenshot.png", "Screenshot captured.")],
        )
        execute.assert_called_once_with(
            "take_screenshot",
            {},
            approved=True,
        )

    def test_send_photo_uploads_multipart_image(self) -> None:
        client = TelegramClient("not-printed")
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "screen.png")
            with open(image_path, "wb") as image:
                image.write(b"png-bytes")

            with patch("local_ai_assistant.telegram_bot.urlopen") as urlopen:
                response = Mock()
                response.__enter__ = Mock(return_value=response)
                response.__exit__ = Mock(return_value=False)
                response.read.return_value = (
                    b'{"ok":true,"result":{"message_id":77}}'
                )
                urlopen.return_value = response

                result = client.send_photo(456, image_path, "Desktop screenshot")

            request = urlopen.call_args.args[0]
            self.assertEqual(result, {"message_id": 77})
            self.assertIn("multipart/form-data; boundary=", request.get_header("Content-type"))
            self.assertIn(b'name="chat_id"', request.data)
            self.assertIn(b"456", request.data)
            self.assertIn(b'name="photo"; filename="screen.png"', request.data)
            self.assertIn(b"png-bytes", request.data)
            self.assertIn(b"Desktop screenshot", request.data)

    def test_message_chunks_respect_telegram_limit(self) -> None:
        chunks = _message_chunks("x" * 8193)
        self.assertEqual([len(chunk) for chunk in chunks], [4096, 4096, 1])

    def test_config_requires_token_and_numeric_user_id(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TELEGRAM_BOT_TOKEN"):
                TelegramConfig.from_environment()

        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "not-printed",
                "TELEGRAM_ALLOWED_USER_ID": "not-a-number",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "Telegram numeric user ID"):
                TelegramConfig.from_environment()

    def test_unauthorized_and_group_messages_are_ignored(self) -> None:
        telegram = FakeTelegram()
        assistant = FakeAssistant()
        message = {
            "from": {"id": 123},
            "chat": {"id": 456, "type": "private"},
            "text": "Open Firefox",
        }
        _handle_message(telegram, assistant, 999, message)
        self.assertEqual(telegram.actions, [])
        self.assertEqual(assistant.reply_calls, [])

        message["from"] = {"id": 999}
        message["chat"] = {"id": -456, "type": "group"}
        _handle_message(telegram, assistant, 999, message)
        self.assertEqual(telegram.actions, [])
        self.assertEqual(assistant.reply_calls, [])

    def test_help_reset_and_chat_are_routed(self) -> None:
        telegram = FakeTelegram()
        assistant = FakeAssistant()
        assistant.histories[456] = ["existing context"]

        base = {"from": {"id": 999}, "chat": {"id": 456, "type": "private"}}
        _handle_message(telegram, assistant, 999, {**base, "text": "/help"})
        self.assertEqual(telegram.actions[0][0], "message")
        self.assertIn("/reset", telegram.actions[0][1][1])

        _handle_message(telegram, assistant, 999, {**base, "text": "/reset"})
        self.assertNotIn(456, assistant.histories)

        _handle_message(
            telegram,
            assistant,
            999,
            {**base, "text": "Open Firefox on my PC"},
        )
        self.assertEqual(assistant.reply_calls, [(456, "Open Firefox on my PC")])
        self.assertEqual(telegram.actions[2][0], "action")
        self.assertEqual(telegram.actions[3], ("message", (456, "Lura is thinking…")))
        self.assertEqual(telegram.actions[-1][0], "edit")
        self.assertEqual(telegram.actions[-1][1][2], "Local reply")

    def test_streamer_sends_placeholder_and_final_chunks(self) -> None:
        telegram = FakeTelegram()
        streamer = TelegramResponseStreamer(telegram, 456)

        streamer.start()
        streamer.append("A response is arriving.")
        streamer.finish("x" * 4097)

        self.assertEqual(telegram.actions[0], ("message", (456, "Lura is thinking…")))
        self.assertEqual(telegram.actions[-2][0], "edit")
        self.assertEqual(len(telegram.actions[-2][1][2]), 4096)
        self.assertEqual(telegram.actions[-1], ("message", (456, "x")))

    def test_slow_generation_is_stopped_before_reply_deadline(self) -> None:
        telegram = FakeTelegram()
        assistant = SlowAssistant()
        message = {
            "from": {"id": 999},
            "chat": {"id": 456, "type": "private"},
            "text": "Slow request",
        }

        with patch("local_ai_assistant.telegram_bot.TELEGRAM_REPLY_TIMEOUT", 0.05):
            started = time.monotonic()
            _handle_message(telegram, assistant, 999, message)
            elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.5)
        self.assertTrue(assistant.ollama.cancelled)
        self.assertIn("stopped after 0 seconds", telegram.actions[-1][1][2])


if __name__ == "__main__":
    unittest.main()