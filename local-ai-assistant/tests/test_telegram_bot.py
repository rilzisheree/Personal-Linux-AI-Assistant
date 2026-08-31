from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from local_ai_assistant.telegram_bot import (
    TelegramConfig,
    TelegramResponseStreamer,
    _handle_message,
    _message_chunks,
)


class FakeTelegram:
    def __init__(self) -> None:
        self.actions: list[tuple[str, object]] = []

    def send_message(self, chat_id: int, text: str) -> None:
        self.actions.append(("message", (chat_id, text)))
        return {"message_id": len(self.actions)}

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        self.actions.append(("action", (chat_id, action)))

    def edit_message_text(self, chat_id: int, message_id: int, text: str) -> None:
        self.actions.append(("edit", (chat_id, message_id, text)))


class FakeAssistant:
    def __init__(self) -> None:
        self.histories: dict[int, list[object]] = {}
        self.reply_calls: list[tuple[int, str]] = []

    def reply(self, chat_id: int, content: str, on_chunk=None) -> str:
        self.reply_calls.append((chat_id, content))
        if on_chunk is not None:
            on_chunk("Local ")
            on_chunk("reply")
        return "Local reply"


class TelegramBotTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()