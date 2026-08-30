from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_ai_assistant.conversations import Conversation, ConversationStore, title_for_messages
from local_ai_assistant.ollama import ChatMessage, ToolCall


class ConversationTests(unittest.TestCase):
    def test_title_uses_first_user_message_and_truncates(self) -> None:
        messages = [
            ChatMessage("assistant", "Welcome"),
            ChatMessage("user", "  Help   me understand local models and their context windows.  "),
        ]
        self.assertEqual(
            title_for_messages(messages),
            "Help me understand local models and their…",
        )

    def test_empty_conversation_has_default_title(self) -> None:
        self.assertEqual(title_for_messages([]), "New chat")

    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "history.json")
            conversation = Conversation.create()
            conversation.update_messages(
                [ChatMessage("user", "Hello"), ChatMessage("assistant", "Hi there")]
            )
            store.save([conversation])
            loaded = store.load()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].id, conversation.id)
        self.assertEqual(loaded[0].title, "Hello")
        self.assertEqual(loaded[0].messages, conversation.messages)

    def test_damaged_history_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(ConversationStore(path).load(), [])

    def test_invalid_conversation_message_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text(
                '{"conversations":['
                '{"id":"broken","messages":[{"role":"user"}]},'
                '{"id":"valid","messages":[{"role":"user","content":"Keep me"}]}'
                ']}',
                encoding="utf-8",
            )
            loaded = ConversationStore(path).load()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].id, "valid")

    def test_tool_call_messages_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "history.json")
            conversation = Conversation.create()
            conversation.update_messages(
                [
                    ChatMessage(
                        "assistant",
                        "",
                        (ToolCall("open_app", {"app": "firefox"}, "call-1"),),
                    ),
                    ChatMessage("tool", "firefox opened.", name="open_app"),
                ]
            )
            store.save([conversation])
            loaded = store.load()

        self.assertEqual(loaded[0].messages, conversation.messages)


if __name__ == "__main__":
    unittest.main()