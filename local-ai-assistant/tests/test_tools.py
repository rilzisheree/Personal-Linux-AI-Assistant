from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_ai_assistant.tools import (
    PermissionLevel,
    ToolConfirmationRequired,
    ToolManager,
)


class ToolManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ToolManager()

    def test_exposes_initial_phase_tools_to_ollama(self) -> None:
        names = {
            tool["function"]["name"]
            for tool in self.manager.definitions_for_ollama()
        }
        self.assertIn("open_app", names)
        self.assertIn("exec", names)
        self.assertIn("get_ram_usage", names)
        self.assertIn("get_gpu_usage", names)
        for name in (
            "list_windows",
            "focus_window",
            "move_window",
            "resize_window",
            "close_window",
            "take_screenshot",
            "search_files",
            "read_file",
            "write_file",
            "create_file",
            "delete_file",
            "move_file",
            "copy_file",
            "mouse_move",
            "mouse_click",
            "keyboard_type",
            "keyboard_press",
        ):
            self.assertIn(name, names)

    def test_safe_system_tool_does_not_need_approval(self) -> None:
        result = self.manager.execute("get_disk_usage", {})
        self.assertIsInstance(result.success, bool)
        self.assertTrue(result.content)

    def test_confirmation_required_tools_are_gated(self) -> None:
        self.assertEqual(
            self.manager.permission_for("exec", {"command": "printf ok"}),
            PermissionLevel.CONFIRMATION_REQUIRED,
        )
        with self.assertRaises(ToolConfirmationRequired):
            self.manager.execute("exec", {"command": "printf should-not-run"})

    def test_destructive_commands_are_classified_as_dangerous(self) -> None:
        self.assertEqual(
            self.manager.permission_for("exec", {"command": "rm -rf ~/Downloads/test"}),
            PermissionLevel.DANGEROUS,
        )

    def test_confirmed_command_returns_output(self) -> None:
        result = self.manager.execute("exec", {"command": "printf tool-ok"}, approved=True)
        self.assertTrue(result.success)
        self.assertEqual(result.content, "tool-ok")

    def test_file_tools_read_and_create_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "note.txt"
            created = self.manager.execute(
                "create_file",
                {"path": str(path), "content": "hello"},
                approved=True,
            )
            self.assertTrue(created.success)
            self.assertEqual(
                self.manager.execute("read_file", {"path": str(path)}).content,
                "hello",
            )
            duplicate = self.manager.execute(
                "create_file",
                {"path": str(path), "content": "changed"},
                approved=True,
            )
            self.assertFalse(duplicate.success)
            self.assertEqual(path.read_text(encoding="utf-8"), "hello")

    def test_file_mutations_require_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "note.txt"
            path.write_text("hello", encoding="utf-8")
            with self.assertRaises(ToolConfirmationRequired):
                self.manager.execute("delete_file", {"path": str(path)})
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()