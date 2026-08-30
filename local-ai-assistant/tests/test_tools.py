from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()