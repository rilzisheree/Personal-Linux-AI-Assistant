from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from local_ai_assistant.applications import ApplicationRecord
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
            "get_identity",
            "get_gpu_info",
            "get_cpu_info",
            "get_ram_info",
            "get_disk_info",
        ):
            self.assertIn(name, names)
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

    def test_direct_dispatch_maps_identity_and_authoritative_facts(self) -> None:
        self.assertEqual(
            self.manager.direct_tool_call_for_request("What's my name?"),
            ("get_identity", {}),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request("What graphics card do I have?"),
            ("get_gpu_info", {}),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request("How many CPU cores do I have?"),
            ("get_cpu_info", {}),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request("How much RAM do I have?"),
            ("get_ram_info", {}),
        )

    def test_direct_dispatch_maps_application_actions_without_guessing_commands(self) -> None:
        self.assertEqual(
            self.manager.direct_tool_call_for_request("Open Firefox"),
            ("open_app", {"app": "Firefox"}),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request("Can you open Firefox?"),
            ("open_app", {"app": "Firefox"}),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request("Please launch Firefox on my computer."),
            ("open_app", {"app": "Firefox"}),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request("Close Discord"),
            ("close_app", {"app": "Discord"}),
        )
        self.assertIsNone(
            self.manager.direct_tool_call_for_request("Run a Python program")
        )

    @patch("local_ai_assistant.tools.subprocess.Popen")
    def test_open_app_executes_discovered_flatpak_id(self, popen_mock) -> None:
        self.manager.application_registry.resolve = unittest.mock.Mock(
            return_value=ApplicationRecord(
                app_id="org.mozilla.firefox",
                name="Firefox",
                kind="flatpak",
                launch_command=("flatpak", "run", "org.mozilla.firefox"),
            )
        )

        result = self.manager.execute("open_app", {"app": "Firefox"})

        self.assertTrue(result.success)
        self.assertEqual(
            popen_mock.call_args.args[0],
            ("flatpak", "run", "org.mozilla.firefox"),
        )

    @patch("local_ai_assistant.tools.shutil.which", return_value="/usr/bin/firefox")
    @patch("local_ai_assistant.tools.subprocess.Popen")
    def test_custom_launcher_executes_direct_command(self, popen_mock, _which_mock) -> None:
        manager = ToolManager(custom_app_commands={"Firefox": "firefox --new-window"})

        result = manager.execute("open_app", {"app": "Firefox"})

        self.assertTrue(result.success)
        self.assertEqual(
            popen_mock.call_args.args[0],
            ("firefox", "--new-window"),
        )
        self.assertFalse(popen_mock.call_args.kwargs["start_new_session"])
        self.assertEqual(json.loads(result.content)["kind"], "custom")

    @patch("local_ai_assistant.tools.shutil.which", return_value="/usr/bin/firefox")
    @patch("local_ai_assistant.tools.subprocess.Popen")
    def test_custom_launcher_accepts_url_arguments_and_articles(
        self, popen_mock, _which_mock
    ) -> None:
        manager = ToolManager(
            custom_app_commands={
                "Youtube tab": "firefox --new-tab https://youtube.com"
            }
        )

        result = manager.execute("open_app", {"app": "my YouTube tab"})

        self.assertTrue(result.success)
        self.assertEqual(
            popen_mock.call_args.args[0],
            ("firefox", "--new-tab", "https://youtube.com"),
        )

    def test_custom_launcher_dispatch_accepts_extra_polite_wording(self) -> None:
        manager = ToolManager(
            custom_app_commands={
                "Youtube tab": "firefox --new-tab https://youtube.com"
            }
        )

        self.assertEqual(
            manager.direct_tool_call_for_request(
                "I need you to please open my YouTube tab now."
            ),
            ("open_app", {"app": "Youtube tab"}),
        )

    def test_custom_launcher_names_are_exposed_to_the_model(self) -> None:
        manager = ToolManager(custom_app_commands={"Youtube tab": "firefox"})

        description = manager.definitions_for_ollama()[0]["function"]["description"]

        self.assertIn('"Youtube tab"', description)
        self.assertIn("Use those names with open_app", description)

    def test_permission_policies_block_or_auto_approve_actions(self) -> None:
        blocked = ToolManager(tool_permissions={"open_app": "blocked"})
        self.assertEqual(
            blocked.permission_for("open_app", {"app": "Firefox"}),
            PermissionLevel.BLOCKED,
        )
        self.assertFalse(blocked.execute("open_app", {"app": "Firefox"}).success)

        allowed = ToolManager(tool_permissions={"exec": "always_allow"})
        self.assertEqual(
            allowed.permission_for("exec", {"command": "printf ok"}),
            PermissionLevel.NORMAL,
        )
        self.assertEqual(allowed.execute("exec", {"command": "printf ok"}).content, "ok")

    def test_identity_tool_keeps_user_and_assistant_names_separate(self) -> None:
        manager = ToolManager(assistant_name="Nova")
        with tempfile.TemporaryDirectory() as directory:
            manager.profile_store = manager.profile_store.__class__(
                Path(directory) / "profile.json"
            )
            manager.profile_store.update({"name": "Alex"})
            result = manager.execute("get_identity", {})
        self.assertTrue(result.success)
        self.assertEqual(
            json.loads(result.content),
            {
                "assistant_name": "Nova",
                "user": {
                    "name": "Alex",
                    "owner": "the current local user",
                    "preferred_address": "Sir",
                },
            },
        )

    def test_confirmation_required_tools_are_gated(self) -> None:
        self.assertEqual(
            self.manager.permission_for("exec", {"command": "printf ok"}),
            PermissionLevel.CONFIRMATION_REQUIRED,
        )
        with self.assertRaises(ToolConfirmationRequired):
            self.manager.execute("exec", {"command": "printf should-not-run"})

    def test_legacy_read_only_commands_are_safe(self) -> None:
        self.assertEqual(
            self.manager.permission_for("exec", {"command": "free -h"}),
            PermissionLevel.SAFE,
        )
        self.assertEqual(
            self.manager.permission_for("exec", {"command": "flatpak list --app"}),
            PermissionLevel.SAFE,
        )

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

    @patch("local_ai_assistant.tools.shutil.which", return_value="/usr/bin/hyprctl")
    @patch("local_ai_assistant.tools.subprocess.run")
    def test_move_window_uses_hyprland_address_dispatch(
        self, run_mock, _which_mock
    ) -> None:
        run_mock.side_effect = [
            unittest.mock.Mock(
                returncode=0,
                stdout='[{"address":"0xabc","title":"Editor","class":"code","workspace":{"name":"1"}}]',
                stderr="",
            ),
            unittest.mock.Mock(returncode=0, stdout="dispatched", stderr=""),
        ]

        result = self.manager.execute(
            "move_window",
            {"window": "Editor", "workspace": "2"},
            approved=True,
        )

        self.assertTrue(result.success)
        self.assertEqual(
            run_mock.call_args_list[1].args[0],
            ["hyprctl", "dispatch", "movetoworkspace", "2,address:0xabc"],
        )

    @patch("local_ai_assistant.tools.Path.home")
    @patch("local_ai_assistant.tools.shutil.which")
    @patch("local_ai_assistant.tools.subprocess.run")
    def test_screenshot_prefers_wayland_backend_and_returns_image(
        self, run_mock, which_mock, home_mock
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home_mock.return_value = Path(directory)

            def available(binary: str) -> str | None:
                return "/usr/bin/hyprshot" if binary == "hyprshot" else None

            which_mock.side_effect = available

            def capture(command, **_kwargs):
                Path(command[-1]).write_bytes(b"png")
                return unittest.mock.Mock(returncode=0, stdout="", stderr="")

            run_mock.side_effect = capture
            result = self.manager.execute("take_screenshot", {}, approved=True)

            self.assertTrue(result.success)
            self.assertEqual(len(result.images), 1)
            self.assertTrue(Path(result.images[0]).is_file())
            self.assertEqual(run_mock.call_args.args[0][0], "hyprshot")

    @patch("local_ai_assistant.tools.shutil.which")
    @patch("local_ai_assistant.tools.subprocess.run")
    def test_keyboard_and_mouse_have_linux_backend_fallbacks(
        self, run_mock, which_mock
    ) -> None:
        which_mock.side_effect = lambda binary: (
            "/usr/bin/xdotool" if binary == "xdotool" else None
        )
        run_mock.return_value = unittest.mock.Mock(returncode=0, stdout="", stderr="")

        typed = self.manager.execute(
            "keyboard_type", {"text": "hello"}, approved=True
        )
        pressed = self.manager.execute(
            "keyboard_press", {"key": "Return"}, approved=True
        )
        clicked = self.manager.execute(
            "mouse_click",
            {"x": 10, "y": 20, "button": "left"},
            approved=True,
        )

        self.assertTrue(typed.success)
        self.assertTrue(pressed.success)
        self.assertTrue(clicked.success)
        self.assertEqual(run_mock.call_args_list[0].args[0][0:2], ["xdotool", "type"])
        self.assertEqual(run_mock.call_args_list[1].args[0][0:2], ["xdotool", "key"])
        self.assertEqual(run_mock.call_args_list[2].args[0][0:2], ["xdotool", "mousemove"])
        self.assertEqual(run_mock.call_args_list[3].args[0][0:2], ["xdotool", "click"])


if __name__ == "__main__":
    unittest.main()
