from __future__ import annotations

import tempfile
import unittest
import json
import subprocess
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch

from local_ai_assistant.applications import ApplicationRecord
from local_ai_assistant.tools import (
    PermissionLevel,
    ToolConfirmationRequired,
    ToolManager,
)
from local_ai_assistant.reminders import ReminderService, ReminderStore


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
            "get_active_model",
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
        for name in (
            "get_weather",
            "search_news",
            "knowledge_search",
            "convert_currency",
            "find_places",
            "get_directions",
            "travel_search",
            "game_search",
            "create_reminder",
        ):
            self.assertIn(name, names)

    def test_safe_system_tool_does_not_need_approval(self) -> None:
        result = self.manager.execute("get_disk_usage", {})
        self.assertIsInstance(result.success, bool)
        self.assertTrue(result.content)

    def test_gpu_tool_returns_structured_complete_metrics(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="NVIDIA GeForce RTX 4060, 89, 55, 1024, 8188\n",
            stderr="",
        )
        with patch("local_ai_assistant.tools.subprocess.run", return_value=completed):
            result = self.manager.execute("get_gpu_info", {})

        self.assertTrue(result.success)
        payload = json.loads(result.content)
        self.assertTrue(payload["available"])
        self.assertEqual(payload["model"], "NVIDIA GeForce RTX 4060")
        self.assertEqual(payload["vram_total_mb"], 8188.0)
        self.assertEqual(payload["temperature_c"], 55.0)
        self.assertEqual(payload["utilization_percent"], 89.0)
        self.assertEqual(len(payload["gpus"]), 1)

    def test_gpu_tool_rejects_malformed_rows_instead_of_skipping_them(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout=(
                "NVIDIA GeForce RTX 4060, 89, 55, 1024, 8188\n"
                "NVIDIA GeForce RTX 3060, 70, 60, unavailable, 12288\n"
            ),
            stderr="",
        )
        with patch("local_ai_assistant.tools.subprocess.run", return_value=completed):
            result = self.manager.execute("get_gpu_info", {})

        self.assertFalse(result.success)
        payload = json.loads(result.content)
        self.assertFalse(payload["available"])
        self.assertEqual(payload["model"], None)
        self.assertEqual(payload["gpus"], [])
        self.assertEqual(payload["error"], "NVIDIA GPU data was malformed.")
        self.assertEqual(len(payload["error_details"]), 1)

    def test_gpu_tool_reports_command_failure_without_inventing_values(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=1,
            stdout="",
            stderr="NVIDIA-SMI has failed.",
        )
        with patch("local_ai_assistant.tools.subprocess.run", return_value=completed):
            result = self.manager.execute("get_gpu_status", {})

        self.assertFalse(result.success)
        payload = json.loads(result.content)
        self.assertFalse(payload["available"])
        self.assertIsNone(payload["model"])
        self.assertIn("unavailable", payload["error"])

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
        self.assertEqual(
            self.manager.direct_tool_call_for_request(
                "What model are you currently using?"
            ),
            ("get_active_model", {}),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request("What GPU do I have?"),
            ("get_gpu_info", {}),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request("What CPU do I have?"),
            ("get_cpu_info", {}),
        )

    def test_active_model_result_uses_runtime_model_state(self) -> None:
        manager = ToolManager(active_model="qwen3.5:2b")

        result = manager.execute("get_active_model", {})

        self.assertTrue(result.success)
        self.assertEqual(
            json.loads(result.content),
            {
                "success": True,
                "data": {"active_model": "qwen3.5:2b", "model_type": "main"},
                "error": None,
            },
        )
        manager.set_active_model("llama3.2:3b")
        self.assertEqual(
            json.loads(manager.execute("get_active_model", {}).content)["data"][
                "active_model"
            ],
            "llama3.2:3b",
        )

    def test_active_model_result_is_explicit_when_unavailable(self) -> None:
        result = self.manager.execute("get_active_model", {})

        self.assertFalse(result.success)
        self.assertEqual(
            json.loads(result.content),
            {
                "success": False,
                "data": None,
                "error": "The active Ollama model is unavailable.",
            },
        )

    def test_direct_dispatch_maps_relative_reminder(self) -> None:
        self.assertEqual(
            self.manager.direct_tool_call_for_request(
                "Remind me in 5 seconds to drink water."
            ),
            (
                "create_reminder",
                {"message": "drink water", "delay_seconds": 5.0},
            ),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request(
                "Please remind me in 2 minutes to stretch"
            ),
            (
                "create_reminder",
                {"message": "stretch", "delay_seconds": 120.0},
            ),
        )

    def test_create_reminder_persists_and_returns_due_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ReminderService(
                ReminderStore(Path(directory) / "reminders.json"),
                notify_send="",
                start_scheduler=False,
            )
            self.manager.reminder_service = service

            result = self.manager.execute(
                "create_reminder",
                {"message": "drink water", "delay_seconds": 5},
            )

            self.assertTrue(result.success)
            self.assertIn("Reminder scheduled for", result.content)
            self.assertIn("drink water", result.content)
            reminders = service.store.list()
            self.assertEqual(len(reminders), 1)
            self.assertEqual(reminders[0].message, "drink water")
            self.assertAlmostEqual(reminders[0].due_at, service._clock() + 5, delta=1)

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
        self.assertTrue(popen_mock.call_args.kwargs["start_new_session"])
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

    def test_direct_dispatch_maps_live_search_requests(self) -> None:
        self.assertEqual(
            self.manager.direct_tool_call_for_request(
                "Search the web for the latest AI news."
            ),
            ("search_news", {"query": "the latest AI news"}),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request(
                "Please look up Spotify release updates."
            ),
            ("web_search", {"query": "Spotify release updates"}),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request(
                "Can you perform a live search for today's AI news?"
            ),
            ("search_news", {"query": "today's AI news"}),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request(
                "What is the latest NVIDIA news?"
            ),
            ("search_news", {"query": "the latest NVIDIA news"}),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request(
                "What is the current price of Bitcoin?"
            ),
            ("web_search", {"query": "the current price of Bitcoin"}),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request(
                "Move my homework reminder to 9 hours from now."
            ),
            (
                "reschedule_reminder",
                {"reminder": "homework", "delay_seconds": 32400.0},
            ),
        )

    def test_direct_dispatch_does_not_search_for_casual_or_general_questions(self) -> None:
        no_direct_tool_requests = (
            "How are you doing today?",
            "How are you?",
            "What's up?",
            "Good morning.",
            "Tell me a joke.",
            "What can you do?",
            "Explain what RAM is.",
            "What is Linux?",
            "What's the weather today?",
        )

        for request in no_direct_tool_requests:
            with self.subTest(request=request):
                self.assertIsNone(
                    self.manager.direct_tool_call_for_request(request)
                )

    def test_direct_dispatch_searches_only_when_current_request_has_external_intent(self) -> None:
        self.assertEqual(
            self.manager.direct_tool_call_for_request(
                "What happened in AI today?"
            ),
            ("web_search", {"query": "What happened in AI today"}),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request(
                "What's the latest NVIDIA news?"
            ),
            ("search_news", {"query": "the latest NVIDIA news"}),
        )
        self.assertEqual(
            self.manager.direct_tool_call_for_request(
                "What are you doing today?"
            ),
            None,
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

    def test_web_search_returns_structured_deduplicated_results(self) -> None:
        html = """
        <html><body>
          <div class="result">
            <a class="result__a" href="https://example.com/a">First result</a>
            <div class="result__snippet">First snippet.</div>
          </div>
          <div class="result">
            <a class="result__a" href="https://example.com/a#fragment">Duplicate result</a>
            <div class="result__snippet">Duplicate snippet.</div>
          </div>
          <div class="result">
            <a class="result__a" href="https://example.org/b">Second result</a>
          </div>
        </body></html>
        """

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def getcode(self):
                return self.status

            def read(self, _limit):
                return html.encode("utf-8")

        with patch("local_ai_assistant.tools.urlopen", return_value=Response()):
            result = self.manager.execute(
                "web_search",
                {"query": "latest AI news", "max_results": 5},
            )

        self.assertTrue(result.success)
        payload = json.loads(result.content)
        self.assertTrue(payload["success"])
        self.assertEqual(len(payload["results"]), 2)
        self.assertEqual(payload["results"][0]["url"], "https://example.com/a")
        self.assertEqual(payload["results"][1]["snippet"], "")
        self.assertEqual(payload["results"][0]["source"], "example.com")

    def test_web_search_failure_states_are_structured_and_accountable(self) -> None:
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def getcode(self):
                return self.status

            def read(self, _limit):
                return b"<html><body>No result cards.</body></html>"

        with patch("local_ai_assistant.tools.urlopen", return_value=Response()):
            empty = self.manager.execute("web_search", {"query": "nothing"})
        self.assertFalse(empty.success)
        self.assertEqual(json.loads(empty.content)["error_code"], "NO_RESULTS")

        with patch(
            "local_ai_assistant.tools.urlopen",
            side_effect=TimeoutError("slow provider"),
        ):
            timed_out = self.manager.execute("web_search", {"query": "latest news"})
        self.assertEqual(json.loads(timed_out.content)["error_code"], "TIMEOUT")

        with patch(
            "local_ai_assistant.tools.urlopen",
            side_effect=URLError("offline"),
        ):
            network_error = self.manager.execute("web_search", {"query": "latest news"})
        self.assertEqual(json.loads(network_error.content)["error_code"], "NETWORK_ERROR")

        class MalformedResponse(Response):
            def read(self, _limit):
                return b"not html"

        with patch(
            "local_ai_assistant.tools.urlopen",
            return_value=MalformedResponse(),
        ):
            malformed = self.manager.execute("web_search", {"query": "latest news"})
        self.assertEqual(json.loads(malformed.content)["error_code"], "PARSER_ERROR")

    def test_web_search_rejects_challenges_and_invalid_urls(self) -> None:
        class Response:
            status = 202

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def getcode(self):
                return self.status

            def read(self, _limit):
                return b'<form id="challenge-form">human check</form>'

        with patch("local_ai_assistant.tools.urlopen", return_value=Response()):
            challenged = self.manager.execute("web_search", {"query": "latest news"})
        self.assertEqual(
            json.loads(challenged.content)["error_code"],
            "WEB_SEARCH_UNAVAILABLE",
        )

        class InvalidResultResponse(Response):
            status = 200

            def read(self, _limit):
                return (
                    b'<html><a class="result__a" href="javascript:alert(1)">'
                    b'Unsafe</a><div class="result__snippet">Nope</div></html>'
                )

        with patch(
            "local_ai_assistant.tools.urlopen",
            return_value=InvalidResultResponse(),
        ):
            invalid = self.manager.execute("web_search", {"query": "latest news"})
        self.assertEqual(json.loads(invalid.content)["error_code"], "INVALID_RESULT")

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
