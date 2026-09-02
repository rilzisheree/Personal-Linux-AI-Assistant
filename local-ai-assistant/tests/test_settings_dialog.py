from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from local_ai_assistant.config import AppConfig
from local_ai_assistant.ui.settings_dialog import SettingsDialog


class SettingsDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_assistant_controls_round_trip_without_crashing(self) -> None:
        dialog = SettingsDialog(AppConfig())
        dialog.assistant_name_input.setText("Nova")
        dialog.wake_word_enabled.setChecked(True)
        dialog.wake_word_input.setText("Hey Nova")
        dialog.active_listening_duration.setValue(30)
        dialog.continuous_conversation_enabled.setChecked(True)
        dialog.conversation_timeout.setValue(20)
        dialog.conversation_transition_delay.setValue(0.6)
        dialog.orb_intensity_input.setValue(80)
        dialog.animation_intensity_input.setValue(45)

        updated = dialog.config()

        self.assertEqual(updated.assistant_name, "Nova")
        self.assertTrue(updated.wake_word_enabled)
        self.assertEqual(updated.wake_word, "Hey Nova")
        self.assertEqual(updated.active_listening_duration, 30)
        self.assertTrue(updated.continuous_conversation_enabled)
        self.assertEqual(updated.conversation_timeout, 20)
        self.assertEqual(updated.conversation_transition_delay, 6 / 10)
        self.assertEqual(updated.orb_intensity, 80)
        self.assertEqual(updated.animation_intensity, 45)
        dialog.close()

    def test_microphone_picker_round_trips_selected_source(self) -> None:
        with patch(
            "local_ai_assistant.ui.settings_dialog.VoiceService.list_microphones",
            return_value=[
                (
                    "alsa_input.usb-hyperx.analog-stereo",
                    "HyperX SoloCast",
                )
            ],
        ):
            dialog = SettingsDialog(AppConfig())
        index = dialog.microphone_input.findData(
            "alsa_input.usb-hyperx.analog-stereo"
        )
        self.assertGreaterEqual(index, 0)
        dialog.microphone_input.setCurrentIndex(index)
        self.assertEqual(
            dialog.config().microphone_device,
            "alsa_input.usb-hyperx.analog-stereo",
        )
        dialog.close()

    def test_permissions_and_custom_launcher_round_trip(self) -> None:
        dialog = SettingsDialog(
            AppConfig(custom_app_commands={"Firefox": "firefox"})
        )
        dialog.permission_inputs["open_app"].setCurrentIndex(
            dialog.permission_inputs["open_app"].findData("always_allow")
        )
        alias_input = dialog.custom_app_command_rows[0][1]
        command_input = dialog.custom_app_command_rows[0][2]
        alias_input.setText("Firefox")
        command_input.setText("firefox --new-window")

        updated = dialog.config()

        self.assertEqual(updated.tool_permissions["open_app"], "always_allow")
        self.assertEqual(
            updated.custom_app_commands,
            {"Firefox": "firefox --new-window"},
        )
        dialog.close()


if __name__ == "__main__":
    unittest.main()