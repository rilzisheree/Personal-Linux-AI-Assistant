from __future__ import annotations

import os
import unittest

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
        dialog.orb_intensity_input.setValue(80)
        dialog.animation_intensity_input.setValue(45)

        updated = dialog.config()

        self.assertEqual(updated.assistant_name, "Nova")
        self.assertTrue(updated.wake_word_enabled)
        self.assertEqual(updated.wake_word, "Hey Nova")
        self.assertEqual(updated.active_listening_duration, 30)
        self.assertEqual(updated.orb_intensity, 80)
        self.assertEqual(updated.animation_intensity, 45)
        dialog.close()


if __name__ == "__main__":
    unittest.main()