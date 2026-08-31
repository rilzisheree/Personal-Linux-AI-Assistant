from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_ai_assistant.desktop_integration import (
    autostart_entry,
    autostart_path,
    is_autostart_enabled,
    set_autostart_enabled,
)


class DesktopIntegrationTests(unittest.TestCase):
    def test_autostart_entry_launches_background_mode(self) -> None:
        entry = autostart_entry("/opt/lura/.venv/bin/python")
        self.assertIn("Type=Application", entry)
        self.assertIn("Exec=/opt/lura/.venv/bin/python -m local_ai_assistant.app --background", entry)
        self.assertIn("Terminal=false", entry)

    def test_autostart_can_be_enabled_and_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)
            path = autostart_path(config_dir)
            self.assertFalse(is_autostart_enabled(config_dir))

            set_autostart_enabled(
                True,
                config_dir=config_dir,
                executable="/usr/bin/python3",
            )
            self.assertTrue(path.is_file())
            self.assertTrue(is_autostart_enabled(config_dir))
            self.assertIn("--background", path.read_text(encoding="utf-8"))

            set_autostart_enabled(False, config_dir=config_dir)
            self.assertFalse(path.exists())
            self.assertFalse(is_autostart_enabled(config_dir))


if __name__ == "__main__":
    unittest.main()