from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from local_ai_assistant.reminders import Reminder, ReminderService, ReminderStore


class ReminderServiceTests(unittest.TestCase):
    def test_store_round_trips_reminders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReminderStore(Path(directory) / "reminders.json")
            reminder = Reminder("abc", "Drink water", 1234.5)

            store.add(reminder)

            self.assertEqual(store.list(), [reminder])
            self.assertTrue(store.remove("abc"))
            self.assertEqual(store.list(), [])

    @patch("local_ai_assistant.reminders.subprocess.run")
    def test_due_notification_uses_notify_send(self, run_mock: Mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess([], 0, "", "")
        with tempfile.TemporaryDirectory() as directory:
            service = ReminderService(
                ReminderStore(Path(directory) / "reminders.json"),
                notify_send="/usr/bin/notify-send",
                start_scheduler=False,
            )

            service._notify(Reminder("abc", "Drink water", 1234.5))

        run_mock.assert_called_once_with(
            [
                "/usr/bin/notify-send",
                "--app-name=Lura",
                "--urgency=normal",
                "--expire-time=10000",
                "Lura reminder",
                "Drink water",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )

    def test_schedule_rejects_empty_message_and_invalid_delay(self) -> None:
        service = ReminderService(start_scheduler=False)
        with self.assertRaises(ValueError):
            service.schedule("", 5)
        with self.assertRaises(ValueError):
            service.schedule("Drink water", 0)


if __name__ == "__main__":
    unittest.main()