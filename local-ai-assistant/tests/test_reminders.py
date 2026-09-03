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

    def test_due_listener_is_called_and_can_be_removed(self) -> None:
        listener = Mock()
        service = ReminderService(start_scheduler=False, on_due=listener)
        reminder = Reminder("abc", "Drink water", 1234.5)

        service._emit_due(reminder)
        service.remove_due_listener(listener)
        service._emit_due(reminder)

        listener.assert_called_once_with(reminder)

    def test_completion_records_history_and_schedules_repeating_next_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReminderStore(Path(directory) / "reminders.json")
            service = ReminderService(store, notify_send="", start_scheduler=False)
            due_at = service._clock() + 3600
            reminder = service.schedule_at(
                "Study mathematics",
                due_at,
                repeat="Daily",
                priority="High",
                strict_mode=True,
            )

            completed = service.complete(reminder.reminder_id)
            saved = store.list()

        self.assertIsNotNone(completed)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(len(saved), 2)
        self.assertEqual(
            [item.status for item in saved],
            ["completed", "upcoming"],
        )
        self.assertEqual(saved[1].message, "Study mathematics")
        self.assertEqual(saved[1].priority, "High")
        self.assertTrue(saved[1].strict_mode)
        self.assertAlmostEqual(saved[1].due_at, due_at + 86400, delta=1)


if __name__ == "__main__":
    unittest.main()