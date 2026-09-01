from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from local_ai_assistant.ui.main_window import MainWindow


class MainWindowVoiceTests(unittest.TestCase):
    @staticmethod
    def _window_with_wake_listener() -> MainWindow:
        window = MainWindow.__new__(MainWindow)
        window.config = SimpleNamespace(voice_input_enabled=True)
        window.chat_worker = None
        window.voice_record_worker = None
        window.voice_transcription_worker = None
        window.wake_word_worker = object()
        window._manual_recording_pending = False
        window._manual_handoff_started_at = None
        window._stop_wake_word_listener = Mock()
        window._start_manual_recording_when_ready = Mock()
        window._set_voice_status = Mock()
        window._set_orb_state = Mock()
        return window

    def test_manual_orb_press_pauses_wake_listener_for_handoff(self) -> None:
        window = self._window_with_wake_listener()

        MainWindow._start_recording(window)

        self.assertTrue(window._manual_recording_pending)
        window._stop_wake_word_listener.assert_called_once_with()
        window._start_manual_recording_when_ready.assert_called_once_with()
        window._set_voice_status.assert_called_once_with(
            "SWITCHING TO MANUAL MICROPHONE…"
        )
        window._set_orb_state.assert_not_called()

    def test_automatic_recording_does_not_compete_with_wake_listener(self) -> None:
        window = self._window_with_wake_listener()

        MainWindow._start_recording(window, automatic=True)

        self.assertFalse(window._manual_recording_pending)
        window._stop_wake_word_listener.assert_not_called()
        window._start_manual_recording_when_ready.assert_not_called()
        window._set_orb_state.assert_called_once_with("idle")

    def test_wake_detection_keeps_command_pending_until_listener_stops(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window._wake_command_pending = False
        window.wake_word_thread = Mock()
        window._set_voice_status = Mock()

        MainWindow._wake_word_detected(window)

        self.assertTrue(window._wake_command_pending)
        window.wake_word_thread.quit.assert_called_once_with()
        window._set_voice_status.assert_called_once_with(
            "WAKE WORD DETECTED // LISTENING…"
        )


if __name__ == "__main__":
    unittest.main()