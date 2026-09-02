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
        window._set_voice_state = Mock()
        return window

    def test_manual_voice_press_pauses_wake_listener_for_handoff(self) -> None:
        window = self._window_with_wake_listener()

        MainWindow._start_recording(window)

        self.assertTrue(window._manual_recording_pending)
        window._stop_wake_word_listener.assert_called_once_with()
        window._start_manual_recording_when_ready.assert_called_once_with()
        window._set_voice_status.assert_called_once_with(
            "SWITCHING TO MANUAL MICROPHONE…"
        )
        window._set_voice_state.assert_not_called()

    def test_automatic_recording_does_not_compete_with_wake_listener(self) -> None:
        window = self._window_with_wake_listener()

        MainWindow._start_recording(window, automatic=True)

        self.assertFalse(window._manual_recording_pending)
        window._stop_wake_word_listener.assert_not_called()
        window._start_manual_recording_when_ready.assert_not_called()
        window._set_voice_state.assert_called_once_with("idle")

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

    def test_conversation_goodbye_is_only_handled_in_active_mode(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window._latency_trace = None
        window.voice_transcription_thread = None
        window._confirmation_recording = False
        window._set_voice_idle = Mock()
        window._conversation_active = True
        window._wake_command_recording = True
        window._end_conversation = Mock()
        window._show_conversation_goodbye = Mock()

        MainWindow._transcription_finished(window, "Goodbye", "/tmp/missing.wav")

        window._end_conversation.assert_called_once_with(
            "CONVERSATION ENDED // GOODBYE"
        )
        window._show_conversation_goodbye.assert_called_once_with("Goodbye")
        self.assertFalse(window._wake_command_recording)

        idle_window = MainWindow.__new__(MainWindow)
        idle_window._latency_trace = None
        idle_window.voice_transcription_thread = None
        idle_window._confirmation_recording = False
        idle_window._set_voice_idle = Mock()
        idle_window._conversation_active = False
        idle_window._wake_command_recording = True
        idle_window.message_input = Mock()
        idle_window._send_message = Mock()

        MainWindow._transcription_finished(idle_window, "Goodbye", "/tmp/missing.wav")

        idle_window._send_message.assert_called_once_with()
        idle_window.message_input.setText.assert_called_once_with("Goodbye")

    def test_conversation_goodbye_requests_audible_response(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window.messages = []
        window.chat_view = Mock()
        window._persist_current_conversation = Mock()
        window._speak_response = Mock()

        MainWindow._show_conversation_goodbye(window, "See you")

        window._speak_response.assert_called_once_with(
            "Goodbye, Sir.", force_voice=True
        )

    def test_one_shot_speech_queues_response_when_chunker_is_empty(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window._speech_chunker = Mock()
        window._speech_chunker.flush.return_value = []
        window.speech_worker = None
        window._queue_speech_chunk = Mock()
        window._set_voice_state = Mock()
        window._latency_trace = None

        MainWindow._finish_speech_stream(
            window, "Goodbye, Sir.", force_voice=True
        )

        window._queue_speech_chunk.assert_called_once_with(
            "Goodbye, Sir.", force_voice=True
        )

    def test_voice_state_updates_embedded_core_and_notification(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window._voice_state = "idle"
        window.config = SimpleNamespace(voice_input_enabled=True)
        window.core_widget = Mock()
        window.status_notification = Mock()
        window.core_status = Mock()
        window._sync_status_notification = Mock()

        MainWindow._set_voice_state(window, "thinking")

        self.assertEqual(window._voice_state, "processing")
        window.core_widget.set_state.assert_called_once_with("processing")
        window.status_notification.set_state.assert_called_once_with("processing")
        window.core_status.setText.assert_called_once_with(
            "PROCESSING // LOCAL MODEL"
        )
        window._sync_status_notification.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()