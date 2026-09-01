from __future__ import annotations

import os
import unittest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from local_ai_assistant.ui.core_widget import CoreWidget


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class CoreWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_orb_emits_release_after_mouse_press(self) -> None:
        widget = CoreWidget()
        widget.resize(310, 310)
        widget.show()
        events: list[str] = []
        widget.pressed.connect(lambda: events.append("pressed"))
        widget.released.connect(lambda: events.append("released"))

        QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=QPoint(155, 155))
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=QPoint(155, 155))
        self.app.processEvents()

        self.assertEqual(events, ["pressed", "released"])
        self.assertFalse(widget._pressed)

    def test_orb_releases_recording_when_focus_is_lost(self) -> None:
        widget = CoreWidget()
        widget.resize(310, 310)
        widget.show()
        other = QWidget()
        other.show()
        released: list[bool] = []
        widget.released.connect(lambda: released.append(True))

        widget.setFocus()
        QTest.keyPress(widget, Qt.Key.Key_Space)
        other.setFocus()
        self.app.processEvents()

        self.assertEqual(released, [True])
        self.assertFalse(widget._pressed)


if __name__ == "__main__":
    unittest.main()