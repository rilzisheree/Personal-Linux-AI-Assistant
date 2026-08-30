"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .config import AppConfig
from .ui.main_window import MainWindow
from .ui.styles import APP_STYLE


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Local AI Assistant")
    app.setOrganizationName("Local AI Assistant")
    app.setStyleSheet(APP_STYLE)
    window = MainWindow(AppConfig.load())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
