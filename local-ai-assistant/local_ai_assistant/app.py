"""Application entry point."""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from .api_store import ApiStore
from .config import AppConfig
from .ui.auth_dialog import UnlockDialog
from .ui.main_window import MainWindow
from .ui.styles import APP_STYLE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lura local AI assistant")
    parser.add_argument(
        "--background",
        action="store_true",
        help="start hidden in the system tray when available",
    )
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    app.setApplicationName("Lura")
    app.setOrganizationName("Lura")
    app.setStyleSheet(APP_STYLE)
    if not _configure_or_unlock_password(ApiStore()):
        return 0
    window = MainWindow(AppConfig.load())
    window.show()
    if args.background:
        window.hide_to_tray(silent=True)
    return app.exec()


def _configure_or_unlock_password(store: ApiStore) -> bool:
    dialog = UnlockDialog(store)
    return dialog.exec() == dialog.DialogCode.Accepted


if __name__ == "__main__":
    raise SystemExit(main())
