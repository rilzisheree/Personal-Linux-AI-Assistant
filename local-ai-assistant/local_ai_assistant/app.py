"""Application entry point."""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication, QInputDialog, QLineEdit, QMessageBox

from .api_store import ApiStore
from .config import AppConfig
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
    if not store.has_password():
        password, accepted = QInputDialog.getText(
            None,
            "Create Lura password",
            "Choose a password for Lura (8–256 characters):",
            QLineEdit.EchoMode.Password,
        )
        if not accepted:
            return False
        confirmation, confirmed = QInputDialog.getText(
            None,
            "Confirm Lura password",
            "Enter the password again:",
            QLineEdit.EchoMode.Password,
        )
        if not confirmed or password != confirmation:
            QMessageBox.warning(None, "Password mismatch", "The passwords did not match.")
            return False
        try:
            store.set_password(password)
        except ValueError as error:
            QMessageBox.warning(None, "Password not saved", str(error))
            return False
        return True

    password, accepted = QInputDialog.getText(
        None,
        "Unlock Lura",
        "Enter your Lura password:",
        QLineEdit.EchoMode.Password,
    )
    if not accepted:
        return False
    if not store.verify_password(password):
        QMessageBox.critical(None, "Lura is locked", "That password is incorrect.")
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
