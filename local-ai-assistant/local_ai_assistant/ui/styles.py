"""Application stylesheet."""

from __future__ import annotations


APP_STYLE = """
QMainWindow, QWidget {
    background: #101419;
    color: #e8edf2;
    font-family: "Inter", "Noto Sans", sans-serif;
    font-size: 14px;
}
QMainWindow {
    background: #101419;
}
QFrame#topBar {
    background: #161c23;
    border-bottom: 1px solid #27313c;
}
QFrame#historyPanel {
    background: #131a21;
    border-right: 1px solid #27313c;
}
QLabel#sectionLabel {
    color: #8794a3;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}
QListWidget#historyList {
    background: transparent;
    border: 0;
    outline: 0;
    color: #bac5cf;
    padding: 2px;
}
QListWidget#historyList::item {
    border-radius: 8px;
    padding: 10px 8px;
    margin: 2px 0;
}
QListWidget#historyList::item:hover {
    background: #1b2730;
}
QListWidget#historyList::item:selected {
    background: #234550;
    color: #effcff;
}
QLabel#appMark {
    color: #f1f5f9;
    font-size: 18px;
    font-weight: 700;
}
QLabel#appSubtitle {
    color: #8794a3;
    font-size: 12px;
}
QLabel#statusLabel {
    border: 1px solid #34404c;
    border-radius: 12px;
    color: #aab6c2;
    padding: 5px 10px;
}
QLabel#statusLabel[status="connected"] {
    border-color: #2c8067;
    color: #8ce0be;
}
QLabel#statusLabel[status="error"] {
    border-color: #8e4045;
    color: #ffaaa7;
}
QComboBox, QLineEdit, QTextEdit {
    background: #182029;
    border: 1px solid #2b3743;
    border-radius: 9px;
    color: #edf3f7;
    padding: 8px 10px;
    selection-background-color: #2a6175;
}
QComboBox:hover, QLineEdit:focus, QTextEdit:focus {
    border-color: #4b9bb1;
}
QComboBox::drop-down {
    border: 0;
    width: 24px;
}
QPushButton {
    background: #274f5c;
    border: 1px solid #376f7f;
    border-radius: 9px;
    color: #effcff;
    padding: 9px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background: #316477;
}
QPushButton:pressed {
    background: #1e424d;
}
QPushButton:disabled {
    background: #202a32;
    border-color: #2b343d;
    color: #65717d;
}
QPushButton#stopButton {
    background: transparent;
    border-color: #7e464d;
    color: #ffb1ae;
}
QPushButton#stopButton:hover {
    background: #3b252b;
}
QPushButton#settingsButton {
    background: transparent;
    border-color: #34404c;
    color: #b6c2cd;
}
QPushButton#newChatButton {
    background: transparent;
    border-color: #34404c;
    color: #b6c2cd;
}
QScrollArea {
    border: 0;
    background: #101419;
}
QScrollBar:vertical {
    background: #101419;
    width: 10px;
    margin: 4px;
}
QScrollBar::handle:vertical {
    background: #2d3b47;
    border-radius: 5px;
    min-height: 28px;
}
QScrollBar::add-line, QScrollBar::sub-line {
    height: 0;
}
QDialog {
    background: #161c23;
}
QDialogButtonBox QPushButton {
    min-width: 80px;
}
"""
