"""Application stylesheet."""

from __future__ import annotations


APP_STYLE = """
QMainWindow, QWidget {
    background: #02060d;
    color: #dceeff;
    font-family: "Noto Sans", "Inter", sans-serif;
    font-size: 14px;
}
QMainWindow {
    background: #02060d;
}
QFrame#topBar {
    background: #040b14;
    border-bottom: 1px solid #12314b;
}
QFrame#historyPanel {
    background: #030a12;
    border-right: 1px solid #12314b;
}
QLabel#appMark {
    color: #7ddcff;
    font-size: 22px;
    font-weight: 700;
}
QLabel#appSubtitle {
    color: #55758f;
    font-size: 12px;
}
QLabel#sectionLabel {
    color: #3c8fba;
    font-size: 11px;
    font-weight: 700;
}
QListWidget#historyList {
    background: transparent;
    border: 0;
    outline: 0;
    color: #7594aa;
    padding: 2px;
}
QListWidget#historyList::item {
    border: 1px solid transparent;
    border-radius: 2px;
    padding: 10px 8px;
    margin: 3px 0;
}
QListWidget#historyList::item:hover {
    background: #071a29;
    border-color: #12466b;
}
QListWidget#historyList::item:selected {
    background: #062a40;
    border-color: #1b9bd1;
    color: #a7eaff;
}
QWidget#emptyState {
    background: #02060d;
}
QLabel#luraVisual {
    background: #02060d;
    color: #24c9ff;
}
QLabel#heroKicker {
    color: #2ddcff;
    font-size: 11px;
    font-weight: 700;
}
QLabel#heroTitle {
    color: #c7edff;
    font-size: 22px;
    font-weight: 600;
}
QLabel#heroCopy {
    color: #527b99;
    font-size: 12px;
}
QFrame#assistantBubble, QFrame#userBubble {
    border-radius: 3px;
}
QFrame#assistantBubble {
    background: #040e18;
    border: 1px solid #12405c;
}
QFrame#userBubble {
    background: #062337;
    border: 1px solid #167aa5;
}
QFrame#assistantBubble QTextBrowser, QFrame#userBubble QTextBrowser {
    background: transparent;
    border: 0;
    color: #dceeff;
}
QFrame#assistantBubble QLabel, QFrame#userBubble QLabel {
    color: #62c9e9;
}
QLabel#statusLabel {
    border: 1px solid #145073;
    border-radius: 2px;
    color: #72a9c4;
    padding: 5px 10px;
}
QLabel#statusLabel[status="connected"] {
    border-color: #169bc5;
    color: #7df0ff;
}
QLabel#statusLabel[status="error"] {
    border-color: #8b5060;
    color: #ff9db5;
}
QComboBox, QLineEdit, QTextEdit {
    background: #030b14;
    border: 1px solid #145073;
    border-radius: 2px;
    color: #dceeff;
    padding: 9px 10px;
    selection-background-color: #0c6589;
}
QComboBox:hover, QLineEdit:focus, QTextEdit:focus {
    border-color: #24c9ff;
}
QComboBox::drop-down {
    border: 0;
    width: 24px;
}
QPushButton {
    background: #073149;
    border: 1px solid #1a8ab7;
    border-radius: 2px;
    color: #b8efff;
    padding: 9px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background: #0a4d6c;
}
QPushButton:pressed {
    background: #062337;
}
QPushButton:disabled {
    background: #07131d;
    border-color: #193344;
    color: #456276;
}
QPushButton#stopButton {
    background: transparent;
    border-color: #8b5060;
    color: #ff9db5;
}
QPushButton#stopButton:hover {
    background: #301827;
}
QPushButton#settingsButton, QPushButton#newChatButton {
    background: transparent;
    border-color: #19425a;
    color: #73c8e8;
}
QScrollArea {
    border: 0;
    background: #02060d;
}
QScrollBar:vertical {
    background: #02060d;
    width: 10px;
    margin: 4px;
}
QScrollBar::handle:vertical {
    background: #12405c;
    border-radius: 2px;
    min-height: 28px;
}
QScrollBar::add-line, QScrollBar::sub-line {
    height: 0;
}
QDialog {
    background: #040b14;
}
QDialogButtonBox QPushButton {
    min-width: 80px;
}
"""
