# Lura

Lura is a native PySide6 Linux desktop assistant that chats with a local Ollama
server and executes a small set of permission-gated tools.

## Run & Operate

- `cd local-ai-assistant && python3 -m unittest discover -s tests -v` — run
  dependency-light tests
- `cd local-ai-assistant && python3 -m local_ai_assistant.app` — launch the
  desktop app
- `cd local-ai-assistant && ./run.sh` — launch through the convenience script
- The app expects a local Ollama service at `http://localhost:11434` by default.
  The URL and model are configurable in Settings.

## Stack

- Python 3.10+
- PySide6 6.7+
- Ollama HTTP API with newline-delimited streaming
- Local JSON persistence for conversations and settings

## Where things live

- `local-ai-assistant/local_ai_assistant/app.py` — Qt entry point
- `local-ai-assistant/local_ai_assistant/ui/` — main window and chat widgets
- `local-ai-assistant/local_ai_assistant/ollama.py` — Ollama HTTP and stream
  protocol adapter
- `local-ai-assistant/local_ai_assistant/tools.py` — native tool schemas,
  implementations, and permission gates
- `local-ai-assistant/local_ai_assistant/conversations.py` — local history
  persistence

## Architecture decisions

- Tool calls use Ollama's native `message.tool_calls` protocol; assistant text
  is never parsed as fake JSON.
- Every non-safe tool is blocked until the Qt UI receives an explicit Allow
  decision.
- Conversation history remains local JSON for the current desktop milestone;
  SQLite and a remote API are later phases from the original handoff.

## Product

- Stream local Ollama chat responses.
- Switch models discovered from the configured Ollama endpoint.
- Save, switch, clear, and export conversations locally.
- Run safe system-information tools and open applications.
- Review and approve confirmation-required or dangerous terminal/app actions.
- Capture screenshots for local vision-capable Ollama models.
- Inspect and manipulate files, Hyprland windows, and pointer/keyboard input
  through explicit tool calls.

## User preferences

- Keep the Linux desktop app as the current scope; do not build the phone/web
  client until the desktop milestones are stable.

## Gotchas

- Ollama must be installed, running, and have the selected model pulled.
- Voice, SQLite memory, and the future API are not implemented yet. Phase 3
  Linux integration is implemented through the tool registry: Hyprland window
  control, Wayland/X11 screenshots, bounded local file operations, and
  pointer/keyboard automation. Tools still report clear unavailable-backend
  errors when the host lacks the required Linux utility.

## Pointers

- See `local-ai-assistant/README.md` for installation, Ollama setup, and
  troubleshooting.
