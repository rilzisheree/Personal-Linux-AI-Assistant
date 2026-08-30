# Lura

Lura is a native-feeling PySide6 desktop chat client for Linux. It talks
directly to an Ollama server running on the same machine (or another URL you
configure) and streams responses into the conversation as they arrive.

The first continuation phase adds a local conversation manager: chats are
restored between launches, can be switched from the sidebar, and can be
started with New chat. Phase 2 adds Ollama-native tool calls with explicit
permission gates. Phase 3 adds Linux integration for Hyprland windows,
screenshots, local files, and pointer/keyboard input. The interface uses a
futuristic local-intelligence HUD visual language, with the supplied Lura core
artwork as its empty-state focus. History is stored as JSON on the local
machine only. There is still no cloud provider, web UI, voice, or phone client.

## Requirements

- Linux with Python 3.10 or newer
- PySide6 6.7 or newer
- Ollama installed and running
- A local Ollama model, with `qwen3.5:4b` as the default

The NVIDIA RTX 4060 and Intel i5-12400F do not require any special code
configuration. Ollama chooses the available acceleration; verify that Ollama
itself sees the GPU on the target machine.

## Install and run

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m local_ai_assistant.app
```

Or use the convenience launcher:

```bash
./run.sh
```

If your distribution manages Python packages externally, use a user
environment or the virtual environment above instead of installing PySide6
into the system Python.

## Prepare Ollama

Start Ollama using the normal CachyOS/Linux service or application method, then
pull the default model:

```bash
ollama serve
ollama pull qwen3.5:4b
```

If `ollama serve` is already managed by your user service, do not start a
second copy. The default endpoint is:

```text
http://localhost:11434
```

The app checks the endpoint on startup and discovers the models available from
`/api/tags`. The model field remains configurable even when the selected model
is not installed.

## Configuration

Use the settings button in the app to change:

- Ollama URL, defaulting to `http://localhost:11434`
- Model name, defaulting to `qwen3.5:4b`

Settings are stored locally at:

```text
~/.config/local-ai-assistant/config.json
```

Only the URL and model name are stored. No credentials or prompts are written
to the settings file. Conversation history is stored separately at
`$XDG_DATA_HOME/local-ai-assistant/conversations.json`, or
`~/.local/share/local-ai-assistant/conversations.json` when `XDG_DATA_HOME` is
not set.

## Troubleshooting

### "Ollama unavailable"

Confirm that the service is running and that the endpoint responds:

```bash
curl http://localhost:11434/api/tags
```

If Ollama is bound to a different host or port, enter that full URL in
Settings. Do not include `/api` in the configured base URL; the app adds API
paths itself.

### "Model not found"

Install the selected model, or choose a model already shown in the model
selector:

```bash
ollama list
ollama pull qwen3.5:4b
```

### Generation is slow

Check Ollama's own logs and model resource requirements first. The app streams
each token-sized response chunk and does not buffer a full answer before
displaying it.

### Stop does not return immediately

Stop closes the active HTTP response and asks the worker to cancel. A network
stack that is currently inside a read may take a short moment to unwind; the
conversation will clearly mark the request as stopped.

## Tests

The focused tests do not need PySide6:

```bash
python -m unittest discover -s tests -v
```

They cover Ollama's newline-delimited JSON stream parsing, configuration
defaults/round-tripping, conversations, permission gates, and the local tool
handlers.

## Project structure

```text
local-ai-assistant/
├── local_ai_assistant/
│   ├── app.py                 # Qt application entry point
│   ├── assistant_core.py      # GUI-independent assistant boundary
│   ├── config.py              # validated local settings
│   ├── conversations.py       # local conversation model and JSON store
│   ├── errors.py              # user-facing error categories
│   ├── ollama.py              # direct Ollama HTTP and streaming adapter
│   ├── tools.py               # tool registry and permission gates
│   ├── workers.py             # background Qt workers and tool-call loop
│   └── ui/
│       ├── chat_view.py       # conversation rendering
│       ├── main_window.py     # application shell and interactions
│       ├── settings_dialog.py # URL/model settings
│       └── styles.py          # Qt stylesheet
├── tests/
│   ├── test_config.py
│   ├── test_conversations.py
│   ├── test_ollama_parser.py
│   └── test_tools.py
├── pyproject.toml
└── run.sh
```

The assistant boundary and Ollama adapter are separate from the widgets.
Ollama-native function calls flow through the tool registry and its permission
gate. Phase 3's Hyprland tools use `hyprctl` JSON/dispatch commands, the
screenshot tool prefers `grim`/`hyprshot` on Wayland, file tools are bounded
to text files with confirmation for mutations, and input tools prefer
`wtype`/`ydotool` with X11 fallbacks where available. Screenshot paths are
attached to the next Ollama tool message so vision-capable models can inspect
the captured image.

The next phase is local voice input/output. The future authenticated API and
phone web client are intentionally not included.
