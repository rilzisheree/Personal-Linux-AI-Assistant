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
artwork as its empty-state focus. History is stored in SQLite on the local
machine only. Voice input and output use optional local host tools; there is
still no cloud provider, web UI, or phone client.

## Authenticated API service (Phase 6)

The optional API service provides authenticated access to conversations without
exposing the desktop control tools. It uses the same local Ollama endpoint and
stores accounts, sessions, and user-owned conversations in a separate SQLite
database.

Start it from this directory with:

```bash
SESSION_SECRET='use-a-long-random-value' python -m local_ai_assistant.api
```

The service listens on `PORT` (default `8000`). Configure these environment
variables when needed:

- `SESSION_SECRET` — required, at least 16 characters; never commit it
- `LURA_OLLAMA_URL` — Ollama base URL, defaulting to `http://localhost:11434`
- `LURA_MODEL` — default model, defaulting to `qwen3.5:4b`
- `LURA_API_DATABASE` — optional API SQLite path
- `LURA_API_HOST` — bind address, defaulting to `0.0.0.0`
- `LURA_ALLOWED_ORIGIN` — optional exact browser origin for credentialed CORS
- `LURA_COOKIE_SECURE` — set to `1` when HTTPS is guaranteed

Available endpoints:

- `POST /api/auth/register` and `POST /api/auth/login`
- `POST /api/auth/logout` and `GET /api/me`
- `GET /api/conversations` and `POST /api/conversations`
- `GET /api/conversations/:id`
- `POST /api/conversations/:id/messages` — server-sent events with streamed tokens
- `GET /api/models`
- `GET /api/health`

Remote API requests only receive chat responses. Desktop tools such as terminal
commands, file mutations, screenshots, Hyprland control, and keyboard/mouse
input remain available only inside the trusted desktop app with its existing
permission dialogs.

## Requirements

- Linux with Python 3.10 or newer
- PySide6 6.7 or newer
- Ollama installed and running
- A local Ollama model, with `qwen3.5:4b` as the default
- For voice input: PipeWire `pw-record` or ALSA `arecord`, plus a local
  Whisper CLI (`whisper` or `whisper-cli`)
- For voice output: `espeak-ng` plus `pw-play`/`aplay`, or Piper with a local
  voice model

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

Voice backends are intentionally host-provided because Whisper and Piper model
downloads are large and model licensing varies. No voice backend is contacted
over the network by Lura. Text chat remains usable when these optional
commands are not installed.

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
- Ollama context size, defaulting to 8,192 tokens
- Push-to-talk microphone input and optional microphone device
- Whisper model and language
- Spoken responses, TTS engine, and voice

When using eSpeak-NG, the voice choices include a British male voice with a
Jarvis-style character and a female-sounding voice. These are local synthetic
voices, not an imitation of a specific actor. The Custom voice option accepts
another eSpeak-NG voice name or a Piper model path.

Piper presets are also available for the British `en_GB-alan-medium` and
female `en_US-amy-medium` voices. Download each model and its matching `.onnx.json`
file from the Piper voice repository, then place them at:

```text
~/Models/piper/en_GB-alan-medium.onnx
~/Models/piper/en_GB-alan-medium.onnx.json
~/Models/piper/en_US-amy-medium.onnx
~/Models/piper/en_US-amy-medium.onnx.json
```

Choosing either Piper preset switches the TTS engine to Piper automatically.

Settings are stored locally at:

```text
~/.config/local-ai-assistant/config.json
```

The Ollama and voice preferences are stored locally. No credentials or prompts
are written to the settings file. Conversation history is stored in SQLite at
`$XDG_DATA_HOME/local-ai-assistant/conversations.db`, or
`~/.local/share/local-ai-assistant/conversations.db` when `XDG_DATA_HOME` is
not set. On first launch after upgrading, an existing `conversations.json` file
is imported automatically and left untouched as a fallback copy.

The context size is a local model setting, not a cloud account quota. It limits
how much conversation history and tool context Ollama sends to the model in one
request. Larger values support longer chats but require more RAM or VRAM. If a
model reaches its configured context limit, start a new chat or increase this
setting in Settings if the computer has enough memory.

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

### Voice input is unavailable

Hold the `MIC` button in the composer to record a message. Release it to run
local transcription and send the resulting text. The recorder prefers
PipeWire's `pw-record` and falls back to `arecord`. Transcription prefers the
OpenAI Whisper CLI and then whisper.cpp. Configure the model name (for
OpenAI Whisper) or model file path (for whisper.cpp) in Settings.

### Voice output is unavailable

Enable spoken responses in Settings. eSpeak-NG is the easiest local option
and uses a voice name such as `en-us`. Piper requires both the `piper`
executable and a local `.onnx` voice model path. Playback prefers `pw-play`,
then `aplay`/`paplay`. Text responses are still displayed if speech fails.

## Tests

Run the focused tests with the project dependencies installed:

```bash
python -m unittest discover -s tests -v
```

They cover Ollama's newline-delimited JSON stream parsing, configuration
defaults/round-tripping, conversations, permission gates, local tool handlers,
and voice backend selection/error handling.

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
│   ├── voice.py               # local recording, Whisper, and TTS adapters
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
│   ├── test_tools.py
│   ├── test_voice.py
│   └── test_workers.py
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

Phase 4 local voice input/output is implemented through the `MIC` push-to-talk
button and optional local backends. The future authenticated API and phone web
client are intentionally not included.
