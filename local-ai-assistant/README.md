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
machine only. Voice input and output use optional local host tools. An optional
Railway web companion and a local Telegram phone bridge are documented below;
the desktop app and Ollama remain local.

## Authenticated API service (Phase 6)

The optional API service provides password-protected access to conversations
and the allowlisted safe `open_app` action. It can use local Ollama or hosted
Google Gemini, and stores the single local user's sessions and conversations
in a separate SQLite database.

Start it manually from this directory with:

```bash
python -m local_ai_assistant.api
```

When started manually without `SESSION_SECRET`, Lura creates a random,
permission-restricted local secret at `~/.config/local-ai-assistant/api-session.secret`.
For a deployed or shared server, set `SESSION_SECRET` explicitly instead. Set
`LURA_API_PASSWORD` to initialize the single API password in a headless
environment. The desktop app prompts for this password on first launch and
starts a localhost API automatically when it opens; set `LURA_API_AUTOSTART=0`
to disable that behavior.

The service listens on `PORT` (default `8000`) when started directly. Configure
these environment variables when needed:

- `SESSION_SECRET` — optional locally; at least 16 characters when supplied
- `LURA_API_PASSWORD` — optional initial password for headless API setup
- `GEMINI_API_KEY` — Google AI Studio key; keep this in a Replit Secret, never in
  source code
- `LURA_AI_PROVIDER` — `gemini` or `ollama`; defaults to `gemini` when
  `GEMINI_API_KEY` is present
- `LURA_GEMINI_MODEL` — Gemini model, defaulting to `gemini-3.6-flash`
- `LURA_OLLAMA_URL` — Ollama base URL, defaulting to `http://localhost:11434`
- `LURA_MODEL` — Ollama model, defaulting to `qwen3.5:4b`
- `LURA_API_DATABASE` — optional API SQLite path
- `LURA_API_HOST` — bind address, defaulting to `0.0.0.0`
- `LURA_ALLOWED_ORIGIN` — optional exact browser origin for credentialed CORS
- `LURA_COOKIE_SECURE` — set to `1` when HTTPS is guaranteed
- `LURA_API_AUTOSTART` — set to `0` to prevent desktop API autostart
- `LURA_API_PORT` — desktop autostart port, defaulting to `8000`

Available endpoints:

- `POST /api/auth/login` with `{ "password": "..." }`
- `POST /api/auth/logout` and `GET /api/me`
- `GET /api/conversations` and `POST /api/conversations`
- `GET /api/conversations/:id`
- `POST /api/conversations/:id/messages` — server-sent events with streamed tokens
- `GET /api/models`
- `GET /api/health`

There are no email accounts or registration flows. Remote API requests can use
the allowlisted safe `open_app` tool so the phone can launch an installed
application on the trusted Linux machine. Telegram can also use app lifecycle
and window tools (`close_app`, `restart_app`, `list_windows`, `focus_window`,
`move_window`, and `resize_window`), take desktop screenshots, system status
tools, and read-only file search/reading. Captured screenshots are uploaded
only to the configured private Telegram chat. Terminal commands, file
mutations, keyboard/mouse input, and destructive window/file tools remain
available only inside the trusted desktop app with its existing permission
dialogs.

## Telegram phone companion

For a simpler phone control channel, Lura includes a Telegram long-polling bot
that runs directly on the same Linux machine as Ollama. This avoids Railway,
Cloudflare, browser CORS, and inbound firewall ports. Telegram messages do pass
through Telegram's servers; this is not an end-to-end encrypted Secret Chat.
Responses are sent immediately as a thinking message and updated while Ollama
generates the answer, rather than waiting for the full response to finish.

The local bot intentionally accepts private messages only from one configured
Telegram numeric user ID. The selected remote tools are explicitly
auto-approved for that one allowlisted user; dangerous tools are not exposed
to Telegram.
The bot token must be stored on the Linux machine, not committed to the
repository or pasted into chat. The desktop app can save the token and start
the listener from Settings; enabling Lura's existing autostart option starts
the Telegram listener as well.

Set these values in the Linux terminal before starting the bot:

```bash
export TELEGRAM_BOT_TOKEN="token-from-BotFather"
export TELEGRAM_ALLOWED_USER_ID="your-numeric-telegram-user-id"
```

Then run:

```bash
python -m local_ai_assistant.telegram_bot
```

The bot uses the local `LURA_OLLAMA_URL` and `LURA_MODEL` settings when
provided. Use `/start`, `/help`, or `/reset` in the Telegram chat. The bot
must be running in the Linux desktop user's session for `open_app` to launch a
visible graphical application.

The Replit Telegram connector can authenticate Telegram API calls from this
workspace, but its credentials are not available to a cloned process running
on the Linux computer. For the no-tunnel architecture, configure the same bot
token locally from BotFather.

## Railway-hosted web companion with a local API

The web companion in `artifacts/lura-web` can be hosted as a static Railway
service while the API and Ollama remain on your Linux computer. The repository
root includes `railway.json` with the build and start commands. In the Railway
service, set:

- `BASE_PATH=/`
- `VITE_LURA_API_URL=https://your-secure-tunnel.example`

`VITE_LURA_API_URL` is the API origin only; do not append `/api`. Railway cannot
reach `localhost` on your computer, so the local API must be reachable through
an HTTPS tunnel or private network gateway. Do not expose the raw API port to
the public internet.

When the Railway origin is known, run the local API with an exact CORS origin:

```bash
LURA_ALLOWED_ORIGIN=https://your-app.up.railway.app \
LURA_COOKIE_SECURE=1 \
python -m local_ai_assistant.api
```

Use a strong `SESSION_SECRET` and `LURA_API_PASSWORD` for any API reachable
through a tunnel. The web client uses the API's expiring browser session
token for cross-origin requests and keeps it in session storage; same-origin
local use continues to work with the HttpOnly cookie.

## Requirements

- Linux with Python 3.10 or newer
- PySide6 6.7 or newer
- Ollama installed and running
- A local Ollama model, with `qwen3.5:4b` as the default
- For voice input: PipeWire `pw-record` or ALSA `arecord`, plus a local
  Whisper CLI (`whisper` or `whisper-cli`)
- For voice output: Piper (`piper-tts`) plus `pw-play`/`aplay`

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

Whisper and Piper model downloads are large, so the Python packages are
installed by the project but the model files are downloaded only when needed.
Text chat remains usable when voice tools are unavailable.

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
- Ollama model, defaulting to `qwen3.5:4b`
- Ollama context size, defaulting to 8,192 tokens
- Push-to-talk microphone input and optional microphone device
- Whisper model and language
- Spoken responses, TTS engine, and voice
- Multiple wake-word aliases, which can be added, edited, or removed
- Optional continuous conversation mode after the wake word
- Conversation timeout and a short voice transition delay

When wake-word listening is enabled, saying any configured alias starts a
conversation. Aliases use the same single recorder stream and overlapping
transcription windows, so adding aliases does not create competing microphone
listeners. Similar aliases are scored together and only the strongest match
activates one handoff.

When continuous conversation mode is enabled, saying any configured wake word
starts a
temporary session. Lura listens for one turn at a time after each response,
uses the existing VAD to detect when you finish, and keeps the microphone
disabled while TTS is speaking. The session ends after the configured quiet
timeout, when you say “goodbye”, “stop listening”, or “go to sleep”, or when
you press Stop. Wake-word-only listening then resumes. If the initial
wake-word recording contains a command, the text after the wake word is used
as the first turn.

Voice responses provide exactly two local Piper choices:

- **Jarvis** — `en_GB-alan-medium`
- **Laura** — `en_US-amy-medium`

On first use, Lura downloads the selected model and its matching `.onnx.json`
file into:

```text
~/.local/share/lura/piper/
```

Choosing either voice uses Piper directly; Lura no longer silently substitutes
eSpeak when Piper or its model is unavailable.

Piper keeps its native WAV sample rate through playback and uses a balanced
normal-speed synthesis profile with normalized audio and reduced stochastic
noise for clearer, more consistent speech without an extra conversion step.

Settings are stored locally at:

```text
~/.config/local-ai-assistant/config.json
```

The native desktop App stores only Ollama and voice preferences locally. It has
no cloud provider settings and never asks for or stores an AI API key. The
separate Railway/web API companion may use `GEMINI_API_KEY` as documented above;
that path is independent of the native desktop App.
Conversation history is stored in SQLite at
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
displaying it. Chat requests send `keep_alive=10m` so Ollama can reuse the
loaded model between nearby messages. The native app also logs one
`VOICE_LATENCY` JSON record per conversation at INFO level; compare
`end_of_user_speech`, `stt_completed`, `ollama_started`, `ollama_first_token`,
`first_sentence_available`, `tts_synthesis_started`, `first_audio_playback`,
and `spoken_response_completed` to locate the slow stage.

For thinking-capable Ollama models such as Qwen3/Qwen3.5, routine short
requests send `think: false` and a 96-token response budget so greetings,
lookups, and one-step actions do not spend time on unnecessary reasoning.
Longer or reasoning-oriented prompts send `think: true`, preserving the
model's ability to work through complex requests. The client logs an
`[Ollama] GENERATION_METRICS` JSON record with time to first model token,
Ollama's aggregate generated-token count, estimated reasoning/output token
counts, generation time, and tokens per second. Ollama currently reports
thinking and output as one aggregate `eval_count`, so the separate reasoning
and output counts are explicitly estimates based on streamed character counts.

The desktop app can use `gemma3:270m` as a local routing model while keeping the
configured Qwen model for complex turns. Install both models once with:

```bash
ollama pull gemma3:270m
ollama pull qwen3.5:2b
```

If Gemma is unavailable, Luna logs the routing failure and falls back to the
configured Qwen model so existing functionality remains available. Ollama
decides GPU placement automatically; use `ollama ps` while testing to confirm
that both models remain resident on an 8 GB GPU.

To evaluate the router on the target machine, run the 30-case benchmark:

```bash
python3 scripts/benchmark_router.py
```

It reports routing accuracy, false-simple classifications, false-reasoning
classifications, function selection accuracy, and average routing latency.

When spoken responses are enabled, complete sentences are sent to the local
TTS worker while Ollama is still generating. The chat continues rendering
tokens independently, and Piper's in-process voice model cache is reused
across those sentence chunks.

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

Enable spoken responses in Settings. Piper requires both the `piper`
executable and a local `.onnx` voice model path. Playback prefers `pw-play`,
then `aplay`/`paplay`. Text responses are still displayed if speech fails.

## Tests

Run the focused tests with the project dependencies installed:

```bash
python -m unittest discover -s tests -v
```

They cover Ollama's newline-delimited JSON stream parsing, configuration
defaults/round-tripping, conversations, permission gates, local tool handlers,
and local voice discovery/error handling.

## Project structure

```text
local-ai-assistant/
├── local_ai_assistant/
│   ├── app.py                 # Qt application entry point
│   ├── assistant_core.py      # GUI-independent assistant boundary
│   ├── config.py              # validated local settings
│   ├── conversations.py       # local conversation model and JSON store
│   ├── errors.py              # user-facing error categories
│   ├── gemini_api.py          # Gemini adapter for the separate web/API path
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
button and optional local backends. A future phone or web client remains
intentionally out of scope for the desktop milestone.
