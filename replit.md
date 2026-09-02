# Lura

Lura is a native PySide6 Linux desktop assistant that chats with local Ollama
or hosted Google Gemini and executes a small set of permission-gated tools.

## Run & Operate

- `cd local-ai-assistant && python3 -m unittest discover -s tests -v` — run
  dependency-light tests
- `cd local-ai-assistant && python3 -m local_ai_assistant.app` — launch the
  desktop app
- `cd local-ai-assistant && python3 -m local_ai_assistant.app --background` —
  launch hidden in the system tray when a desktop tray is available
- `cd local-ai-assistant && ./run.sh` — launch through the convenience script
- `cd local-ai-assistant && python3 -m local_ai_assistant.api` — launch the
  password-protected API service; the desktop app initializes its password on
  first launch
- The app expects a local Ollama service at `http://localhost:11434` by default.
  The URL and model are configurable in Settings.
- The API service uses Gemini when `GEMINI_API_KEY` is configured. Set
  `LURA_AI_PROVIDER=gemini` and optionally `LURA_GEMINI_MODEL` to select it
  explicitly. Keep the key in Replit Secrets. The default is
  `gemini-3.6-flash`.
- Settings includes optional system-tray background mode and user-level Linux
  autostart. Autostart creates `~/.config/autostart/lura.desktop` and launches
  Lura with `--background`; both options are disabled by default.
- Settings can enable multiple persistent wake-word aliases and a temporary
  continuous conversation session after any configured alias. The session
  reuses VAD/STT/TTS, pauses the microphone during TTS, supports explicit
  goodbye phrases and manual Stop, and returns to wake-word listening after
  its timeout.
- Settings includes an editable User Profile section for the user's name,
  preferred address, assistant role, owner label, and application-install
  preference. These values are saved separately from live system status.
- Settings includes a Permissions tab for per-action default/always-allow/
  ask/block policies and custom app launchers. A launcher such as `Firefox` →
  `firefox` is executed as a direct process with no shell interpretation.

## Stack

- Python 3.10+
- PySide6 6.7+
- Ollama HTTP API with newline-delimited streaming
- Google Gemini REST API with server-sent event streaming
- Local SQLite persistence for conversations and JSON persistence for settings
- Dependency-free single-user password-protected HTTP API with server-sent event
  chat streaming
- Standard-library current-information clients for weather, search, maps, and
  currency conversion

## Where things live

- `local-ai-assistant/local_ai_assistant/app.py` — Qt entry point
- `local-ai-assistant/local_ai_assistant/ui/` — main window and chat widgets
- `local-ai-assistant/local_ai_assistant/ollama.py` — Ollama HTTP and stream
  protocol adapter
- `local-ai-assistant/local_ai_assistant/tools.py` — native tool schemas,
  implementations, and permission gates
- `local-ai-assistant/local_ai_assistant/information_tools.py` — current
  information HTTP/RSS clients and response normalization
- `local-ai-assistant/local_ai_assistant/conversations.py` — local history
  persistence

## Architecture decisions

- Tool calls use Ollama's native `message.tool_calls` protocol; assistant text
  is never parsed as fake JSON.
- Every non-safe tool is blocked until the Qt UI receives an explicit Allow
  decision.
- Conversation history remains local SQLite for the current desktop milestone;
  the first launch imports the previous local JSON format, while a remote API
  provides a separate single-user SQLite store for API clients. The remote API
  only allowlists the safe `open_app` action; confirmation-required and dangerous
  desktop tools remain local-only.

## Product

- Stream local Ollama or Google Gemini chat responses.
- Switch models discovered from the configured Ollama endpoint.
- Keep an editable local user profile in
  `~/.config/local-ai-assistant/user_profile.json`; profile facts are kept
  separate from live system probes.
- Discover Flatpak applications through `flatpak list` and native applications
  through freedesktop desktop entries before launching or closing them.
- Query structured current PC state through controlled GPU, CPU, RAM, storage,
  process, window, display, network, power, and application tools.
- Retrieve current weather, news, web and Wikipedia knowledge, currency rates,
  places, directions, travel research, and game information through safe
  model-selected information tools.
- Save, switch, clear, and export conversations locally.
- Run safe system-information tools and open applications.
- Run normal reversible actions automatically, while confirmation-required and
  dangerous actions use a short-lived exact-call voice approval through local
  Piper and Whisper when available.
- Capture screenshots for local vision-capable Ollama models.
- Inspect and manipulate files, Hyprland windows, and pointer/keyboard input
  through explicit tool calls.

## User preferences

- Keep the Linux desktop app as the current scope; do not build the phone/web
  client until the desktop milestones are stable.

## Gotchas

- Ollama must be installed, running, and have the selected model pulled when
  using the local provider. Hosted Gemini requires a valid `GEMINI_API_KEY`
  Secret and a model available to that Google API key.
- The Phase 6 API uses one local password, keeps API data separate from the
  desktop history database, streams chat over SSE, and only exposes the
  allowlisted safe `open_app` action remotely. The desktop app asks for the password
  on launch and starts a localhost API automatically unless
  `LURA_API_AUTOSTART=0`. Phase 3 Linux
  integration is implemented through the tool registry: Hyprland window
  control, Wayland/X11 screenshots, bounded local file operations, and
  pointer/keyboard automation. Phase 4 voice is implemented as optional local
  PipeWire/ALSA recording, Whisper/whisper.cpp transcription, and
  Piper/eSpeak-NG playback. Tools and voice report clear unavailable-backend
  errors when the host lacks the required Linux utility.
- The Phase 7 web companion is `artifacts/lura-web/`; it uses the existing
  cookie-authenticated HTTP/SSE API and intentionally does not expose desktop
  control tools remotely.
- Arbitrary shell execution is not available to the model. The legacy `exec`
  seam accepts only a small read-only command allowlist without approval;
  mutating or unknown commands remain confirmation-gated, and the structured
  tools are the preferred interface.
- Unambiguous requests such as “can you open Firefox?” are dispatched by the
  application before the response model runs, so the model cannot replace a
  local launch with a generic permission refusal. Custom launchers use the same
  `open_app` permission policy.
- The Gemini provider is text/SSE based. The reference Mark-LI project uses
  Gemini Live for realtime audio, which is a different API and is not used by
  this text chat path.
- Current-information tools use public Open-Meteo, Google News RSS, Wikipedia,
  ExchangeRate-API, OpenStreetMap/OSRM, and DuckDuckGo endpoints. They require
  outbound internet access, return explicit failures, and never substitute
  guessed live data.

## Pointers

- See `local-ai-assistant/README.md` for installation, Ollama setup, and
  troubleshooting.

## Gemma router training

- Generate the balanced, scenario-isolated dataset with
  `cd local-ai-assistant && python3 scripts/generate_router_dataset.py`.
- Install optional training packages from `requirements-training.txt`, then
  run `python3 scripts/train_router.py`. The default base checkpoint is
  `google/gemma-3-270m-it`; Hugging Face access requires accepting Google's
  Gemma terms.
- Evaluate only the untouched test split with
  `python3 scripts/evaluate_router.py --backend transformers --model-path
  training/router_lora`, or evaluate an Ollama model with
  `--backend ollama --ollama-model <model>`.
- The report includes accuracy, macro precision/recall/F1, per-label metrics,
  a confusion matrix, and average/p50/p95 latency. The script exits non-zero
  below 90% accuracy instead of hiding a poor model.
- LoRA output is an adapter, not an Ollama model by itself. Use
  `python3 scripts/create_ollama_modelfile.py --adapter <converted-adapter>`
  and let `ollama create` validate the adapter format. If it succeeds, set
  `LURA_ROUTER_MODEL` to the created model name before launching Lura. The
  default remains `gemma3:270m`.
