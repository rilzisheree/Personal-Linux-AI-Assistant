---
name: PipeWire recorder exit status
description: The local voice recorder's stop behavior and valid WAV handling.
---

`pw-record` may exit with status 1 after an intentional SIGINT stop even when it has successfully finalized a usable WAV file. Treat the audio file as the source of truth for an intentional stop; only report failure when the file is missing or too small.

**Why:** The local voice input worked from the terminal with Ctrl+C, but the application rejected the same recording because it treated the recorder's nonzero stop status as fatal.

**How to apply:** When changing local recorder shutdown or validation, preserve SIGINT-based finalization and validate the resulting WAV before surfacing a recorder failure.