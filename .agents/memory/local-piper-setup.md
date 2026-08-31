---
name: Local Piper setup
description: The Python environment requirement for Lura's offline Piper TTS.
---

Install Piper inside Lura's project virtual environment rather than the system Python:

```bash
cd ~/Personal-Linux-AI-Assistant/local-ai-assistant
source .venv/bin/activate
python -m pip install piper-tts
```

**Why:** Arch Linux protects its system Python with PEP 668, so installing `piper-tts` from the repository root with system Python fails as an externally managed environment.

**How to apply:** Activate Lura's `.venv` before installing Piper or launching the app. The `piper` executable must be discoverable from that environment.