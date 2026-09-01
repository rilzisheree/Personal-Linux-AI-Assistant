---
name: Desktop Ollama-only runtime
description: The native desktop App is intentionally local-only; hosted AI belongs only to the separate web/API companion.
---

The native desktop App uses Ollama as its only AI runtime. Do not add cloud-provider selectors, model fields, API-key inputs, or hosted credential storage back into the desktop configuration or UI.

**Why:** The user chose to avoid Gemini and other hosted AI providers after free-tier quota and latency issues, and wants the desktop assistant to remain local.

**How to apply:** Keep any hosted-provider code isolated to the separate web/API companion path. Desktop settings should expose only Ollama connection/model/context controls plus other local features.