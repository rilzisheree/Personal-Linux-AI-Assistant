---
name: Stream truncation diagnostics
description: Live stream-versus-non-stream comparisons require the Linux host that runs Ollama; Replit localhost is a separate environment.
---

Run the truncation matrix and the stream=false comparison on the same Linux machine as Ollama, not from the Replit API runtime.

**Why:** The API and Ollama processes may both use localhost while running in different environments, so a successful Replit API health check does not prove the model transport is reachable.

**How to apply:** Use the count-only diagnostic command on the Ollama host, then compare Ollama, backend, SSE, and Qt display counters before changing generation settings.