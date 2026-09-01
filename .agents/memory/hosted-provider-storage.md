---
name: Hosted provider storage
description: Selection and local storage boundary for hosted AI providers in the desktop app.
---

Hosted AI access should remain a selectable backend alongside Ollama, with API keys kept outside the general JSON settings and conversation history.

**Why:** The desktop app needs user-configurable hosted access, but provider credentials should not be mixed into ordinary settings or persisted with conversations.

**How to apply:** Add new hosted providers through the backend interface and keep credentials in the owner-only local credential store; preserve Ollama as the default and existing local path.