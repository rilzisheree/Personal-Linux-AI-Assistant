---
name: Gemini OpenAI compatibility tools
description: Compatibility constraint for sending Lura's local tool registry through Google's OpenAI-compatible endpoint.
---

Google's Gemini OpenAI-compatible chat endpoint may reject multiple function declarations with only `400 INVALID_ARGUMENT`, even though ordinary text chat and the native Gemini API support tool calling.

**Why:** Lura's desktop assistant exposes many local tools to the model, so forwarding the entire registry can make every hosted Gemini message fail before generation starts.

**How to apply:** Keep the compatibility adapter's multi-tool fallback disabled or route Gemini tool-enabled chat through the native Gemini API; do not change Ollama or OpenAI/OpenRouter tool behavior.