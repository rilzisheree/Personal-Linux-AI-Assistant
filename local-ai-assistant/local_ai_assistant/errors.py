"""Typed errors shared by the Ollama adapter and the UI."""

from __future__ import annotations


class OllamaError(Exception):
    """Base class for expected Ollama failures."""


class OllamaUnavailableError(OllamaError):
    """The configured Ollama endpoint could not be reached."""


class OllamaModelNotFoundError(OllamaError):
    """The requested model is not installed in Ollama."""


class OllamaProtocolError(OllamaError):
    """Ollama returned an invalid or explicitly failed response."""


class OllamaCancelledError(OllamaError):
    """The active generation was cancelled by the user."""


def format_ollama_error(error: Exception) -> str:
    """Return a concise message suitable for displaying in the chat UI."""

    if isinstance(error, OllamaModelNotFoundError):
        return f"Model not found: {error}. Pull it with `ollama pull <model>` or choose another model."
    if isinstance(error, OllamaUnavailableError):
        return f"Ollama unavailable: {error}. Check the service and the URL in Settings."
    if isinstance(error, OllamaCancelledError):
        return "Generation stopped."
    if isinstance(error, OllamaProtocolError):
        return f"Ollama returned an error: {error}"
    return f"Unexpected error: {error}"
