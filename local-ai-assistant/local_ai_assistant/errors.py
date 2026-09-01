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


class AssistantBackendError(Exception):
    """Base class for failures from any selectable AI backend."""


class AssistantUnavailableError(AssistantBackendError):
    """A selected hosted backend could not be reached."""


class AssistantAuthenticationError(AssistantBackendError):
    """A selected hosted backend rejected or lacks credentials."""


class AssistantProtocolError(AssistantBackendError):
    """A selected hosted backend returned an invalid or failed response."""


class AssistantCancelledError(AssistantBackendError):
    """A hosted generation was cancelled by the user."""


def format_backend_error(error: Exception, backend_name: str) -> str:
    if isinstance(error, AssistantAuthenticationError):
        return f"{backend_name} authentication failed: {error}"
    if isinstance(error, AssistantUnavailableError):
        return f"{backend_name} unavailable: {error}"
    if isinstance(error, AssistantCancelledError):
        return "Generation stopped."
    if isinstance(error, AssistantProtocolError):
        return f"{backend_name} returned an error: {error}"
    return format_ollama_error(error)


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
