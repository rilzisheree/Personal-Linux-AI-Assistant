"""Permission-restricted local storage for provider credentials."""

from __future__ import annotations

import os
from pathlib import Path


HOSTED_API_KEY_PATH = (
    Path.home() / ".config" / "local-ai-assistant" / "hosted-api.key"
)
GEMINI_API_KEY_PATH = (
    Path.home() / ".config" / "local-ai-assistant" / "gemini-api.key"
)


def load_hosted_api_key() -> str:
    try:
        return HOSTED_API_KEY_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except OSError as error:
        raise RuntimeError(
            f"Could not read the hosted API key file: {error}"
        ) from error


def save_hosted_api_key(api_key: str) -> None:
    api_key = api_key.strip()
    if not api_key:
        raise ValueError("Hosted API key cannot be empty.")
    HOSTED_API_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = HOSTED_API_KEY_PATH.with_suffix(".tmp")
    temporary_path.write_text(api_key + "\n", encoding="utf-8")
    temporary_path.chmod(0o600)
    temporary_path.replace(HOSTED_API_KEY_PATH)


def load_gemini_api_key() -> str:
    """Prefer the deployment secret, then use the local desktop credential."""
    environment_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    try:
        return GEMINI_API_KEY_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except OSError as error:
        raise RuntimeError(
            f"Could not read the Gemini API key file: {error}"
        ) from error


def save_gemini_api_key(api_key: str) -> None:
    api_key = api_key.strip()
    if not api_key:
        raise ValueError("Gemini API key cannot be empty.")
    GEMINI_API_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = GEMINI_API_KEY_PATH.with_suffix(".tmp")
    temporary_path.write_text(api_key + "\n", encoding="utf-8")
    temporary_path.chmod(0o600)
    temporary_path.replace(GEMINI_API_KEY_PATH)