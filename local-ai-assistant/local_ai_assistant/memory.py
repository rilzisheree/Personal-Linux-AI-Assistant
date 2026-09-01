"""Small explicit local memory store for user-approved assistant facts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class MemoryStore:
    """Persist only facts the user explicitly asks Lura to remember."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (
            Path.home() / ".config" / "local-ai-assistant" / "memory.json"
        )

    def facts(self) -> list[str]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
            return []
        if not isinstance(raw, list):
            return []
        return [item.strip() for item in raw if isinstance(item, str) and item.strip()]

    def add(self, fact: str) -> str:
        fact = fact.strip()
        if not fact:
            raise ValueError("Memory fact cannot be empty.")
        facts = self.facts()
        if fact.casefold() not in {item.casefold() for item in facts}:
            facts.append(fact)
        self._save(facts[-100:])
        return fact

    def forget(self, fact: str) -> bool:
        normalized = fact.strip().casefold()
        facts = self.facts()
        remaining = [item for item in facts if item.casefold() != normalized]
        if len(remaining) == len(facts):
            return False
        self._save(remaining)
        return True

    def context(self, limit: int = 30) -> str:
        return "\n".join(f"- {fact}" for fact in self.facts()[-limit:])

    def _save(self, facts: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f"{self.path.name}.",
            dir=self.path.parent,
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(facts, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            temporary_path.chmod(0o600)
            temporary_path.replace(self.path)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass