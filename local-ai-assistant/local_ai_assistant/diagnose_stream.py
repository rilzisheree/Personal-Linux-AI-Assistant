"""Compare one Ollama prompt over streaming and non-streaming transports.

The report intentionally contains counts and termination metadata only. It
does not print the prompt, response, credentials, or request body.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from threading import Event

from .ollama import ChatMessage, OllamaClient


DEFAULT_PROMPT = (
    "Explain Linux in detail. Cover its architecture, security model, "
    "package management, and common desktop use cases."
)


def run_once(client: OllamaClient, prompt: str, model: str, context_size: int) -> dict:
    result = {
        "model": model,
        "prompt_chars": len(prompt),
        "stream": {
            "done": False,
            "done_reason": "",
            "chunks": 0,
            "characters": 0,
            "generated_tokens": None,
            "error": None,
        },
        "non_stream": {
            "done": False,
            "done_reason": "",
            "chunks": 0,
            "characters": 0,
            "generated_tokens": None,
            "error": None,
        },
    }
    messages = [ChatMessage("user", prompt)]

    try:
        for event in client.stream_chat(messages, model, Event(), context_size=context_size):
            result["stream"]["chunks"] += 1
            result["stream"]["characters"] += len(event.content)
            if event.done:
                result["stream"]["done"] = True
                result["stream"]["done_reason"] = event.done_reason
                result["stream"]["generated_tokens"] = (event.metrics or {}).get("eval_count")
    except Exception as error:
        result["stream"]["error"] = type(error).__name__ + ": " + str(error)

    try:
        event = client.chat_once(messages, model, Event(), context_size=context_size)
        result["non_stream"].update(
            {
                "done": event.done,
                "done_reason": event.done_reason,
                "chunks": 1,
                "characters": len(event.content),
                "generated_tokens": (event.metrics or {}).get("eval_count"),
            }
        )
    except Exception as error:
        result["non_stream"]["error"] = type(error).__name__ + ": " + str(error)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--model",
        default=os.environ.get("LURA_MODEL", "qwen3.5:4b"),
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("LURA_OLLAMA_URL", "http://localhost:11434"),
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=int(os.environ.get("LURA_CONTEXT_SIZE", "8192")),
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.debug else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    report = run_once(
        OllamaClient(args.ollama_url),
        args.prompt,
        args.model,
        args.context_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["stream"]["error"] and not report["non_stream"]["error"] else 1


if __name__ == "__main__":
    raise SystemExit(main())