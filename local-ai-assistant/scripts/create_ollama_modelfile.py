#!/usr/bin/env python3
"""Create a cautious Ollama Modelfile for a converted Gemma router adapter.

The LoRA trainer saves a Hugging Face/PEFT adapter. Ollama may require a
GGUF-converted adapter depending on its installed version. This helper does
not pretend to convert weights; it writes the exact Modelfile and leaves
ollama create to validate compatibility on the target Ollama build.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_ai_assistant.assistant_core import ROUTER_SYSTEM_PROMPT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="gemma3:270m")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--name", default="lura-gemma-router")
    parser.add_argument("--output", type=Path, default=Path("training/Modelfile.router"))
    args = parser.parse_args()
    if not args.adapter.exists():
        parser.error(f"adapter path does not exist: {args.adapter}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompt = ROUTER_SYSTEM_PROMPT.replace('"""', '\\"\\"\\"')
    content = (
        f"FROM {args.base}\n"
        f"ADAPTER {args.adapter}\n\n"
        f'SYSTEM """{prompt}"""\n\n'
        "PARAMETER temperature 0\n"
        "PARAMETER num_predict 8\n"
        'FORMAT """{"type":"string","enum":["SIMPLE","FUNCTION","REASONING"]}"""\n'
    )
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Validate with: ollama create {args.name} -f {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())