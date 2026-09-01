#!/usr/bin/env python3
"""Create a cautious Ollama Modelfile for a Gemma router adapter.

The LoRA trainer saves a Hugging Face/PEFT adapter. For architectures supported
by the target Ollama build, the adapter directory can be referenced directly.
This helper does not convert weights; for Gemma 3 builds that report
"unsupported architecture", use merge_router_adapter.py and import the merged
GGUF model instead.
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
    if not args.adapter.is_dir():
        parser.error("--adapter must point to an adapter directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompt = ROUTER_SYSTEM_PROMPT.replace('"""', '\\"\\"\\"')
    content = (
        f"FROM {args.base}\n"
        f"ADAPTER {args.adapter}\n\n"
        f'SYSTEM """{prompt}"""\n\n'
        "PARAMETER temperature 0\n"
        "PARAMETER num_predict 8\n"
    )
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Validate with: ollama create {args.name} -f {args.output}")
    print(
        "If Ollama reports 'unsupported architecture' for Gemma 3, merge the "
        "adapter first with scripts/merge_router_adapter.py and import the "
        "resulting GGUF model."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())