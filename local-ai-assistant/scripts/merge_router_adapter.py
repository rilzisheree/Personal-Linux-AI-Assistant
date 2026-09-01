#!/usr/bin/env python3
"""Merge a PEFT router adapter into its Hugging Face base model.

Ollama can run Gemma 3 models, but some Ollama builds do not accept Gemma 3
LoRA adapters through the Modelfile ADAPTER instruction. Merging first and
exporting the result to GGUF avoids that adapter-import limitation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path


def _adapter_source(adapter: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Return a PEFT-compatible adapter directory.

    PEFT normally writes adapter_model.safetensors. Some Transformers/Trainer
    combinations write model.safetensors instead, so normalize that filename
    in a temporary directory without modifying the user's training output.
    """

    expected = adapter / "adapter_model.safetensors"
    if expected.exists() or (adapter / "adapter_model.bin").exists():
        return adapter, None
    alternate = adapter / "model.safetensors"
    if not alternate.exists():
        raise SystemExit(
            f"Adapter weights not found in {adapter}; expected "
            "adapter_model.safetensors, adapter_model.bin, or model.safetensors."
        )

    temporary = tempfile.TemporaryDirectory(prefix="lura-router-adapter-")
    normalized = Path(temporary.name)
    for source in adapter.iterdir():
        if source.is_file():
            shutil.copy2(source, normalized / source.name)
    shutil.copy2(alternate, normalized / "adapter_model.safetensors")
    return normalized, temporary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-model", default="google/gemma-3-270m-it")
    parser.add_argument("--output-dir", type=Path, default=Path("training/router_merged"))
    args = parser.parse_args()

    if not args.adapter.is_dir():
        parser.error(f"adapter directory does not exist: {args.adapter}")
    if (args.output_dir / "config.json").exists():
        parser.error(
            f"{args.output_dir} already contains a model; choose a new output "
            "directory or remove the old export first."
        )

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "Merging requires requirements-training.txt. Install it in the "
            "training virtual environment first."
        ) from error

    adapter_source, temporary = _adapter_source(args.adapter)
    try:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(base, str(adapter_source))
        merged = model.merge_and_unload()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(args.output_dir), safe_serialization=True)
        AutoTokenizer.from_pretrained(args.base_model).save_pretrained(
            str(args.output_dir)
        )
    finally:
        if temporary is not None:
            temporary.cleanup()

    (args.output_dir / "router_merge_config.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "adapter": str(args.adapter),
                "method": "PEFT LoRA merge",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Merged model written to {args.output_dir}")
    print(
        "Next: convert it with llama.cpp's convert_hf_to_gguf.py, then use "
        "FROM /path/to/router.gguf in an Ollama Modelfile."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())