---
name: Ollama Gemma adapter import
description: Gemma 3 can run in Ollama while its LoRA adapter import remains unsupported on some Ollama builds.
---

When Ollama reports `unsupported architecture` while importing a Gemma 3
Safetensors LoRA adapter, merge the PEFT adapter into the matching Hugging Face
base model and export the merged model to GGUF before importing it.

**Why:** Ollama's Safetensors adapter importer has narrower architecture support
than its model runner; a successful `gemma3:270m` base-model pull does not prove
that Gemma 3 adapter application is supported.

**How to apply:** Keep the adapter and base model exactly matched. Use the
project's merge helper, then a current llama.cpp `convert_hf_to_gguf.py`, and
import the resulting standalone GGUF with `FROM`.