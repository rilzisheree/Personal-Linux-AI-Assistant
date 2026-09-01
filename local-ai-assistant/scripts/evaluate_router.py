#!/usr/bin/env python3
"""Evaluate a router on the untouched test split with full classification metrics."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_ai_assistant.assistant_core import (
    DEFAULT_ROUTER_MODEL,
    ROUTER_SYSTEM_PROMPT,
    RoutedAssistantService,
)
from local_ai_assistant.ollama import ChatMessage, OllamaClient
from local_ai_assistant.router_dataset import LABELS


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _metric_summary(rows: list[dict], latencies: list[float]) -> dict:
    matrix = {label: {predicted: 0 for predicted in LABELS} for label in LABELS}
    invalid = 0
    for row in rows:
        expected = row["expected"]
        actual = row["actual"]
        if actual not in LABELS:
            invalid += 1
            continue
        matrix[expected][actual] += 1
    total = len(rows)
    correct = sum(matrix[label][label] for label in LABELS)
    per_label = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(matrix[label].values()),
        }
    return {
        "cases": total,
        "correct": correct,
        "invalid_outputs": invalid,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_precision": round(statistics.mean(item["precision"] for item in per_label.values()), 4),
        "macro_recall": round(statistics.mean(item["recall"] for item in per_label.values()), 4),
        "macro_f1": round(statistics.mean(item["f1"] for item in per_label.values()), 4),
        "per_label": per_label,
        "confusion_matrix": {"labels": list(LABELS), "rows_expected": matrix},
        "latency_ms": {
            "average": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "p50": round(statistics.median(latencies), 2) if latencies else 0.0,
            "p95": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2) if latencies else 0.0,
            "min": round(min(latencies), 2) if latencies else 0.0,
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
    }


def _parse_transformers_output(text: str) -> str:
    normalized = text.strip().upper().strip("`'\".,:; ")
    for label in LABELS:
        if normalized == label or normalized.startswith(label + "\n"):
            return label
    return ""


def _evaluate_ollama(rows: list[dict], model: str, url: str) -> tuple[list[dict], list[float]]:
    service = RoutedAssistantService(OllamaClient(url), router_model=model)
    tools: list[dict] = []
    evaluated: list[dict] = []
    latencies: list[float] = []
    for row in rows:
        started = time.perf_counter()
        decision = service.route_request([ChatMessage("user", row["text"])], tools)
        latency = (time.perf_counter() - started) * 1000
        actual = {"simple": "SIMPLE", "function": "FUNCTION", "reasoning": "REASONING"}.get(decision.route, "")
        latencies.append(latency)
        evaluated.append({**row, "expected": row["label"], "actual": actual, "used_fallback": decision.used_fallback})
    return evaluated, latencies


def _evaluate_transformers(rows: list[dict], model_path: Path, base_model: str, max_new_tokens: int) -> tuple[list[dict], list[float]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Transformers evaluation requires requirements-training.txt.") from error
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(base_model)
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(model_path))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    evaluated: list[dict] = []
    latencies: list[float] = []
    for row in rows:
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": row["text"]},
        ]
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"System: {messages[0]['content']}\\nUser: {row['text']}\\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = (time.perf_counter() - started) * 1000
        output = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        actual = _parse_transformers_output(output)
        latencies.append(latency)
        evaluated.append({**row, "expected": row["label"], "actual": actual, "raw_output": output})
    return evaluated, latencies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("training/router_data"))
    parser.add_argument("--backend", choices=("ollama", "transformers"), default="ollama")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default=DEFAULT_ROUTER_MODEL)
    parser.add_argument("--model-path", type=Path, help="LoRA adapter or merged Transformers model directory")
    parser.add_argument("--base-model", default="google/gemma-3-270m-it")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    test_path = args.data_dir / "test.jsonl"
    if not test_path.exists():
        raise SystemExit("Test split is missing; run generate_router_dataset.py first.")
    rows = _load_rows(test_path)
    if args.backend == "ollama":
        evaluated, latencies = _evaluate_ollama(rows, args.ollama_model, args.ollama_url)
        model_name = args.ollama_model
    else:
        if not args.model_path:
            parser.error("--model-path is required for transformers backend")
        evaluated, latencies = _evaluate_transformers(rows, args.model_path, args.base_model, args.max_new_tokens)
        model_name = str(args.model_path)
    report = {
        "model": model_name,
        "backend": args.backend,
        "split": "test",
        "metrics": _metric_summary(evaluated, latencies),
        "results": evaluated,
    }
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if report["metrics"]["accuracy"] >= 0.90 else 1


if __name__ == "__main__":
    raise SystemExit(main())