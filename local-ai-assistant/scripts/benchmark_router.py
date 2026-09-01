#!/usr/bin/env python3
"""Measure Gemma router accuracy and latency against a local Ollama server."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_ai_assistant.assistant_core import DEFAULT_ROUTER_MODEL, RoutedAssistantService
from local_ai_assistant.ollama import ChatMessage, OllamaClient
from local_ai_assistant.tools import ToolManager
from tests.router_benchmark_cases import (
    NATURAL_CONVERSATIONAL_CASES,
    ROUTER_BENCHMARK_CASES,
)


def main() -> int:
    client = OllamaClient("http://localhost:11434")
    try:
        installed_models = client.list_models()
    except Exception as error:
        print(
            json.dumps(
                {
                    "error": "Ollama is unavailable",
                    "detail": str(error),
                    "router_model": DEFAULT_ROUTER_MODEL,
                },
                indent=2,
            )
        )
        return 2
    if DEFAULT_ROUTER_MODEL not in installed_models:
        print(
            json.dumps(
                {
                    "error": "Router model is not installed",
                    "router_model": DEFAULT_ROUTER_MODEL,
                    "installed_models": installed_models,
                },
                indent=2,
            )
        )
        return 2

    service = RoutedAssistantService(client)
    tools = ToolManager().definitions_for_ollama()

    def evaluate(cases: tuple[tuple[str, str, str], ...]) -> list[dict]:
        results: list[dict] = []
        for prompt, expected_route, expected_function in cases:
            started = time.monotonic()
            decision = service.route_request(
                [ChatMessage("user", prompt)],
                tools,
            )
            latency_ms = round((time.monotonic() - started) * 1000, 1)
            route_correct = decision.route == expected_route
            results.append(
                {
                    "prompt": prompt,
                    "expected": expected_route,
                    "actual": decision.route,
                    "expected_function": expected_function,
                    "router_output_valid": not decision.used_fallback,
                    "function_classification_correct": (
                        route_correct if expected_route == "function" else None
                    ),
                    "correct": route_correct,
                    "latency_ms": latency_ms,
                }
            )
        return results

    results = evaluate(ROUTER_BENCHMARK_CASES)
    natural_results = evaluate(NATURAL_CONVERSATIONAL_CASES)

    def summarize(results: list[dict]) -> dict:
        total = len(results)
        correct = sum(result["correct"] for result in results)
        false_simple = sum(
            result["actual"] == "simple" and result["expected"] != "simple"
            for result in results
        )
        false_reasoning = sum(
            result["actual"] == "reasoning" and result["expected"] == "simple"
            for result in results
        )
        function_cases = [result for result in results if result["expected"] == "function"]
        function_correct = sum(
            result["function_classification_correct"] for result in function_cases
        )
        invalid_outputs = sum(
            not result["router_output_valid"] for result in results
        )
        return {
            "cases": total,
            "routing_accuracy": round(correct / total, 4) if total else 0,
            "invalid_router_outputs": invalid_outputs,
            "false_simple": false_simple,
            "false_reasoning": false_reasoning,
            "function_classification_accuracy": (
                round(function_correct / len(function_cases), 4)
                if function_cases
                else 0
            ),
            "average_routing_latency_ms": round(
                sum(result["latency_ms"] for result in results) / total, 1
            )
            if total
            else 0,
        }

    summary = {
        "router_model": DEFAULT_ROUTER_MODEL,
        **summarize(results),
        "results": results,
        "natural_conversational": {
            **summarize(natural_results),
            "results": natural_results,
        },
    }
    print(json.dumps(summary, indent=2))
    return 0 if all(result["correct"] for result in results + natural_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())