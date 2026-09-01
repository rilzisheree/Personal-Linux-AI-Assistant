#!/usr/bin/env python3
"""Generate the balanced, scenario-isolated Lura router dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_ai_assistant.router_dataset import generate_examples, write_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("training/router_data"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-per-scenario", type=int, default=60)
    parser.add_argument("--validation-per-scenario", type=int, default=50)
    parser.add_argument("--test-per-scenario", type=int, default=70)
    args = parser.parse_args()
    if min(args.train_per_scenario, args.validation_per_scenario, args.test_per_scenario) < 1:
        parser.error("per-scenario counts must be positive")
    summary = write_dataset(
        args.output_dir,
        generate_examples(
            seed=args.seed,
            train_per_scenario=args.train_per_scenario,
            validation_per_scenario=args.validation_per_scenario,
            test_per_scenario=args.test_per_scenario,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())