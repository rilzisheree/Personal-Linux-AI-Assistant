from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_ai_assistant.router_dataset import LABELS, SCENARIOS, generate_examples, write_dataset


class RouterDatasetTests(unittest.TestCase):
    def test_generation_is_balanced_and_reproducible(self) -> None:
        first = generate_examples(
            seed=7, train_per_scenario=5, validation_per_scenario=3, test_per_scenario=4
        )
        second = generate_examples(
            seed=7, train_per_scenario=5, validation_per_scenario=3, test_per_scenario=4
        )
        self.assertEqual(first, second)
        self.assertEqual({example.label for example in first}, set(LABELS))
        for split in ("train", "validation", "test"):
            rows = [example for example in first if example.split == split]
            self.assertEqual(len({example.label for example in rows}), 3)

    def test_scenarios_never_cross_splits(self) -> None:
        examples = generate_examples(
            seed=9, train_per_scenario=4, validation_per_scenario=2, test_per_scenario=3
        )
        split_by_scenario = {}
        for example in examples:
            split_by_scenario.setdefault(example.scenario, set()).add(example.split)
        self.assertEqual(len(split_by_scenario), len(SCENARIOS))
        self.assertTrue(all(len(splits) == 1 for splits in split_by_scenario.values()))

    def test_writer_creates_jsonl_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = write_dataset(output, generate_examples(seed=3, train_per_scenario=2, validation_per_scenario=1, test_per_scenario=1))
            self.assertEqual(set(summary["splits"]), {"train", "validation", "test"})
            self.assertTrue((output / "train.jsonl").exists())
            self.assertTrue((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()