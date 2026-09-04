#!/usr/bin/env python3
"""Print incorrect benchmark generations for quick manual auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from math_verify import parse


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "math500"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=10)
    args = parser.parse_args()
    data = load_jsonl(args.data)
    predictions = load_jsonl(args.predictions)
    if len(data) != len(predictions):
        raise RuntimeError(f"Data/prediction length mismatch: {len(data)} != {len(predictions)}")
    shown = 0
    for index, (sample, result) in enumerate(zip(data, predictions)):
        if result["correct"]:
            continue
        problem = sample["question"] if args.dataset == "gsm8k" else sample["problem"]
        gold = sample["answer"]
        prediction = result["prediction"]
        print(f"\n{'=' * 24} index={index} {'=' * 24}")
        print(f"finish_reason: {result['finish_reason']}")
        print(f"parser_empty: {not bool(parse(prediction))}")
        print(f"problem:\n{problem}")
        print(f"gold:\n{gold}")
        print(f"raw prediction:\n{prediction}")
        shown += 1
        if shown == args.num_cases:
            break


if __name__ == "__main__":
    main()
