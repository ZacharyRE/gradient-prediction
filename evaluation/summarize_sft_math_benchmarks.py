#!/usr/bin/env python3
"""Aggregate SFT benchmark summaries and paired selected-vs-random tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import binomtest


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--datasets", nargs="*", default=["gsm8k", "math500"])
    args = parser.parse_args()
    variants = args.variants or [
        "base",
        "selected_epoch1", "selected_epoch2", "selected_epoch3",
        "random_epoch1", "random_epoch2", "random_epoch3",
    ]
    results = {}
    for variant in variants:
        results[variant] = {}
        for dataset in args.datasets:
            summary = json.loads((args.root / variant / f"{dataset}_summary.json").read_text())
            predictions = load_jsonl(args.root / variant / f"{dataset}_predictions.jsonl")
            expected = summary["samples"]
            if summary["status"] != "passed" or len(predictions) != expected:
                raise RuntimeError(f"Incomplete result: {variant}/{dataset}")
            if summary["correct"] != sum(int(row["correct"]) for row in predictions):
                raise RuntimeError(f"Incorrect summary count: {variant}/{dataset}")
            results[variant][dataset] = summary

    paired = []
    for epoch in (1, 2, 3):
        if f"selected_epoch{epoch}" not in results or f"random_epoch{epoch}" not in results:
            continue
        for dataset in args.datasets:
            selected = load_jsonl(args.root / f"selected_epoch{epoch}" / f"{dataset}_predictions.jsonl")
            random = load_jsonl(args.root / f"random_epoch{epoch}" / f"{dataset}_predictions.jsonl")
            selected_only = sum(bool(a["correct"]) and not bool(b["correct"]) for a, b in zip(selected, random))
            random_only = sum(bool(b["correct"]) and not bool(a["correct"]) for a, b in zip(selected, random))
            discordant = selected_only + random_only
            paired.append({
                "epoch": epoch,
                "dataset": dataset,
                "selected_accuracy": results[f"selected_epoch{epoch}"][dataset]["accuracy"],
                "random_accuracy": results[f"random_epoch{epoch}"][dataset]["accuracy"],
                "selected_minus_random": (
                    results[f"selected_epoch{epoch}"][dataset]["accuracy"]
                    - results[f"random_epoch{epoch}"][dataset]["accuracy"]
                ),
                "selected_only_correct": selected_only,
                "random_only_correct": random_only,
                "exact_mcnemar_p": binomtest(selected_only, discordant, 0.5).pvalue if discordant else 1.0,
            })
    output = {
        "status": "passed",
        "evaluation_root": str(args.root.resolve()),
        "variants": results,
        "paired_selected_vs_random": paired,
        "best_sft": {
            dataset: max(
                (
                    {"variant": variant, "accuracy": results[variant][dataset]["accuracy"]}
                    for variant in variants if variant != "base"
                ),
                key=lambda item: item["accuracy"],
            )
            for dataset in args.datasets
        },
    }
    path = args.root / "comparison_summary.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
