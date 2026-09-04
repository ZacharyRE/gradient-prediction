#!/usr/bin/env python3
"""Build gradient-selected and random MATH-only SFT sets from cached scores."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=43)
    args = parser.parse_args()
    candidates = read_jsonl(args.cached_selection / "math_candidate_scores.jsonl")
    target = read_jsonl(args.cached_selection / "math_target_200.jsonl")
    if args.train_size > len(candidates):
        raise ValueError("train-size exceeds the MATH candidate pool")
    if any(row["source"] != "DigitalLearningGmbH/MATH-lighteval:train" for row in candidates + target):
        raise RuntimeError("Non-MATH row found in MATH-only inputs")
    if len({row["sample_id"] for row in candidates}) != len(candidates):
        raise RuntimeError("Duplicate candidate sample IDs")
    target_ids = {row["sample_id"] for row in target}
    if target_ids & {row["sample_id"] for row in candidates}:
        raise RuntimeError("Target/candidate leakage")

    ranked = sorted(
        candidates,
        key=lambda row: (-float(row["predicted_target_similarity"]), row["sample_id"]),
    )
    selected = [{**row, "role": "selected_math_only"} for row in ranked[: args.train_size]]
    rng = np.random.default_rng(args.random_seed)
    random_indices = rng.choice(len(candidates), size=args.train_size, replace=False)
    random_rows = [{**candidates[int(i)], "role": "random_math_only"} for i in random_indices]
    dev = [{**row, "role": "sft_dev_math_target"} for row in target]

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / f"selected_{args.train_size}.jsonl", selected)
    write_jsonl(args.output / f"random_{args.train_size}.jsonl", random_rows)
    write_jsonl(args.output / "sft_dev_200.jsonl", dev)
    selected_ids = {row["sample_id"] for row in selected}
    random_ids = {row["sample_id"] for row in random_rows}
    summary = {
        "status": "passed",
        "method": "MATH-only ranking by predicted gradient similarity to a held-out MATH target anchor",
        "candidate_pool": len(candidates),
        "target_anchor_samples": len(target),
        "selected_samples": len(selected),
        "random_samples": len(random_rows),
        "random_seed": args.random_seed,
        "selected_random_overlap": len(selected_ids & random_ids),
        "pool_score_mean": float(np.mean([row["predicted_target_similarity"] for row in candidates])),
        "selected_score_mean": float(np.mean([row["predicted_target_similarity"] for row in selected])),
        "random_score_mean": float(np.mean([row["predicted_target_similarity"] for row in random_rows])),
        "selected_type_counts": dict(Counter(row["type"] for row in selected)),
        "random_type_counts": dict(Counter(row["type"] for row in random_rows)),
        "leakage_checks": {
            "target_candidate_overlap": 0,
            "selected_unique": len(selected_ids) == len(selected),
            "random_unique": len(random_ids) == len(random_rows),
            "all_rows_math": True,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
