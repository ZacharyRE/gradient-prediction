#!/usr/bin/env python3
"""Build a common unseen MATH-train evaluation set for two SFT variants."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--selected-train", type=Path, required=True)
    parser.add_argument("--random-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    candidates = read_jsonl(args.candidate_scores)
    selected_ids = {row["sample_id"] for row in read_jsonl(args.selected_train)}
    random_ids = {row["sample_id"] for row in read_jsonl(args.random_train)}
    common_unseen = [
        row for row in candidates
        if row["sample_id"] not in selected_ids and row["sample_id"] not in random_ids
    ]
    if len(common_unseen) < args.size:
        raise RuntimeError(f"Only {len(common_unseen)} common unseen rows for requested {args.size}")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(common_unseen), size=args.size, replace=False)
    evaluation_rows = [{**common_unseen[int(i)], "role": "common_unseen_math_train_eval"} for i in indices]
    eval_ids = {row["sample_id"] for row in evaluation_rows}
    if eval_ids & (selected_ids | random_ids):
        raise RuntimeError("Training/evaluation leakage")

    write_jsonl(args.output, evaluation_rows)
    summary = {
        "status": "passed",
        "candidate_samples": len(candidates),
        "selected_train_samples": len(selected_ids),
        "random_train_samples": len(random_ids),
        "selected_random_overlap": len(selected_ids & random_ids),
        "common_unseen_available": len(common_unseen),
        "evaluation_samples": len(evaluation_rows),
        "seed": args.seed,
        "training_evaluation_overlap": len(eval_ids & (selected_ids | random_ids)),
        "output": str(args.output.resolve()),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
