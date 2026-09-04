from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


def stable_sample_id(problem: str, solution: str) -> str:
    payload = (problem.strip() + "\0" + solution.strip()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_math_train(path: Path) -> list[dict]:
    columns = ["problem", "solution", "type", "level"]
    table = pq.read_table(path, columns=columns).to_pydict()
    rows = []
    for index in range(len(table["problem"])):
        row = {column: table[column][index] for column in columns}
        rows.append(
            {
                **row,
                "sample_id": stable_sample_id(row["problem"], row["solution"]),
                "source": "DigitalLearningGmbH/MATH-lighteval:train",
                "source_row_index": index,
            }
        )
    return rows


def load_math500(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    **row,
                    "sample_id": stable_sample_id(row["problem"], row["solution"]),
                    "source": "HuggingFaceH4/MATH-500:test",
                    "source_row_index": index,
                    "type": row["subject"],
                    "level": f"Level {row['level']}",
                }
            )
    return rows


def build_fixed_splits(
    math_train: list[dict],
    math500: list[dict],
    predictor_train_size: int,
    candidate_test_size: int,
    seed: int,
) -> dict[str, list[dict]]:
    indices = np.arange(len(math_train))
    # The source contains two "Geometry / Level ?" rows. Merge only that
    # underspecified label into the same subject's largest level stratum so the
    # two-stage stratified split remains well-defined.
    strata = np.asarray(
        [
            f"{row['type']}::{row['level'] if row['level'] != 'Level ?' else 'Level 5'}"
            for row in math_train
        ]
    )
    predictor_indices, remaining = train_test_split(
        indices,
        train_size=predictor_train_size,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )
    candidate_indices, _ = train_test_split(
        remaining,
        train_size=candidate_test_size,
        random_state=seed + 1,
        shuffle=True,
        stratify=strata[remaining],
    )
    splits = {
        "predictor_train": [math_train[int(i)] for i in sorted(predictor_indices)],
        "candidate_test": [math_train[int(i)] for i in sorted(candidate_indices)],
        "target": list(math500),
    }
    validate_splits(splits)
    return splits


def validate_splits(splits: dict[str, list[dict]]) -> None:
    ids = {role: [row["sample_id"] for row in rows] for role, rows in splits.items()}
    for role, role_ids in ids.items():
        if len(role_ids) != len(set(role_ids)):
            raise ValueError(f"Duplicate sample IDs in {role}")
    if set(ids["predictor_train"]) & set(ids["candidate_test"]):
        raise ValueError("predictor_train and candidate_test overlap")
    train_candidate_problems = {
        row["problem"] for role in ("predictor_train", "candidate_test") for row in splits[role]
    }
    if train_candidate_problems & {row["problem"] for row in splits["target"]}:
        raise ValueError("MATH train and MATH-500 have exact problem overlap")


def select_warmup_rows(
    math_train: list[dict],
    fixed_splits: dict[str, list[dict]],
    size: int,
    seed: int,
) -> list[dict]:
    excluded = {
        row["sample_id"]
        for role in ("predictor_train", "candidate_test")
        for row in fixed_splits[role]
    }
    unused = [row for row in math_train if row["sample_id"] not in excluded]
    strata = np.asarray(
        [
            f"{row['type']}::{row['level'] if row['level'] != 'Level ?' else 'Level 5'}"
            for row in unused
        ]
    )
    indices = np.arange(len(unused))
    selected, _ = train_test_split(
        indices,
        train_size=size,
        random_state=seed + 2,
        shuffle=True,
        stratify=strata,
    )
    rows = [unused[int(i)] for i in sorted(selected)]
    selected_ids = {row["sample_id"] for row in rows}
    if selected_ids & excluded:
        raise ValueError("Warm-up rows overlap predictor train or candidate test")
    if {row["problem"] for row in rows} & {row["problem"] for row in fixed_splits["target"]}:
        raise ValueError("Warm-up rows overlap MATH-500 by exact problem text")
    if len(selected_ids) != size:
        raise ValueError("Warm-up rows are not unique")
    return rows


def smoke_subset(splits: dict[str, list[dict]], train_n: int, candidate_n: int, target_n: int):
    def diverse_take(rows: list[dict], count: int) -> list[dict]:
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(row["type"], []).append(row)
        selected = []
        depth = 0
        while len(selected) < count:
            added = False
            for group_name in sorted(groups):
                if depth < len(groups[group_name]):
                    selected.append(groups[group_name][depth])
                    added = True
                    if len(selected) == count:
                        break
            if not added:
                raise ValueError(f"Could not select {count} diverse rows")
            depth += 1
        return selected

    subset = {
        "predictor_train": diverse_take(splits["predictor_train"], train_n),
        "candidate_test": diverse_take(splits["candidate_test"], candidate_n),
        "target": diverse_take(splits["target"], target_n),
    }
    validate_splits(subset)
    return subset


def public_manifest_row(row: dict, role: str) -> dict:
    return {
        "sample_id": row["sample_id"],
        "role": role,
        "source": row["source"],
        "source_row_index": row["source_row_index"],
        "type": row["type"],
        "level": row["level"],
        "unique_id": row.get("unique_id"),
    }
