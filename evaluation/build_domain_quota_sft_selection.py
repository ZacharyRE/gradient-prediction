#!/usr/bin/env python3
"""Build balanced MATH/GSM8K SFT sets using a frozen gradient predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from sklearn.model_selection import train_test_split
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gradient_geometry.compression import row_cosine  # noqa: E402
from gradient_geometry.data import (  # noqa: E402
    build_fixed_splits,
    load_math500,
    load_math_train,
    select_warmup_rows,
)
from gradient_geometry.extraction import (  # noqa: E402
    extract_one,
    extract_prompt_only_hidden,
    load_model_and_tokenizer,
    set_global_seed,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def gsm_id(question: str, answer: str) -> str:
    return hashlib.sha256((question.strip() + "\0" + answer.strip()).encode()).hexdigest()


def load_gsm(path: Path) -> list[dict]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        item = json.loads(line)
        rows.append({
            "problem": item["question"],
            "solution": item["answer"],
            "sample_id": gsm_id(item["question"], item["answer"]),
            "source": "openai/grade-school-math:train",
            "source_row_index": index,
            "type": "GSM8K",
            "level": "grade_school",
        })
    return rows


def cached_target_gradients(model, tokenizer, rows: list[dict], config: dict, device: str,
                            output: Path, gradient_dim: int) -> np.ndarray:
    path = output / "math_target_raw_gradients.npy"
    progress = output / "math_target_progress.json"
    completed = 0
    if path.exists():
        gradients = np.load(path, mmap_mode="r+")
        if gradients.shape != (len(rows), gradient_dim):
            raise RuntimeError(f"Unexpected target gradient shape: {gradients.shape}")
        if progress.exists():
            completed = int(json.loads(progress.read_text())["completed"])
    else:
        gradients = open_memmap(path, mode="w+", dtype=np.float32,
                                shape=(len(rows), gradient_dim))
    for index in tqdm(range(completed, len(rows)), initial=completed, total=len(rows),
                      desc="MATH target gradients"):
        gradients[index] = extract_one(
            model, tokenizer, rows[index], config, device, include_hidden=False
        ).raw_gradient
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            gradients.flush()
            write_json(progress, {"completed": index + 1})
    gradients.flush()
    return gradients


def cached_hidden(model, tokenizer, rows: list[dict], config: dict, device: str,
                  output: Path, stem: str, hidden_dim: int) -> np.ndarray:
    path = output / f"{stem}_hidden.npy"
    progress = output / f"{stem}_hidden_progress.json"
    completed = 0
    if path.exists():
        hidden = np.load(path, mmap_mode="r+")
        if hidden.shape != (len(rows), hidden_dim):
            raise RuntimeError(f"Unexpected {stem} hidden shape: {hidden.shape}")
        if progress.exists():
            completed = int(json.loads(progress.read_text())["completed"])
    else:
        hidden = open_memmap(path, mode="w+", dtype=np.float32,
                             shape=(len(rows), hidden_dim))
    for index in tqdm(range(completed, len(rows)), initial=completed, total=len(rows), desc=stem):
        hidden[index] = extract_prompt_only_hidden(model, tokenizer, rows[index], config, device)
        if (index + 1) % 20 == 0 or index + 1 == len(rows):
            hidden.flush()
            write_json(progress, {"completed": index + 1})
    hidden.flush()
    return hidden


def predict(hidden: np.ndarray, experiment: Path, device: str, batch_size: int = 512):
    mean = np.load(experiment / "ridge/scaler_mean.npy")
    scale = np.load(experiment / "ridge/scaler_scale.npy")
    coefficient = torch.from_numpy(
        np.array(np.load(experiment / "ridge/coefficient.npy", mmap_mode="r"), copy=True)
    ).to(device)
    intercept = torch.from_numpy(np.load(experiment / "ridge/intercept.npy")).to(device)
    outputs = []
    for start in range(0, len(hidden), batch_size):
        x = (np.asarray(hidden[start:start + batch_size]) - mean) / scale
        x_tensor = torch.from_numpy(x.astype(np.float32)).to(device)
        outputs.append((x_tensor @ coefficient + intercept).cpu().numpy().astype(np.float32))
    del coefficient, intercept
    torch.cuda.empty_cache()
    return np.concatenate(outputs)


def public_row(row: dict, role: str, score: float | None = None) -> dict:
    result = {
        "sample_id": row["sample_id"],
        "role": role,
        "source": row["source"],
        "source_row_index": row["source_row_index"],
        "type": row["type"],
        "level": row["level"],
        "problem": row["problem"],
        "solution": row["solution"],
    }
    if score is not None:
        result["predicted_target_similarity"] = float(score)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--gsm-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--math-target-size", type=int, default=200)
    parser.add_argument("--quota-per-domain", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=43)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.time()
    experiment = args.experiment.resolve()
    audit = args.gsm_audit.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((experiment / "config_resolved.json").read_text())

    math_train = load_math_train(Path(config["data"]["math_train_path"]))
    math500 = load_math500(Path(config["data"]["math500_path"]))
    fixed = build_fixed_splits(math_train, math500, 2000, 1000, args.seed)
    warmup = select_warmup_rows(math_train, fixed, 128, args.seed)
    excluded_math = {
        row["sample_id"]
        for group in (fixed["predictor_train"], fixed["candidate_test"], warmup)
        for row in group
    }
    math_fresh = [row for row in math_train if row["sample_id"] not in excluded_math]
    strata = np.asarray([
        f"{row['type']}::{row['level'] if row['level'] != 'Level ?' else 'Level 5'}"
        for row in math_fresh
    ])
    indices = np.arange(len(math_fresh))
    target_indices, candidate_indices = train_test_split(
        indices, train_size=args.math_target_size, random_state=args.seed + 3,
        shuffle=True, stratify=strata,
    )
    math_target = [math_fresh[int(i)] for i in sorted(target_indices)]
    math_candidates = [math_fresh[int(i)] for i in sorted(candidate_indices)]

    gsm = load_gsm(ROOT / "data/GSM8K/train.jsonl")
    audit_manifest = read_jsonl(audit / "manifest.jsonl")
    audit_ids = {row["sample_id"] for row in audit_manifest}
    gsm_target_indices = [
        int(row["source_row_index"]) for row in audit_manifest if row["role"] == "target_anchor"
    ]
    gsm_target = [gsm[index] for index in gsm_target_indices]
    gsm_candidates = [row for row in gsm if row["sample_id"] not in audit_ids]
    if len(gsm_target) != 100 or len(gsm_candidates) != len(gsm) - len(audit_ids):
        raise RuntimeError("Unexpected GSM8K target/audit partition")

    set_global_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(config, args.device)
    layout = json.loads((experiment / "parameter_layout.json").read_text())
    math_target_gradients = cached_target_gradients(
        model, tokenizer, math_target, config, args.device, output,
        int(layout["raw_gradient_dim"]),
    )
    math_hidden = cached_hidden(
        model, tokenizer, math_candidates, config, args.device, output,
        "math_candidates", int(layout["hidden_dim"]),
    )
    gsm_hidden = cached_hidden(
        model, tokenizer, gsm_candidates, config, args.device, output,
        "gsm_candidates", int(layout["hidden_dim"]),
    )
    del model
    torch.cuda.empty_cache()

    math_prediction = predict(math_hidden, experiment, args.device)
    gsm_prediction = predict(gsm_hidden, experiment, args.device)
    math_anchor = np.asarray(math_target_gradients, dtype=np.float64).mean(axis=0).astype(np.float32)
    gsm_anchor = np.load(audit / "target_raw_mean_gradient.npy")
    math_scores = row_cosine(math_prediction, math_anchor[None, :])
    gsm_scores = row_cosine(gsm_prediction, gsm_anchor[None, :])
    np.save(output / "math_target_raw_mean_gradient.npy", math_anchor)
    np.save(output / "gsm_target_raw_mean_gradient.npy", gsm_anchor)
    np.save(output / "math_candidate_scores.npy", math_scores.astype(np.float32))
    np.save(output / "gsm_candidate_scores.npy", gsm_scores.astype(np.float32))

    quota = args.quota_per_domain
    math_top = np.argsort(-math_scores, kind="stable")[:quota]
    gsm_top = np.argsort(-gsm_scores, kind="stable")[:quota]
    selected = [public_row(math_candidates[int(i)], "selected", math_scores[int(i)])
                for i in math_top]
    selected += [public_row(gsm_candidates[int(i)], "selected", gsm_scores[int(i)])
                 for i in gsm_top]
    random_rng = np.random.default_rng(args.random_seed)
    math_random = random_rng.choice(len(math_candidates), size=quota, replace=False)
    gsm_random = random_rng.choice(len(gsm_candidates), size=quota, replace=False)
    random_rows = [public_row(math_candidates[int(i)], "random", math_scores[int(i)])
                   for i in math_random]
    random_rows += [public_row(gsm_candidates[int(i)], "random", gsm_scores[int(i)])
                    for i in gsm_random]
    dev = [public_row(row, "sft_dev_target") for row in math_target + gsm_target]
    write_jsonl(output / "selected_2000.jsonl", selected)
    write_jsonl(output / "random_2000.jsonl", random_rows)
    write_jsonl(output / "sft_dev_300.jsonl", dev)
    write_jsonl(output / "math_target_200.jsonl",
                [public_row(row, "math_target") for row in math_target])
    write_jsonl(output / "math_candidate_scores.jsonl", [
        public_row(row, "candidate", score) for row, score in zip(math_candidates, math_scores)
    ])
    write_jsonl(output / "gsm_candidate_scores.jsonl", [
        public_row(row, "candidate", score) for row, score in zip(gsm_candidates, gsm_scores)
    ])

    selected_ids = {row["sample_id"] for row in selected}
    random_ids = {row["sample_id"] for row in random_rows}
    summary = {
        "status": "passed",
        "method": "within-domain gradient ranking with fixed equal source quota",
        "predictor_experiment": str(experiment),
        "seed": args.seed,
        "random_seed": args.random_seed,
        "partition": {
            "math_predictor_train": 2000,
            "math_predictor_audit": 1000,
            "math_warmup": 128,
            "math_target": len(math_target),
            "math_candidates": len(math_candidates),
            "gsm_predictor_audit_excluded": len(audit_ids),
            "gsm_target_within_excluded_audit": len(gsm_target),
            "gsm_candidates": len(gsm_candidates),
        },
        "training_sets": {
            "selected": {"total": len(selected), "math": quota, "gsm8k": quota},
            "random": {"total": len(random_rows), "math": quota, "gsm8k": quota},
            "selected_random_overlap": len(selected_ids & random_ids),
        },
        "score_statistics": {
            "math_pool_mean": float(math_scores.mean()),
            "math_selected_mean": float(math_scores[math_top].mean()),
            "gsm_pool_mean": float(gsm_scores.mean()),
            "gsm_selected_mean": float(gsm_scores[gsm_top].mean()),
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
