#!/usr/bin/env python3
"""Extend a direct-gradient training split while preserving its exact held-out set."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from numpy.lib.format import open_memmap
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gradient_geometry.data import load_math_train, public_manifest_row
from gradient_geometry.extraction import (
    extract_one,
    load_model_and_tokenizer,
    set_global_seed,
    trainable_lora_parameters,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def normalized_stratum(row: dict) -> str:
    level = "Level 5" if row["level"] == "Level ?" else row["level"]
    return f"{row['type']}::{level}"


def extract_shard(payload: dict) -> dict:
    shard_index = payload["shard_index"]
    rows = payload["rows"]
    config = payload["config"]
    device = payload["device"]
    shard_dir = Path(payload["shard_dir"])
    shard_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(config["experiment"]["seed"]))
    model, tokenizer = load_model_and_tokenizer(config, device)
    hidden_dim = int(model.config.hidden_size)
    gradient_dim = sum(parameter.numel() for _, parameter in trainable_lora_parameters(model))
    hidden_path = shard_dir / "hidden.npy"
    gradient_path = shard_dir / "raw_gradients.npy"
    metadata_path = shard_dir / "metadata.jsonl"
    metadata = read_jsonl(metadata_path) if metadata_path.exists() else []
    expected_ids = [row["sample_id"] for row in rows]
    if [row["sample_id"] for row in metadata] != expected_ids[: len(metadata)]:
        raise RuntimeError(f"shard {shard_index}: existing metadata does not match its split")
    if hidden_path.exists() != gradient_path.exists():
        raise RuntimeError(f"shard {shard_index}: incomplete array pair")
    if hidden_path.exists():
        hidden = np.load(hidden_path, mmap_mode="r+")
        gradients = np.load(gradient_path, mmap_mode="r+")
        if hidden.shape != (len(rows), hidden_dim):
            raise RuntimeError(f"shard {shard_index}: incompatible hidden shape {hidden.shape}")
        if gradients.shape != (len(rows), gradient_dim):
            raise RuntimeError(f"shard {shard_index}: incompatible gradient shape {gradients.shape}")
    else:
        hidden = open_memmap(
            hidden_path, mode="w+", dtype=np.float32, shape=(len(rows), hidden_dim)
        )
        gradients = open_memmap(
            gradient_path, mode="w+", dtype=np.float32, shape=(len(rows), gradient_dim)
        )
    with metadata_path.open("a", encoding="utf-8") as handle:
        for index in range(len(metadata), len(rows)):
            result = extract_one(model, tokenizer, rows[index], config, device)
            if result.metadata["gradient_representation"] != "raw":
                raise RuntimeError("Extended direct probe requires raw gradients")
            if result.metadata["truncated"]:
                raise RuntimeError(f"shard {shard_index}: sample {index} was truncated")
            hidden[index] = result.hidden
            gradients[index] = result.raw_gradient
            handle.write(json.dumps(result.metadata, ensure_ascii=False) + "\n")
            handle.flush()
            metadata.append(result.metadata)
            if (index + 1) % 10 == 0:
                hidden.flush()
                gradients.flush()
            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                print(
                    json.dumps(
                        {
                            "shard": shard_index,
                            "device": device,
                            "completed": index + 1,
                            "total": len(rows),
                        }
                    ),
                    flush=True,
                )
    hidden.flush()
    gradients.flush()
    return {
        "shard_index": shard_index,
        "rows": len(rows),
        "hidden_dim": hidden_dim,
        "gradient_dim": gradient_dim,
    }


def copy_array_prefix(source: Path, destination: np.memmap, stop: int) -> None:
    source_array = np.load(source, mmap_mode="r")
    if len(source_array) != stop or source_array.shape[1:] != destination.shape[1:]:
        raise RuntimeError(f"Cannot copy incompatible source array {source_array.shape}")
    destination[:stop] = source_array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    args = parser.parse_args()
    source = args.source_experiment.resolve()
    output = args.output.resolve()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ("math_train_path", "math500_path"):
        path = Path(config["data"][key])
        if not path.is_absolute():
            config["data"][key] = str((PROJECT_ROOT / path).resolve())
    config["lora"]["checkpoint"] = str(args.adapter.resolve())
    config["data"]["predictor_train_size"] = args.train_size
    config["data"]["candidate_test_size"] = len(
        read_jsonl(source / "splits" / "candidate_test.jsonl")
    )

    math_train = load_math_train(Path(config["data"]["math_train_path"]))
    row_by_id = {row["sample_id"]: row for row in math_train}
    source_train_manifest = read_jsonl(source / "splits" / "predictor_train.jsonl")
    candidate_manifest = read_jsonl(source / "splits" / "candidate_test.jsonl")
    source_train_ids = [row["sample_id"] for row in source_train_manifest]
    candidate_ids = [row["sample_id"] for row in candidate_manifest]
    warmup_manifest_path = args.adapter.resolve().parent.parent / "warmup_manifest.jsonl"
    if not warmup_manifest_path.exists():
        warmup_manifest_path = args.adapter.resolve().parent / "warmup_manifest.jsonl"
    warmup_ids = {
        row["sample_id"] for row in read_jsonl(warmup_manifest_path)
    } if warmup_manifest_path.exists() else set()
    excluded = set(source_train_ids) | set(candidate_ids) | warmup_ids
    additional_size = args.train_size - len(source_train_ids)
    if additional_size <= 0:
        raise RuntimeError("train size must exceed the source training size")
    eligible = [row for row in math_train if row["sample_id"] not in excluded]
    if additional_size > len(eligible):
        raise RuntimeError(
            f"Requested {additional_size} additional rows but only {len(eligible)} are eligible"
        )
    eligible_indices = np.arange(len(eligible))
    selected_indices, _ = train_test_split(
        eligible_indices,
        train_size=additional_size,
        random_state=int(config["experiment"]["seed"]) + 3,
        shuffle=True,
        stratify=np.asarray([normalized_stratum(row) for row in eligible]),
    )
    additional_rows = [eligible[int(index)] for index in sorted(selected_indices)]
    source_rows = [row_by_id[sample_id] for sample_id in source_train_ids]
    candidate_rows = [row_by_id[sample_id] for sample_id in candidate_ids]
    all_train_rows = source_rows + additional_rows
    if set(row["sample_id"] for row in all_train_rows) & set(candidate_ids):
        raise RuntimeError("Extended training split overlaps the preserved candidate split")

    plan = {
        "source_experiment": str(source),
        "adapter": str(args.adapter.resolve()),
        "source_train_size": len(source_rows),
        "additional_train_size": len(additional_rows),
        "final_train_size": len(all_train_rows),
        "candidate_test_size": len(candidate_rows),
        "candidate_test_preserved_exactly": True,
        "warmup_rows_excluded": len(warmup_ids),
        "devices": args.devices,
        "additional_sample_ids": [row["sample_id"] for row in additional_rows],
    }
    plan_path = output / "extension_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise RuntimeError("Existing extension plan differs from the requested experiment")
    write_json(plan_path, plan)
    write_json(output / "config_resolved.json", config)
    write_jsonl(
        output / "splits" / "predictor_train.jsonl",
        [public_manifest_row(row, "predictor_train") for row in all_train_rows],
    )
    write_jsonl(
        output / "splits" / "candidate_test.jsonl",
        [public_manifest_row(row, "candidate_test") for row in candidate_rows],
    )

    shard_count = len(args.devices)
    boundaries = np.linspace(0, len(additional_rows), shard_count + 1, dtype=int)
    payloads = []
    for shard_index, device in enumerate(args.devices):
        shard_rows = additional_rows[boundaries[shard_index] : boundaries[shard_index + 1]]
        payloads.append(
            {
                "shard_index": shard_index,
                "rows": shard_rows,
                "config": copy.deepcopy(config),
                "device": device,
                "shard_dir": str(output / "extraction_shards" / f"shard_{shard_index}"),
            }
        )
    context = mp.get_context("spawn")
    with context.Pool(processes=shard_count) as pool:
        shard_results = pool.map(extract_shard, payloads)
    shard_results.sort(key=lambda row: row["shard_index"])
    hidden_dim = shard_results[0]["hidden_dim"]
    gradient_dim = shard_results[0]["gradient_dim"]
    if any(
        row["hidden_dim"] != hidden_dim or row["gradient_dim"] != gradient_dim
        for row in shard_results
    ):
        raise RuntimeError("Extraction shards produced inconsistent dimensions")

    extraction = output / "extraction"
    extraction.mkdir(parents=True, exist_ok=True)
    train_hidden = open_memmap(
        extraction / "predictor_train_hidden.npy",
        mode="w+",
        dtype=np.float32,
        shape=(args.train_size, hidden_dim),
    )
    train_gradients = open_memmap(
        extraction / "predictor_train_raw_gradients.npy",
        mode="w+",
        dtype=np.float32,
        shape=(args.train_size, gradient_dim),
    )
    source_size = len(source_rows)
    copy_array_prefix(source / "extraction" / "predictor_train_hidden.npy", train_hidden, source_size)
    copy_array_prefix(
        source / "extraction" / "predictor_train_raw_gradients.npy", train_gradients, source_size
    )
    offset = source_size
    additional_metadata = []
    for shard_index, shard_result in enumerate(shard_results):
        shard_dir = output / "extraction_shards" / f"shard_{shard_index}"
        shard_hidden = np.load(shard_dir / "hidden.npy", mmap_mode="r")
        shard_gradients = np.load(shard_dir / "raw_gradients.npy", mmap_mode="r")
        stop = offset + shard_result["rows"]
        train_hidden[offset:stop] = shard_hidden
        train_gradients[offset:stop] = shard_gradients
        additional_metadata.extend(read_jsonl(shard_dir / "metadata.jsonl"))
        offset = stop
    train_hidden.flush()
    train_gradients.flush()
    source_metadata = read_jsonl(source / "extraction" / "predictor_train_metadata.jsonl")
    write_jsonl(
        extraction / "predictor_train_metadata.jsonl", source_metadata + additional_metadata
    )
    for filename in (
        "candidate_test_hidden.npy",
        "candidate_test_raw_gradients.npy",
        "candidate_test_metadata.jsonl",
    ):
        shutil.copy2(source / "extraction" / filename, extraction / filename)
    shutil.copy2(source / "parameter_layout.json", output / "parameter_layout.json")
    write_json(
        output / "extension_summary.json",
        {
            **plan,
            "hidden_shape": list(train_hidden.shape),
            "gradient_shape": list(train_gradients.shape),
            "shards": shard_results,
            "status": "passed",
        },
    )
    print(json.dumps({"status": "passed", **plan}, indent=2), flush=True)


if __name__ == "__main__":
    main()
