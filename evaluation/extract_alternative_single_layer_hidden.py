#!/usr/bin/env python3
"""Extract an alternative prompt representation while reusing fixed raw gradients.

This is intended for matched hidden-source ablations.  The source experiment owns
the adapter, split, per-example LoRA gradients, and target-validation gradients;
only the prompt-only hidden feature is recomputed here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gradient_geometry.data import build_fixed_splits, load_math500, load_math_train  # noqa: E402
from gradient_geometry.extraction import (  # noqa: E402
    extract_prompt_only_hidden,
    load_model_and_tokenizer,
    set_global_seed,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def link_or_verify(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise RuntimeError(f"Refusing to replace existing file: {destination}")
    os.link(source, destination)


def copy_or_verify(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == source.read_bytes():
            return
        raise RuntimeError(f"Refusing to replace different existing file: {destination}")
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hidden-source", choices=("target_module_input", "hidden_states_tuple"), required=True
    )
    parser.add_argument("--hidden-module-suffix")
    parser.add_argument("--hidden-states-tuple-index", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    source = args.source_experiment.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    config = json.loads((source / "config_resolved.json").read_text(encoding="utf-8"))
    config["experiment"]["name"] = output.parent.name
    config["model"]["hidden_source"] = args.hidden_source
    config["model"]["hidden_token"] = "last_prompt_token"
    if args.hidden_source == "target_module_input":
        if not args.hidden_module_suffix:
            parser.error("--hidden-module-suffix is required for target_module_input")
        config["model"]["hidden_module_suffix"] = args.hidden_module_suffix
    else:
        if args.hidden_states_tuple_index is None:
            parser.error("--hidden-states-tuple-index is required for hidden_states_tuple")
        config["model"]["hidden_states_tuple_index"] = args.hidden_states_tuple_index
        config["model"].pop("hidden_module_suffix", None)

    resolved = output / "config_resolved.json"
    if resolved.exists():
        if json.loads(resolved.read_text(encoding="utf-8")) != config:
            raise RuntimeError("Refusing to resume with a different alternative-hidden config")
    else:
        write_json(resolved, config)

    for relative in (
        "parameter_layout.json",
        "prompt.json",
        "splits/predictor_train.jsonl",
        "splits/candidate_test.jsonl",
    ):
        copy_or_verify(source / relative, output / relative)
    for role in ("predictor_train", "candidate_test"):
        for suffix in ("raw_gradients.npy", "metadata.jsonl"):
            link_or_verify(
                source / "extraction" / f"{role}_{suffix}",
                output / "extraction" / f"{role}_{suffix}",
            )

    # The target gradient is representation-independent, so reuse it too.  The
    # alignment evaluator will see complete metadata and skip all backward passes.
    for name in (
        "target_validation_raw_gradients.npy",
        "target_validation_metadata.jsonl",
        "target_validation_manifest.jsonl",
    ):
        link_or_verify(source / "target_alignment" / name, output / "target_alignment" / name)

    for key in ("math_train_path", "math500_path"):
        path = Path(config["data"][key])
        if not path.is_absolute():
            config["data"][key] = str((PROJECT_ROOT / path).resolve())
    math_train = load_math_train(Path(config["data"]["math_train_path"]))
    math500 = load_math500(Path(config["data"]["math500_path"]))
    all_splits = build_fixed_splits(
        math_train,
        math500,
        int(config["data"]["predictor_train_size"]),
        int(config["data"]["candidate_test_size"]),
        int(config["experiment"]["seed"]),
    )
    splits = {role: all_splits[role] for role in ("predictor_train", "candidate_test")}
    for role, rows in splits.items():
        manifest = read_jsonl(source / "splits" / f"{role}.jsonl")
        if [row["sample_id"] for row in rows] != [row["sample_id"] for row in manifest]:
            raise RuntimeError(f"{role}: rebuilt split does not match source experiment")

    set_global_seed(int(config["experiment"]["seed"]))
    model, tokenizer = load_model_and_tokenizer(config, args.device)
    hidden_dim = int(model.config.hidden_size)
    for role, rows in splits.items():
        path = output / "extraction" / f"{role}_hidden.npy"
        progress_path = output / "extraction" / f"{role}_hidden_progress.json"
        completed = 0
        if path.exists():
            hidden = np.load(path, mmap_mode="r+")
            if hidden.shape != (len(rows), hidden_dim):
                raise RuntimeError(f"{role}: unexpected hidden shape {hidden.shape}")
            if progress_path.exists():
                completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"])
        else:
            hidden = open_memmap(path, mode="w+", dtype=np.float32, shape=(len(rows), hidden_dim))
        for index in tqdm(range(completed, len(rows)), initial=completed, total=len(rows), desc=role):
            hidden[index] = extract_prompt_only_hidden(model, tokenizer, rows[index], config, args.device)
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                hidden.flush()
                write_json(progress_path, {"completed": index + 1})
        hidden.flush()

    train_hidden = np.load(output / "extraction" / "predictor_train_hidden.npy", mmap_mode="r")
    test_hidden = np.load(output / "extraction" / "candidate_test_hidden.npy", mmap_mode="r")
    train_gradient = np.load(
        output / "extraction" / "predictor_train_raw_gradients.npy", mmap_mode="r"
    )
    test_gradient = np.load(
        output / "extraction" / "candidate_test_raw_gradients.npy", mmap_mode="r"
    )
    write_json(
        output / "extraction_summary.json",
        {
            "status": "passed",
            "experiment": "matched_alternative_single_layer_hidden",
            "source_experiment": str(source),
            "reused_raw_gradients": True,
            "hidden_source": args.hidden_source,
            "hidden_module_suffix": config["model"].get("hidden_module_suffix"),
            "hidden_states_tuple_index": config["model"].get("hidden_states_tuple_index"),
            "hidden_token": config["model"]["hidden_token"],
            "shapes": {
                "train_hidden": list(train_hidden.shape),
                "train_raw_gradient": list(train_gradient.shape),
                "test_hidden": list(test_hidden.shape),
                "test_raw_gradient": list(test_gradient.shape),
            },
            "elapsed_seconds": time.time() - started,
        },
    )


if __name__ == "__main__":
    main()
