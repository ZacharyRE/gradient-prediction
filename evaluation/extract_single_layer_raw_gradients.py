#!/usr/bin/env python3
"""Extract paired single-layer hidden states and raw LoRA gradients."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml
from numpy.lib.format import open_memmap
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gradient_geometry.data import (  # noqa: E402
    build_fixed_splits,
    load_math500,
    load_math_train,
    public_manifest_row,
)
from gradient_geometry.extraction import (  # noqa: E402
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    extract_one,
    load_model_and_tokenizer,
    set_global_seed,
    trainable_lora_parameters,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_paths(config: dict) -> None:
    for key in ("math_train_path", "math500_path"):
        path = Path(config["data"][key])
        if not path.is_absolute():
            config["data"][key] = str((PROJECT_ROOT / path).resolve())


def hidden_input_dim(model, config: dict) -> int:
    if config["model"].get("hidden_source", "hidden_states_tuple") != "target_module_input":
        return int(model.config.hidden_size)
    suffix = config["model"]["hidden_module_suffix"]
    matches = [module for name, module in model.named_modules() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one module ending in {suffix!r}, found {len(matches)}")
    dimension = getattr(matches[0], "in_features", None)
    if dimension is None and hasattr(matches[0], "base_layer"):
        dimension = getattr(matches[0].base_layer, "in_features", None)
    if dimension is None:
        raise RuntimeError(f"Cannot infer input dimension for module ending in {suffix!r}")
    return int(dimension)


def extract_role(model, tokenizer, rows, role: str, config: dict, device: str, output: Path,
                 hidden_dim: int, gradient_dim: int):
    directory = output / "extraction"
    directory.mkdir(parents=True, exist_ok=True)
    hidden_path = directory / f"{role}_hidden.npy"
    gradient_path = directory / f"{role}_raw_gradients.npy"
    metadata_path = directory / f"{role}_metadata.jsonl"
    metadata = read_jsonl(metadata_path)
    if len(metadata) > len(rows):
        raise RuntimeError(f"{role}: metadata has more rows than the fixed split")
    if [row["sample_id"] for row in metadata] != [row["sample_id"] for row in rows[:len(metadata)]]:
        raise RuntimeError(f"{role}: existing metadata does not match the fixed split")
    if hidden_path.exists() != gradient_path.exists():
        raise RuntimeError(f"{role}: incomplete hidden/gradient array pair")
    if hidden_path.exists():
        hidden = np.load(hidden_path, mmap_mode="r+")
        gradients = np.load(gradient_path, mmap_mode="r+")
        if hidden.shape != (len(rows), hidden_dim):
            raise RuntimeError(f"{role}: incompatible hidden shape {hidden.shape}")
        if gradients.shape != (len(rows), gradient_dim):
            raise RuntimeError(f"{role}: incompatible gradient shape {gradients.shape}")
    else:
        if metadata:
            raise RuntimeError(f"{role}: metadata exists without arrays")
        hidden = open_memmap(hidden_path, mode="w+", dtype=np.float32,
                             shape=(len(rows), hidden_dim))
        gradients = open_memmap(gradient_path, mode="w+", dtype=np.float32,
                                shape=(len(rows), gradient_dim))
    with metadata_path.open("a", encoding="utf-8") as handle:
        for index in tqdm(range(len(metadata), len(rows)), initial=len(metadata), total=len(rows),
                          desc=f"extract {role}"):
            result = extract_one(model, tokenizer, rows[index], config, device)
            if result.metadata["gradient_representation"] != "raw":
                raise RuntimeError("Direct probe requires raw, unsketched gradients")
            hidden[index] = result.hidden
            gradients[index] = result.raw_gradient
            handle.write(json.dumps(result.metadata, ensure_ascii=False) + "\n")
            handle.flush()
            metadata.append(result.metadata)
            if (index + 1) % 10 == 0:
                hidden.flush()
                gradients.flush()
    hidden.flush()
    gradients.flush()
    return hidden, gradients, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    resolve_paths(config)
    if config.get("gradient_sketch", {}).get("enabled", False):
        raise ValueError("gradient_sketch must be disabled")
    config["lora"]["checkpoint"] = str(args.adapter.resolve())
    seed = int(config["experiment"]["seed"])
    set_global_seed(seed)
    started = time.time()
    math_train = load_math_train(Path(config["data"]["math_train_path"]))
    math500 = load_math500(Path(config["data"]["math500_path"]))
    all_splits = build_fixed_splits(
        math_train, math500, int(config["data"]["predictor_train_size"]),
        int(config["data"]["candidate_test_size"]), seed,
    )
    splits = {role: all_splits[role] for role in ("predictor_train", "candidate_test")}
    resolved_path = args.output / "config_resolved.json"
    if resolved_path.exists():
        if json.loads(resolved_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError("Refusing to resume with a different resolved config")
    else:
        write_json(resolved_path, config)
        for role, rows in splits.items():
            write_jsonl(args.output / "splits" / f"{role}.jsonl",
                        (public_manifest_row(row, role) for row in rows))
        write_json(args.output / "prompt.json", {
            "system_prompt": SYSTEM_PROMPT,
            "user_template": USER_TEMPLATE,
            "assistant_target_field": "solution",
            "loss_masking": "prompt_tokens_are_-100; assistant_solution_tokens_only",
        })
        write_json(args.output / "environment.json", {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "device": args.device, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(args.device),
            "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        })
    model, tokenizer = load_model_and_tokenizer(config, args.device)
    hidden_dim = hidden_input_dim(model, config)
    named_parameters = trainable_lora_parameters(model)
    gradient_dim = sum(parameter.numel() for _, parameter in named_parameters)
    parameter_entries = []
    offset = 0
    for name, parameter in named_parameters:
        factor = "A" if "lora_A" in name else "B"
        stop = offset + parameter.numel()
        parameter_entries.append({
            "name": name,
            "factor": factor,
            "shape": list(parameter.shape),
            "numel": parameter.numel(),
            "flat_start": offset,
            "flat_stop": stop,
        })
        offset = stop
    write_json(args.output / "parameter_layout.json", {
        "hidden_layer_zero_based": int(config["model"]["hidden_layer_zero_based"]),
        "hidden_states_tuple_index": int(config["model"]["hidden_states_tuple_index"]),
        "hidden_dim": hidden_dim,
        "raw_gradient_dim": gradient_dim,
        "parameters": parameter_entries,
    })
    extracted = {}
    for role, rows in splits.items():
        extracted[role] = extract_role(model, tokenizer, rows, role, config, args.device,
                                       args.output, hidden_dim, gradient_dim)
        if any(row["truncated"] for row in extracted[role][2]):
            raise RuntimeError(f"{role}: at least one sample was truncated")
    del model
    torch.cuda.empty_cache()
    train_hidden, train_gradient, train_metadata = extracted["predictor_train"]
    test_hidden, test_gradient, test_metadata = extracted["candidate_test"]
    summary = {
        "status": "passed",
        "experiment": "extract_single_layer_hidden_and_raw_lora_gradient",
        "no_gradient_projection_or_sketch": True,
        "layer_zero_based": int(config["model"]["hidden_layer_zero_based"]),
        "hidden_states_tuple_index": int(config["model"]["hidden_states_tuple_index"]),
        "lora": config["lora"],
        "shapes": {
            "train_hidden": list(train_hidden.shape), "train_raw_gradient": list(train_gradient.shape),
            "test_hidden": list(test_hidden.shape), "test_raw_gradient": list(test_gradient.shape),
        },
        "gradient_norms": {
            "train_mean": float(np.mean([row["gradient_norm"] for row in train_metadata])),
            "test_mean": float(np.mean([row["gradient_norm"] for row in test_metadata])),
            "train_A_mean": float(np.mean([row["gradient_a_norm"] for row in train_metadata])),
            "train_B_mean": float(np.mean([row["gradient_b_norm"] for row in train_metadata])),
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output / "extraction_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
