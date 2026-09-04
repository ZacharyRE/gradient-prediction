#!/usr/bin/env python3
"""Cache ragged prompt-token hidden sequences for representation ablations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gradient_geometry.data import build_fixed_splits, load_math500, load_math_train  # noqa: E402
from gradient_geometry.extraction import (  # noqa: E402
    extract_prompt_hidden_sequence,
    load_model_and_tokenizer,
    set_global_seed,
)


ROLES = ("predictor_train", "candidate_test")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_data_paths(config: dict) -> None:
    for key in ("math_train_path", "math500_path"):
        path = Path(config["data"][key])
        if not path.is_absolute():
            config["data"][key] = str((PROJECT_ROOT / path).resolve())


def fixed_rows(config: dict) -> dict[str, list[dict]]:
    rows = build_fixed_splits(
        load_math_train(Path(config["data"]["math_train_path"])),
        load_math500(Path(config["data"]["math500_path"])),
        int(config["data"]["predictor_train_size"]),
        int(config["data"]["candidate_test_size"]),
        int(config["experiment"]["seed"]),
    )
    return {role: rows[role] for role in ROLES}


def validate_role_inputs(experiment: Path, role: str, rows: list[dict]) -> tuple[list[dict], np.ndarray]:
    extraction = experiment / "extraction"
    metadata = read_jsonl(extraction / f"{role}_metadata.jsonl")
    manifest = read_jsonl(experiment / "splits" / f"{role}.jsonl")
    expected_ids = [row["sample_id"] for row in rows]
    if len(metadata) != len(rows) or [row["sample_id"] for row in metadata] != expected_ids:
        raise RuntimeError(f"{role}: extraction metadata does not match the fixed split")
    if len(manifest) != len(rows) or [row["sample_id"] for row in manifest] != expected_ids:
        raise RuntimeError(f"{role}: saved manifest does not match the fixed split")
    token_counts = np.asarray([int(row["prompt_token_count"]) for row in metadata], dtype=np.int64)
    if np.any(token_counts <= 0):
        raise RuntimeError(f"{role}: prompt token counts must be positive")
    offsets = np.empty(len(rows) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(token_counts, out=offsets[1:])
    return metadata, offsets


def open_role_cache(
    output: Path, role: str, offsets: np.ndarray, hidden_dim: int
) -> tuple[np.memmap, list[dict]]:
    token_path = output / f"{role}_tokens.npy"
    offset_path = output / f"{role}_offsets.npy"
    progress_path = output / f"{role}_metadata.jsonl"
    progress = read_jsonl(progress_path)
    if len(progress) > len(offsets) - 1:
        raise RuntimeError(f"{role}: cache metadata has too many rows")
    if offset_path.exists():
        saved_offsets = np.load(offset_path)
        if not np.array_equal(saved_offsets, offsets):
            raise RuntimeError(f"{role}: cached offsets do not match extraction metadata")
    else:
        np.save(offset_path, offsets)
    shape = (int(offsets[-1]), hidden_dim)
    if token_path.exists():
        tokens = np.load(token_path, mmap_mode="r+")
        if tokens.shape != shape or tokens.dtype != np.float32:
            raise RuntimeError(f"{role}: incompatible token cache {tokens.shape} {tokens.dtype}")
    else:
        if progress:
            raise RuntimeError(f"{role}: cache metadata exists without token array")
        tokens = open_memmap(token_path, mode="w+", dtype=np.float32, shape=shape)
    return tokens, progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--last-hidden-atol", type=float, default=1e-5)
    parser.add_argument(
        "--expected-module-suffix",
        default=None,
        help="Optional exact guard, e.g. layers.11.self_attn.q_proj.",
    )
    args = parser.parse_args()
    started = time.time()
    experiment = args.experiment.resolve()
    config = json.loads((experiment / "config_resolved.json").read_text(encoding="utf-8"))
    resolve_data_paths(config)
    if config["model"].get("hidden_source") != "target_module_input":
        raise RuntimeError("This ablation requires hidden_source=target_module_input")
    module_suffix = config["model"].get("hidden_module_suffix")
    if args.expected_module_suffix and module_suffix != args.expected_module_suffix:
        raise RuntimeError(
            f"Expected module suffix {args.expected_module_suffix!r}, got {module_suffix!r}"
        )
    layout = json.loads((experiment / "parameter_layout.json").read_text(encoding="utf-8"))
    hidden_dim = int(layout["hidden_dim"])
    rows_by_role = fixed_rows(config)
    output = experiment / "prompt_hidden"
    output.mkdir(parents=True, exist_ok=True)

    role_state = {}
    for role in ROLES:
        _, offsets = validate_role_inputs(experiment, role, rows_by_role[role])
        tokens, progress = open_role_cache(output, role, offsets, hidden_dim)
        existing_last = np.load(
            experiment / "extraction" / f"{role}_hidden.npy", mmap_mode="r"
        )
        if existing_last.shape != (len(rows_by_role[role]), hidden_dim):
            raise RuntimeError(f"{role}: unexpected existing last-hidden shape {existing_last.shape}")
        if [row["sample_id"] for row in progress] != [
            row["sample_id"] for row in rows_by_role[role][: len(progress)]
        ]:
            raise RuntimeError(f"{role}: cached sequence metadata does not match fixed split")
        role_state[role] = {
            "offsets": offsets,
            "tokens": tokens,
            "progress": progress,
            "existing_last": existing_last,
        }

    set_global_seed(int(config["experiment"]["seed"]))
    model, tokenizer = load_model_and_tokenizer(config, args.device)
    try:
        for role in ROLES:
            state = role_state[role]
            offsets = state["offsets"]
            tokens = state["tokens"]
            progress = state["progress"]
            metadata_path = output / f"{role}_metadata.jsonl"
            with metadata_path.open("a", encoding="utf-8") as handle:
                for index in tqdm(
                    range(len(progress), len(rows_by_role[role])),
                    initial=len(progress),
                    total=len(rows_by_role[role]),
                    desc=f"cache {role} prompt hidden",
                ):
                    row = rows_by_role[role][index]
                    sequence = extract_prompt_hidden_sequence(
                        model, tokenizer, row, config, args.device
                    )
                    start, stop = int(offsets[index]), int(offsets[index + 1])
                    if sequence.shape != (stop - start, hidden_dim):
                        raise RuntimeError(
                            f"{role}[{index}]: expected {(stop - start, hidden_dim)}, "
                            f"got {sequence.shape}"
                        )
                    last_max_abs_difference = float(
                        np.max(np.abs(sequence[-1] - state["existing_last"][index]))
                    )
                    if last_max_abs_difference > args.last_hidden_atol:
                        raise RuntimeError(
                            f"{role}[{index}]: H[-1] differs from existing last hidden by "
                            f"{last_max_abs_difference:.6g} > {args.last_hidden_atol:.6g}"
                        )
                    tokens[start:stop] = sequence
                    record = {
                        "sample_id": row["sample_id"],
                        "token_count": stop - start,
                        "flat_start": start,
                        "flat_stop": stop,
                        "last_hidden_max_abs_difference": last_max_abs_difference,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    progress.append(record)
                    if (index + 1) % 10 == 0:
                        tokens.flush()
            tokens.flush()
    finally:
        del model
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    summary = {
        "status": "passed",
        "hidden_source": "target_module_input",
        "hidden_module_suffix": module_suffix,
        "hidden_dim": hidden_dim,
        "storage_dtype": "float32",
        "same_tensor_contract": {
            "last": "H[-1]",
            "mean": "H.mean(axis=0)",
            "second_order": "(H @ A2.T).T @ (H @ B1)",
        },
        "roles": {
            role: {
                "samples": len(rows_by_role[role]),
                "total_tokens": int(role_state[role]["offsets"][-1]),
                "minimum_tokens": int(np.diff(role_state[role]["offsets"]).min()),
                "maximum_tokens": int(np.diff(role_state[role]["offsets"]).max()),
                "last_hidden_max_abs_difference": max(
                    float(row["last_hidden_max_abs_difference"])
                    for row in role_state[role]["progress"]
                ),
            }
            for role in ROLES
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
