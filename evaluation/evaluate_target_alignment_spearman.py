#!/usr/bin/env python3
"""Evaluate whether predicted gradients preserve target-gradient alignment rankings."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from numpy.lib.format import open_memmap
from scipy.stats import spearmanr
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gradient_geometry.compression import row_cosine  # noqa: E402
from gradient_geometry.data import (  # noqa: E402
    build_fixed_splits,
    load_math500,
    load_math_train,
    public_manifest_row,
)
from gradient_geometry.extraction import (  # noqa: E402
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


def resolve_paths(config: dict) -> None:
    for key in ("math_train_path", "math500_path"):
        path = Path(config["data"][key])
        if not path.is_absolute():
            config["data"][key] = str((PROJECT_ROOT / path).resolve())


def extract_target_gradients(config: dict, adapter: Path, experiment: Path, device: str):
    seed = int(config["experiment"]["seed"])
    math_train = load_math_train(Path(config["data"]["math_train_path"]))
    math500 = load_math500(Path(config["data"]["math500_path"]))
    splits = build_fixed_splits(
        math_train, math500, int(config["data"]["predictor_train_size"]),
        int(config["data"]["candidate_test_size"]), seed,
    )
    target_rows = splits["target"]
    output = experiment / "target_alignment"
    output.mkdir(parents=True, exist_ok=True)
    gradient_path = output / "target_validation_raw_gradients.npy"
    metadata_path = output / "target_validation_metadata.jsonl"
    manifest_path = output / "target_validation_manifest.jsonl"
    metadata = read_jsonl(metadata_path)
    if [row["sample_id"] for row in metadata] != [
        row["sample_id"] for row in target_rows[:len(metadata)]
    ]:
        raise RuntimeError("Existing target metadata does not match the fixed target split")
    if not manifest_path.exists():
        write_jsonl(manifest_path, (public_manifest_row(row, "target_validation")
                                    for row in target_rows))
    layout = json.loads((experiment / "parameter_layout.json").read_text(encoding="utf-8"))
    gradient_dim = int(layout["raw_gradient_dim"])
    if gradient_path.exists():
        gradients = np.load(gradient_path, mmap_mode="r+")
        if gradients.shape != (len(target_rows), gradient_dim):
            raise RuntimeError(f"Unexpected target-gradient shape: {gradients.shape}")
    else:
        if metadata:
            raise RuntimeError("Target metadata exists without its gradient array")
        gradients = open_memmap(gradient_path, mode="w+", dtype=np.float32,
                                shape=(len(target_rows), gradient_dim))
    if len(metadata) < len(target_rows):
        config["lora"]["checkpoint"] = str(adapter.resolve())
        set_global_seed(seed)
        model, tokenizer = load_model_and_tokenizer(config, device)
        actual_dim = sum(parameter.numel() for _, parameter in trainable_lora_parameters(model))
        if actual_dim != gradient_dim:
            raise RuntimeError(f"Expected raw gradient dim {gradient_dim}, got {actual_dim}")
        with metadata_path.open("a", encoding="utf-8") as handle:
            for index in tqdm(range(len(metadata), len(target_rows)), initial=len(metadata),
                              total=len(target_rows), desc="extract target validation"):
                result = extract_one(
                    model, tokenizer, target_rows[index], config, device, include_hidden=False
                )
                gradients[index] = result.raw_gradient
                handle.write(json.dumps(result.metadata, ensure_ascii=False) + "\n")
                handle.flush()
                metadata.append(result.metadata)
                if (index + 1) % 10 == 0:
                    gradients.flush()
        gradients.flush()
        del model
        torch.cuda.empty_cache()
    if len(metadata) != len(target_rows):
        raise RuntimeError("Target-gradient extraction is incomplete")
    return gradients, metadata, output


def finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def summary_statistics(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "min": float(values.min()), "median": float(np.median(values)),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.time()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    resolve_paths(config)
    target_gradients, target_metadata, output = extract_target_gradients(
        config, args.adapter, args.experiment, args.device
    )
    target_mean = np.asarray(target_gradients, dtype=np.float64).mean(axis=0).astype(np.float32)
    np.save(output / "target_validation_mean_raw_gradient.npy", target_mean)
    candidate_true = np.load(
        args.experiment / "extraction" / "candidate_test_raw_gradients.npy", mmap_mode="r"
    )
    true_similarity = row_cosine(np.asarray(candidate_true), target_mean[None, :])
    np.save(output / "candidate_true_target_similarity.npy", true_similarity.astype(np.float32))
    predictors = {
        "ridge_shared_alpha": (
            args.experiment / "ridge" / "candidate_ridge_raw_gradient.npy",
            args.experiment / "ridge" / "summary.json",
        ),
        "ridge_factor_alpha": (
            args.experiment / "factor_ridge" / "candidate_factor_ridge_raw_gradient.npy",
            args.experiment / "factor_ridge" / "summary.json",
        ),
        "mlp_width512": (
            args.experiment / "mlp_width512" / "candidate_mlp_raw_gradient.npy",
            args.experiment / "mlp_width512" / "summary.json",
        ),
        "structured_bottleneck_width64": (
            args.experiment
            / "structured_bottleneck_width64"
            / "candidate_structured_bottleneck_raw_gradient.npy",
            args.experiment / "structured_bottleneck_width64" / "summary.json",
        ),
        "ridge_bottleneck_rank64": (
            args.experiment
            / "ridge_bottleneck_rank64"
            / "candidate_ridge_bottleneck_raw_gradient.npy",
            args.experiment / "ridge_bottleneck_rank64" / "summary.json",
        ),
        "separate_ridge_bottlenecks_rank64_64": (
            args.experiment
            / "separate_ridge_bottlenecks_rank64_64"
            / "candidate_separate_ridge_bottlenecks_raw_gradient.npy",
            args.experiment / "separate_ridge_bottlenecks_rank64_64" / "summary.json",
        ),
    }
    results = {}
    for name, (prediction_path, summary_path) in predictors.items():
        if not prediction_path.exists():
            continue
        prediction = np.load(prediction_path, mmap_mode="r")
        if prediction.shape != candidate_true.shape:
            raise RuntimeError(f"{name}: prediction shape {prediction.shape} is incompatible")
        predicted_similarity = row_cosine(np.asarray(prediction), target_mean[None, :])
        correlation = spearmanr(true_similarity, predicted_similarity)
        metric = {
            "spearman": finite_or_none(correlation.statistic),
            "spearman_pvalue": finite_or_none(correlation.pvalue),
            "true_target_similarity": summary_statistics(true_similarity),
            "predicted_target_similarity": summary_statistics(predicted_similarity),
            "definition": (
                "Spearman across candidate_test examples between cosine(true_raw_gradient, "
                "mean_target_validation_raw_gradient) and cosine(predicted_raw_gradient, "
                "mean_target_validation_raw_gradient)"
            ),
        }
        results[name] = metric
        np.save(output / f"candidate_{name}_predicted_target_similarity.npy",
                predicted_similarity.astype(np.float32))
        if summary_path.exists():
            predictor_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            predictor_summary["target_alignment_ranking"] = metric
            write_json(summary_path, predictor_summary)
    summary = {
        "status": "passed",
        "target_validation": {
            "source": "MATH-500",
            "num_samples": len(target_gradients),
            "aggregation": "mean raw LoRA A+B gradient",
            "gradient_shape": list(target_gradients.shape),
            "mean_gradient_norm": float(np.linalg.norm(target_mean)),
            "truncated_samples": int(sum(bool(row["truncated"]) for row in target_metadata)),
            "role_warning": (
                "MATH-500 supplies the target gradient and is not an untouched final benchmark."
            ),
        },
        "candidate_test_samples": len(candidate_true),
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
