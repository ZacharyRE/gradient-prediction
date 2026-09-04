#!/usr/bin/env python3
"""Audit a frozen single-layer raw-gradient predictor on GSM8K train."""

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
from scipy.stats import spearmanr
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_direct_raw_gradient_ridge import metric_summary  # noqa: E402
from gradient_geometry.compression import row_cosine  # noqa: E402
from gradient_geometry.extraction import (  # noqa: E402
    extract_one,
    load_model_and_tokenizer,
    set_global_seed,
)


def write_json(path: Path, value) -> None:
    def sanitize(item):
        if isinstance(item, dict):
            return {key: sanitize(child) for key, child in item.items()}
        if isinstance(item, list):
            return [sanitize(child) for child in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sample_id(question: str, answer: str) -> str:
    return hashlib.sha256((question.strip() + "\0" + answer.strip()).encode()).hexdigest()


def load_gsm8k(path: Path) -> list[dict]:
    rows = []
    for source_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        raw = json.loads(line)
        question, answer = raw["question"], raw["answer"]
        rows.append(
            {
                "problem": question,
                "solution": answer,
                "sample_id": sample_id(question, answer),
                "source": "openai/grade-school-math:train",
                "source_row_index": source_index,
            }
        )
    return rows


def finite(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def ranking_summary(true_gradient: np.ndarray, predicted_gradient: np.ndarray,
                    anchor: np.ndarray) -> dict:
    true_score = row_cosine(true_gradient, anchor[None, :])
    predicted_score = row_cosine(predicted_gradient, anchor[None, :])
    correlation = spearmanr(true_score, predicted_score)
    top_k_diagnostics = {}
    for fraction in (0.10, 0.25, 0.50):
        k = max(1, int(round(len(true_score) * fraction)))
        true_top = set(np.argpartition(true_score, -k)[-k:].tolist())
        predicted_top = set(np.argpartition(predicted_score, -k)[-k:].tolist())
        selected_true_scores = true_score[list(predicted_top)]
        top_k_diagnostics[str(fraction)] = {
            "k": k,
            "predicted_true_top_overlap": len(true_top & predicted_top),
            "random_expected_overlap": k * k / len(true_score),
            "selected_true_score_mean": float(selected_true_scores.mean()),
            "pool_true_score_mean": float(true_score.mean()),
            "selected_true_score_uplift": float(
                selected_true_scores.mean() - true_score.mean()
            ),
        }
    return {
        "spearman": finite(correlation.statistic),
        "spearman_pvalue": finite(correlation.pvalue),
        "num_candidates": len(true_score),
        "true_score_mean": float(true_score.mean()),
        "true_score_std": float(true_score.std()),
        "predicted_score_mean": float(predicted_score.mean()),
        "predicted_score_std": float(predicted_score.std()),
        "top_k_diagnostics": top_k_diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--gsm8k", type=Path, default=PROJECT_ROOT / "data/GSM8K/train.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--target-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.time()
    experiment = args.experiment.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not 0 < args.target_samples < args.num_samples:
        parser.error("target-samples must be between zero and num-samples")
    all_rows = load_gsm8k(args.gsm8k)
    rng = np.random.default_rng(args.seed)
    selected_indices = rng.choice(len(all_rows), size=args.num_samples, replace=False)
    rows = [all_rows[int(index)] for index in selected_indices]
    roles = ["target_anchor" if i < args.target_samples else "ranking_audit"
             for i in range(args.num_samples)]

    manifest_path = output / "manifest.jsonl"
    expected_manifest = [
        {
            "audit_index": i,
            "role": roles[i],
            "sample_id": row["sample_id"],
            "source": row["source"],
            "source_row_index": row["source_row_index"],
        }
        for i, row in enumerate(rows)
    ]
    if manifest_path.exists():
        if read_jsonl(manifest_path) != expected_manifest:
            raise RuntimeError("Existing manifest does not match requested GSM8K audit split")
    else:
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in expected_manifest:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = json.loads((experiment / "config_resolved.json").read_text(encoding="utf-8"))
    layout = json.loads((experiment / "parameter_layout.json").read_text(encoding="utf-8"))
    hidden_dim = int(layout["hidden_dim"])
    gradient_dim = int(layout["raw_gradient_dim"])
    hidden_path = output / "hidden.npy"
    gradient_path = output / "true_raw_gradients.npy"
    metadata_path = output / "metadata.jsonl"
    metadata = read_jsonl(metadata_path)
    if [row["sample_id"] for row in metadata] != [row["sample_id"] for row in rows[:len(metadata)]]:
        raise RuntimeError("Existing audit metadata does not match the fixed sample order")
    if hidden_path.exists() != gradient_path.exists():
        raise RuntimeError("Incomplete audit array pair")
    if hidden_path.exists():
        hidden = np.load(hidden_path, mmap_mode="r+")
        gradients = np.load(gradient_path, mmap_mode="r+")
        if hidden.shape != (args.num_samples, hidden_dim):
            raise RuntimeError(f"Unexpected hidden shape: {hidden.shape}")
        if gradients.shape != (args.num_samples, gradient_dim):
            raise RuntimeError(f"Unexpected gradient shape: {gradients.shape}")
    else:
        hidden = open_memmap(hidden_path, mode="w+", dtype=np.float32,
                             shape=(args.num_samples, hidden_dim))
        gradients = open_memmap(gradient_path, mode="w+", dtype=np.float32,
                                shape=(args.num_samples, gradient_dim))

    if len(metadata) < args.num_samples:
        set_global_seed(args.seed)
        model, tokenizer = load_model_and_tokenizer(config, args.device)
        with metadata_path.open("a", encoding="utf-8") as handle:
            for index in tqdm(range(len(metadata), args.num_samples), initial=len(metadata),
                              total=args.num_samples, desc="GSM8K frozen-predictor audit"):
                result = extract_one(model, tokenizer, rows[index], config, args.device)
                hidden[index] = result.hidden
                gradients[index] = result.raw_gradient
                item = {"sample_id": rows[index]["sample_id"], **result.metadata}
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                metadata.append(item)
                if (index + 1) % 10 == 0:
                    hidden.flush()
                    gradients.flush()
        hidden.flush()
        gradients.flush()
        del model
        torch.cuda.empty_cache()

    scaler_mean = np.load(experiment / "ridge/scaler_mean.npy")
    scaler_scale = np.load(experiment / "ridge/scaler_scale.npy")
    coefficient = np.load(experiment / "ridge/coefficient.npy", mmap_mode="r")
    intercept = np.load(experiment / "ridge/intercept.npy")
    standardized = (np.asarray(hidden) - scaler_mean) / scaler_scale
    x = torch.from_numpy(standardized.astype(np.float32)).to(args.device)
    weight = torch.from_numpy(np.array(coefficient, copy=True)).to(args.device)
    bias = torch.from_numpy(intercept).to(args.device)
    prediction = (x @ weight + bias).cpu().numpy().astype(np.float32)
    del x, weight, bias
    torch.cuda.empty_cache()
    np.save(output / "predicted_raw_gradients.npy", prediction)

    truth = np.asarray(gradients)
    baseline = np.repeat(intercept[None, :], args.num_samples, axis=0)
    baseline_mse = float(np.mean(np.square(truth.astype(np.float64) - baseline)))
    results = {
        "frozen_ridge": metric_summary(truth, prediction, baseline_mse),
        "math_train_mean_gradient": metric_summary(truth, baseline, baseline_mse),
    }
    factor_results = {}
    for entry in layout["parameters"]:
        factor = entry["factor"]
        block = slice(int(entry["flat_start"]), int(entry["flat_stop"]))
        factor_baseline_mse = float(np.mean(np.square(
            truth[:, block].astype(np.float64) - baseline[:, block]
        )))
        factor_results[factor] = {
            "frozen_ridge": metric_summary(
                truth[:, block], prediction[:, block], factor_baseline_mse
            ),
            "math_train_mean_gradient": metric_summary(
                truth[:, block], baseline[:, block], factor_baseline_mse
            ),
        }

    target_truth = truth[:args.target_samples]
    candidate_truth = truth[args.target_samples:]
    candidate_prediction = prediction[args.target_samples:]
    raw_anchor = target_truth.astype(np.float64).mean(axis=0).astype(np.float32)
    target_norms = np.linalg.norm(target_truth, axis=1, keepdims=True)
    normalized_anchor = (target_truth / np.maximum(target_norms, 1e-12)).mean(axis=0)
    np.save(output / "target_raw_mean_gradient.npy", raw_anchor)
    np.save(output / "target_mean_normalized_gradient.npy", normalized_anchor)
    ranking = {
        "raw_mean_anchor": ranking_summary(candidate_truth, candidate_prediction, raw_anchor),
        "mean_normalized_anchor": ranking_summary(
            candidate_truth, candidate_prediction, normalized_anchor
        ),
    }
    source_summary = json.loads((experiment / "ridge/summary.json").read_text(encoding="utf-8"))
    summary = {
        "status": "passed",
        "audit": "frozen_MATH_trained_gradient_predictor_on_GSM8K_train",
        "source_experiment": str(experiment),
        "gsm8k_path": str(args.gsm8k.resolve()),
        "seed": args.seed,
        "num_samples": args.num_samples,
        "target_anchor_samples": args.target_samples,
        "ranking_audit_samples": args.num_samples - args.target_samples,
        "predictor_was_not_refit": True,
        "selected_alpha": source_summary["selection"]["selected_alpha"],
        "results": results,
        "results_by_lora_factor": factor_results,
        "target_ranking": ranking,
        "hidden_distribution_shift": {
            "standardized_mean_rms": float(np.sqrt(np.mean(standardized.mean(axis=0) ** 2))),
            "standardized_scale_mean": float(standardized.std(axis=0).mean()),
            "fraction_abs_z_greater_than_3": float(np.mean(np.abs(standardized) > 3)),
        },
        "math_held_out_reference": source_summary["results"]["ridge_test"],
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
